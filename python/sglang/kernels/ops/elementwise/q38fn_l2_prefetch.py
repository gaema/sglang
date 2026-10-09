# _q38fn_l2pf: warm L2 with the NEXT layer's attention-side HC mix weights while
# the main stream waits on the routed-MoE chain (tools/l2_prefetch_patch.py +
# tools/q38fn_l2_prefetch.py in the qwen3-8-flash-next model dir; installed into a
# worktree as sglang/kernels/ops/elementwise/q38fn_l2_prefetch.py).
import os

import torch
import triton
import triton.language as tl

_ATTN_HC = {}  # layer_id -> that layer's attn_hyper_connection (first registration wins)
_SINK = {}
_CTAS = int(os.environ.get("SGLANG_Q38FN_L2PF_CTAS", "48") or 48)


def q38fn_l2pf_enabled() -> bool:
    return os.environ.get("SGLANG_Q38FN_L2PF", "0") == "1"


def q38fn_l2pf_register(layer, layer_id) -> None:
    """Called once per decoder layer at construction."""
    if layer_id is None or layer_id in _ATTN_HC:
        return
    hc = getattr(layer, "attn_hyper_connection", None)
    if hc is None:
        return
    _ATTN_HC[layer_id] = hc
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        mlp._q38fn_l2pf_layer = layer_id


@triton.jit
def _q38fn_l2_touch_kernel(src_ptr, n_words, sink_ptr, BLOCK: tl.constexpr, NCTA: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for start in range(pid * BLOCK, n_words, NCTA * BLOCK):
        idx = start + offs
        acc = acc ^ tl.load(src_ptr + idx, mask=idx < n_words, other=0,
                            eviction_policy="evict_last")
    # One store keeps the loads live; nothing reads the sink.
    tl.store(sink_ptr + pid, tl.reduce(acc, 0, _xor))


@triton.jit
def _xor(a, b):
    return a ^ b


def q38fn_l2_touch(*tensors) -> None:
    for t in tensors:
        if t is None or not t.is_cuda or not t.is_contiguous() or (t.numel() * t.element_size()) % 4:
            continue
        sink = _SINK.get(t.device)
        if sink is None:
            sink = torch.zeros(1024, dtype=torch.int32, device=t.device)
            _SINK[t.device] = sink
        words = t.view(torch.int32) if t.dim() == 1 else t.reshape(-1).view(torch.int32)
        _q38fn_l2_touch_kernel[(_CTAS,)](words, words.numel(), sink, BLOCK=2048, NCTA=_CTAS,
                                         num_warps=4)


def q38fn_l2pf_next(moe_block) -> None:
    """Touch the next layer's FP8 HC mix weights (no-op until they exist)."""
    lid = getattr(moe_block, "_q38fn_l2pf_layer", None)
    if lid is None:
        return
    hc = _ATTN_HC.get(lid + 1)
    w = getattr(hc, "_q38fn_fp8w", None) if hc is not None else None
    if w is not None:
        q38fn_l2_touch(w[0], w[2])
