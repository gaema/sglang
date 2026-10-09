# _q38fn_lmfp8: per-row e4m3 lm_head for decode-sized logits
# (tools/lm_head_fp8_patch.py in the qwen3-8-flash-next model dir).
import torch
import triton
import triton.language as tl


@triton.jit
def _q38fn_lm_head_fp8_kernel(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, sxm, swn, som,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                              STAGES: tl.constexpr):
    pn = tl.program_id(0)
    om = tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    mm = om < M
    mn = on < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        k = k0 + ok
        x = tl.load(x_ptr + om[:, None] * sxm + k[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + on[:, None].to(tl.int64) * swn + k[None, :], mask=mn[:, None],
                    other=0.0).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc)
    acc = acc * tl.load(s_ptr + on, mask=mn, other=0.0)[None, :]
    tl.store(o_ptr + om[:, None] * som + on[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


def q38fn_quant_rows_e4m3(w: torch.Tensor):
    out_q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    out_s = torch.empty((w.shape[0],), dtype=torch.float32, device=w.device)
    step = 8192  # bound the fp32 temporaries
    for r0 in range(0, w.shape[0], step):
        wf = w[r0:r0 + step].float()
        s = wf.abs().amax(dim=1).clamp_min(1e-12) / 448.0
        out_q[r0:r0 + step] = (wf / s[:, None]).to(torch.float8_e4m3fn)
        out_s[r0:r0 + step] = s
    return out_q, out_s


def q38fn_lm_head_fp8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor,
                      bn: int = 64, bk: int = 256, warps: int = 4, stages: int = 3) -> torch.Tensor:
    m, k = x.shape
    n = q.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    _q38fn_lm_head_fp8_kernel[(triton.cdiv(n, bn),)](
        x, q, s, out, m, n, k, x.stride(0), q.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk, STAGES=stages, num_warps=warps,
    )
    return out


def q38fn_lm_head_logits(hidden_states: torch.Tensor, lm_head):
    w = getattr(lm_head, "weight", None)
    x = hidden_states
    if (
        w is None or w.dtype != torch.bfloat16 or not w.is_contiguous()
        or x.dim() != 2 or not (0 < x.shape[0] <= 64) or x.shape[1] != w.shape[1]
        or w.shape[1] % 256 != 0
    ):
        return None
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if not x.is_contiguous():
        x = x.contiguous()
    cache = getattr(lm_head, "_q38fn_fp8", None)
    if cache is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        cache = q38fn_quant_rows_e4m3(w)
        lm_head._q38fn_fp8 = cache
        print(f"[q38fn-lmfp8] fp8 lm_head built: {tuple(w.shape)}", flush=True)
    return q38fn_lm_head_fp8(x, cache[0], cache[1])
