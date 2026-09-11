"""QSA indexer as a tc_piecewise / breakable prefill-graph split op (R2d).

Gated by ``SGLANG_QWEN4_QSA_SPLIT_OP`` (default OFF); with the gate off nothing
in this module is reached and the model's eager indexer path is byte-for-byte
the shipped one.

Why a split op: ``tc_piecewise`` compiles the language model ``fullgraph=True``
and splits the FX graph only at registered split ops, which then run eagerly
between the captured pieces. The compressed QSA indexer (``QSAIndexer``) cannot
be traced -- it reads per-forward host metadata, calls JIT kernels, loops over
row chunks and (without the ``max_position`` hoist) performs
``positions.max().item()`` host syncs -- so it has to be one of those seams,
exactly as DeepSeek's DSA indexer already is
(``layers/attention/dsa/dsa_prefill_cuda_graph.py``).

Output contract (mirrors the DSA op): a split op returns ``None``; the result is
delivered by mutating ``topk_result`` in place. The call site pre-allocates it
at the static, padded shape ``(hidden_states.shape[0], token_topk +
compress_ratio - 1)`` (int32, filled with -1) inside the traced body, so the
downstream captured piece reads it at a fixed address. Only the first
``num_valid_tokens`` rows (the indexer metadata's request mapping) are written;
the attention split op (``radix_attention.unified_attention_with_output``)
already narrows ``topk_indices`` to the real query token count before the
sparse backend sees it, so padded rows never reach a kernel.
"""

from __future__ import annotations

import torch

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()


def qsa_split_op_enabled() -> bool:
    """The R2d gate, read once per call so tests can flip it in-process."""
    return envs.SGLANG_QWEN4_QSA_SPLIT_OP.get()


def is_graph_qsa_split_op_surface(forward_batch) -> bool:
    """True when the QSA indexer must run as a split op for this forward.

    Same predicate as the DSA surface (``dsa/utils.is_graph_dsa_split_op_surface``)
    plus the R2d gate: CUDA, inside a tc_piecewise or breakable prefill graph,
    and a plain (non-speculative) extend. Decode graphs never see the indexer
    through here (the decode runner captures it whole), and speculative extend
    modes keep the eager path.
    """
    return (
        qsa_split_op_enabled()
        and _is_cuda
        and (is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph())
        and forward_batch.forward_mode.is_extend_without_speculative()
    )


def qsa_topk_result_width(indexer) -> int:
    """Row width of the indexer's output: ``token_topk + compress_ratio - 1``.

    Read off the indexer instance so the buffer the call site allocates and the
    tensor ``QSAIndexer.select_prefill_tokens`` produces agree by construction
    (``qsa/kernel.py`` ``expand_qsa_block_indices`` -> ``final_topk``).
    """
    return int(indexer.token_topk) + int(indexer.compress_ratio) - 1


def _pcg_qsa_indexer_prefill_split_fake_impl(
    layer_id: int,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    topk_result: torch.Tensor,
) -> None:
    # The op mutates ``topk_result`` and returns nothing, so shape propagation
    # has nothing to produce; what the fake impl CAN do is refuse a buffer that
    # violates the contract at trace time rather than at replay.
    if topk_result.dtype != torch.int32:
        raise TypeError(
            "pcg_qsa_indexer_prefill_split: topk_result must be int32, got "
            f"{topk_result.dtype}"
        )
    if topk_result.ndim != 2 or topk_result.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "pcg_qsa_indexer_prefill_split: topk_result must be "
            f"[{hidden_states.shape[0]}, K], got {tuple(topk_result.shape)}"
        )
    num_position_tokens = (
        positions.shape[-1] if positions.ndim == 2 else positions.numel()
    )
    if num_position_tokens != hidden_states.shape[0]:
        raise ValueError(
            "pcg_qsa_indexer_prefill_split: positions cover "
            f"{num_position_tokens} tokens against {hidden_states.shape[0]} rows"
        )
    return None


@register_custom_op(
    mutates_args=["topk_result"],
    fake_impl=_pcg_qsa_indexer_prefill_split_fake_impl,
)
@register_split_op()
def pcg_qsa_indexer_prefill_split(
    layer_id: int,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    topk_result: torch.Tensor,
) -> None:
    """Run the whole compressed QSA indexer for one layer as an eager split op.

    Everything the traced body must not contain lives here: the metadata
    fetch (built by the backend's per-forward ``init_forward_metadata``, only
    read here), the indexer forward (projection, key-state store, compression,
    MQA scoring, top-k, block expansion) and the MTP seed capture hook.
    """
    assert _is_cuda, "Internal error: QSA graph dispatch is only supported on CUDA"
    from sglang.srt.layers.attention.qsa.glue import (
        get_qsa_indexer_metadata,
        resolve_qsa_sparse_backend,
    )

    forward_context = get_tc_piecewise_forward_context()
    if forward_context is None:
        raise RuntimeError(
            "pcg_qsa_indexer_prefill_split called outside a prefill graph "
            "forward context"
        )
    forward_batch = forward_context.forward_batch
    indexers = forward_context.dsa_indexers
    indexer = indexers[layer_id] if indexers is not None else None
    if indexer is None:
        raise RuntimeError(
            f"pcg_qsa_indexer_prefill_split: no QSA indexer registered for layer "
            f"{layer_id}; layer_setup collects `layer.indexer` only when "
            "SGLANG_QWEN4_QSA_SPLIT_OP is set at model-runner init"
        )
    backend = get_attn_backend()
    indexer_metadata = get_qsa_indexer_metadata(backend, layer_id, forward_batch)

    topk_indices = indexer(hidden_states, positions, forward_batch, indexer_metadata)

    rows = topk_indices.shape[0]
    if rows > topk_result.shape[0] or topk_indices.shape[1] != topk_result.shape[1]:
        raise ValueError(
            "pcg_qsa_indexer_prefill_split: indexer produced "
            f"{tuple(topk_indices.shape)} against a {tuple(topk_result.shape)} "
            "result buffer"
        )
    if rows:
        topk_result[:rows].copy_(topk_indices)
    if rows < topk_result.shape[0]:
        topk_result[rows:].fill_(-1)

    sparse_backend = resolve_qsa_sparse_backend(backend)
    should_capture = getattr(sparse_backend, "should_capture_mtp_sparse_indices", None)
    if should_capture is not None and should_capture(forward_batch):
        sparse_backend.capture_mtp_sparse_indices(
            topk_indices, forward_batch, layer_id, metadata=indexer_metadata
        )


__all__ = [
    "is_graph_qsa_split_op_surface",
    "pcg_qsa_indexer_prefill_split",
    "qsa_split_op_enabled",
    "qsa_topk_result_width",
]
