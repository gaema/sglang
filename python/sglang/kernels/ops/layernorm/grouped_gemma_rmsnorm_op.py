"""Grouped Gemma RMSNorm as a registered torch custom op.

R2d (SGLANG_QWEN4_QSA_SPLIT_OP, fn:N285 obstacle 3): `grouped_gemma_rmsnorm`
calls the tvm_ffi JIT module's `Function` directly, and Dynamo cannot trace
that call (`Dynamo does not know how to trace method __call__ of class
Function`), so the hyper-connection norm broke every tc_piecewise prefill
capture of Qwen4Exp. The kernel is shape- and dtype-preserving, so its fake
impl is exactly `empty_like(input)`; the eager impl is the shipped wrapper,
untouched. This module is imported ONLY by a model that opted in (the norm's
`use_traceable_kernel` flag), so the shipped path never registers the op.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.srt.utils.custom_op import register_custom_op


def _grouped_gemma_rmsnorm_fake_impl(
    input: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    if input.size(-1) % group_size != 0:
        raise ValueError(
            f"grouped_gemma_rmsnorm_op: hidden {input.size(-1)} is not a "
            f"multiple of group_size {group_size}"
        )
    if weight.ndim != 1 or weight.shape[0] != input.size(-1):
        raise ValueError(
            f"grouped_gemma_rmsnorm_op: weight must be [{input.size(-1)}], "
            f"got {tuple(weight.shape)}"
        )
    return torch.empty_like(input)


@register_custom_op(mutates_args=[], fake_impl=_grouped_gemma_rmsnorm_fake_impl)
def grouped_gemma_rmsnorm_op(
    input: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    return grouped_gemma_rmsnorm(input, weight, group_size, eps)
