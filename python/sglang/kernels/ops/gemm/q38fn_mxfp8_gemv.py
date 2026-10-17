# _q38fn_mxgemv: small-M MXFP8 dense linear as a weight-streaming W8A16 GEMM (SGLANG_Q38FN_MXFP8_GEMV).
# FlashInfer's MXFP8 CUTLASS GEMM runs a 128-row tile and a separate activation quantize for every call; at decode
# row counts (one request's 3 verify rows up to ~32) it computes padding and the quantize is a kernel of its own.
# This reads the e4m3 weight [n, k] and its UE8M0 scales [n, k / 32] (the unswizzled `weight_scale_inv` the CUTLASS
# backend keeps), dequantizes each tile to bf16 -- exact: an e4m3 value times a power of two fits bf16 -- and
# multiplies the bf16 activation on the tensor cores with fp32 accumulation. Split-K partials land in their own slots
# and are summed in a fixed order, so the output is run-to-run repeatable and matches F.linear on the dequantized
# weight up to fp32 summation order. The activation is never quantized, so it differs from the W8A8 CUTLASS path by
# that quantization error.
import torch
import triton
import triton.language as tl

MAX_M = 32


@triton.jit
def _q38fn_mxfp8_gemv_kernel(x_ptr, w_ptr, s_ptr, part_ptr, M, N, K, sxm, swn, ssn, spk, spm,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    om = tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    mm = om < M
    mn = on < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_per = tl.cdiv(tl.cdiv(K, BK), SPLIT_K) * BK
    k_lo = pk * k_per
    for k0 in range(k_lo, k_lo + k_per, BK):
        k = k0 + tl.arange(0, BK)
        mk = k < K
        x = tl.load(x_ptr + om[:, None] * sxm + k[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        w = tl.load(w_ptr + on[:, None] * swn + k[None, :], mask=mn[:, None] & mk[None, :], other=0.0)
        e = tl.load(s_ptr + on[:, None] * ssn + (k[None, :] // 32), mask=mn[:, None] & mk[None, :], other=127)
        scale = tl.exp2(e.to(tl.float32) - 127.0)
        wd = (w.to(tl.float32) * scale).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(wd), acc)
    tl.store(part_ptr + pk * spk + om[:, None] * spm + on[None, :], acc, mask=mm[:, None] & mn[None, :])


@triton.jit
def _q38fn_mxfp8_gemv_reduce(part_ptr, out_ptr, total, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + p * total + i, mask=m, other=0.0)
    tl.store(out_ptr + i, acc.to(out_ptr.dtype.element_ty), mask=m)


def q38fn_mxfp8_gemv_ok(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> bool:
    n, k = weight.shape
    return (
        x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == k
        and 0 < x.numel() // k <= MAX_M
        and weight.dtype == torch.float8_e4m3fn and weight.is_contiguous()
        and scale.dtype == torch.uint8 and scale.dim() == 2 and scale.shape[0] == n
        and scale.shape[1] >= (k + 31) // 32 and scale.stride(1) == 1
        and k % 32 == 0
    )


def _split_k(n: int, k: int, bn: int, bk: int, sms: int) -> int:
    tiles = triton.cdiv(n, bn)
    sk = 1
    while sk < 8 and tiles * sk * 2 <= 2 * sms and triton.cdiv(triton.cdiv(k, bk), sk * 2) >= 1:
        sk *= 2
    return sk


_SMS = {}


def q38fn_mxfp8_gemv(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                     bias=None, bn: int = 64, bk: int = 128) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    m, k = x2.shape
    n = weight.shape[0]
    dev = x2.device.index
    if dev not in _SMS:
        _SMS[dev] = torch.cuda.get_device_properties(x2.device).multi_processor_count
    sk = _split_k(n, k, bn, bk, _SMS[dev])
    bm = 16 if m <= 16 else 32
    parts = torch.empty((sk, m, n), device=x2.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x2.device, dtype=x2.dtype)
    _q38fn_mxfp8_gemv_kernel[(triton.cdiv(n, bn), sk)](
        x2, weight, scale, parts, m, n, k, x2.stride(0), weight.stride(0), scale.stride(0),
        parts.stride(0), parts.stride(1), BM=bm, BN=bn, BK=bk, SPLIT_K=sk, num_warps=4, num_stages=3)
    _q38fn_mxfp8_gemv_reduce[(triton.cdiv(m * n, 1024),)](parts, out, m * n, SPLIT_K=sk, BLOCK=1024)
    if bias is not None:
        out += bias
    return out.view(*lead, n)
