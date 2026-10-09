# _q38fn_bagemm: small-M GEMM for the GDN in_proj_ba projection
# (tools/gdn_ba_gemm_patch.py in the qwen3-8-flash-next model dir).
import torch
import triton
import triton.language as tl


@triton.jit
def _q38fn_ba_gemm_kernel(x_ptr, w_ptr, o_ptr, M, N, K, sxm, swn, som,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          STAGES: tl.constexpr):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    mm = om < M
    mn = on < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        k = k0 + ok
        mk = k < K
        x = tl.load(x_ptr + om[:, None] * sxm + k[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        w = tl.load(w_ptr + on[:, None] * swn + k[None, :], mask=mn[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(o_ptr + om[:, None] * som + on[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


def q38fn_ba_gemm_ok(x: torch.Tensor, layer) -> bool:
    w = getattr(layer, "weight", None)
    return (
        w is not None
        and getattr(layer, "bias", None) is None
        and x.is_cuda and x.dim() == 2 and 0 < x.shape[0] <= 256
        and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
        and x.is_contiguous() and w.is_contiguous() and w.shape[1] == x.shape[1]
    )


def q38fn_ba_gemm(x: torch.Tensor, w: torch.Tensor, bn: int = 16, bk: int = 128,
                  warps: int = 4, stages: int = 3) -> torch.Tensor:
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    _q38fn_ba_gemm_kernel[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
        x, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk, STAGES=stages, num_warps=warps,
    )
    return out
