# _q38fn_gdnq: GDN gated RMSNorm (sigmoid gate, norm before gate, one group =
# one 128-wide head) fused with the 128-group FP8 activation quant
# (tools/gdn_out_bundle_patch.py in the qwen3-8-flash-next model dir).
import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_norm_quant_kernel(X, Z, W, Q, S, M, eps,
                           N: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, N)
    rmask = rows < M
    mask = rmask[:, None]
    off = rows[:, None] * N + cols[None, :]
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    # the FLA kernel: var over the row, rstd, x_hat * w, then * sigmoid(z)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols).to(tl.float32)
    y = x * rstd[:, None]
    y = y * w[None, :]
    z = tl.load(Z + off, mask=mask, other=0.0).to(tl.float32)
    y *= tl.sigmoid(z)
    # the norm writes bf16; the quant reads it back
    y = y.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10)
    scale = amax * (1.0 / 448.0)
    qs = tl.math.div_rn(tl.full(amax.shape, 448.0, tl.float32), amax)
    q = tl.minimum(y * qs[:, None], 448.0)
    tl.store(Q + off, q.to(tl.float8e4nv), mask=mask)
    tl.store(S + rows, scale, mask=rmask)


def gdn_norm_quant(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float):
    m, n = x.shape
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    rows = 4
    _gdn_norm_quant_kernel[(triton.cdiv(m, rows),)](
        x, z, weight, q, s, m, float(eps), N=n, ROWS=rows, num_warps=1,
        enable_fp_fusion=False,
    )
    return q, s
