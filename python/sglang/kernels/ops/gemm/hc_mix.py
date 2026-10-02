"""Fused HC low-rank mix for decode-size batches.

One persistent kernel replaces the five-kernel `GatedResidual._mix_compute` chain.
One CTA per SM keeps every CTA resident, so the software grid barrier cannot deadlock;
the last CTA to finish resets the barrier counters,
so a captured CUDA graph replays with them in their initial state.
Row counts beyond ``_FUSED_MIX_MAX_ROWS`` stay on the torch.compile path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FUSED_MIX_MAX_ROWS = 16

# N53 -- ENV-GATED row gate, DEFAULT UNCHANGED (unset = 16 = upstream). The DP
# decode batch is 128 rows/rank; raising this sends those mixes to the
# persistent kernel instead of the five-kernel cuBLAS/compile chain.
import os as _n53_os
_n53_rows = _n53_os.environ.get("SGLANG_HC_FUSED_MIX_MAX_ROWS", "")
if _n53_rows:
    _FUSED_MIX_MAX_ROWS = int(_n53_rows)
    # Readback: one line per importing process, so a server log proves the gate
    # reached the scheduler; the launcher prints its first call's row count too.
    print(f"[N53] hc_mix_triton gate raised: _FUSED_MIX_MAX_ROWS={_FUSED_MIX_MAX_ROWS} pid={_n53_os.getpid()}", flush=True)
_n53_first_call = [True]
_n53_seen_shapes = set()


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _hc_mix_persistent_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    # R2d (fn:N286 obstacle 6): named `num_ctas` until 2026-09-11. That is a
    # Triton LAUNCH option name; positional eager launches never collide, but
    # a Dynamo-captured launch (`triton_kernel_wrapper_mutation`) re-issues
    # every argument by keyword and the binder then sees `num_ctas` twice.
    # A rename of a kernel parameter changes no generated code.
    grid_ctas,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    zero_span = ROWS * LOWRANK
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, grid_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    _grid_barrier(counters_ptr + 0, grid_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, grid_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = tl.dot(xt, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, grid_ctas)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, grid_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == grid_ctas - 1:
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


_counters_cache = {}


def _get_counters(device: torch.device) -> torch.Tensor:
    buf = _counters_cache.get(device)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[device] = buf
    return buf


def _deterministic_inference() -> bool:
    from sglang.srt.runtime_context import get_exec

    try:
        exec_cfg = get_exec()
    except ValueError:
        return False
    return bool(exec_cfg.deterministic.enable_deterministic_inference)


def fused_hc_mix_supported(
    hyper_input_normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    # The persistent kernel accumulates the down projection with
    # device-scope atomics, so summation order varies across replays.
    if _deterministic_inference():
        return False
    if _n53_rows and tuple(hyper_input_normed.shape) not in _n53_seen_shapes:
        _n53_seen_shapes.add(tuple(hyper_input_normed.shape))
        # Readback of the first call per SHAPE this process sees, accepted or
        # declined -- a rank whose 128-row shape never reaches "fused_hc_mix
        # first call" is declining it on one of the predicates below.
        print(f"[N53] fused_hc_mix_supported new shape: shape={tuple(hyper_input_normed.shape)} "
              f"dtype={hyper_input_normed.dtype} contig={hyper_input_normed.is_contiguous()} "
              f"dim={hyper_input_normed.dim()} gate={_FUSED_MIX_MAX_ROWS} pid={_n53_os.getpid()}", flush=True)
    return (
        hyper_input_normed.is_cuda
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_down.dtype == hyper_input_normed.dtype
        and w_up.dtype == hyper_input_normed.dtype
        and hyper_input_normed.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input_normed.dim() == 2
        and hyper_input_normed.shape[1] % 2048 == 0
        and hyper_input_normed.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


def fused_hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    # N53: ROWS is a tl.arange extent, so pad to the next power of two >= 16.
    # With the gate at upstream's 16 this loop never runs (rows <= 16).
    while rows_pad < rows:
        rows_pad *= 2
    if _n53_rows and _n53_first_call[0]:
        _n53_first_call[0] = False
        print(f"[N53] fused_hc_mix first call: rows={rows} rows_pad={rows_pad} pid={_n53_os.getpid()}", flush=True)
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    _hc_mix_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        t_raw,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=32,
        # N53: at ROWS > 16 the [ROWS, HC*BLOCK_J] accumulator must shrink or the
        # CTA runs out of shared memory (BLOCK_J=32 at 128 rows: 114688 B needed
        # against 101376); BLOCK_K=128 was the best of a 7-config sweep at 128
        # rows (19.23 us vs 25.95 at 64 and 31.25 at 32). 16 rows keep upstream's.
        BLOCK_K=256 if rows_pad <= 16 else 128,
        BLOCK_J=32 if rows_pad <= 16 else 16,
        BLOCK_R=64,
        num_warps=8,
    )
    return out
