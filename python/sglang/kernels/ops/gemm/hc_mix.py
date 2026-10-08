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
    w_gate_ptr,  # _q38fn_fold: [GATE, K] inject weight (unused when GATE == 0)
    partials_ptr,  # _q38fn_fold: [rows, SPLIT, GATE] combine partials
    sd_ptr,  # _q38fn_mixfp8: [LOWRANK] fp32 down-projection row scales
    su_ptr,  # _q38fn_mixfp8: [HC * HS] fp32 up-projection row scales
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
    GATE: tl.constexpr = 0,  # _q38fn_fold
    WFP8: tl.constexpr = 0,  # _q38fn_mixfp8
    S1: tl.constexpr = 1,  # _q38fn_mixtune: phase-1 pipeline stages
    S2: tl.constexpr = 1,  # _q38fn_mixtune: phase-2 pipeline stages
    SPLIT: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    zero_span = ROWS * (LOWRANK + GATE)  # _q38fn_fold
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, grid_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    _grid_barrier(counters_ptr + 0, grid_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks_down = tl.cdiv(LOWRANK, BLOCK_N)  # _q38fn_fold
    n_blocks = n_blocks_down + (1 if GATE > 0 else 0)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in tl.range(pid, n_blocks * k_chunks, grid_ctas, num_stages=S1):  # _q38fn_mixtune
        nb = tile % n_blocks
        kc = tile // n_blocks
        k = kc * BLOCK_K + offs_k
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        if nb < n_blocks_down:
            n = nb * BLOCK_N + offs_n
            mask_n = n < LOWRANK
            w = tl.load(
                w_down_ptr + n[:, None] * K + k[None, :],
                mask=mask_n[:, None],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)  # _q38fn_mixfp8: fp8 -> bf16 in registers
        else:
            n = LOWRANK + offs_n
            mask_n = offs_n < GATE
            w = tl.load(
                w_gate_ptr + offs_n[:, None] * K + k[None, :],
                mask=mask_n[:, None],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)  # _q38fn_mixfp8: GATE == 0 passes fp8 w_down here
        acc = tl.dot(xt, tl.trans(w))
        if WFP8 > 0:  # _q38fn_mixfp8: per-row scale of the down projection
            if nb < n_blocks_down:
                acc = acc * tl.load(sd_ptr + n, mask=mask_n, other=0.0)[None, :]
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * (LOWRANK + GATE) + n[None, :],
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
        for r0 in tl.range(0, LOWRANK, BLOCK_R, num_stages=S2):  # _q38fn_mixtune
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * (LOWRANK + GATE) + r[None, :],  # _q38fn_fold
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)  # _q38fn_mixfp8
            acc = tl.dot(t, tl.trans(w), acc)
        if WFP8 > 0:  # _q38fn_mixfp8: per-row scale of the up projection
            acc = acc * tl.load(su_ptr + gj_flat, mask=mask_gj, other=0.0)[None, :]
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
        if GATE > 0:  # _q38fn_fold
            offs_c = tl.arange(0, GATE)
            logits = tl.load(
                t_raw_ptr + offs_m[:, None] * (LOWRANK + GATE) + LOWRANK + offs_c[None, :]
            )
            base = offs_m[:, None] * (SPLIT * GATE) + offs_c[None, :]
            tl.store(partials_ptr + base, logits, mask=mask_m[:, None])
            for s in tl.static_range(1, SPLIT):
                tl.store(
                    partials_ptr + base + s * GATE,
                    tl.zeros((ROWS, GATE), dtype=tl.float32),
                    mask=mask_m[:, None],
                )
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


_counters_cache = {}


def q38fn_quant_rows_fp8(w: torch.Tensor):  # _q38fn_mixfp8
    """Per-row e4m3: scale = amax(row) / 448; returns (q, scale fp32)."""
    wf = w.float()
    s = wf.abs().amax(dim=1).clamp_min(1e-12) / 448.0
    q = (wf / s[:, None]).to(torch.float8_e4m3fn)
    return q.contiguous(), s.contiguous()


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
    w_gate: torch.Tensor = None,  # _q38fn_fold
    partials: torch.Tensor = None,
    w_down_scale: torch.Tensor = None,  # _q38fn_mixfp8: w_down/w_up are e4m3 when set
    w_up_scale: torch.Tensor = None,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    gate = int(w_gate.shape[0]) if w_gate is not None else 0  # _q38fn_fold
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
    t_raw = torch.empty((rows_pad, lowrank + gate), dtype=torch.float32, device=device)  # _q38fn_fold
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    _tc = _n53_os.environ.get("SGLANG_Q38FN_MIX_CFG", "")  # _q38fn_mixtune
    _tc = [int(v) for v in _tc.split(",")] if _tc else None
    _hc_mix_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        t_raw,
        out,
        _get_counters(device),
        w_gate if w_gate is not None else w_down,  # _q38fn_fold
        partials if partials is not None else t_raw,
        w_down_scale if w_down_scale is not None else t_raw,  # _q38fn_mixfp8
        w_up_scale if w_up_scale is not None else t_raw,
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=_tc[1] if _tc else 32,  # _q38fn_mixtune
        # N53: at ROWS > 16 the [ROWS, HC*BLOCK_J] accumulator must shrink or the
        # CTA runs out of shared memory (BLOCK_J=32 at 128 rows: 114688 B needed
        # against 101376); BLOCK_K=128 was the best of a 7-config sweep at 128
        # rows (19.23 us vs 25.95 at 64 and 31.25 at 32). 16 rows keep upstream's.
        BLOCK_K=_tc[0] if _tc else ((128 if _q38fn_sl_hc_v2(rows_pad) else 256) if rows_pad <= 16 else 128),  # _q38fn_mixtune
        BLOCK_J=_tc[2] if _tc else ((16 if _q38fn_sl_hc_v2(rows_pad) else 32) if rows_pad <= 16 else 16),
        BLOCK_R=_tc[3] if _tc else 64,
        S1=_tc[5] if _tc else 1,
        S2=_tc[6] if _tc else 1,
        GATE=gate,  # _q38fn_fold
        WFP8=1 if w_down_scale is not None else 0,  # _q38fn_mixfp8
        num_warps=_tc[4] if _tc else (4 if _q38fn_sl_hc_v2(rows_pad) else 8),  # _q38fn_mixtune
    )
    return out


_q38fn_sl_hc_said = [False]


def _q38fn_sl_hc_v2(rows_pad):  # _q38fn_sl
    on = _n53_os.environ.get("SGLANG_HC_MIX_TILE_V2", "1") == "1" and rows_pad <= 16
    if on and not _q38fn_sl_hc_said[0]:
        _q38fn_sl_hc_said[0] = True
        print(f"[q38fn-sl] hc_mix tile v2 on: BLOCK_K=128 BLOCK_J=16 num_warps=4 pid={_n53_os.getpid()}", flush=True)
    return on
