# N54 milestone 1 -- fused GDN chunk kernel: chunk_h + chunk_o (+ optionally
# recompute_w_u) in ONE Triton kernel, so the per-chunk state h and the
# corrected values v_new never leave the SM.
#
# Installed into the sglang fork by gdn_fused_h_o_patch.py as
#   sglang/kernels/ops/attention/fla/chunk_delta_h_o.py
# and selected from chunk.py ONLY when SGLANG_GDN_FUSED_HO=1 (unset = upstream
# path byte-for-byte). The upstream three-kernel chain it replaces, for one
# sequence of T tokens, H value heads, Hg key heads, K/V head dims, chunk BT:
#   recompute_w_u   w = A (k*beta*exp(g)),  u = A (v*beta)          [wy_fast.py]
#   chunk_h         per chunk: v_new = u - w h^T ; h <- exp(g_last) h + k^T (v_new*exp(g_last-g))
#                   stores h (every chunk) and v_new                  [chunk_delta_h.py]
#   chunk_o         o = scale*( (q h^T)*exp(g) + tril((q k^T)*exp(g_i-g_j)) v_new )   [chunk_o.py]
# This kernel keeps the chunk_h grid (V/BV, N*H) and its serial recurrence over
# chunks; at every chunk step the CTA already holds h (pre-update) and v_new in
# registers, so it computes its [BT, BV] tile of o there. With FUSE_WU it also
# forms w and u from A, k, v, beta, g instead of reading them (w is then formed
# once per CTA, i.e. V/BV times per head -- compute traded for bytes).
#
# Contract kept: h (optional, STORE_H), v_new (optional), o, and the final state
# written IN PLACE into initial_state exactly as chunk_h does -- which is why
# this kernel, like chunk_h, must never be autotuned over multiple configs (the
# benchmark phase would re-run the in-place state update; see chunk_delta_h.py).
# Scalar g only (no gk), K <= 128 (two 64-wide state blocks), V % BV == 0; the
# wrapper falls back to the unfused chain otherwise.
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.op import exp, safe_exp

CHUNK_SIZE = 64
GDN_FUSED_HO_BV = int(os.getenv("SGLANG_GDN_FUSED_HO_BV", "32"))
GDN_FUSED_HO_NUM_WARPS = int(os.getenv("SGLANG_GDN_FUSED_HO_NUM_WARPS", "4"))
GDN_FUSED_HO_NUM_STAGES = int(os.getenv("SGLANG_GDN_FUSED_HO_NUM_STAGES", "2"))
GDN_FUSED_HO_FUSE_WU = os.getenv("SGLANG_GDN_FUSED_HO_FUSE_WU", "0") == "1"
# Measured (gdn_fused_h_o_bench.py): folding chunk_o in is SLOWER than the chain
# (255 regs + spills, chunk_o's parallel CTAs serialised into the recurrence),
# so the gate's default keeps chunk_o as upstream's separate kernel.
GDN_FUSED_HO_FUSE_O = os.getenv("SGLANG_GDN_FUSED_HO_FUSE_O", "0") == "1"
GDN_FUSED_HO_STORE_H = os.getenv("SGLANG_GDN_FUSED_HO_STORE_H", "1") == "1"


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_o(
    q,
    k,
    v,
    w,
    u,
    A,
    beta,
    v_new,
    g,
    h,
    o,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STORE_H: tl.constexpr,
    FUSE_WU: tl.constexpr,
    FUSE_O: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, 64] state blocks (K <= 128)
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    if FUSE_O:
        q += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
        o += ((bos * H + i_h) * V).to(tl.int64)
    if FUSE_WU:
        A += ((bos * H + i_h) * BT).to(tl.int64)
        beta += (bos * H + i_h).to(tl.int64)
    else:
        w += ((bos * H + i_h) * K).to(tl.int64)
        u += ((bos * H + i_h) * V).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    g += (bos * H + i_h).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    h0 = initial_state + index * stride_init_state
    ht = initial_state + index * stride_init_state
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    if USE_INITIAL_STATE and valid_state:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]

    for i_t in range(NT):
        if STORE_H:
            p_h1 = tl.make_block_ptr(h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        # gates for this chunk
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * H)
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        # k blocks, [64, BT] (the layout the state update wants)
        p_k1 = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        if K > 64:
            p_k2 = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))

        # w = A (k*beta*exp(g)) and u = A (v*beta), or read them
        if FUSE_WU:
            p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            b_A_wu = tl.load(p_A, boundary_check=(0, 1))
            p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_bg = b_beta * tl.exp(b_g)
            b_w1 = tl.dot(b_A_wu, (tl.trans(b_k1) * b_bg[:, None]).to(b_k1.dtype))
            if K > 64:
                b_w2 = tl.dot(b_A_wu, (tl.trans(b_k2) * b_bg[:, None]).to(b_k2.dtype))
            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_vin = tl.load(p_v, boundary_check=(0, 1))
            b_u = tl.dot(b_A_wu, (b_vin * b_beta[:, None]).to(b_vin.dtype), allow_tf32=False)
            b_w1 = b_w1.to(b_k1.dtype)
            if K > 64:
                b_w2 = b_w2.to(b_k1.dtype)
            b_u = b_u.to(b_k1.dtype).to(tl.float32)
        else:
            p_w1 = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w1 = tl.load(p_w1, boundary_check=(0, 1))
            if K > 64:
                p_w2 = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            p_u = tl.make_block_ptr(u, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_u = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)

        # v_new = u - w h^T   (h pre-update)
        b_v = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))
        if K > 64:
            b_v += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))
        b_v = b_u - b_v
        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))
        if FUSE_O:
            b_vb = b_v.to(b_k1.dtype)  # the bf16 v_new chunk_o would have read back
            # o tile = scale*( (q h^T)*exp(g) + tril((q k^T)*exp(g_i-g_j)) v_new )
            p_q1 = tl.make_block_ptr(q, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_q1 = tl.load(p_q1, boundary_check=(0, 1))
            b_o = tl.dot(b_q1, tl.trans(b_h1).to(b_q1.dtype))
            b_A = tl.dot(b_q1, b_k1)
            if K > 64:
                p_q2 = tl.make_block_ptr(q, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_q2 = tl.load(p_q2, boundary_check=(0, 1))
                b_o += tl.dot(b_q2, tl.trans(b_h2).to(b_q2.dtype))
                b_A += tl.dot(b_q2, b_k2)
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])
            b_A = tl.where(m_A, b_A, 0)
            b_o = b_o * scale + tl.dot(b_A.to(b_vb.dtype), b_vb) * scale
            p_o = tl.make_block_ptr(o, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # state update
        b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
        b_g_last = exp(b_g_last)
        b_h1 = b_h1 * b_g_last
        if K > 64:
            b_h2 = b_h2 * b_g_last
        b_v = b_v.to(b_k1.dtype)
        b_h1 += tl.trans(tl.dot(b_k1, b_v))
        if K > 64:
            b_h2 += tl.trans(tl.dot(b_k2, b_v))

    if INPLACE_UPDATE and valid_state:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def fused_h_o_supported(k: torch.Tensor, g, gk) -> bool:
    K = k.shape[-1]
    return g is not None and gk is None and K <= 128 and K % 64 == 0


def chunk_gated_delta_rule_fwd_h_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    w: Optional[torch.Tensor],
    u: Optional[torch.Tensor],
    scale: float,
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    save_new_value: bool = True,
    store_h: bool = GDN_FUSED_HO_STORE_H,
    fuse_wu: bool = GDN_FUSED_HO_FUSE_WU,
    fuse_o: bool = GDN_FUSED_HO_FUSE_O,
    BV: int = GDN_FUSED_HO_BV,
    num_warps: int = GDN_FUSED_HO_NUM_WARPS,
    num_stages: int = GDN_FUSED_HO_NUM_STAGES,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Returns (o, h, v_new). o is None unless fuse_o (the caller then runs
    chunk_fwd_o on h and v_new, as upstream does); h is the per-chunk state
    tensor chunk_h returns (garbage-free only when store_h); v_new is None
    unless save_new_value."""
    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = CHUNK_SIZE
    assert fused_h_o_supported(k, g, None)
    assert V % BV == 0, "fused h+o needs V % BV == 0"
    if not fuse_wu:
        assert w is not None and u is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(v) if (save_new_value or not fuse_o) else None
    o = torch.empty_like(v) if fuse_o else None
    grid = (V // BV, N * H)
    chunk_gated_delta_rule_fwd_kernel_h_o[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        u=u,
        A=A,
        beta=beta,
        v_new=v_new,
        g=g,
        h=h,
        o=o,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        stride_init_state=(initial_state.stride(0) if initial_state is not None else 0),
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=v_new is not None,
        STORE_H=store_h,
        FUSE_WU=fuse_wu,
        FUSE_O=fuse_o,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, h, v_new


def fused_gdn_fwd(q, k, v, g, beta, scale, initial_state, initial_state_indices,
                  cu_seqlens=None, chunk_indices=None):
    """Drop-in for the body of chunk.py's chunk_gated_delta_rule_fwd AFTER the
    g cumsum, selected only under SGLANG_GDN_FUSED_HO=1. Runs the fork's
    kkt+solve kernel for A, then the fused kernel with recompute_w_u folded in
    (w and u are never materialised; the returned w slot is None), and
    chunk_fwd_o separately unless SGLANG_GDN_FUSED_HO_FUSE_O=1. Falls back to
    the upstream chain for shapes the fused kernel does not cover."""
    from sglang.kernels.ops.attention.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_intra,
        chunk_gated_delta_rule_fwd_kkt_solve_kernel,
    )
    from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o

    BT = CHUNK_SIZE
    if not fused_h_o_supported(k, g, None) or v.shape[-1] % GDN_FUSED_HO_BV != 0:
        w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
        h, v_new = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=initial_state, initial_state_indices=initial_state_indices, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens)
        return g, o, A, w, h, v_new
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k, g=g, beta=beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, Hg=Hg, K=K, BT=BT, BC=16,
    )
    o, h, v_new = chunk_gated_delta_rule_fwd_h_o(
        q=q, k=k, v=v, g=g, beta=beta, A=A, w=None, u=None, scale=scale,
        initial_state=initial_state, initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, save_new_value=True,
        store_h=True, fuse_wu=True, fuse_o=GDN_FUSED_HO_FUSE_O,
    )
    if o is None:
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens)
    if not _readback[0]:
        _readback[0] = True
        print(f"[N54] fused GDN chunk kernel engaged: fuse_wu=1 fuse_o={int(GDN_FUSED_HO_FUSE_O)} "
              f"BV={GDN_FUSED_HO_BV} warps={GDN_FUSED_HO_NUM_WARPS} stages={GDN_FUSED_HO_NUM_STAGES} "
              f"T={T} H={H} Hg={Hg} K={K} V={v.shape[-1]} pid={os.getpid()}", flush=True)
    return g, o, A, None, h, v_new


_readback = [False]
