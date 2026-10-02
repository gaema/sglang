# N57 milestone 1 -- multi-row tiling for the QSA sparse full-attention prefill
# kernel: R consecutive query rows of one KV group share every gathered K/V row.
#
# Installed into the sglang fork by qsa_multirow_patch.py as
#   sglang/srt/layers/attention/qsa/sparse_attn_multirow.py
# and selected from sparse_attn.py's sparse_gqa_fwd_interface_triton_ck ONLY
# when SGLANG_QSA_MULTIROW=1 (unset = upstream kernel byte-for-byte).
#
# Upstream `_sparse_gqa_chunk_prefill` runs one CTA per (query row, KV group):
# it walks that row's top-k token list (indexer output, 512 blocks x 4 tokens +
# <= 3 tail tokens = 2051 entries, in score order) in BLOCK_N slices, gathering
# K and V rows for every entry -- 2.0 MB per CTA out of an L2-resident working
# set, at 79.5% of the L2 read roof (N55), 4 resident warps per SM (N57).
# Adjacent rows select overlapping token sets but in different orders, so the
# share has to be built:
#   prep 1  `_qsa_row_bitmap`   one CTA per query row: OR each valid entry of
#           the row's list (first row_limit entries, token >= 0 -- exactly the
#           entries upstream reads) into a [max_kv/32] int32 bitmap.
#   prep 2  `_qsa_tile_union`   one CTA per tile of R rows: walk the R lists
#           concatenated; an entry is KEPT iff no earlier row of the tile has
#           its token (bitmap test), its OWNER mask is the R bits of rows that
#           have it; compact the kept entries (cumsum) into union[tile] and
#           owner[tile], count[tile] = union size.
#   main    `_sparse_gqa_chunk_prefill_multirow`  one CTA per (tile, group):
#           Q tile [R*BLOCK_M, D] (BLOCK_M = padded GQA group); per BLOCK_N
#           slice of the union: gather K/V ONCE, scores for all R*BLOCK_M rows,
#           mask column n out of row-block r unless owner bit r is set, online
#           softmax per row as upstream. A row's tokens are the same SET as
#           upstream's, summed in union order instead of list order (fp32
#           accumulation-order difference only).
# Contract: same out tensor as upstream from the same (q, k, v, indices, cu_q,
# cu_k, kv_lens, scale). Indices must be 2-D [rows, topk] (the served shape;
# 3-D per-group indices fall back to upstream).
import os
from typing import Tuple

import torch
import triton
import triton.language as tl

# Defaults = the best measured config (qsa_multirow_bench.py, R=4 / BLOCK_N 64 /
# 4 warps / 1 stage: 74.6% of shipped at 97% adjacent-row overlap, 86.3% at 90%,
# short of the bar below ~89%).
QSA_MULTIROW_R = int(os.getenv("SGLANG_QSA_MULTIROW_R", "4"))
QSA_MULTIROW_BLOCK_N = int(os.getenv("SGLANG_QSA_MULTIROW_BLOCK_N", "64"))
QSA_MULTIROW_NUM_WARPS = int(os.getenv("SGLANG_QSA_MULTIROW_NUM_WARPS", "4"))
QSA_MULTIROW_NUM_STAGES = int(os.getenv("SGLANG_QSA_MULTIROW_NUM_STAGES", "1"))


@triton.jit
def _qsa_row_bitmap(
    indices,
    bitmap,
    cu_q,
    cu_k,
    kv_lens,
    topk,
    si_m: tl.constexpr,
    si_n: tl.constexpr,
    NW: tl.constexpr,
    NUM_BATCH: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    query = tl.program_id(0).to(tl.int64)
    # which batch owns this global query row: count the batch ends <= query
    offs_b = tl.arange(0, BLOCK_B)
    ends = tl.load(cu_q + 1 + offs_b, mask=offs_b < NUM_BATCH, other=2**62).to(tl.int64)
    batch = tl.sum((query >= ends).to(tl.int32), 0)
    q_start = tl.load(cu_q + batch).to(tl.int64)
    q_end = tl.load(cu_q + batch + 1).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    query_relative = query - q_start
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs = tl.arange(0, BLOCK_T)
    token = tl.load(indices + query * si_m + offs * si_n, mask=(offs < row_limit) & (offs < topk), other=-1)
    valid = token >= 0
    tok = tl.where(valid, token, 0).to(tl.int32)
    tl.atomic_or(bitmap + query * NW + (tok // 32), (1 << (tok % 32)).to(tl.int32), mask=valid)


@triton.jit
def _qsa_tile_union(
    indices,
    bitmap,
    union,
    owner,
    count,
    cu_q,
    cu_k,
    kv_lens,
    topk,
    si_m: tl.constexpr,
    si_n: tl.constexpr,
    NW: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    MAXU: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1)
    utile = batch * TILES_PER_BATCH + tile
    q_start = tl.load(cu_q + batch).to(tl.int64)
    q_end = tl.load(cu_q + batch + 1).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    row0 = q_start + tile * R
    if row0 >= q_end:
        return
    offs = tl.arange(0, BLOCK_E)
    r = offs // topk
    p = offs % topk
    query = row0 + r
    in_tile = (offs < R * topk) & (query < q_end)
    query_relative = query - q_start
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    token = tl.load(indices + query * si_m + p * si_n, mask=in_tile & (p < row_limit), other=-1)
    valid = in_tile & (p < row_limit) & (token >= 0)
    tok = tl.where(valid, token, 0).to(tl.int32)
    word = tok // 32
    bit = (1 << (tok % 32)).to(tl.int32)
    kept = valid
    own = tl.zeros([BLOCK_E], tl.int32)
    for rr in tl.static_range(R):
        qrr = row0 + rr
        w = tl.load(bitmap + qrr * NW + word, mask=valid & (qrr < q_end), other=0)
        has = (w & bit) != 0
        own = own | tl.where(has, 1 << rr, 0)
        # an earlier row of the tile already holds this token -> not kept here
        kept = kept & ((rr >= r) | (~has))
    pos = tl.cumsum(kept.to(tl.int32), 0) - 1
    n = tl.sum(kept.to(tl.int32), 0)
    tl.store(union + (utile * MAXU + pos).to(tl.int64), tok, mask=kept)
    tl.store(owner + (utile * MAXU + pos).to(tl.int64), own, mask=kept)
    tl.store(count + utile, n)


@triton.jit
def _sparse_gqa_chunk_prefill_multirow(
    q,
    k,
    v,
    out,
    union,
    owner,
    count,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAXU: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch).to(tl.int64)
    q_end = tl.load(cu_q + batch + 1).to(tl.int64)
    row0 = q_start + tile * R
    if row0 >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    utile = batch * TILES_PER_BATCH + tile
    n_union = tl.load(count + utile)
    u_limit = ((n_union + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    offs_m = tl.arange(0, R * BLOCK_M)
    rows = offs_m // BLOCK_M
    heads = offs_m % BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)
    query = row0 + rows
    m_valid = (heads < GROUP_SIZE) & (query < q_end)
    q_values = tl.load(
        q + query[:, None] * sq_m + (group * GROUP_SIZE + heads[:, None]) * sq_h + offs_d[None, :] * sq_d,
        mask=m_valid[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    u_base = union + utile * MAXU
    o_base = owner + utile * MAXU
    # -1e30 instead of -inf: a row whose first member arrives in a later slice
    # must not see exp2(-inf - -inf); with a finite floor alpha = 1, p = 0.
    max_value = tl.full([R * BLOCK_M], -1.0e30, tl.float32)
    normalizer = tl.zeros([R * BLOCK_M], tl.float32)
    accumulator = tl.zeros([R * BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, u_limit, BLOCK_N):
        current = start + offs_n
        in_u = current < n_union
        token = tl.load(u_base + current, mask=in_u, other=0)
        own = tl.load(o_base + current, mask=in_u, other=0)
        keys = tl.load(
            k_base + token[None, :].to(tl.int64) * sk_n + offs_d[:, None] * sk_d,
            mask=in_u[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None].to(tl.int64) * sv_n + offs_d[None, :] * sv_d,
            mask=in_u[:, None],
            other=0.0,
        )
        member = ((own[None, :] >> rows[:, None]) & 1) != 0
        # Pool-gathered K/V may carry the fp8 storage dtype; the QSA pool is
        # written by a scale-free cast, so this upcast is the dequant (no-op for
        # bf16). Unconditional, matching upstream's _sparse_gqa_chunk_prefill.
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(member & in_u[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(probabilities.to(values.dtype), values, accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out + query[:, None] * so_m + (group * GROUP_SIZE + heads[:, None]) * so_h + offs_d[None, :] * so_d,
        output,
        mask=m_valid[:, None],
    )


def qsa_multirow_supported(q, indices) -> bool:
    return indices.ndim == 2 and q.is_cuda


def qsa_multirow_prep(indices, cu_q, cu_k, kv_lens, max_q: int, max_kv: int, R: int, block_n: int, workspace=None):
    """Returns (union, owner, count, tiles_per_batch). workspace: optional dict
    to reuse the scratch tensors across calls of the same shape."""
    rows, topk = indices.shape
    n_batch = cu_q.shape[0] - 1
    NW = (max_kv + 31) // 32
    tiles_per_batch = triton.cdiv(max_q, R)
    MAXU = R * topk
    dev = indices.device
    key = (rows, topk, n_batch, NW, tiles_per_batch, MAXU)
    if workspace is not None and workspace.get("key") == key:
        bitmap, union, owner, count = workspace["t"]
        bitmap.zero_()
    else:
        bitmap = torch.zeros(rows, NW, dtype=torch.int32, device=dev)
        union = torch.empty(n_batch * tiles_per_batch, MAXU, dtype=torch.int32, device=dev)
        owner = torch.empty_like(union)
        count = torch.zeros(n_batch * tiles_per_batch, dtype=torch.int32, device=dev)
        if workspace is not None:
            workspace["key"] = key
            workspace["t"] = (bitmap, union, owner, count)
    _qsa_row_bitmap[(rows,)](
        indices, bitmap, cu_q, cu_k, kv_lens, topk,
        indices.stride(0), indices.stride(1),
        NW=NW, NUM_BATCH=n_batch, BLOCK_B=triton.next_power_of_2(n_batch), BLOCK_N=block_n,
        BLOCK_T=triton.next_power_of_2(topk), num_warps=4,
    )
    _qsa_tile_union[(tiles_per_batch, n_batch)](
        indices, bitmap, union, owner, count, cu_q, cu_k, kv_lens, topk,
        indices.stride(0), indices.stride(1),
        NW=NW, R=R, BLOCK_N=block_n, BLOCK_E=triton.next_power_of_2(R * topk), MAXU=MAXU,
        TILES_PER_BATCH=tiles_per_batch, num_warps=8,
    )
    return union, owner, count, tiles_per_batch


def sparse_gqa_fwd_multirow(q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_kv: int,
                            R: int = QSA_MULTIROW_R, block_n: int = QSA_MULTIROW_BLOCK_N,
                            num_warps: int = QSA_MULTIROW_NUM_WARPS, num_stages: int = QSA_MULTIROW_NUM_STAGES,
                            workspace=None, prep=None):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    if prep is None:
        prep = qsa_multirow_prep(indices, cu_q, cu_k, kv_lens, max_q, max_kv, R, block_n, workspace)
    union, owner, count, tiles_per_batch = prep
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill_multirow[(tiles_per_batch, (cu_q.shape[0] - 1) * num_kv_heads)](
        q, k, v, out, union, owner, count, cu_q, cu_k, kv_lens, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_KV_HEADS=num_kv_heads, GROUP_SIZE=group_size, BLOCK_M=block_m, R=R, BLOCK_N=block_n,
        HEAD_DIM=head_dim, MAXU=union.shape[1], TILES_PER_BATCH=tiles_per_batch,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


_workspace = {}
_readback = [False]


def sparse_gqa_fwd_interface_multirow(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    """Drop-in for sparse_attn.py's sparse_gqa_fwd_interface_triton_ck under
    SGLANG_QSA_MULTIROW=1 (the caller has already checked qsa_multirow_supported).
    Scratch is reused across calls of one shape; also prints one engagement
    readback line per process."""
    max_kv = int(kv_lens.max().item())
    out = sparse_gqa_fwd_multirow(q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_kv, workspace=_workspace)
    if not _readback[0]:
        _readback[0] = True
        print(f"[N57] multirow QSA prefill engaged: R={QSA_MULTIROW_R} BLOCK_N={QSA_MULTIROW_BLOCK_N} "
              f"warps={QSA_MULTIROW_NUM_WARPS} stages={QSA_MULTIROW_NUM_STAGES} q={tuple(q.shape)} "
              f"indices={tuple(indices.shape)} max_kv={max_kv} pid={os.getpid()}", flush=True)
    return out
