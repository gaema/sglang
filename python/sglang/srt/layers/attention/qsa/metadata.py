"""Metadata owned by the simple QSA implementation.

QSA intentionally does not inherit the NSA metadata abstraction. This module
contains only fields and transforms consumed by the indexer.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk


def qsa_ring_span(compress_ratio: int) -> int:
    """Pending index-key ring positions a request: one group (the default) or, with SGLANG_QSA_RING2=1, two.

    The ring holds a request's not-yet-compressed index keys at slot req * span + position % span. One group (span =
    ratio) admits a verify window of at most `ratio` rows: a wider window completes two groups in one forward, and
    the second group's members would share ring slots with the first's. Two groups (span = 2 ratio) admit a window
    of span - ratio + 1 rows (5 at ratio 4): the open group's members are at most ratio - 1 positions before the
    window, so every slot the forward writes is distinct from every member a compress in it still reads, and no row
    has to be parked."""
    return compress_ratio * (2 if os.environ.get("SGLANG_QSA_RING2", "0") == "1" else 1)


def qsa_max_verify_rows(compress_ratio: int) -> int:
    """The widest verify window (draft tokens a request) the pending ring admits."""
    return qsa_ring_span(compress_ratio) - compress_ratio + 1 if qsa_ring_span(compress_ratio) > compress_ratio \
        else compress_ratio


def build_qsa_row_ranges(
    sequence_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    query_sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed compressed-key ranges for prefill scoring."""

    sequence_lengths = sequence_lengths.to(dtype=torch.int32)
    compressed_lengths = torch.div(
        sequence_lengths, compress_ratio, rounding_mode="floor"
    )
    compressed_cu_seqlens = torch.nn.functional.pad(
        compressed_lengths.cumsum(0), (1, 0)
    ).to(torch.int32)
    query_sequence_ids = query_sequence_ids.to(
        device=sequence_lengths.device, dtype=torch.long
    )
    row_starts = compressed_cu_seqlens.index_select(0, query_sequence_ids)
    visible_blocks = torch.div(
        query_positions.to(device=sequence_lengths.device, dtype=torch.int32) + 1,
        compress_ratio,
        rounding_mode="floor",
    )
    max_blocks = compressed_lengths.index_select(0, query_sequence_ids)
    row_ends = row_starts + torch.minimum(visible_blocks, max_blocks)
    return row_starts, row_ends, compressed_cu_seqlens


class QSAIndexerMetadata(msgspec.Struct, frozen=True):
    """All per-forward metadata consumed specifically by ``QSAIndexer``.

    Row layout contract:

    - ``sequence_lengths``/``token_slot_table`` carry one row per *sequence*
      for extend modes and one row per *query token* for the paged modes
      (decode, target_verify, draft_extend).
    - ``token_to_batch_idx`` maps every query/token row handled by the indexer
      onto a row of ``sequence_lengths``/``token_slot_table``; DP attention
      token padding adds physical rows beyond this mapping, never inside it.
      For CP extend, it covers the global packed *new* tokens used by the
      compression write plan. Query selection uses a zigzag-sharded copy;
      cached prefix tokens are K/V context, not rows in this mapping.
    - For the paged modes the mapping is the identity
      (``arange(num_query_rows)``), so page-table/MQA inputs built per
      ``sequence_lengths`` row line up with per-query sparse-attention rows.
    """

    sequence_lengths: torch.Tensor
    token_to_batch_idx: torch.Tensor
    token_slot_table: torch.Tensor
    out_cache_loc: torch.Tensor
    token_to_kv_pool: object
    compress_ratio: int
    block_topk: int
    req_pool_indices: Optional[torch.Tensor] = None
    # Parallel per-group arrays for the groups compressed this forward:
    # slot, sequence-local group-end position, and owning metadata row.
    # The first member's token row in this forward's packed tensors is extend only,
    # where group-aligned chunks keep every member in-chunk; None on paged forwards.
    write_locs: Optional[torch.Tensor] = None
    compress_group_positions: Optional[torch.Tensor] = None
    compress_sequence_ids: Optional[torch.Tensor] = None
    compress_member_rows: Optional[torch.Tensor] = None
    is_cuda_graph: bool = False
    graph_write_locs: Optional[torch.Tensor] = None
    graph_compressed_page_table: Optional[torch.Tensor] = None
    graph_compressed_lengths: Optional[torch.Tensor] = None
    graph_prefix_lengths: Optional[torch.Tensor] = None
    decode_page_table: Optional[torch.Tensor] = None
    decode_lengths: Optional[torch.Tensor] = None
    decode_logical_positions: Optional[torch.Tensor] = None
    pending_ring_slots: Optional[torch.Tensor] = None
    # Multi-row decode (target verify, draft-extend): where each row's raw index key is stored BEFORE the
    # compress -- its ring slot, or for a row past a group close in this step, request slot 0's inert rows
    # (build_pending_store_slots); the indexer re-stores every row at pending_ring_slots after the compress.
    pending_store_slots: Optional[torch.Tensor] = None
    compress_group_ring_locs: Optional[torch.Tensor] = None
    extend_rope_matrix: Optional[torch.Tensor] = None
    graph_ring_group_locs: Optional[torch.Tensor] = None
    defer_block_expansion: bool = False
    # Host copy of the extend sequence_lengths; lets prefill MQA skip a device sync.
    prefill_sequence_lengths_cpu: Optional[Tuple[int, ...]] = None
    # R2d (SGLANG_QWEN4_QSA_SPLIT_OP): an upper bound on every RoPE position
    # this forward will index, known on the host when the metadata is built
    # (the backend's ``max_length - 1``, read from ``seq_lens_cpu``). When set,
    # ``QSAIndexer`` grows the cos/sin cache from it instead of syncing on
    # ``positions.max().item()``; None (the default) keeps the sync.
    max_position: Optional[int] = None
    # _q38fn_nosync: host copy of sequence_lengths (non-speculative eager path)
    sequence_lengths_cpu: Optional[tuple] = None

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.sequence_lengths.to(torch.int32)

    def get_token_slot_table(self) -> torch.Tensor:
        return self.token_slot_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.get_seqlens_int32().index_select(
            0, self.get_token_to_batch_idx().long()
        )

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
        row_ends: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk != self.block_topk:
            raise ValueError(
                f"QSA compressed top-k must be {self.block_topk}, got {topk}"
            )
        if row_starts is None or row_ends is None:
            raise ValueError("QSA top-k transform requires row_starts and row_ends")
        return qsa_fast_topk(logits, row_starts, row_ends, topk=self.block_topk)

    def get_prefill_mqa_inputs(
        self,
        layer_id: int,
        positions: torch.Tensor,
        query_sequence_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather packed compressed K and ragged ranges for prefill MQA.

        ``query_sequence_ids`` overrides ``token_to_batch_idx`` for callers that
        score a subset of the batch's rows (prefill CP: this rank's zigzag rows);
        ``positions`` must then be those rows' logical positions."""

        pool = self.token_to_kv_pool
        ratio = self.compress_ratio
        compressed_buffer = pool.get_qsa_compressed_k_buffer(layer_id)
        parts = []
        sequence_lengths = self.sequence_lengths.to(torch.int32)
        sequence_lengths_list = self.prefill_sequence_lengths_cpu
        if sequence_lengths_list is None and self.sequence_lengths_cpu is not None:
            sequence_lengths_list = list(self.sequence_lengths_cpu)  # _q38fn_nosync
        if sequence_lengths_list is None:
            sequence_lengths_list = sequence_lengths.tolist()
        for sequence_id in range(len(sequence_lengths_list)):
            complete_blocks = int(sequence_lengths_list[sequence_id]) // ratio
            if complete_blocks == 0:
                continue
            # compressed slot = first raw slot // ratio; the allocator is page-aligned,
            # so each group is contiguous in one page (see QSATokenToKVPool).
            compressed_locs = (
                self.token_slot_table[
                    sequence_id, : complete_blocks * ratio : ratio
                ].long()
                // ratio
            )
            parts.append(compressed_buffer.index_select(0, compressed_locs))
        compressed_keys = (
            torch.cat(parts, dim=0)
            if parts
            else compressed_buffer.new_empty(
                (0, pool.qsa_index_kv_heads, pool.qsa_index_head_dim)
            )
        )
        if query_sequence_ids is None:
            query_sequence_ids = self.token_to_batch_idx
        num_valid_tokens = query_sequence_ids.numel()
        if positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA prefill positions are shorter than the request mapping: "
                f"positions={positions.numel()}, mapping={num_valid_tokens}"
            )
        positions = positions[:num_valid_tokens]
        row_starts, row_ends, _ = build_qsa_row_ranges(
            sequence_lengths,
            positions.to(sequence_lengths.device),
            query_sequence_ids.to(sequence_lengths.device),
            self.compress_ratio,
        )
        return compressed_keys, row_starts, row_ends, sequence_lengths

    def get_decode_mqa_inputs(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Paged compressed-K cache inputs for decode MQA, one row per query row."""

        pool = self.token_to_kv_pool
        num_rows = self.sequence_lengths.numel()
        if self.token_slot_table.shape[0] != num_rows:
            raise ValueError(
                "QSA decode page-table rows must match the per-query sequence "
                f"lengths: table_rows={self.token_slot_table.shape[0]}, "
                f"rows={num_rows}"
            )
        compressed_cache = pool.get_qsa_compressed_k_buffer(layer_id).reshape(
            -1,
            pool.qsa_compressed_page_size,
            pool.qsa_index_kv_heads,
            pool.qsa_index_head_dim,
        )
        if self.is_cuda_graph:
            if (
                self.graph_compressed_page_table is None
                or self.graph_compressed_lengths is None
            ):
                raise RuntimeError("QSA CUDA graph decode metadata is incomplete")
            return (
                compressed_cache,
                self.graph_compressed_page_table,
                self.graph_compressed_lengths,
                self.graph_compressed_page_table.shape[1]
                * pool.qsa_compressed_page_size,
            )
        if self.decode_page_table is not None and self.decode_lengths is not None:
            return (
                compressed_cache,
                self.decode_page_table,
                self.decode_lengths,
                self.decode_page_table.shape[1] * pool.qsa_compressed_page_size,
            )
        compressed_page_table, compressed_lengths = compressed_decode_view(
            compressed_page_size=pool.qsa_compressed_page_size,
            compress_ratio=self.compress_ratio,
            sequence_lengths=self.sequence_lengths,
            token_slot_table=self.token_slot_table,
        )
        return (
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            compressed_page_table.shape[1] * pool.qsa_compressed_page_size,
        )


def build_pending_ring_slots(
    *,
    token_to_batch_idx: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sequence_lengths: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
    is_extend: bool,
) -> torch.Tensor:
    """Pending-ring slot ``req_pool_idx * span + position % span`` per token (span: qsa_ring_span).
    On extend, tokens before the pending tail dump into rows [0, span),
    which no request owns (request slot 0 is never allocated); CUDA-graph safe."""
    span = qsa_ring_span(compress_ratio)
    rows = token_to_batch_idx.long()[: logical_positions.numel()]
    requests = req_pool_indices.long()[rows]
    positions = logical_positions.long()
    slots = requests * span + positions % span
    if is_extend:
        lengths = sequence_lengths.long()[rows]
        pending = positions >= (lengths // compress_ratio) * compress_ratio
        slots = torch.where(pending, slots, positions % span)
    return slots


def build_pending_store_slots(
    *,
    pending_ring_slots: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    req_pool_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Multi-row decode: the slot each row's raw index key is stored at BEFORE the compress.

    A request's rows are consecutive positions, and a row past a group close inside this step lands on a ring
    slot of that group (position 4b + 4 + k shares 4b + k's slot): stored first, it would replace the group's
    own key before the compress reads it. Such a row is parked in rows [0, ratio) of request slot 0 (never
    allocated: the inert dump) and re-stored at its ring slot after the compress. A row is past a close when its
    group index exceeds that of its request's first row in this forward. CUDA-graph safe (no host sync).

    With a two-group ring (qsa_ring_span > ratio) no row of an admitted window shares a slot with a member a
    compress still reads, so every row is stored at its ring slot directly."""
    if qsa_ring_span(compress_ratio) > compress_ratio:
        return pending_ring_slots.long()[: logical_positions.numel()]
    n = logical_positions.numel()
    positions = logical_positions.long()[:n]
    requests = req_pool_indices.long()[token_to_batch_idx.long()[:n]]
    idx = torch.arange(n, device=positions.device)
    start = torch.ones(n, dtype=torch.bool, device=positions.device)
    if n > 1:
        start[1:] = requests[1:] != requests[:-1]
    first = positions[torch.cummax(torch.where(start, idx, 0), 0).values]
    past_close = positions // compress_ratio > first // compress_ratio
    return torch.where(past_close, positions % compress_ratio, pending_ring_slots.long()[:n])


def build_group_ring_slots(
    *,
    req_pool_indices: torch.Tensor,
    group_end_positions: torch.Tensor,
    sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Ring slots of a planned group's members, oldest first."""
    span = qsa_ring_span(compress_ratio)
    requests = req_pool_indices.long()[sequence_ids]
    offsets = torch.arange(
        compress_ratio - 1,
        -1,
        -1,
        device=group_end_positions.device,
        dtype=torch.long,
    )
    positions = (group_end_positions[:, None] - offsets[None, :]).clamp_min(0)
    return requests[:, None] * span + positions % span


def build_rope_position_matrix(
    rope_positions: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """This forward's RoPE coordinates as the [tokens, 3] layout the fused
    compress kernel reads."""
    if rope_positions.ndim == 1:
        return (
            rope_positions[:num_tokens].long().unsqueeze(1).expand(-1, 3)
        ).contiguous()
    return rope_positions[:, :num_tokens].long().transpose(0, 1).contiguous()


def compressed_decode_view(
    *,
    compressed_page_size: int,
    compress_ratio: int,
    sequence_lengths: torch.Tensor,
    token_slot_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compressed page table and lengths for decode MQA.

    Page-table entries are full-KV page ids read off the page-aligned
    token-slot rows; the scoring kernel converts them to compressed
    slots as page_id * compressed_page_size + block_in_page. Entries
    past a row's compressed length are stale-but-unread (bounded by
    compressed_lengths); clamp keeps them non-negative.
    """
    full_page = compressed_page_size * compress_ratio
    compressed_lengths = torch.div(
        sequence_lengths.to(torch.int32),
        compress_ratio,
        rounding_mode="floor",
    )
    compressed_page_table = (
        (token_slot_table[:, ::full_page].long() // full_page)
        .clamp_min(0)
        .to(torch.int32)
    )
    return compressed_page_table, compressed_lengths


__all__ = [
    "QSAIndexerMetadata",
    "build_qsa_row_ranges",
    "build_pending_ring_slots",
    "build_pending_store_slots",
    "build_group_ring_slots",
    "build_rope_position_matrix",
    "compressed_decode_view",
    "qsa_max_verify_rows",
    "qsa_ring_span",
]
