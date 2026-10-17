#!/usr/bin/env python3
"""Relay one-shot RDMA all-reduce on the 4-cycle CX7 ring: runtime + sglang adapter.

Design: ../design/rdma-allreduce-ring.md (option (a)). Every rank RDMA-writes its
input to both ring neighbours; the host proxy of each rank forwards the copy it got
from `prev` on to `next` (so the diagonal arrives from `prev` after two hops); one
GPU kernel per op stages, rings the doorbell, waits for 3 peers x rails x chunks
flags in pinned host memory, reduces the four inputs in fixed rank order (fp32
accumulate, one bf16 rounding) and advances a device-resident epoch so the op
replays inside a CUDA graph with no host call. Protocol shape = b12x RoCEnante
(`b12x/comm/roce/_roce_proxy.c`, `_oneshot_cute.py`) plus the relay duty.

`SGLANG_RDMA_RING_DIAG=1` replaces the second hop with the proven hardware diagonal: every
rank ALSO opens one marked RC QP to `DIAG=(rank+2)%4` and writes its own input DIRECTLY into
that rank's `recv[me][slot]`, and the proxy forwards nothing. Same recv slot, same flag index,
same bytes as the relay, so the reduced result is bit-identical between the two paths and the
CUDA kernel below is unchanged by it. Default 0 = the relay path, no diagonal QP at all.
`SGLANG_RDMA_RING_DIAG_MAX_BYTES=N` (with DIAG=1) makes that choice PER OP: nbytes <= N takes the
diagonal, a larger op the relay forward, in the same runtime (both the diag QP and the forward path
exist). The proxy derives it from the per-slot nbytes doorbell word the kernel already writes, so a
CUDA-graph replay carries it with no host call and the kernel source does not change at all.

No GPUDirect is used: the transport region is pinned host memory that the NIC writes
and the GB10 GPU reads in place (unified addressing). pyverbs is absent on the
Sparks, so the posting loop is a small C shim (RING_PROXY_C below) built with the
host gcc against libibverbs into the cache dir (host-local, `~/local/build/`) and
bound with ctypes; the kernel is CUDA C compiled at runtime with NVRTC (cuda.bindings).

This file is ONE source with two homes: tracked here, and deployed byte-identical as
`python/sglang/srt/distributed/device_communicators/rdma_ring_allreduce.py` in the
sglang tree (tools/sglang-dsv41-rdma-ring-allreduce.patch hooks it into
`PyNcclCommunicator.all_reduce`). The 4-rank microbench `rdma_allreduce_ring_bench.py`
imports the runtime from here.

sglang contract (env, read once at TP-group construction; default OFF = pristine):
  SGLANG_RDMA_RING_ALLREDUCE=1      route eligible TP all-reduces (SUM, bf16/fp32,
                                    contiguous, 16 B multiple, nbytes <= max) here;
                                    everything else stays on NCCL and is counted
  SGLANG_RDMA_RING_MAX_SIZE         bytes, default 2097152 (2 MiB)
   SGLANG_RDMA_RING_RAILS / _CHUNKS  default 2 / 1 (the N21 bench's best arm)
   SGLANG_RDMA_RING_DIAG_CHUNKS     step-2 chunk count (the diag post, and the relay forward that
                                    must land identical bytes): 0 (default) = same as _CHUNKS,
                                    byte-identical to today INCLUDING post order; nonzero = the
                                    step-2 stripe is cut into that many chunks with one flag each
                                    and (diag mode) is posted BEFORE step (1), so the marked
                                    diagonal stops queueing behind the rank's own neighbour writes
                                    behind it (audit 2026-09-28-diag-slow-rootcause). Part of the
                                    all-rank config match; ring step (1) keeps _CHUNKS geometry.
  SGLANG_RDMA_RING_BLOCKS / _THREADS  default 8 / 512
   SGLANG_RDMA_RING_NEXT_HCA / _PREV_HCA  comma lists, default the f0 / f1 PF pairs
   SGLANG_RDMA_RING_DIAG             0 (default) = the relay path above, unchanged and with NO
                                     diagonal QP created or connected, so an image without the
                                     marked-diagonal underlay keeps running today's path.
                                     1 = each rank ALSO opens one marked RC QP to the opposite
                                     rank DIAG=(rank+2)%4 and writes its input DIRECTLY into that
                                     rank's recv[me][slot], and the proxy forwards NOTHING. The
                                     mark is flow_label 16383 -> UDP sport 65535 -> ethertype
                                     0x88b5 -> intermediate ASIC L2 forward, GID index 3
                                     (::ffff:10.226.x) -- audit
                                     ../audit/2026-09-26-cx7-diagonal-hw-e2e.md. recv slot, flag
                                     index and payload bytes are the same as in relay mode, so the
                                     reduced bits are IDENTICAL between the two (proved in
                                     --self-test); the kernel is byte-for-byte the same.
   SGLANG_RDMA_RING_DIAG_MAX_BYTES   per-op size cutoff, bytes. 0 (default) = today exactly: DIAG=1
                                     sends EVERY op over the diagonal, DIAG=0 every op over the relay,
                                     and the proxy source (so the .so hash) is RING_PROXY_C unchanged.
                                     N > 0 with DIAG=1: an op with nbytes <= N takes the diagonal
                                     second hop, a larger op the relay forward; the diag QP and the
                                     forward path both exist (proxy built from build_proxy_source(
                                     True), which splices the per-op choice in). The choice is
                                     op_uses_diag(nbytes, diag, N) -- a pure function of nbytes, which
                                     every rank shares -- read by the proxy from the per-slot nbytes
                                     doorbell word (ctrl[4+slot]) the kernel already writes, so it is
                                     graph-replay safe and the kernel is untouched. Ignored with DIAG=0.
                                     Part of the all-rank config match: ranks with different cutoffs
                                     would disagree on who delivers the opposite rank's bytes (one
                                     side waits for a relay nobody forwards -> spin-limit poison), so
                                     a mixed farm fails-stop at setup instead.
   SGLANG_RDMA_RING_DIAG_HCA        comma list of the diag PFs, one per rail. Default = the same
                                     PFs the endpoint's underlay sits on (the _NEXT_HCA list).
                                     VERIFY: the apply round must pass the real netdev-backed IB
                                     device carrying the 10.226.x/29 GID at index 3 -- which PF
                                     differs per rank (audit: d47b f0, b4b3 f1).
   SGLANG_RDMA_RING_GID_INDEX        default NCCL_IB_GID_INDEX, else 1 (the RING gid; the diagonal
                                     always uses index 3, independent of this knob)
  SGLANG_RDMA_RING_SPIN_LIMIT       GPU polls before poison (~1 us each), default 20 M
  SGLANG_RDMA_RING_PROXY_CPU        pin the proxy thread (-1 = no pin)
  SGLANG_RDMA_RING_CACHE_DIR        proxy .so cache, default ~/local/build/rdma-ring-proxy
  SGLANG_RDMA_RING_LOG_EVERY_S      seconds between stats log lines, default 300 (0 = off;
                                    the scheduler's idle loop calls the per-step check
                                    thousands of times a second, so the cadence is wall time)
   SGLANG_RDMA_RING_TIMING          0 (default) = the NVRTC kernel source byte-for-byte today's (no
                                    instrumentation compiled in); 1 = the one-shot kernel stamps four
                                    %globaltimer times per op (entry, own doorbell rung, last peer
                                    flag seen, done) into a graph-replay-safe device ring indexed by
                                    seq % TIMING_SLOTS, and the host reader logs one
                                    `[rdma-ring] timing ...` line (arrival_wait_us med/p90, op_us)
                                    next to the stats line. Part of the all-rank config match.

Failure model is fail-stop, as RoCEnante: a timed-out wait poisons the runtime and
later launches no-op; `check_health()` raises on the next eager op and at every
scheduler step (`check_health_all()`), so a poisoned serve dies with a named cause
rather than serving garbage. There is deliberately NO per-op fallback to NCCL after a
timeout: a rank that switched an op to NCCL while its peers still ran it on the ring
would pair NCCL op s with the peers' NCCL op s+1 (collective-order mismatch). NCCL
stays one env change away (flag off).

  --self-test   protocol simulator on one host: 4 simulated ranks, in-order QP
                delivery with randomized interleaving, slot-overwrite detectors,
                missed-doorbell catch-up, chunk/rail stripe coverage, the bf16
                reduction against an fp32-accumulate-once reference, the diagonal run
                against the relay run (identical bits through either second hop), the
                per-op DIAG_MAX_BYTES cutoff on mixed op sizes against an all-relay run
                (plus a must-fire arm: an inconsistent per-rank cutoff fails the
                all-rank config match, and run anyway it stalls -- never corrupts), and
                the adapter's env parsing + routing counters. Needs numpy only; also
                compiles the C shim / CUDA kernel when a toolchain is present (SKIP
                otherwise). No NIC, no GPU.
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

WORLD = 4
SLOTS = 2
FLAG_STRIDE = 128
CTRL_BYTES = 128
PACK_BYTES = 16
MAX_RAILS = 2
MAX_CHUNKS = 8
# Bumped for the diagonal descriptor appended to ring_blob_t (the ring fields keep their byte
# offsets, so the relay path's wire behaviour is unchanged). load_proxy() refuses a stale .so.
RING_ABI = 3
DEFAULT_SPIN_LIMIT = 20_000_000  # GPU polls before poison (~1 us each -> ~20 s)
DEFAULT_MAX_SIZE = 2 * 1024 * 1024
# SGLANG_RDMA_RING_TIMING=1 device ring: TIMING_SLOTS ops x 5 u64 {seq, t_entry, t_posted,
# t_arrived, t_done} (%globaltimer ns), indexed by seq % TIMING_SLOTS so CUDA-graph replays write
# their own slot with no host call. 4096 slots = 160 KiB, allocated only when the gate is on.
TIMING_SLOTS = 4096
TIMING_FIELDS = 5
# One-shot callables run by the next captured all_reduce on its stream right before the kernel launch
# (dsv41_ar_preload.py forks a weight preload there, so it overlaps the op's wait), and right after it
# (dsv41_l2_collective.py joins its side stream there). Empty = no effect.
PRE_LAUNCH: list = []
POST_LAUNCH: list = []


def kernel_timing_slot(epoch: int) -> int:
    """Pure mirror of the gated kernel's record slot: the op's seq is epoch + 1 and the slot is seq % TIMING_SLOTS."""
    return (epoch + 1) % TIMING_SLOTS


def timing_records(buf, seen: int, e: int) -> list[tuple[int, int, int, int, int]]:
    """Pure: the valid (seq, t_entry, t_posted, t_arrived, t_done) rows of a (TIMING_SLOTS, TIMING_FIELDS)
    buffer, in slot order. A slot is a record iff seq in (seen, e], seq % TIMING_SLOTS == slot index, and
    the four stamps form a monotonic chain; that drops in-flight, poisoned and buffer-wrap torn records."""
    out = []
    for slot, row in enumerate(buf):
        seq, t0, t1, t2, t3 = (int(v) for v in row)
        if not (seen < seq <= e) or seq % TIMING_SLOTS != slot:
            continue
        if 0 < t0 <= t1 <= t2 <= t3:
            out.append((seq, t0, t1, t2, t3))
    return out


def parse_timing_buf(buf, seen: int, e: int) -> tuple[list[int], list[int]]:
    """Pure parse of a (TIMING_SLOTS, TIMING_FIELDS) buffer -> (arrival_wait_us list, op_us list)."""
    recs = timing_records(buf, seen, e)
    return [(t2 - t1) // 1000 for _, _, t1, t2, _ in recs], [(t3 - t0) // 1000 for _, t0, _, _, t3 in recs]


DEFAULT_NEXT_HCAS = ("roceP2p1s0f0", "rocep1s0f0")
DEFAULT_PREV_HCAS = ("roceP2p1s0f1", "rocep1s0f1")
# The diagonal QP's GRH, set literally in the C shim (see connect_diag_qp) and recorded here so the
# Python side documents the same contract: flow_label 16383 -> UDP sport 65535 -> ethertype 0x88b5
# -> intermediate ASIC L2 forward; GID index 3 = the ::ffff:10.226.x underlay GID, NOT the ring's.
DIAG_FLOW_LABEL = 16383
DIAG_GID_INDEX = 3

# --------------------------------------------------------------------------------------
# Geometry shared by the C proxy, the CUDA kernel, the Python runtime and the simulator.
# --------------------------------------------------------------------------------------


def stripe(total_packs: int, chunks: int, rails: int, chunk: int, rail: int) -> tuple[int, int]:
    """(pack_offset, packs) of one (chunk, rail) stripe of a `total_packs` message.

    Chunk c covers packs [c*cp, min((c+1)*cp, total)) with cp = ceil(total/chunks);
    inside a chunk rail r takes len//rails (+1 for r < len%rails) packs, in rail order.
    Empty stripes are legal (their flag is still written) so the wait set is static.
    """
    cp = (total_packs + chunks - 1) // chunks if total_packs else 0
    start = min(chunk * cp, total_packs)
    length = min(cp, total_packs - start) if start < total_packs else 0
    base, extra = divmod(length, rails)
    off = start + rail * base + min(rail, extra)
    packs = base + (1 if rail < extra else 0)
    return off, packs


class Layout:
    """Byte offsets inside one rank's pinned region."""

    def __init__(self, slot_bytes: int, rails: int, chunks: int, diag_chunks: int = 0) -> None:
        if slot_bytes % 4096 or slot_bytes <= 0:
            raise ValueError("slot_bytes must be a positive multiple of 4096")
        if not 1 <= rails <= MAX_RAILS or not 1 <= chunks <= MAX_CHUNKS or not 0 <= diag_chunks <= MAX_CHUNKS:
            raise ValueError("rails in 1..2, chunks in 1..8, diag_chunks in 0..8")
        self.slot_bytes = slot_bytes
        self.rails = rails
        self.chunks = chunks
        self.diag_chunks = int(diag_chunks)
        self.chunk_dim = max(chunks, self.diag_chunks)   # the flag array's chunk dimension
        self.recv_off = 0
        self.send_off = WORLD * SLOTS * slot_bytes
        self.flag_off = self.send_off + SLOTS * slot_bytes
        self.ctrl_off = self.flag_off + WORLD * SLOTS * rails * self.chunk_dim * FLAG_STRIDE
        self.total_bytes = self.ctrl_off + CTRL_BYTES

    def recv(self, src: int, slot: int) -> int:
        return self.recv_off + (src * SLOTS + slot) * self.slot_bytes

    def send(self, slot: int) -> int:
        return self.send_off + slot * self.slot_bytes

    def flag(self, src: int, slot: int, rail: int, chunk: int) -> int:
        return self.flag_off + (
            ((src * SLOTS + slot) * self.rails + rail) * self.chunk_dim + chunk
        ) * FLAG_STRIDE

    def as_u64_array(self):
        return (ctypes.c_uint64 * 10)(
            self.recv_off, self.send_off, self.flag_off, self.ctrl_off,
            self.slot_bytes, FLAG_STRIDE, SLOTS, self.rails, self.chunks, self.diag_chunks,
        )


def op_uses_diag(nbytes: int, diag: bool, diag_max_bytes: int) -> bool:
    """Second hop of one op: True = the direct diagonal, False = the relay forward.

    Pure function of nbytes (shared by all ranks) and the all-rank-matched config, so every rank
    makes the same choice for the same op. diag_max_bytes == 0 = today: diag decides for every op.
    The C twin is the `use_diag` line build_proxy_source(True) splices into post_op.
    """
    return bool(diag) and (int(diag_max_bytes) == 0 or int(nbytes) <= int(diag_max_bytes))


def ring_config(*, slot_bytes: int, rails: int, chunks: int, max_size: int, threads: int, blocks: int,
                spin_limit: int, diag: bool, diag_chunks: int, timing: bool, diag_max_bytes: int) -> dict:
    """The dict every rank all-gathers at setup; any difference between ranks fails the group stop."""
    return dict(abi=RING_ABI, slot_bytes=slot_bytes, rails=rails, chunks=chunks, max_size=max_size,
                threads=threads, blocks=blocks, spin_limit=spin_limit,
                diag=1 if diag else 0, diag_chunks=diag_chunks,
                timing=1 if timing else 0, diag_max_bytes=int(diag_max_bytes))


def config_mismatches(configs: Sequence[dict]) -> list[str]:
    """Fail-stop verdicts of the all-rank config match ([] = every rank agrees with rank 0)."""
    return [f"rank {i} config differs: {c}" for i, c in enumerate(configs) if c != configs[0]]


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(f: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bits (NaN -> 0x7fc0); the kernel does the same."""
    u = f.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    r = ((u + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16)
    return np.where(np.isnan(f), np.uint16(0x7FC0), r).astype(np.uint16)


def reference_sum_bf16(inputs_bits: Sequence[np.ndarray]) -> np.ndarray:
    """fp32 accumulate in rank order 0..3, round once: what the kernel must reproduce bit for bit."""
    acc = bf16_to_f32(inputs_bits[0]).copy()
    for b in inputs_bits[1:]:
        acc = acc + bf16_to_f32(b)
    return f32_to_bf16_bits(acc)


# --------------------------------------------------------------------------------------
# C proxy shim (libibverbs). Built with the host gcc at first use; plain C, no CUDA.
# --------------------------------------------------------------------------------------

RING_PROXY_C = r"""
// Relay one-shot RDMA proxy for the 4-cycle ring (rdma_ring_allreduce.py).
// Region: recv[src][slot] | send[slot] | flag[src][slot][rail][chunk] | ctrl.
// ctrl u32 words: 0 seq, 1 nbytes, 2 error_seq, 3 error_peer, 4 nbytes[slot0],
// 5 nbytes[slot1], 6 error_rail, 7 error_chunk.  The GPU kernel writes 4/5, then 0.
// Per op s (slot = s&1) this thread (1) posts send[slot] to next and prev, striped
// over the rails of that side, one data WR chained to one inline 4-byte flag WR of s
// on the same RC QP per (chunk, rail); (2) for each (chunk, rail) waits until
// flag[prev][slot][rail][chunk] == s and forwards recv[prev][slot] stripe to next's
// recv[prev][slot] + next's flag[prev][slot][rail][chunk] (next's diagonal).
// With diag=1 step (2) instead posts send[slot] DIRECTLY to the opposite rank (rank+2) on
// a marked diagonal QP (flow_label 16383, sgid_index 3) and forwards nothing: the opposite
// rank gets my bytes in its recv[me][slot] + flag[me][slot][rail][chunk], which is exactly
// where relay mode put them (its prev forwarded them), so the kernel reduces identical bits
// either way. diag=0 keeps the relay code below verbatim and never creates the diag QP.
#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RING_ABI 3
#define SIDES 2          /* 0 = next (rank+1), 1 = prev (rank-1) */
#define DIAG_SIDE 2      /* hca[] slot of the diagonal (rank+2) QP; used only when diag=1 */
#define NSIDES 3         /* SIDES + the diagonal slot (never touched when diag=0) */
#define DIAG_GID_INDEX 3 /* the diagonal underlay GID (::ffff:10.226.x), NOT the ring gid_index */
#define MAX_RAILS 2
#define MAX_CHUNKS 8
#define SEND_DEPTH 256
#define IDLE_SPINS 20000000ULL
#define ROCE_PORT 1

typedef struct {
    uint64_t region_addr;
    uint32_t rkey[SIDES][MAX_RAILS];
    uint32_t qp_num[SIDES][MAX_RAILS];
    uint32_t mtu[SIDES][MAX_RAILS];
    uint16_t lid[SIDES][MAX_RAILS];
    uint8_t gid[SIDES][MAX_RAILS][16];
    // Diagonal descriptor (the rank+2 QP), filled ONLY when diag=1 and zeroed otherwise. The
    // ring fields above keep their exact offsets, so relay mode's wire behaviour is unchanged.
    // diag_gid[r] is the GID read at index DIAG_GID_INDEX (3) on the diag device: the
    // ::ffff:10.226.x IPv4-mapped address of the marked underlay, not the ring's link-local GID.
    uint8_t diag_gid[MAX_RAILS][16];
    uint32_t diag_qp_num[MAX_RAILS];
    uint32_t diag_rkey[MAX_RAILS];
    uint16_t diag_lid[MAX_RAILS];
    uint32_t diag_mtu[MAX_RAILS];
} ring_blob_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    uint32_t outstanding;
    uint64_t writes_completed;
    uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
    int gid_index;       // index this QP's GRH uses: the ring gid_index, or DIAG_GID_INDEX for the diag side
} ring_hca_t;

typedef struct {
    int world, rank, rails, chunks, slots, gid_index, proxy_cpu, diag;
    ring_hca_t hca[NSIDES][MAX_RAILS];
    uint8_t *region;
    size_t region_bytes, slot_bytes, recv_off, send_off, flag_off, ctrl_off, flag_stride;
    uint64_t peer_addr[NSIDES];
    uint32_t peer_rkey[NSIDES][MAX_RAILS];
    pthread_t thread;
    atomic_int running, failed;
    int started, diag_chunks, chunk_dim, diag_first;
    uint32_t last_seq;
    uint64_t spin_limit;
    uint64_t ops_posted, forwards_posted, writes_completed, relay_polls, diag_posts;
    volatile uint32_t phase, phase_seq;   /* diagnostics: 0 idle, 1 posting, 2 relay wait, 3 send-queue drain */
    char err[512];
} ring_ctx_t;

void ring_destroy(ring_ctx_t *c);
static void set_err(ring_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}
int ring_abi_version(void) { return RING_ABI; }
uint64_t ring_blob_bytes(void) { return sizeof(ring_blob_t); }

static void stripe(uint32_t total, int chunks, int rails, int chunk, int rail,
                   uint32_t *off, uint32_t *packs) {
    uint32_t cp = total ? (total + chunks - 1) / (uint32_t)chunks : 0;
    uint32_t start = (uint32_t)chunk * cp;
    if (start > total) start = total;
    uint32_t len = (start < total) ? ((cp < total - start) ? cp : total - start) : 0;
    uint32_t base = len / (uint32_t)rails, extra = len % (uint32_t)rails;
    *off = start + (uint32_t)rail * base + ((uint32_t)rail < extra ? (uint32_t)rail : extra);
    *packs = base + ((uint32_t)rail < extra ? 1 : 0);
}

static uint64_t flag_off(ring_ctx_t *c, int src, int slot, int rail, int chunk) {
    return c->flag_off +
           ((((uint64_t)src * c->slots + slot) * c->rails + rail) * c->chunk_dim + chunk) *
               c->flag_stride;
}

static int open_hca(ring_ctx_t *c, int side, int rail, const char *name, int gid_index) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (!list) { set_err(c, "ibv_get_device_list", errno); return -1; }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++)
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) { dev = list[i]; break; }
    if (!dev) { ibv_free_device_list(list); snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name); return -1; }
    ring_hca_t *h = &c->hca[side][rail];
    h->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (!h->ctx) { set_err(c, "ibv_open_device", errno); return -1; }
    struct ibv_port_attr port;
    if (ibv_query_port(h->ctx, ROCE_PORT, &port)) { set_err(c, "ibv_query_port", errno); return -1; }
    if (port.state != IBV_PORT_ACTIVE) { snprintf(c->err, sizeof(c->err), "%s port not active", name); return -1; }
    h->lid = port.lid;
    h->mtu = port.active_mtu;
    if (ibv_query_gid(h->ctx, ROCE_PORT, gid_index, &h->gid)) { set_err(c, "ibv_query_gid", errno); return -1; }
    h->gid_index = gid_index;
    h->pd = ibv_alloc_pd(h->ctx);
    if (!h->pd) { set_err(c, "ibv_alloc_pd", errno); return -1; }
    h->mr = ibv_reg_mr(h->pd, c->region, c->region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!h->mr) { set_err(c, "ibv_reg_mr(pinned region)", errno); return -1; }
    h->cq = ibv_create_cq(h->ctx, SEND_DEPTH, NULL, NULL, 0);
    if (!h->cq) { set_err(c, "ibv_create_cq", errno); return -1; }
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = h->cq; attr.recv_cq = h->cq; attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = SEND_DEPTH; attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1; attr.cap.max_recv_sge = 1; attr.cap.max_inline_data = 16;
    h->qp = ibv_create_qp(h->pd, &attr);
    if (!h->qp) { set_err(c, "ibv_create_qp", errno); return -1; }
    struct ibv_qp_attr init;
    memset(&init, 0, sizeof(init));
    init.qp_state = IBV_QPS_INIT; init.pkey_index = 0; init.port_num = ROCE_PORT;
    init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    int rc = ibv_modify_qp(h->qp, &init, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (rc) { set_err(c, "ibv_modify_qp(INIT)", rc); return -1; }
    return 0;
}

// layout = {recv_off, send_off, flag_off, ctrl_off, slot_bytes, flag_stride, slots, rails, chunks, diag_chunks}
// diag_chunks (layout[9]) is the step-2 chunk count, 0 = same as chunks (byte-identical to ABI 2);
// the flag array's chunk dimension is max(chunks, diag_chunks) so both geometries index one table.
// diag_names[] are the PFs the marked diagonal rides (one per rail, the ones that carry the
// 10.226.x underlay GID at index 3); they are read ONLY when diag != 0.
ring_ctx_t *ring_create(int world, int rank, const char *const *next_names, const char *const *prev_names,
                        const char *const *diag_names, int diag,
                        int rails, int gid_index, void *region, uint64_t region_bytes,
                        const uint64_t *layout, uint64_t spin_limit, int proxy_cpu,
                        char *err, uint64_t err_len) {
    if (world != 4 || rank < 0 || rank >= world || rails < 1 || rails > MAX_RAILS ||
        layout[6] != 2 || layout[7] != (uint64_t)rails || layout[8] < 1 || layout[8] > MAX_CHUNKS ||
        layout[9] > (uint64_t)MAX_CHUNKS ||
        layout[3] + 128 > region_bytes) {
        snprintf(err, err_len, "invalid ring geometry"); return NULL;
    }
    ring_ctx_t *c = calloc(1, sizeof(*c));
    if (!c) { snprintf(err, err_len, "out of memory"); return NULL; }
    c->world = world; c->rank = rank; c->rails = rails; c->gid_index = gid_index; c->proxy_cpu = proxy_cpu;
    c->diag = diag ? 1 : 0;
    c->diag_chunks = layout[9] ? (int)layout[9] : (int)layout[8];
    c->chunk_dim = (int)(layout[8] > layout[9] ? layout[8] : layout[9]);
    c->diag_first = diag && layout[9] ? 1 : 0;
    c->region = region; c->region_bytes = region_bytes;
    c->recv_off = layout[0]; c->send_off = layout[1]; c->flag_off = layout[2]; c->ctrl_off = layout[3];
    c->slot_bytes = layout[4]; c->flag_stride = layout[5]; c->slots = (int)layout[6]; c->chunks = (int)layout[8];
    c->spin_limit = spin_limit;
    for (int r = 0; r < rails; r++) {
        if (open_hca(c, 0, r, next_names[r], gid_index) || open_hca(c, 1, r, prev_names[r], gid_index)) {
            snprintf(err, err_len, "%s", c->err); ring_destroy(c); return NULL;
        }
    }
    // The diagonal HCA/QP exists ONLY when diag is on: with diag=0 nothing below this line is
    // opened, connected or posted, so an image without the marked underlay keeps relay behaviour
    // and can never hit a diagonal RTR failure.
    if (c->diag)
        for (int r = 0; r < rails; r++)
            if (open_hca(c, DIAG_SIDE, r, diag_names[r], DIAG_GID_INDEX)) {
                snprintf(err, err_len, "%s", c->err); ring_destroy(c); return NULL;
            }
    return c;
}

int ring_local_blob(ring_ctx_t *c, void *out, uint64_t out_len) {
    if (out_len < sizeof(ring_blob_t)) return -1;
    ring_blob_t b; memset(&b, 0, sizeof(b));   // zeroed: with diag=0 the diag descriptor stays zero
    b.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int s = 0; s < SIDES; s++)
        for (int r = 0; r < c->rails; r++) {
            ring_hca_t *h = &c->hca[s][r];
            b.rkey[s][r] = h->mr->rkey; b.qp_num[s][r] = h->qp->qp_num; b.mtu[s][r] = (uint32_t)h->mtu;
            b.lid[s][r] = h->lid; memcpy(b.gid[s][r], h->gid.raw, 16);
        }
    if (c->diag)
        for (int r = 0; r < c->rails; r++) {
            ring_hca_t *h = &c->hca[DIAG_SIDE][r];
            memcpy(b.diag_gid[r], h->gid.raw, 16);   // h->gid is the index-3 GID (see open_hca)
            b.diag_qp_num[r] = h->qp->qp_num; b.diag_rkey[r] = h->mr->rkey;
            b.diag_lid[r] = h->lid; b.diag_mtu[r] = (uint32_t)h->mtu;
        }
    memcpy(out, &b, sizeof(b));
    return 0;
}

static int connect_qp(ring_ctx_t *c, int side, int rail, const ring_blob_t *peer) {
    // My side-`side` rail-`rail` QP talks to the peer's opposite side, same rail index.
    int pside = 1 - side;
    ring_hca_t *h = &c->hca[side][rail];
    struct ibv_qp_attr rtr; memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[pside][rail] < (uint32_t)h->mtu ? peer->mtu[pside][rail] : (uint32_t)h->mtu);
    rtr.dest_qp_num = peer->qp_num[pside][rail];
    rtr.rq_psn = 0; rtr.max_dest_rd_atomic = 1; rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1; rtr.ah_attr.dlid = peer->lid[pside][rail]; rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0; rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[pside][rail], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)h->gid_index; rtr.ah_attr.grh.hop_limit = 64;
    int rc = ibv_modify_qp(h->qp, &rtr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                           IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc) { snprintf(c->err, sizeof(c->err), "ibv_modify_qp(RTR) side %d rail %d: %s (no L2 path to that peer?)", side, rail, strerror(rc)); return -1; }
    struct ibv_qp_attr rts; memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS; rts.timeout = 14; rts.retry_cnt = 7; rts.rnr_retry = 7; rts.sq_psn = 0; rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(h->qp, &rts, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc) { set_err(c, "ibv_modify_qp(RTS)", rc); return -1; }
    c->peer_addr[side] = peer->region_addr;
    c->peer_rkey[side][rail] = peer->rkey[pside][rail];
    return 0;
}

// The diagonal QP, mirroring the proven marked path (audit 2026-09-26-cx7-diagonal-hw-e2e.md):
// flow_label 16383 makes mlx5 emit UDP sport 65535, the on-board marker rewrites the ethertype
// to 0x88b5, and the intermediate ASIC L2-forwards it to the opposite rank -- so the only routing
// state is the GRH below plus the underlay GID at index 3. Everything else matches connect_qp.
static int connect_diag_qp(ring_ctx_t *c, int rail, const ring_blob_t *peer) {
    ring_hca_t *h = &c->hca[DIAG_SIDE][rail];
    struct ibv_qp_attr rtr; memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->diag_mtu[rail] < (uint32_t)h->mtu ? peer->diag_mtu[rail] : (uint32_t)h->mtu);
    rtr.dest_qp_num = peer->diag_qp_num[rail];
    rtr.rq_psn = 0; rtr.max_dest_rd_atomic = 1; rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1; rtr.ah_attr.dlid = peer->diag_lid[rail]; rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0; rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->diag_gid[rail], 16);   // opposite's ::ffff:10.226.x GID (index 3)
    rtr.ah_attr.grh.flow_label = 16383; rtr.ah_attr.grh.sgid_index = 3; rtr.ah_attr.grh.hop_limit = 64;
    int rc = ibv_modify_qp(h->qp, &rtr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                           IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc) {
        snprintf(c->err, sizeof(c->err),
                 "ibv_modify_qp(RTR) DIAG rail %d: %s (marked diagonal underlay missing? needs the "
                 "10.226.x/29 GID at index 3 on both ends plus the ASIC tc+mirred forward rules; "
                 "leave SGLANG_RDMA_RING_DIAG=0 until then)", rail, strerror(rc));
        return -1;
    }
    struct ibv_qp_attr rts; memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS; rts.timeout = 14; rts.retry_cnt = 7; rts.rnr_retry = 7; rts.sq_psn = 0; rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(h->qp, &rts, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc) { set_err(c, "ibv_modify_qp(RTS) DIAG", rc); return -1; }
    c->peer_addr[DIAG_SIDE] = peer->region_addr;
    c->peer_rkey[DIAG_SIDE][rail] = peer->diag_rkey[rail];
    return 0;
}

int ring_connect(ring_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (blobs_len < sizeof(ring_blob_t) * (uint64_t)c->world) { snprintf(c->err, sizeof(c->err), "peer blob buffer too small"); return -1; }
    const ring_blob_t *all = (const ring_blob_t *)blobs;
    int next = (c->rank + 1) % c->world, prev = (c->rank + c->world - 1) % c->world;
    for (int r = 0; r < c->rails; r++)
        if (connect_qp(c, 0, r, &all[next]) || connect_qp(c, 1, r, &all[prev])) return -1;
    // The diagonal is symmetric: (rank+2)+2 == rank, so both ends use their side-2 QP and the
    // peer's diag descriptor. Never touched when diag=0.
    if (c->diag) {
        int opposite = (c->rank + 2) % c->world;
        for (int r = 0; r < c->rails; r++)
            if (connect_diag_qp(c, r, &all[opposite])) return -1;
    }
    return 0;
}

static int drain_cq(ring_ctx_t *c, int side, int rail) {
    struct ibv_wc wc[32];
    ring_hca_t *h = &c->hca[side][rail];
    int n = ibv_poll_cq(h->cq, 32, wc);
    if (n < 0) { set_err(c, "ibv_poll_cq", errno); return -1; }
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(c->err, sizeof(c->err), "RDMA write side %d rail %d failed: %s (vendor 0x%x, seq %u)",
                     side, rail, ibv_wc_status_str(wc[i].status), wc[i].vendor_err, c->last_seq);
            return -1;
        }
        h->outstanding -= 1; h->writes_completed += 1; c->writes_completed += 1;
    }
    return 0;
}

static int drain_all(ring_ctx_t *c) {
    // The diagonal CQ is only polled when the diag QP exists (diag=0 keeps today's exact set).
    for (int s = 0; s < (c->diag ? NSIDES : SIDES); s++) for (int r = 0; r < c->rails; r++) if (drain_cq(c, s, r)) return -1;
    return 0;
}

// One (data?, flag) pair on side/rail: data WR (unsignaled) chained to an inline signaled flag WR.
static int post_pair(ring_ctx_t *c, int side, int rail, uint64_t local_off, uint32_t bytes,
                     uint64_t remote_data_off, uint64_t remote_flag_off, uint32_t seq) {
    ring_hca_t *h = &c->hca[side][rail];
    while (h->outstanding >= SEND_DEPTH / 4) {
        c->phase = 3;
        if (drain_cq(c, side, rail)) return -1;
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            snprintf(c->err, sizeof(c->err), "proxy stopped with %u writes outstanding", h->outstanding); return -1;
        }
    }
    uint32_t seq_copy = seq;
    struct ibv_sge flag_sge = { .addr = (uint64_t)(uintptr_t)&seq_copy, .length = 4, .lkey = 0 };
    struct ibv_send_wr flag_wr; memset(&flag_wr, 0, sizeof(flag_wr));
    flag_wr.wr_id = seq; flag_wr.sg_list = &flag_sge; flag_wr.num_sge = 1; flag_wr.opcode = IBV_WR_RDMA_WRITE;
    flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    flag_wr.wr.rdma.remote_addr = c->peer_addr[side] + remote_flag_off; flag_wr.wr.rdma.rkey = c->peer_rkey[side][rail];
    struct ibv_send_wr data_wr; struct ibv_sge data_sge; struct ibv_send_wr *first = &flag_wr;
    if (bytes) {
        data_sge = (struct ibv_sge){ .addr = (uint64_t)(uintptr_t)(c->region + local_off), .length = bytes, .lkey = h->mr->lkey };
        memset(&data_wr, 0, sizeof(data_wr));
        data_wr.wr_id = seq; data_wr.next = &flag_wr; data_wr.sg_list = &data_sge; data_wr.num_sge = 1;
        data_wr.opcode = IBV_WR_RDMA_WRITE;
        data_wr.wr.rdma.remote_addr = c->peer_addr[side] + remote_data_off; data_wr.wr.rdma.rkey = c->peer_rkey[side][rail];
        first = &data_wr;
    }
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(h->qp, first, &bad);
    if (rc) { set_err(c, "ibv_post_send", rc); return -1; }
    h->outstanding += 1; h->bytes_posted += bytes;
    return 0;
}

static int post_op(ring_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    if (nbytes == 0 || nbytes % 16) { snprintf(c->err, sizeof(c->err), "payload must be a positive multiple of 16 B, got %u", nbytes); return -1; }
    c->phase = 1; c->phase_seq = seq;
    // SGLANG_RDMA_RING_HOP_PROBE (ctrl+124): time entry -> step (1) posted (the direct writes' CPU cost,
    // not the relay forward's wait); sum ns at ctrl+112, count at ctrl+120
    volatile uint32_t *hctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    const int hop = hctrl[31] != 0u;
    struct timespec hp0, hp1;
    if (hop) clock_gettime(CLOCK_MONOTONIC, &hp0);
    int slot = (int)(seq & 1u), prev = (c->rank + c->world - 1) % c->world;
    uint32_t total = nbytes / 16;
    int dc = c->diag_chunks ? c->diag_chunks : c->chunks;   // step-(2) geometry; == chunks unless DIAG_CHUNKS is set
    // (0) DIAG_CHUNKS (diag mode only): post the marked diagonal BEFORE its neighbour writes, so
    // the stripe bound for the opposite rank stops queueing behind this rank's own 1-hop writes
    // on the same card's egress queue (audit 2026-09-28-diag-slow-rootcause, cause C1).
    if (c->diag && c->diag_first) {
        for (int ch = 0; ch < dc; ch++)
            for (int r = 0; r < c->rails; r++) {
                uint32_t off, packs; stripe(total, dc, c->rails, ch, r, &off, &packs);
                if (post_pair(c, DIAG_SIDE, r, c->send_off + (uint64_t)slot * c->slot_bytes + (uint64_t)off * 16, packs * 16,
                              c->recv_off + ((uint64_t)c->rank * c->slots + slot) * c->slot_bytes + (uint64_t)off * 16,
                              flag_off(c, c->rank, slot, r, ch), seq)) return -1;
                c->diag_posts += 1;
            }
    }
    // (1) own input to both neighbours: lands in their recv[me][slot] + flag[me][slot][rail][chunk]
    for (int side = 0; side < SIDES; side++)
        for (int ch = 0; ch < c->chunks; ch++)
            for (int r = 0; r < c->rails; r++) {
                uint32_t off, packs; stripe(total, c->chunks, c->rails, ch, r, &off, &packs);
                if (post_pair(c, side, r, c->send_off + (uint64_t)slot * c->slot_bytes + (uint64_t)off * 16, packs * 16,
                              c->recv_off + ((uint64_t)c->rank * c->slots + slot) * c->slot_bytes + (uint64_t)off * 16,
                              flag_off(c, c->rank, slot, r, ch), seq)) return -1;
            }
    c->ops_posted += 1;
    if (hop) {
        clock_gettime(CLOCK_MONOTONIC, &hp1);
        *(volatile uint64_t *)(c->region + c->ctrl_off + 112) +=
            (uint64_t)((hp1.tv_sec - hp0.tv_sec) * 1000000000LL + (hp1.tv_nsec - hp0.tv_nsec));
        hctrl[30] += 1u;
    }
    // (2) the diagonal, two ways. diag=0: RELAY -- wait for prev's stripe to land here, then forward
    //     it on to next. diag=1: post MY OWN send[slot] straight to the opposite rank on the marked
    //     diag QP and forward NOTHING. The opposite rank sees it in recv[me][slot] with
    //     flag[me][slot][rail][chunk] -- the identical slot, flag index and bytes relay mode delivered
    //     there (its prev forwarded them), so the kernel's reduce is bit-identical either way (proved
    //     device-free in --self-test). With DIAG_CHUNKS set, step (2) of BOTH modes cuts its stripe
    //     into `dc` chunks with one flag each (the kernel waits src-specific chunk counts), which is
    //     what pipelines the 2-hop write; step (1) keeps the ring chunks geometry.
    if (c->diag) {
        if (!c->diag_first)
            for (int ch = 0; ch < dc; ch++)
                for (int r = 0; r < c->rails; r++) {
                    uint32_t off, packs; stripe(total, dc, c->rails, ch, r, &off, &packs);
                    if (post_pair(c, DIAG_SIDE, r, c->send_off + (uint64_t)slot * c->slot_bytes + (uint64_t)off * 16, packs * 16,
                                  c->recv_off + ((uint64_t)c->rank * c->slots + slot) * c->slot_bytes + (uint64_t)off * 16,
                                  flag_off(c, c->rank, slot, r, ch), seq)) return -1;
                    c->diag_posts += 1;
                }
        return drain_all(c);
    }
    if (dc == c->chunks) {
        // (2) relay: prev's input, once it has landed here, on to next (next's diagonal)
        for (int ch = 0; ch < c->chunks; ch++)
            for (int r = 0; r < c->rails; r++) {
                volatile uint32_t *flag = (volatile uint32_t *)(c->region + flag_off(c, prev, slot, r, ch));
                uint64_t polls = 0;
                // Bound the relay wait in TIME, not host polls: a host poll is ~30 ns where a
                // GPU poll is ~1 us, so `spin_limit` host polls expired in ~1 s and starved
                // `next` whenever a peer was still in its per-rank NVRTC compile at op 1
                // (N21 round, 2 of 4 first runs). Budget = spin_limit us, the GPU's bound.
                struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
                c->phase = 2;
                while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != seq) {
                    if ((++polls & 63) == 0) {
                        if (drain_all(c)) return -1;
                        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) { snprintf(c->err, sizeof(c->err), "proxy stopped during relay wait"); return -1; }
                        if ((polls & 4095) == 0) {
                            struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
                            int64_t us = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000 + ((int64_t)t1.tv_nsec - (int64_t)t0.tv_nsec) / 1000;
                            if (us > (int64_t)c->spin_limit) {
                                uint32_t plain = __atomic_load_n(flag, __ATOMIC_ACQUIRE);
                                // the same word after invalidating its cache line: a newer value here than in
                                // `plain` means this CPU kept a stale copy of memory the NIC had written
                                __asm__ volatile("dc civac, %0\n\tdsb sy" :: "r"(flag) : "memory");
                                uint32_t fresh = __atomic_load_n(flag, __ATOMIC_ACQUIRE);
                                snprintf(c->err, sizeof(c->err), "relay wait for rank %d rail %d chunk %d seq %u timed out after %lld us (flag read %u, after dc civac %u, doorbell seq %u, ops_posted %llu, writes_completed %llu, outstanding next %u/%u prev %u/%u)", prev, r, ch, seq, (long long)us, plain, fresh, *(volatile uint32_t *)(c->region + c->ctrl_off), (unsigned long long)c->ops_posted, (unsigned long long)c->writes_completed, c->hca[0][0].outstanding, c->rails > 1 ? c->hca[0][1].outstanding : 0u, c->hca[1][0].outstanding, c->rails > 1 ? c->hca[1][1].outstanding : 0u); return -1; }
                        }
                    }
                }
                c->relay_polls += polls;
                uint32_t off, packs; stripe(total, c->chunks, c->rails, ch, r, &off, &packs);
                if (post_pair(c, 0, r, c->recv_off + ((uint64_t)prev * c->slots + slot) * c->slot_bytes + (uint64_t)off * 16, packs * 16,
                              c->recv_off + ((uint64_t)prev * c->slots + slot) * c->slot_bytes + (uint64_t)off * 16,
                              flag_off(c, prev, slot, r, ch), seq)) return -1;
                c->forwards_posted += 1;
            }
        return drain_all(c);
    }
    // (2) relay with DIAG_CHUNKS: wait for EVERY step-1 flag of prev first (chunks geometry), then
    // forward prev's bytes sliced in the same dc geometry a direct diagonal write uses, so
    // recv[DIAG]/flag[DIAG] lands bit-identical in both modes for either step-2 geometry. The
    // time-bound convention above applies to the whole wait-all (one budget, not one per flag).
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int ch = 0; ch < c->chunks; ch++)
        for (int r = 0; r < c->rails; r++) {
            volatile uint32_t *flag = (volatile uint32_t *)(c->region + flag_off(c, prev, slot, r, ch));
            uint64_t polls = 0;
            while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != seq) {
                if ((++polls & 63) == 0) {
                    if (drain_all(c)) return -1;
                    if (!atomic_load_explicit(&c->running, memory_order_relaxed)) { snprintf(c->err, sizeof(c->err), "proxy stopped during relay wait"); return -1; }
                    if ((polls & 4095) == 0) {
                        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
                        int64_t us = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000 + ((int64_t)t1.tv_nsec - (int64_t)t0.tv_nsec) / 1000;
                        if (us > (int64_t)c->spin_limit) { snprintf(c->err, sizeof(c->err), "relay wait for rank %d rail %d chunk %d seq %u timed out after %lld us (flag read %u, doorbell seq %u)", prev, r, ch, seq, (long long)us, __atomic_load_n(flag, __ATOMIC_ACQUIRE), *(volatile uint32_t *)(c->region + c->ctrl_off)); return -1; }
                    }
                }
            }
            c->relay_polls += polls;
        }
    for (int ch = 0; ch < dc; ch++)
        for (int r = 0; r < c->rails; r++) {
            uint32_t off, packs; stripe(total, dc, c->rails, ch, r, &off, &packs);
            if (post_pair(c, 0, r, c->recv_off + ((uint64_t)prev * c->slots + slot) * c->slot_bytes + (uint64_t)off * 16, packs * 16,
                          c->recv_off + ((uint64_t)prev * c->slots + slot) * c->slot_bytes + (uint64_t)off * 16,
                          flag_off(c, prev, slot, r, ch), seq)) return -1;
            c->forwards_posted += 1;
        }
    return drain_all(c);
}

static void *proxy_main(void *arg) {
    ring_ctx_t *c = (ring_ctx_t *)arg;
    if (c->proxy_cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(c->proxy_cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    uint64_t idle = 0;
    const struct timespec nap = {0, 20000};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
        if (seq == c->last_seq) {
            c->phase = 0;
            idle++;
            if (idle % 64 == 0 && drain_all(c)) { atomic_store(&c->failed, 1); return NULL; }
            if (idle >= IDLE_SPINS) nanosleep(&nap, NULL);
            continue;
        }
        idle = 0;
        uint32_t pending = seq - c->last_seq;   // doorbell is a level: catch up in order
        if (pending > (uint32_t)c->slots) {
            snprintf(c->err, sizeof(c->err), "doorbell skipped %u ops (last %u, now %u)", pending, c->last_seq, seq);
            atomic_store(&c->failed, 1); return NULL;
        }
        // SGLANG_RDMA_RING_HOP_PROBE=1 (ctrl+124 set by the host): echo seq to ctrl+96 for the kernel's
        // round-trip probe; post_op times its own direct posts (ctrl+112 / +120)
        if (ctrl[31] != 0u) __atomic_store_n(&ctrl[24], seq, __ATOMIC_RELEASE);
        for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
            if (post_op(c, s, ctrl[4 + (s & 1u)])) { atomic_store(&c->failed, 1); return NULL; }
            c->last_seq = s;
        }
    }
    return NULL;
}

int ring_start(ring_ctx_t *c) {
    if (atomic_load(&c->running)) return 0;
    if (!c->started) { c->last_seq = ((volatile uint32_t *)(c->region + c->ctrl_off))[0]; c->started = 1; }
    atomic_store(&c->failed, 0); atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc) { atomic_store(&c->running, 0); set_err(c, "pthread_create", rc); return -1; }
    return 0;
}
void ring_stop(ring_ctx_t *c) { if (atomic_exchange(&c->running, 0)) pthread_join(c->thread, NULL); }
int ring_failed(ring_ctx_t *c) { return atomic_load(&c->failed); }
const char *ring_error(ring_ctx_t *c) { return c->err; }
uint64_t ring_stat(ring_ctx_t *c, int which) {
    switch (which) {
    case 16: return c->phase; case 17: return c->phase_seq;
    case 18: { uint64_t o = 0; for (int s = 0; s < NSIDES; s++) for (int r = 0; r < c->rails; r++) o += c->hca[s][r].outstanding; return o; }
    case 0: return c->ops_posted; case 1: return c->forwards_posted; case 2: return c->writes_completed;
    case 3: return c->last_seq; case 4: return c->relay_polls; case 5: return c->diag_posts; default: return 0;
    }
}
void ring_destroy(ring_ctx_t *c) {
    if (!c) return;
    ring_stop(c);
    for (int s = 0; s < NSIDES; s++) for (int r = 0; r < MAX_RAILS; r++) {   // diag slot is NULL when diag=0
        ring_hca_t *h = &c->hca[s][r];
        if (h->qp) ibv_destroy_qp(h->qp);
        if (h->cq) ibv_destroy_cq(h->cq);
        if (h->mr) ibv_dereg_mr(h->mr);
        if (h->pd) ibv_dealloc_pd(h->pd);
        if (h->ctx) ibv_close_device(h->ctx);
    }
    free(c);
}
"""

# --------------------------------------------------------------------------------------
# Per-op diag/relay cutoff (SGLANG_RDMA_RING_DIAG=1 + SGLANG_RDMA_RING_DIAG_MAX_BYTES > 0). Each
# entry is (anchor, replaced-by); build_proxy_source(False) returns RING_PROXY_C byte-for-byte, so
# the gate-off .so (named by the source hash) is today's, while build_proxy_source(True) splices
# exactly these edits in (each anchor occurs once; drift raises). The cutoff is SET once before the
# proxy thread starts (ring_set_diag_max_bytes) and is all-rank matched; per op the proxy reads the
# op's nbytes from the per-slot doorbell word the kernel writes (ctrl[4+slot], already the size it
# posts), so the choice rides every graph replay with no host call. Relay ops in this mode run the
# relay branch verbatim; the diag CQ is still drained (c->diag stays 1, the QP exists).
# --------------------------------------------------------------------------------------

_DIAG_CUTOFF_EDITS: tuple[tuple[str, str], ...] = (
    ("#define RING_ABI 3\n",
     "#define RING_ABI 3\n"
     "#define RING_DIAG_CUTOFF 1   /* per-op diag/relay choice by nbytes (build_proxy_source(True)) */\n"),
    ("    int started, diag_chunks, chunk_dim, diag_first;\n",
     "    int started, diag_chunks, chunk_dim, diag_first;\n"
     "    uint64_t diag_max_bytes;   /* 0 = every op diag (when diag=1); else nbytes <= it -> diag, larger -> relay */\n"
     "    uint64_t diag_ops, relay_ops;\n"),
    ("    int dc = c->diag_chunks ? c->diag_chunks : c->chunks;   // step-(2) geometry; == chunks unless DIAG_CHUNKS is set\n",
     "    int dc = c->diag_chunks ? c->diag_chunks : c->chunks;   // step-(2) geometry; == chunks unless DIAG_CHUNKS is set\n"
     "    // Per-op second hop (twin of op_uses_diag): a pure function of nbytes, which every rank shares,\n"
     "    // and of the all-rank-matched cutoff, so all four proxies pick the same path for this seq.\n"
     "    int use_diag = c->diag && (c->diag_max_bytes == 0 || (uint64_t)nbytes <= c->diag_max_bytes);\n"
     "    if (use_diag) c->diag_ops += 1; else c->relay_ops += 1;\n"),
    ("    if (c->diag && c->diag_first) {\n",
     "    if (use_diag && c->diag_first) {\n"),
    ("    if (c->diag) {\n        if (!c->diag_first)\n",
     "    if (use_diag) {\n        if (!c->diag_first)\n"),
    ("int ring_failed(ring_ctx_t *c) { return atomic_load(&c->failed); }\n",
     "int ring_failed(ring_ctx_t *c) { return atomic_load(&c->failed); }\n"
     "int ring_set_diag_max_bytes(ring_ctx_t *c, uint64_t v) {   /* before ring_start only */\n"
     "    if (atomic_load(&c->running) || !c->diag) return -1;\n"
     "    c->diag_max_bytes = v; return 0;\n"
     "}\n"),
    ("    case 3: return c->last_seq; case 4: return c->relay_polls; case 5: return c->diag_posts; default: return 0;\n",
     "    case 3: return c->last_seq; case 4: return c->relay_polls; case 5: return c->diag_posts;\n"
     "    case 6: return c->diag_ops; case 7: return c->relay_ops; default: return 0;\n"),
)


def build_proxy_source(diag_cutoff: bool) -> str:
    """C source of the proxy shim. Gate off = RING_PROXY_C unchanged, byte for byte."""
    if not diag_cutoff:
        return RING_PROXY_C
    src = RING_PROXY_C
    for anchor, repl in _DIAG_CUTOFF_EDITS:
        if src.count(anchor) != 1:
            raise RuntimeError(f"diag-cutoff patch anchor hit {src.count(anchor)}x, want 1: {anchor[:64]!r}")
        src = src.replace(anchor, repl)
    return src


def proxy_uses_cutoff(diag: bool, diag_max_bytes: int) -> bool:
    """Whether the cutoff proxy variant is built: only DIAG=1 with a positive cutoff."""
    return bool(diag) and int(diag_max_bytes) > 0

# --------------------------------------------------------------------------------------
# CUDA kernel (NVRTC). Constants injected with -D: RANK RAILS CHUNKS THREADS DTYPE_BF16.
# --------------------------------------------------------------------------------------

RING_KERNEL_CU = r"""
typedef unsigned int u32; typedef unsigned long long u64;
#define WORLD 4
#define SLOTS 2
#define FLAG_STRIDE 128
#define NEXT ((RANK + 1) % WORLD)
#define PREV ((RANK + WORLD - 1) % WORLD)
#define DIAG ((RANK + 2) % WORLD)
#ifndef DIAG_CHUNKS
#define DIAG_CHUNKS CHUNKS   /* unset = today's uniform geometry, macro-for-macro */
#endif
#ifndef POLL_BACKOFF_NS
#define POLL_BACKOFF_NS 0    /* unset = poll at full rate (today's loop) */
#endif
#ifndef POLL_HEARTBEAT
#define POLL_HEARTBEAT 0     /* unset = no heartbeat stores (today's loop) */
#endif
#ifndef HOP_PROBE
#define HOP_PROBE 0          /* unset = no proxy-hop round-trip probe (today's doorbell) */
#endif
#define CHUNK_DIM (CHUNKS > DIAG_CHUNKS ? CHUNKS : DIAG_CHUNKS)
#define NRFLAGS (RAILS * CHUNKS)
#define NFLAGS (2 * NRFLAGS + RAILS * DIAG_CHUNKS)

__device__ __forceinline__ u32 ld_acquire_sys(u64 a) { u32 v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
// diagnostics: a cache-volatile load (fetch again, do not use a cached copy) of the same word
__device__ __forceinline__ u32 ld_cv(u64 a) { u32 v; asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ u32 ld_relaxed_sys(u64 a) { u32 v; asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ u32 ld_relaxed_gpu(u64 a) { u32 v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void st_relaxed_sys(u64 a, u32 v) { asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_gpu(u64 a, u32 v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_v4(u64 a) { uint4 v; asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(u64 a) { uint4 v; asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void st_v4(u64 a, uint4 v) { asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }

#if DTYPE_BF16
#define ELEMS 8
__device__ __forceinline__ void unpack(u32 w, float *lo, float *hi) { *lo = __uint_as_float(w << 16); *hi = __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ u32 f2bf(float f) {  // RNE, NaN -> 0x7fc0 (same as the numpy reference)
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ void accum(float *acc, uint4 w, bool init) {
    u32 ws[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int i = 0; i < 4; i++) { float lo, hi; unpack(ws[i], &lo, &hi);
        if (init) { acc[2*i] = lo; acc[2*i+1] = hi; } else { acc[2*i] += lo; acc[2*i+1] += hi; } }
}
__device__ __forceinline__ uint4 pack(const float *acc) {
    uint4 v; v.x = f2bf(acc[0]) | (f2bf(acc[1]) << 16); v.y = f2bf(acc[2]) | (f2bf(acc[3]) << 16);
    v.z = f2bf(acc[4]) | (f2bf(acc[5]) << 16); v.w = f2bf(acc[6]) | (f2bf(acc[7]) << 16); return v;
}
#else
#define ELEMS 4
__device__ __forceinline__ void accum(float *acc, uint4 w, bool init) {
    float f[4] = {__uint_as_float(w.x), __uint_as_float(w.y), __uint_as_float(w.z), __uint_as_float(w.w)};
    #pragma unroll
    for (int i = 0; i < 4; i++) { if (init) acc[i] = f[i]; else acc[i] += f[i]; }
}
__device__ __forceinline__ uint4 pack(const float *acc) {
    uint4 v; v.x = __float_as_uint(acc[0]); v.y = __float_as_uint(acc[1]); v.z = __float_as_uint(acc[2]); v.w = __float_as_uint(acc[3]); return v;
}
#endif

extern "C" __global__ void __launch_bounds__(THREADS)
ring_allreduce(u64 in_base, u64 out_base, int size_packs, int nbytes,
               u64 recv_base, u64 flag_base, u64 send_base, u64 ctrl_base, u64 slot_bytes,
               u64 epoch_ptr, u64 stage_ptr, u64 tail_ptr, u64 poison_ptr, u64 poll_ptr, u32 spin_limit) {
    const int tidx = threadIdx.x, gdim = gridDim.x;
    // Every block reads the epoch before any block can advance it (advance is after the tail barrier).
    const u32 epoch = ld_relaxed_gpu(epoch_ptr);
    const u32 seq = epoch + 1u;
    const u64 slot = (u64)(seq & 1u);
    const u64 send_slot = send_base + slot * slot_bytes;
    const int index = blockIdx.x * THREADS + tidx, stride = gdim * THREADS;
    const u32 poisoned = ld_relaxed_gpu(poison_ptr);
    if (poisoned != 0u) return;   // a recorded timeout: later launches do nothing until the host raises

    // 1. stage the input into the pinned send slot (host memory, unified addressing)
    for (int i = index; i < size_packs; i += stride) st_v4(send_slot + (u64)i * 16, ld_v4(in_base + (u64)i * 16));
    __syncthreads();
    // 2. the last block to finish staging rings the proxy doorbell: nbytes per slot, then seq
    if (tidx == 0) {
        fence_sc_sys();
        u32 prior = atomicAdd((u32 *)stage_ptr, 1u);
        if ((prior + 1u) % (u32)gdim == 0u) {
            st_relaxed_sys(ctrl_base + 4, (u32)nbytes);
            st_relaxed_sys(ctrl_base + 16 + slot * 4, (u32)nbytes);
            fence_sc_sys();
#if HOP_PROBE
            // SGLANG_RDMA_RING_HOP_PROBE=1 (diagnostic): time doorbell -> the proxy's echo of seq at ctrl+96
            // (GPU -> CPU -> GPU through host memory, under the serve's load); count at ctrl+100, sum ns at
            // ctrl+104. One op at a time, so this single thread owns the accumulators.
            unsigned long long hp0, hp1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(hp0));
#endif
            st_relaxed_sys(ctrl_base, seq);
#if HOP_PROBE
            while (ld_acquire_sys(ctrl_base + 96) != seq) { }
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(hp1));
            volatile unsigned int *hc = (volatile unsigned int *)(ctrl_base + 100);
            volatile unsigned long long *hs = (volatile unsigned long long *)(ctrl_base + 104);
            *hc = *hc + 1u;
            *hs = *hs + (hp1 - hp0);
#endif
        }
    }
    // 3. wait for every (peer, rail, chunk) flag: next and prev arrive direct, diag via prev's relay
    if (tidx < NFLAGS) {
        int idx = tidx, rail, chunk; u64 src;
        if (idx < NRFLAGS) { src = NEXT; rail = idx / CHUNKS; chunk = idx % CHUNKS; }
        else if (idx < 2 * NRFLAGS) { idx -= NRFLAGS; src = PREV; rail = idx / CHUNKS; chunk = idx % CHUNKS; }
        else { idx -= 2 * NRFLAGS; src = DIAG; rail = idx / DIAG_CHUNKS; chunk = idx % DIAG_CHUNKS; }
        const u64 flag_addr = flag_base + ((((u64)src * SLOTS + slot) * RAILS + rail) * CHUNK_DIM + chunk) * FLAG_STRIDE;
        u32 polls = 0; bool timed_out = false;
        while (ld_acquire_sys(flag_addr) != seq) {
            if (++polls >= spin_limit) { timed_out = true; break; }
#if POLL_BACKOFF_NS > 0
            // SGLANG_RDMA_RING_POLL_BACKOFF_NS: after a short burst, space the polls out so a spinning
            // reader does not hold the flag's line against the NIC's write to it
            if (polls > 32u) __nanosleep(POLL_BACKOFF_NS);
#endif
#if POLL_HEARTBEAT
            // SGLANG_RDMA_RING_POLL_HEARTBEAT=1 (diagnostic): each waiting thread publishes its poll count to
            // host memory (ctrl+64+4*tidx) every 1024 polls, so an outside reader can tell a GPU that keeps
            // polling a stale flag from one whose load of the flag never returns
            if ((polls & 1023u) == 0u) st_relaxed_sys(ctrl_base + 64 + 4 * (u64)tidx, polls);
#endif
        }
        atomicAdd((u32 *)poll_ptr, polls);
        if (timed_out) {
            st_relaxed_sys(ctrl_base + 12, (u32)src);
            st_relaxed_sys(ctrl_base + 24, (u32)rail);
            st_relaxed_sys(ctrl_base + 28, (u32)chunk);
            st_relaxed_sys(ctrl_base + 32, ld_acquire_sys(flag_addr));   // the flag value seen at the timeout
            st_relaxed_sys(ctrl_base + 36, ld_cv(flag_addr));            // ... and re-fetched past the GPU caches
            st_relaxed_sys(ctrl_base + 8, seq);
            st_release_gpu(poison_ptr, seq);
        }
    }
    __syncthreads();
    const u32 failed = ld_relaxed_gpu(poison_ptr);
    if (failed == 0u) {
        // 4. reduce in fixed rank order 0..3 so every rank stores identical bits
        for (int i = index; i < size_packs; i += stride) {
            float acc[ELEMS];
            const u64 off = (u64)i * 16;
            #pragma unroll
            for (int src = 0; src < WORLD; src++) {
                uint4 w = (src == RANK) ? ld_v4(in_base + off)
                                        : ld_relaxed_sys_v4(recv_base + ((u64)src * SLOTS + slot) * slot_bytes + off);
                accum(acc, w, src == 0);
            }
            st_v4(out_base + off, pack(acc));
        }
    }
    // 5. the last block to finish publishes the next epoch (a failed seq keeps it: later launches no-op)
    fence_sc_gpu();
    __syncthreads();
    if (tidx == 0) {
        u32 prior = atomicAdd((u32 *)tail_ptr, 1u);
        if ((prior + 1u) % (u32)gdim == 0u) {
            fence_sc_gpu();
            if (ld_relaxed_sys(ctrl_base + 8) == 0u) st_release_gpu(epoch_ptr, seq);
        }
    }
}
"""


# --------------------------------------------------------------------------------------
# Instrumented kernel source (SGLANG_RDMA_RING_TIMING=1). Each entry is (anchor, replaced-by);
# build_kernel_source(False) returns RING_KERNEL_CU byte-for-byte -- NO instrumentation is
# compiled when the gate is off -- while build_kernel_source(True) splices the stamps below
# into exactly those seven spots (each anchor occurs once; a drift raises rather than
# silently skipping a stamp). Sites: entry (block 0, before staging), own doorbell rung (the
# staging counter's last block), all-flags-arrived (block 0, right after the post-wait
# barrier, so it is the LAST peer flag seen), done (the tail barrier's last block). The slot
# carries seq first so the host reader can attribute (seq % TIMING_SLOTS == slot index) and
# drop torn/in-flight/poisoned records on the monotonicity chain.
# --------------------------------------------------------------------------------------

_TIMING_EDITS: tuple[tuple[str, str], ...] = (
    ("__device__ __forceinline__ void st_v4(u64 a, uint4 v) { asm volatile(\"st.global.v4.u32 [%0], {%1,%2,%3,%4};\" :: \"l\"(a), \"r\"(v.x), \"r\"(v.y), \"r\"(v.z), \"r\"(v.w) : \"memory\"); }",
     "__device__ __forceinline__ void st_v4(u64 a, uint4 v) { asm volatile(\"st.global.v4.u32 [%0], {%1,%2,%3,%4};\" :: \"l\"(a), \"r\"(v.x), \"r\"(v.y), \"r\"(v.z), \"r\"(v.w) : \"memory\"); }\n"
     "__device__ __forceinline__ void st_relaxed_gpu_u64(u64 a, u64 v) { asm volatile(\"st.relaxed.gpu.global.u64 [%0], %1;\" :: \"l\"(a), \"l\"(v) : \"memory\"); }\n"
     "__device__ __forceinline__ u64 rd_globaltimer() { u64 v; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(v) : : \"memory\"); return v; }"),
    ("u64 epoch_ptr, u64 stage_ptr, u64 tail_ptr, u64 poison_ptr, u64 poll_ptr, u32 spin_limit) {",
     "u64 epoch_ptr, u64 stage_ptr, u64 tail_ptr, u64 poison_ptr, u64 poll_ptr, u64 timing_ptr, u32 spin_limit) {"),
    ("    const u64 slot = (u64)(seq & 1u);",
     "    const u64 slot = (u64)(seq & 1u);\n"
     "    const u64 ts_base = timing_ptr + (u64)(seq % (u64)4096u) * 40u;   // TIMING_SLOTS x 5 u64; seq = epoch + 1, the reader's invariant\n"
     "    const u64 t_entry = rd_globaltimer();"),
    ("    if (poisoned != 0u) return;   // a recorded timeout: later launches do nothing until the host raises",
     "    if (poisoned != 0u) return;   // a recorded timeout: later launches do nothing until the host raises\n"
     "    if (tidx == 0 && blockIdx.x == 0u) { st_relaxed_gpu_u64(ts_base, (u64)seq); st_relaxed_gpu_u64(ts_base + 8, t_entry); }"),
    ("            st_relaxed_sys(ctrl_base, seq);",
     "            st_relaxed_sys(ctrl_base, seq);\n"
     "            st_relaxed_gpu_u64(ts_base + 16, rd_globaltimer());"),
    ("    __syncthreads();\n    const u32 failed = ld_relaxed_gpu(poison_ptr);",
     "    __syncthreads();\n"
     "    if (tidx == 0 && blockIdx.x == 0u) st_relaxed_gpu_u64(ts_base + 24, rd_globaltimer());\n"
     "    const u32 failed = ld_relaxed_gpu(poison_ptr);"),
    ("            if (ld_relaxed_sys(ctrl_base + 8) == 0u) st_release_gpu(epoch_ptr, seq);",
     "            st_relaxed_gpu_u64(ts_base + 32, rd_globaltimer());\n"
     "            if (ld_relaxed_sys(ctrl_base + 8) == 0u) st_release_gpu(epoch_ptr, seq);"),
)


def build_kernel_source(timing: bool) -> str:
    """NVRTC source for ring_allreduce. Gate off = RING_KERNEL_CU unchanged, byte for byte."""
    if not timing:
        return RING_KERNEL_CU
    src = RING_KERNEL_CU
    for anchor, repl in _TIMING_EDITS:
        if src.count(anchor) != 1:
            raise RuntimeError(f"timing patch anchor hit {src.count(anchor)}x, want 1: {anchor[:64]!r}")
        src = src.replace(anchor, repl)
    return src


# --------------------------------------------------------------------------------------
# Proxy binding.
# --------------------------------------------------------------------------------------

_LIB_LOCK = threading.Lock()
_LIBS: dict[bool, ctypes.CDLL] = {}   # keyed by diag_cutoff: one loaded .so per proxy source variant


def default_cache_dir() -> Path:
    return Path(os.environ.get("SGLANG_RDMA_RING_CACHE_DIR",
                               os.environ.get("RING_PROXY_CACHE_DIR", os.path.expanduser("~/local/build/rdma-ring-proxy"))))


def build_proxy(cache_dir: Path, diag_cutoff: bool = False) -> Path:
    """Compile build_proxy_source(diag_cutoff) once per source hash into cache_dir; returns the .so path.

    diag_cutoff=False compiles RING_PROXY_C itself, so the .so name (source hash) is today's."""
    src = build_proxy_source(diag_cutoff).encode()
    digest = hashlib.sha256(src).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"ring_proxy-{digest}.so"
    if target.exists():
        return target
    cc = next((c for c in (os.environ.get("CC"), "gcc", "cc", "clang") if c and shutil.which(c)), None)
    if cc is None:
        raise RuntimeError("no C compiler (gcc/cc/clang) for the RDMA proxy shim")
    source = cache_dir / f"ring_proxy-{digest}.c"
    source.write_bytes(src)
    tmp = cache_dir / f".ring_proxy-{digest}-{os.getpid()}.so"
    cmd = [cc, "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(tmp), str(source), "-libverbs", "-lpthread"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("proxy build failed: " + " ".join(cmd) + "\n" + proc.stderr)
    os.replace(tmp, target)
    return target


def load_proxy(cache_dir: Path, diag_cutoff: bool = False) -> ctypes.CDLL:
    diag_cutoff = bool(diag_cutoff)
    with _LIB_LOCK:
        if diag_cutoff in _LIBS:
            return _LIBS[diag_cutoff]
        lib = ctypes.CDLL(str(build_proxy(cache_dir, diag_cutoff)), use_errno=True)
        p, u64 = ctypes.c_void_p, ctypes.c_uint64
        lib.ring_abi_version.restype = ctypes.c_int
        lib.ring_blob_bytes.restype = u64
        lib.ring_create.restype = p
        lib.ring_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
                                    ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, p, u64, ctypes.POINTER(u64), u64, ctypes.c_int, ctypes.c_char_p, u64]
        lib.ring_local_blob.restype = ctypes.c_int
        lib.ring_local_blob.argtypes = [p, p, u64]
        lib.ring_connect.restype = ctypes.c_int
        lib.ring_connect.argtypes = [p, p, u64]
        lib.ring_start.restype = ctypes.c_int
        lib.ring_start.argtypes = [p]
        lib.ring_stop.restype = None
        lib.ring_stop.argtypes = [p]
        lib.ring_failed.restype = ctypes.c_int
        lib.ring_failed.argtypes = [p]
        lib.ring_error.restype = ctypes.c_char_p
        lib.ring_error.argtypes = [p]
        lib.ring_stat.restype = u64
        lib.ring_stat.argtypes = [p, ctypes.c_int]
        lib.ring_destroy.restype = None
        lib.ring_destroy.argtypes = [p]
        if diag_cutoff:   # exported only by the cutoff variant
            lib.ring_set_diag_max_bytes.restype = ctypes.c_int
            lib.ring_set_diag_max_bytes.argtypes = [p, u64]
        if lib.ring_abi_version() != RING_ABI:
            raise RuntimeError("ring proxy ABI mismatch")
        _LIBS[diag_cutoff] = lib
        return lib


class Proxy:
    """ctypes binding of the C proxy shim. `diag=False` (default) never creates the diagonal QP."""

    def __init__(self, *, rank: int, next_hcas: Sequence[str], prev_hcas: Sequence[str], gid_index: int,
                 region_ptr: int, region_bytes: int, layout: Layout, spin_limit: int, proxy_cpu: int,
                 cache_dir: Path, diag: bool = False, diag_hcas: Optional[Sequence[str]] = None,
                 diag_max_bytes: int = 0) -> None:
        # The cutoff variant is built only for DIAG=1 with a positive cutoff; every other config
        # loads RING_PROXY_C itself (the same .so hash as before the cutoff existed).
        self.diag_cutoff = proxy_uses_cutoff(diag, diag_max_bytes)
        self.diag_max_bytes = int(diag_max_bytes) if self.diag_cutoff else 0
        self._lib = load_proxy(cache_dir, self.diag_cutoff)
        self.so_path = build_proxy(cache_dir, self.diag_cutoff)
        rails = layout.rails
        if len(next_hcas) != rails or len(prev_hcas) != rails:
            raise ValueError(f"need {rails} next and {rails} prev HCA names")
        # The diag list is one PF per rail, defaulting to the PF the endpoint's underlay sits on.
        # VERIFY: the apply round passes the real netdev-backed IB device holding the 10.226.x/29
        # GID at index 3 -- which PF differs per rank (audit 2026-09-26-cx7-diagonal-hw-e2e.md).
        diag_names = list(next_hcas) if diag_hcas is None else list(diag_hcas)
        if diag and len(diag_names) != rails:
            raise ValueError(f"need {rails} diag HCA names")
        nxt = (ctypes.c_char_p * rails)(*[n.encode() for n in next_hcas])
        prv = (ctypes.c_char_p * rails)(*[n.encode() for n in prev_hcas])
        # built even when diag=0 (the shim ignores it then) so the pointer is never dangling
        dgc = (ctypes.c_char_p * rails)(*[n.encode() for n in diag_names])
        err = ctypes.create_string_buffer(512)
        self.diag = bool(diag)
        self._ctx = self._lib.ring_create(WORLD, rank, nxt, prv, dgc, 1 if self.diag else 0,
                                          rails, gid_index, ctypes.c_void_p(region_ptr), region_bytes,
                                          layout.as_u64_array(), spin_limit, proxy_cpu, err, len(err))
        if not self._ctx:
            raise RuntimeError(f"ring proxy setup failed: {err.value.decode(errors='replace')}")
        if self.diag_cutoff and self._lib.ring_set_diag_max_bytes(self._ctx, self.diag_max_bytes) != 0:
            ctx, self._ctx = self._ctx, None
            self._lib.ring_destroy(ctx)
            raise RuntimeError("ring proxy setup failed: ring_set_diag_max_bytes refused (diag off or proxy running)")

    def local_blob(self) -> bytes:
        """This rank's ring descriptor, plus the diagonal descriptor when diag=1 (zeroed otherwise)."""
        n = int(self._lib.ring_blob_bytes())
        buf = ctypes.create_string_buffer(n)
        if self._lib.ring_local_blob(self._ctx, buf, n) != 0:
            raise RuntimeError("blob size mismatch")
        return buf.raw

    def connect(self, blobs: list[bytes]) -> None:
        """Ring QPs to next/prev, and (diag=1 only) the marked diagonal QP to rank+2, from the
        all-gathered blobs. The diagonal fields ride the same opaque blob, so no Python-side
        decode is needed; the shim reads peer_blob.diag_* only when it was created with diag=1."""
        n = int(self._lib.ring_blob_bytes())
        if len(blobs) != WORLD or any(len(b) != n for b in blobs):
            raise RuntimeError("blob size mismatch")
        joined = b"".join(blobs)
        buf = ctypes.create_string_buffer(joined, len(joined))
        if self._lib.ring_connect(self._ctx, buf, len(joined)) != 0:
            raise RuntimeError(f"ring QP connect failed: {self.error()}")

    def start(self) -> None:
        if self._lib.ring_start(self._ctx) != 0:
            raise RuntimeError(f"proxy thread failed to start: {self.error()}")

    def failed(self) -> bool:
        return bool(self._lib.ring_failed(self._ctx))

    def error(self) -> str:
        raw = self._lib.ring_error(self._ctx)
        return raw.decode(errors="replace") if raw else ""

    def diag_state(self) -> str:
        """Diagnostics: the proxy thread's phase (0 idle, 1 posting, 2 relay wait, 3 send-queue drain), the
        seq it is on, its op / write counters and the writes still outstanding on every QP."""
        s = self._lib.ring_stat
        return "proxy phase %d seq %d last_seq %d ops_posted %d writes_completed %d outstanding %d" % (
            s(self._ctx, 16), s(self._ctx, 17), s(self._ctx, 3), s(self._ctx, 0), s(self._ctx, 2), s(self._ctx, 18))

    def stats(self) -> dict:
        names = ("ops_posted", "forwards_posted", "writes_completed", "last_seq", "relay_polls", "diag_posts")
        if self.diag_cutoff:   # per-op path counts exist only in the cutoff variant
            names += ("diag_ops", "relay_ops")
        return {n: int(self._lib.ring_stat(self._ctx, i)) for i, n in enumerate(names)}

    def close(self) -> None:
        ctx, self._ctx = self._ctx, None
        if ctx:
            self._lib.ring_destroy(ctx)


# --------------------------------------------------------------------------------------
# Runtime (GPU + NIC). Mirrors the b12x RoceOneshotAllReduce surface.
# --------------------------------------------------------------------------------------


def _grid_blocks(size_packs: int, threads: int, max_blocks: int) -> int:
    """Power-of-two grid with about two packs per thread, capped (as b12x)."""
    per_block = 2 * threads
    required = max(1, (size_packs + per_block - 1) // per_block)
    return min(1 << (required - 1).bit_length(), max_blocks)


class RingOneshotAllReduce:
    """Relay one-shot all-reduce over the 4-cycle ring; constructed collectively by all 4 ranks.

    `diag=False` (default) is the shipped relay path byte-for-byte: SIDES=2 and no diagonal QP is
    created or connected, so an image without the marked underlay keeps today's behaviour and can
    never hit a diagonal RTR failure. `diag=True` adds the direct 2-hop diagonal QP to rank+2 and
    drops the proxy's forward; the reduced bits are identical between the two (the kernel is the
    same, and the diagonal data lands in the same recv slot with the same flag index). Every rank
    must agree: `diag` is in the all-rank config-match dict, so a rank on relay cannot pair with a
    rank on diag -- they fail-stop together.

    `diag_max_bytes` > 0 (with diag=True) makes the second hop per op: op_uses_diag(nbytes) picks
    the diagonal for nbytes <= diag_max_bytes and the relay forward otherwise; the proxy reads the
    op's nbytes from its doorbell word, so graph replays carry it and the kernel is unchanged. 0 =
    today's whole-runtime choice. It is in the config-match dict too (mixed cutoffs fail-stop).
    """

    def __init__(self, *, exchange_group, device, next_hcas: Sequence[str], prev_hcas: Sequence[str],
                 gid_index: int, rails: int = 2, chunks: int = 1, threads: int = 512, blocks: int = 8,
                 max_size: int = DEFAULT_MAX_SIZE, spin_limit: int = DEFAULT_SPIN_LIMIT, proxy_cpu: int = -1,
                 cache_dir: Optional[Path] = None, diag: bool = False, diag_chunks: int = 0,
                 diag_hcas: Optional[Sequence[str]] = None, timing: bool = False,
                 diag_max_bytes: int = 0) -> None:
        import torch
        import torch.distributed as dist

        self._torch = torch
        self.device = torch.device(device)
        self.rank = dist.get_rank(group=exchange_group)
        if dist.get_world_size(group=exchange_group) != WORLD:
            raise ValueError("the relay one-shot is specific to a 4-cycle: world size must be 4")
        if threads % 32 or not 32 <= threads <= 1024 or blocks < 1 or blocks & (blocks - 1):
            raise ValueError("threads multiple of 32 in 32..1024; blocks a power of two")
        if not 0 <= int(diag_chunks) <= MAX_CHUNKS:
            raise ValueError("diag_chunks must be 0 (default, = chunks) or in 1..8")
        self.diag_chunks = int(diag_chunks)
        if threads < rails * (2 * chunks + max(chunks, self.diag_chunks)):
            raise ValueError("threads must cover rails*(2*chunks + max(chunks, diag_chunks)) flag waiters")
        self._group = exchange_group
        self.max_size = int(max_size)
        self.rails, self.chunks, self.threads, self.blocks = rails, chunks, threads, blocks
        self.spin_limit = int(spin_limit)
        self.diag = bool(diag)
        if int(diag_max_bytes) < 0 or int(diag_max_bytes) >= 1 << 32:
            raise ValueError("diag_max_bytes must be 0 (off) or a positive byte count below 4 GiB")
        self.diag_max_bytes = int(diag_max_bytes)
        self.timing = bool(timing)
        self._timing_buf = None
        self._timing_addr = 0
        self._timing_seen = 0
        self.diag_hcas = None if diag_hcas is None else list(diag_hcas)
        self._closed = False
        self._lock = threading.Lock()
        self._counter_classes = blocks.bit_length()
        self._stream_event = torch.cuda.Event()
        self._last_stream = None
        self._capture_stream = None
        self._capture_id = 0
        self._kernels: dict = {}
        self._align_buffers = None
        self._proxy = None
        slot_bytes = (self.max_size + 4095) // 4096 * 4096
        self.layout = Layout(slot_bytes, rails, chunks, self.diag_chunks)
        with torch.cuda.device(self.device):
            self._region = torch.zeros(self.layout.total_bytes, dtype=torch.uint8, pin_memory=True)
            # [0] epoch, [1..cc] stage counters, [cc+1..2cc] tail counters, [2cc+1] poison, [2cc+2] polls
            self._counters = torch.zeros(3 + 2 * self._counter_classes, dtype=torch.int32, device=self.device)
            if self.timing:   # gate off (default) allocates NOTHING here and the kernel never sees it
                self._timing_buf = torch.zeros(TIMING_SLOTS * TIMING_FIELDS, dtype=torch.int64, device=self.device)
                self._timing_addr = self._timing_buf.data_ptr()
        host_ptr = self._region.data_ptr()
        if self._device_pointer(host_ptr) != host_ptr:
            raise RuntimeError("needs an integrated GPU whose device pointer of pinned memory equals the host pointer")
        self._base = host_ptr
        self._ctrl = self._region[self.layout.ctrl_off:self.layout.ctrl_off + 64].view(torch.int32).numpy()
        self._epoch_addr = self._counters.data_ptr()
        self._poison_addr = self._epoch_addr + 4 * (1 + 2 * self._counter_classes)
        self._poll_addr = self._poison_addr + 4
        cache_dir = cache_dir or default_cache_dir()
        error = None
        try:
            self._proxy = Proxy(rank=self.rank, next_hcas=next_hcas, prev_hcas=prev_hcas, gid_index=gid_index,
                                region_ptr=host_ptr, region_bytes=self.layout.total_bytes, layout=self.layout,
                                spin_limit=self.spin_limit, proxy_cpu=proxy_cpu, cache_dir=cache_dir,
                                diag=self.diag, diag_hcas=self.diag_hcas, diag_max_bytes=self.diag_max_bytes)
            blob = self._proxy.local_blob()
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error, blob, self._proxy = str(exc), b"", None
        # `diag` and `diag_max_bytes` are part of the all-rank match: a rank on relay cannot pair
        # with a rank on diag, nor two ranks that would split one op size differently, so the
        # "rank i config differs" path below fails the whole group stop.
        config = ring_config(slot_bytes=slot_bytes, rails=rails, chunks=chunks, max_size=self.max_size,
                             threads=threads, blocks=blocks, spin_limit=self.spin_limit, diag=self.diag,
                             diag_chunks=self.diag_chunks, timing=self.timing, diag_max_bytes=self.diag_max_bytes)
        statuses = [None] * WORLD
        dist.all_gather_object(statuses, (error, blob, config), group=exchange_group)
        failures = [f"rank {i}: {s[0]}" for i, s in enumerate(statuses) if s[0] is not None]
        if not failures:
            failures += config_mismatches([s[2] for s in statuses])
        if failures:
            self.close()
            raise RuntimeError("ring setup failed: " + "; ".join(failures))
        try:
            self._proxy.connect([s[1] for s in statuses])
            self._proxy.start()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        verdicts = [None] * WORLD
        dist.all_gather_object(verdicts, error, group=exchange_group)
        failures = [f"rank {i}: {v}" for i, v in enumerate(verdicts) if v is not None]
        if failures:
            self.close()
            raise RuntimeError("ring connect failed: " + "; ".join(failures))

    @staticmethod
    def _device_pointer(host_ptr: int) -> int:
        from cuda.bindings import runtime as cudart

        err, ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        return int(ptr)

    @classmethod
    def from_exchange_group(cls, *, exchange_group, device, max_size=DEFAULT_MAX_SIZE, **kw):
        return cls(exchange_group=exchange_group, device=device, max_size=max_size, **kw)

    @property
    def proxy_so_path(self) -> Optional[Path]:
        return self._proxy.so_path if self._proxy is not None else None

    # -- kernel -----------------------------------------------------------------------
    def _compile(self, dtype):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        torch = self._torch
        key = str(dtype)
        if key in self._kernels:
            return self._kernels[key]
        major, minor = torch.cuda.get_device_capability(self.device)
        defines = dict(RANK=self.rank, RAILS=self.rails, CHUNKS=self.chunks, THREADS=self.threads,
                       DIAG_CHUNKS=self.diag_chunks or self.chunks,
                       DTYPE_BF16=1 if dtype == torch.bfloat16 else 0,
                       POLL_BACKOFF_NS=int(os.environ.get("SGLANG_RDMA_RING_POLL_BACKOFF_NS", "0")),
                       POLL_HEARTBEAT=int(os.environ.get("SGLANG_RDMA_RING_POLL_HEARTBEAT", "0")),
                       HOP_PROBE=int(os.environ.get("SGLANG_RDMA_RING_HOP_PROBE", "0")))
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"-default-device"]
        opts += [f"-D{k}={v}".encode() for k, v in defines.items()]
        err, prog = nvrtc.nvrtcCreateProgram(build_kernel_source(self.timing).encode(), b"ring_allreduce.cu", 0, [], [])
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcCreateProgram: {err}")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC compile failed:\n" + log.decode(errors="replace"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, module = cu.cuModuleLoadData(cubin)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleLoadData: {err}")
        err, fn = cu.cuModuleGetFunction(module, b"ring_allreduce")
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction: {err}")
        self._kernels[key] = (module, fn)
        return self._kernels[key]

    def prepare(self, dtypes=None) -> None:
        """Compile the kernels and allocate alignment scratch ahead of CUDA graph capture."""
        torch = self._torch
        with torch.cuda.device(self.device):
            for dt in dtypes or (torch.bfloat16,):
                self._compile(dt)
            if self._align_buffers is None:
                self._align_buffers = (torch.empty(self.max_size, dtype=torch.uint8, device=self.device),
                                       torch.empty(self.max_size, dtype=torch.uint8, device=self.device))

    def should_allreduce(self, inp) -> bool:
        torch = self._torch
        if self._closed or self._proxy is None:
            raise RuntimeError("ring runtime is closed")
        if inp.dtype not in (torch.bfloat16, torch.float32) or not inp.is_cuda or inp.device != self.device:
            return False
        if not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_size and nbytes % PACK_BYTES == 0

    def _counter_addresses(self, grid: int) -> tuple[int, int]:
        cls = grid.bit_length() - 1
        return self._epoch_addr + 4 * (1 + cls), self._epoch_addr + 4 * (1 + self._counter_classes + cls)

    def _aligned(self, which: int, like):
        if self._align_buffers is None:
            if self._torch.cuda.is_current_stream_capturing():
                raise RuntimeError("call prepare() before capture (alignment scratch)")
            self.prepare((like.dtype,))
        n = like.numel() * like.element_size()
        return self._align_buffers[which][:n].view(like.dtype).view(like.shape)

    def _capture_id_of(self, stream) -> int:
        from cuda.bindings import runtime as cudart

        info = cudart.cudaStreamGetCaptureInfo(stream.cuda_stream)
        if info[0] != cudart.cudaError_t.cudaSuccess or info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            return 0
        return int(info[2])

    def _order_stream(self, capturing: bool) -> None:
        torch = self._torch
        current = torch.cuda.current_stream(self.device)
        if capturing:
            cid = self._capture_id_of(current)
            if cid != self._capture_id:
                self._capture_id, self._capture_stream = cid, current
            elif current != self._capture_stream:
                raise RuntimeError("ring collectives must all be captured on one stream")
            return
        if self._last_stream is not None and current != self._last_stream:
            current.wait_event(self._stream_event)

    def _mark_stream(self, capturing: bool) -> None:
        if capturing:
            return
        current = self._torch.cuda.current_stream(self.device)
        self._stream_event.record(current)
        self._last_stream = current

    def all_reduce(self, inp, *, out=None, stream=None):
        from cuda.bindings import driver as cu

        torch = self._torch
        with self._lock:
            self.check_health()
            if not self.should_allreduce(inp):
                raise ValueError("input not eligible for the ring one-shot all-reduce")
            if out is not None and (out.shape != inp.shape or out.dtype != inp.dtype or out.device != inp.device or not out.is_contiguous()):
                raise ValueError("out must match the input and be contiguous")
            nbytes = inp.numel() * inp.element_size()
            ctx = torch.cuda.stream(stream) if stream is not None else _Null()
            with torch.cuda.device(self.device), ctx:
                capturing = torch.cuda.is_current_stream_capturing()
                if capturing and str(inp.dtype) not in self._kernels:
                    raise RuntimeError("kernel must be prepared before CUDA graph capture")
                _, fn = self._compile(inp.dtype)
                if out is None:
                    out = torch.empty_like(inp)
                src = inp if inp.data_ptr() % PACK_BYTES == 0 else self._aligned(0, inp)
                if src is not inp:
                    src.copy_(inp)
                dst = out if out.data_ptr() % PACK_BYTES == 0 else self._aligned(1, out)
                size_packs = nbytes // PACK_BYTES
                grid = _grid_blocks(size_packs, self.threads, self.blocks)
                stage_ptr, tail_ptr = self._counter_addresses(grid)
                self._order_stream(capturing)
                while capturing and PRE_LAUNCH:
                    PRE_LAUNCH.pop(0)()
                args = [src.data_ptr(), dst.data_ptr(), size_packs, nbytes,
                        self._base + self.layout.recv_off, self._base + self.layout.flag_off,
                        self._base + self.layout.send_off, self._base + self.layout.ctrl_off, self.layout.slot_bytes,
                        self._epoch_addr, stage_ptr, tail_ptr, self._poison_addr, self._poll_addr]
                types = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32) + (ctypes.c_uint64,) * 10
                if self.timing:   # the gated kernel build takes timing_ptr just before spin_limit
                    args.append(self._timing_addr)
                    types += (ctypes.c_uint64,)
                args.append(self.spin_limit)
                types += (ctypes.c_uint32,)
                args, types = tuple(args), tuple(types)
                sh = torch.cuda.current_stream(self.device).cuda_stream
                (err,) = cu.cuLaunchKernel(fn, grid, 1, 1, self.threads, 1, 1, 0, cu.CUstream(sh), (args, types), 0)
                if err != cu.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"cuLaunchKernel: {err}")
                while capturing and POST_LAUNCH:
                    POST_LAUNCH.pop(0)()
                if dst is not out:
                    out.copy_(dst)
                self._mark_stream(capturing)
        if not capturing:
            self.check_health()
        return out

    def check_health(self) -> None:
        if self._proxy is not None and self._proxy.failed():
            raise RuntimeError(f"ring proxy failed: {self._proxy.error()}")
        failed_seq = int(self._ctrl[2])
        if failed_seq:
            src, rail, chunk = int(self._ctrl[3]), int(self._ctrl[6]), int(self._ctrl[7])
            # the same flag word read by the HOST now (the kernel's read is ctrl[8]): equal = the write never
            # arrived; host newer = the GPU kept reading a stale copy of host memory the NIC had written
            off = self.layout.flag(src, failed_seq & 1, rail, chunk)
            host_flag = int(self._region[off:off + 4].view(self._torch.int32)[0]) & 0xffffffff
            proxy = self._proxy.diag_state() if self._proxy is not None else "no proxy"
            raise RuntimeError(f"ring collective on rank {self.rank} timed out waiting for rank {src} "
                               f"rail {rail} chunk {chunk} at seq {failed_seq} "
                               f"(flag read {int(self._ctrl[8]) & 0xffffffff}, ld.cv {int(self._ctrl[9]) & 0xffffffff}, "
                               f"host reads {host_flag}, "
                               f"doorbell seq {int(self._ctrl[0]) & 0xffffffff}; {proxy}); runtime poisoned")

    def capture(self, stream=None):
        return _Capture(self)

    def polls(self) -> int:
        """GPU flag polls accumulated since the last reset (device word)."""
        return int(self._counters[2 + 2 * self._counter_classes].item())

    def reset_polls(self) -> None:
        self._counters[2 + 2 * self._counter_classes] = 0

    def epoch(self) -> int:
        """Ops completed on the GPU (device word; a graph replay advances it, the host never sees the launch)."""
        return int(self._counters[0].item())

    def timing_stats(self) -> Optional[dict]:
        """Per-op %globaltimer deltas since the last call (None = gate off; dict(n=0) = no new ops).

        A slot is a record iff seq in (seen, epoch], seq % TIMING_SLOTS == slot index, and the four
        stamps form a monotonic chain; that drops in-flight, poisoned and buffer-wrap torn records.
        arrival_wait_us = t_all_arrived - t_local_posted (the wait for the LAST peer flag after this
        rank rang its own doorbell), op_us = t_done - t_entry. Nearest-rank median/p90, in us.
        """
        if self._timing_buf is None:
            return None
        e = self.epoch()
        if e <= self._timing_seen:
            return dict(n=0)
        buf = self._timing_buf.cpu().numpy().reshape(TIMING_SLOTS, TIMING_FIELDS)
        recs = timing_records(buf, self._timing_seen, e)
        waits = [(t2 - t1) // 1000 for _, _, t1, t2, _ in recs]
        ops = [(t3 - t0) // 1000 for _, t0, _, _, t3 in recs]
        dump_dir = os.environ.get("SGLANG_RDMA_RING_TIMING_DUMP", "")
        if dump_dir and recs:
            # Raw records for a cross-rank join on seq (the ranks' %globaltimer clocks are not synchronized,
            # so only per-rank deltas are comparable): seq, posted-entry, arrived-posted, done-arrived in ns.
            with open(os.path.join(dump_dir, "timing-rank%s.csv" % getattr(self, "rank", "x")), "a") as f:
                for seq, t0, t1, t2, t3 in recs:
                    f.write("%d,%d,%d,%d,%d\n" % (seq, t0, t1 - t0, t2 - t1, t3 - t2))
        self._timing_seen = e
        if not waits:
            return dict(n=0)
        def _q(xs: list[int], num: int, den: int) -> int:
            s = sorted(xs)
            return s[max(0, (num * len(s) + den - 1) // den - 1)]
        return dict(n=len(waits),
                    arrival_wait_us_med=_q(waits, 1, 2), arrival_wait_us_p90=_q(waits, 9, 10),
                    op_us_med=_q(ops, 1, 2), op_us_p90=_q(ops, 9, 10))

    def stats(self) -> dict:
        info = dict(rank=self.rank, rails=self.rails, chunks=self.chunks, epoch=self.epoch(),
                    error_seq=int(self._ctrl[2]), ctrl_seq=int(self._ctrl[0]), polls=self.polls(),
                    diag=self.diag)
        if self.diag_max_bytes:   # absent at 0 so today's stats line is unchanged
            info["diag_max_bytes"] = self.diag_max_bytes
        if self._proxy is not None:
            info.update(self._proxy.stats())
        return info

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._torch.cuda.synchronize(self.device)
            except Exception:  # noqa: BLE001
                pass
            if self._proxy is not None:
                self._proxy.close()
                self._proxy = None


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _Capture:
    def __init__(self, rt: RingOneshotAllReduce) -> None:
        self.rt = rt

    def __enter__(self):
        self.rt._last_stream = None
        self.rt._capture_stream = None
        self.rt._capture_id = 0
        return self.rt

    def __exit__(self, *exc):
        self.rt._capture_stream = None
        self.rt._capture_id = 0
        self.rt._last_stream = None
        return False


# --------------------------------------------------------------------------------------
# sglang adapter: env contract, eligibility + counters, health registry.
# --------------------------------------------------------------------------------------


def env_config(env=None) -> dict:
    """The SGLANG_RDMA_RING_* contract (module docstring), parsed once."""
    env = os.environ if env is None else env

    def _int(name: str, default: int) -> int:
        return int(env.get(name, default))

    return dict(
        enabled=env.get("SGLANG_RDMA_RING_ALLREDUCE", "0") == "1",
        max_size=_int("SGLANG_RDMA_RING_MAX_SIZE", DEFAULT_MAX_SIZE),
        rails=_int("SGLANG_RDMA_RING_RAILS", 2),
        chunks=_int("SGLANG_RDMA_RING_CHUNKS", 1),
        diag_chunks=_int("SGLANG_RDMA_RING_DIAG_CHUNKS", 0),
        blocks=_int("SGLANG_RDMA_RING_BLOCKS", 8),
        threads=_int("SGLANG_RDMA_RING_THREADS", 512),
        spin_limit=_int("SGLANG_RDMA_RING_SPIN_LIMIT", DEFAULT_SPIN_LIMIT),
        proxy_cpu=_int("SGLANG_RDMA_RING_PROXY_CPU", -1),
        gid_index=int(env.get("SGLANG_RDMA_RING_GID_INDEX", env.get("NCCL_IB_GID_INDEX", "1"))),
        next_hcas=tuple(env.get("SGLANG_RDMA_RING_NEXT_HCA", ",".join(DEFAULT_NEXT_HCAS)).split(",")),
        prev_hcas=tuple(env.get("SGLANG_RDMA_RING_PREV_HCA", ",".join(DEFAULT_PREV_HCAS)).split(",")),
        # 0 = relay path (no diagonal QP at all); nonzero = the direct marked diagonal to rank+2.
        # Carried into the runtime's all-rank config match, so a mixed farm fails-stop, never mixes.
        diag=_int("SGLANG_RDMA_RING_DIAG", 0),
        # 0 = today (DIAG decides for every op); N > 0 with DIAG=1 = per-op: nbytes <= N diag, else
        # relay (op_uses_diag). All-rank matched, so a mixed farm fails-stop, never splits an op.
        diag_max_bytes=_int("SGLANG_RDMA_RING_DIAG_MAX_BYTES", 0),
        # empty = default to the next/PF set the underlay sits on. VERIFY: the apply round passes the
        # real netdev-backed IB device holding the 10.226.x/29 GID at index DIAG_GID_INDEX (3).
        diag_hcas=tuple(s for s in env.get("SGLANG_RDMA_RING_DIAG_HCA", "").split(",") if s),
        # 0 = today's kernel byte-for-byte; 1 = per-op %globaltimer stamps (see build_kernel_source).
        timing=_int("SGLANG_RDMA_RING_TIMING", 0),
        cache_dir=Path(env.get("SGLANG_RDMA_RING_CACHE_DIR", os.path.expanduser("~/local/build/rdma-ring-proxy"))),
        log_every_s=float(env.get("SGLANG_RDMA_RING_LOG_EVERY_S", 300)),
    )


def install_kernel_load_sync(device, log: Callable[[str], None]) -> bool:
    """SGLANG_RDMA_RING_SYNC_ON_KERNEL_LOAD=1: synchronize `device` before Triton loads a kernel module.

    A lazy Triton module load (cuModuleLoadData at a kernel's first launch, e.g. a new batch shape) while a
    relay kernel is in flight on this GPU stops that GPU's host-memory traffic for ~210-230 s: the relay
    kernel's staging, doorbell and polls stay invisible, its writes never leave, and every rank stalls
    (audit 2026-10-08-sglang-rebase-relay-c16-crash). With the GPU idle at the load, nothing is in flight.
    Skipped while a stream is capturing (a sync would invalidate the capture). Added to Triton's hook chain
    (a HookChain in this Triton; an older Triton's single callable is wrapped instead)."""
    if os.environ.get("SGLANG_RDMA_RING_SYNC_ON_KERNEL_LOAD", "0") != "1":
        return False
    try:
        from triton import knobs  # noqa: PLC0415
    except ImportError:
        log("[rdma-ring] kernel-load sync: triton.knobs not importable, not installed")
        return False
    import torch  # noqa: PLC0415

    prior = knobs.runtime.kernel_load_start_hook
    state = {"n": 0}

    def hook(module, function, name, metadata_group, hash_):
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize(device)
            state["n"] += 1
            if state["n"] <= 3 or state["n"] % 100 == 0:
                log(f"[rdma-ring] kernel-load sync #{state['n']} before loading {name}")

    if hasattr(prior, "add"):   # HookChain: runs every added hook in order
        prior.add(hook)
    else:
        def chained(*args):
            hook(*args)
            if prior is not None:
                prior(*args)
        knobs.runtime.kernel_load_start_hook = chained
    log("[rdma-ring] kernel-load sync installed (SGLANG_RDMA_RING_SYNC_ON_KERNEL_LOAD=1)")
    return True


_LIVE: "weakref.WeakSet[RingAllReduce]" = weakref.WeakSet()
_HEALTH_CALLS = 0
_LAST_LOG_T = 0.0


class RingAllReduce:
    """What PyNcclCommunicator holds: eligibility + routed/fallback counters over the runtime."""

    FALLBACK_REASONS = ("not_sum", "dtype", "too_large", "not_contiguous", "unaligned", "device")

    def __init__(self, rt, config: dict, log: Optional[Callable[[str], None]] = None) -> None:
        self._rt = rt
        self.config = config
        self.log = log or (lambda msg: None)
        self.routed_calls = 0            # host-side calls routed here (eager + one per captured graph node)
        self.fallback_calls = 0          # host-side calls left on NCCL (ineligible)
        self.fallback_reasons = {r: 0 for r in self.FALLBACK_REASONS}
        self.max_nbytes = 0
        self._closed = False
        _LIVE.add(self)

    @classmethod
    def from_group(cls, exchange_group, device, config: Optional[dict] = None,
                   log: Optional[Callable[[str], None]] = None) -> "RingAllReduce":
        """Build the runtime collectively over the TP group's Gloo cpu_group (all 4 ranks call this)."""
        import torch
        import torch.distributed as dist

        cfg = dict(config or env_config())
        rails = int(cfg["rails"])
        # SGLANG_RDMA_RING_DIAG_HCA_BY_RANK="<rank0 list>;<rank1 list>;...": the diagonal PF differs per rank
        # (d47b/6a8e f0, b4b3/d7ae f1) while one environment reaches every rank; picked here by group rank
        by_rank = os.environ.get("SGLANG_RDMA_RING_DIAG_HCA_BY_RANK", "")
        if by_rank and not cfg.get("diag_hcas"):
            entries = by_rank.split(";")
            r = dist.get_rank(group=exchange_group)
            if r < len(entries) and entries[r]:
                cfg["diag_hcas"] = tuple(entries[r].split(","))
        rt = RingOneshotAllReduce(
            exchange_group=exchange_group, device=device,
            next_hcas=list(cfg["next_hcas"])[:rails], prev_hcas=list(cfg["prev_hcas"])[:rails],
            gid_index=int(cfg["gid_index"]), rails=rails, chunks=int(cfg["chunks"]),
            threads=int(cfg["threads"]), blocks=int(cfg["blocks"]), max_size=int(cfg["max_size"]),
            spin_limit=int(cfg["spin_limit"]), proxy_cpu=int(cfg["proxy_cpu"]), cache_dir=Path(cfg["cache_dir"]),
            diag=bool(int(cfg.get("diag", 0))), diag_chunks=int(cfg.get("diag_chunks", 0)),
            diag_hcas=(list(cfg["diag_hcas"]) if cfg.get("diag_hcas") else None),
            timing=bool(int(cfg.get("timing", 0))),
            diag_max_bytes=int(cfg.get("diag_max_bytes", 0)),
        )
        # prepare() is a per-rank NVRTC compile after the last exchange; line the ranks up
        # before op 1 so the first relay wait measures the protocol, not compile skew.
        rt.prepare((torch.bfloat16, torch.float32))
        dist.barrier(group=exchange_group)
        obj = cls(rt, cfg, log)
        install_kernel_load_sync(device, obj.log)
        if os.environ.get("SGLANG_RDMA_RING_HOP_PROBE", "0") == "1":
            # arm the proxy half of the hop probe (ctrl+124); the kernel half is the HOP_PROBE define.
            # Read the totals from outside with rebase-ab/hop-probe-read.py (ctrl+96..127).
            ctypes.c_uint32.from_address(rt._base + rt.layout.ctrl_off + 124).value = 1
            obj.log("[rdma-ring] hop probe armed (SGLANG_RDMA_RING_HOP_PROBE=1)")
        atexit.register(obj.close)
        cut = f"diag_max_bytes={rt.diag_max_bytes} " if rt.diag_max_bytes else ""   # absent at 0 = today's line
        # diagnostics: where the NIC-registered region lives, so its physical pages can be read from
        # /proc/<pid>/pagemap (rdma-region-pfns.py) at attach and again at a failure
        obj.log(f"[rdma-ring] region pid={os.getpid()} va=0x{rt._base:x} bytes={rt.layout.total_bytes} "
                f"flag_off={rt.layout.flag_off} ctrl_off={rt.layout.ctrl_off}")
        obj.log(f"[rdma-ring] attached rank={rt.rank} rails={rt.rails} chunks={rt.chunks} blocks={rt.blocks} "
                f"threads={rt.threads} max_size={rt.max_size} gid={cfg['gid_index']} diag={int(rt.diag)} {cut}"
                f"next={','.join(cfg['next_hcas'][:rails])} prev={','.join(cfg['prev_hcas'][:rails])} "
                f"diag_hca={','.join(rt.diag_hcas) if rt.diag_hcas else ('(= next)' if rt.diag else '(none, no diag QP)')} "
                f"spin_limit={rt.spin_limit} proxy_cpu={cfg['proxy_cpu']} proxy={rt.proxy_so_path} "
                f"kernels={sorted(rt._kernels)}")
        return obj

    # -- routing --------------------------------------------------------------------
    def route(self, tensor, is_sum: bool = True) -> Optional[str]:
        """None = take this op (counted routed); else the counted reason it stays on NCCL."""
        reason = self._reason(tensor, is_sum)
        if reason is None:
            self.routed_calls += 1
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes > self.max_nbytes:
                self.max_nbytes = nbytes
            return None
        self.fallback_calls += 1
        self.fallback_reasons[reason] += 1
        return reason

    def _reason(self, tensor, is_sum: bool) -> Optional[str]:
        if not is_sum:
            return "not_sum"
        dt = str(tensor.dtype)
        if dt not in ("torch.bfloat16", "torch.float32"):
            return "dtype"
        if not tensor.is_contiguous():
            return "not_contiguous"
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes <= 0 or nbytes > int(self.config["max_size"]):
            return "too_large"
        if nbytes % PACK_BYTES:
            return "unaligned"
        if not self._rt.should_allreduce(tensor):
            return "device"
        return None

    def all_reduce(self, tensor, out=None, stream=None):
        # diagnostics: the relay kernel takes its seq from one device epoch and assumes no two relay ops run at
        # once; log every distinct CUDA stream it is launched on (capture or eager), with the caller, once each
        import torch  # noqa: PLC0415
        sid = (stream or torch.cuda.current_stream()).cuda_stream
        seen = self.__dict__.setdefault("_streams_seen", set())
        if sid not in seen:
            seen.add(sid)
            f, where = sys._getframe(1), "?"
            while f is not None:
                s = f.f_locals.get("self")
                if s is not None and type(s).__module__.startswith("sglang.srt.models"):
                    where = "%s.%s layer=%s" % (type(s).__name__, f.f_code.co_name, getattr(s, "layer_id", "-"))
                    break
                f = f.f_back
            self.log("[rdma-ring] stream #%d 0x%x first used by %s (capturing=%s, nbytes=%d)" % (
                len(seen), sid, where, torch.cuda.is_current_stream_capturing(),
                tensor.numel() * tensor.element_size()))
        if _CALLSITE["path"] is None:
            _CALLSITE["path"] = os.environ.get("SGLANG_RDMA_RING_CALLSITE_LOG", "")
        if _CALLSITE["path"]:
            _log_callsite(tensor)
        return self._rt.all_reduce(tensor, out=out, stream=stream)

    def capture(self, stream=None):
        return self._rt.capture(stream)

    def check_health(self) -> None:
        if not self._closed:
            self._rt.check_health()

    def stats(self) -> dict:
        info = dict(routed_calls=self.routed_calls, fallback_calls=self.fallback_calls,
                    fallback_reasons={k: v for k, v in self.fallback_reasons.items() if v},
                    max_nbytes=self.max_nbytes)
        if not self._closed:
            info.update(self._rt.stats())
        return info

    def summary(self) -> str:
        return "[rdma-ring] stats " + " ".join(f"{k}={v}" for k, v in self.stats().items())

    def timing_summary_line(self) -> Optional[str]:
        """The `[rdma-ring] timing ...` line beside the stats line, or None (gate off / no new ops)."""
        if self._closed:
            return None
        fn = getattr(self._rt, "timing_stats", None)
        s = fn() if fn is not None else None
        if not s or not s.get("n"):
            return None
        return f"[rdma-ring] timing rank={getattr(self._rt, 'rank', '?')} " + " ".join(f"{k}={v}" for k, v in s.items())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.log(self.summary().replace("stats", "closing", 1))
        except Exception:  # noqa: BLE001
            pass
        self._rt.close()


def check_health_all(log_every_s: Optional[float] = None, now: Optional[float] = None) -> None:
    """Per scheduler step: raise on any poisoned runtime; log stats every `log_every_s` seconds."""
    global _HEALTH_CALLS, _LAST_LOG_T
    _HEALTH_CALLS += 1
    t = time.monotonic() if now is None else now
    for ring in list(_LIVE):
        if ring._closed:
            continue
        ring.check_health()
        every = float(ring.config.get("log_every_s", 300)) if log_every_s is None else log_every_s
        if every > 0 and t - _LAST_LOG_T >= every:
            _LAST_LOG_T = t
            ring.log(ring.summary() + f" health_calls={_HEALTH_CALLS}")
            line = ring.timing_summary_line()
            if line is not None:
                ring.log(line)


# --------------------------------------------------------------------------------------
# Protocol simulator (--self-test): no NIC, no GPU. Same layout, same proxy logic, same
# reduction; QPs deliver in order but the scheduler interleaves ranks and QPs at random.
# --------------------------------------------------------------------------------------


class SimFabric:
    """Side 0 = next, 1 = prev, and (diag=True only) 2 = the diagonal QP to rank+2."""

    def __init__(self, layout: Layout, rng: random.Random, diag: bool = False) -> None:
        self.layout = layout
        self.rng = rng
        self.diag = bool(diag)
        self.regions = [np.zeros(layout.total_bytes, dtype=np.uint8) for _ in range(WORLD)]
        # one in-order queue per (src rank, side, rail) = one RC QP; the diag side exists only in diag mode
        self.qps: dict[tuple[int, int, int], list] = {(r, s, rl): [] for r in range(WORLD)
                                                     for s in range(3 if self.diag else 2) for rl in range(layout.rails)}
        # detectors: generation tag of the data last staged/landed in each slot
        self.send_tag = [[0] * SLOTS for _ in range(WORLD)]
        self.recv_tag = [[[0] * SLOTS for _ in range(WORLD)] for _ in range(WORLD)]
        self.delivered_writes = 0

    def u32(self, rank: int, off: int) -> int:
        return int(self.regions[rank][off:off + 4].view(np.uint32)[0])

    def set_u32(self, rank: int, off: int, v: int) -> None:
        self.regions[rank][off:off + 4].view(np.uint32)[0] = v

    def post(self, src: int, side: int, rail: int, local_off: int, nbytes: int, remote_data_off: int,
             remote_flag_off: int, seq: int, src_tag: tuple) -> None:
        # side 2 (diag mode only) is the direct QP to the opposite rank; 0/1 are unchanged
        dst = ((src + 1) % WORLD, (src - 1) % WORLD, (src + 2) % WORLD)[side]
        self.qps[(src, side, rail)].append((dst, local_off, nbytes, remote_data_off, remote_flag_off, seq, src_tag))

    def deliver_one(self) -> bool:
        """Execute the head WR pair of one random non-empty QP (the NIC reads the source at execution time)."""
        live = [k for k, q in self.qps.items() if q]
        if not live:
            return False
        k = self.rng.choice(live)
        dst, local_off, nbytes, rdata, rflag, seq, src_tag = self.qps[k].pop(0)
        src = k[0]
        kind, a, b = src_tag
        # slot-overwrite detector: the bytes the NIC reads must still belong to `seq`
        if kind == "send":
            assert self.send_tag[src][a] == seq, f"send slot {a} of rank {src} overwritten before the NIC read it (seq {seq})"
        else:
            assert self.recv_tag[src][a][b] == seq, f"recv[{a}][{b}] on rank {src} overwritten before the relay read it (seq {seq})"
        if nbytes:
            self.regions[dst][rdata:rdata + nbytes] = self.regions[src][local_off:local_off + nbytes]
            # who lands where: recv[orig][slot] on dst; tag it with seq
            orig = (rdata - self.layout.recv_off) // (SLOTS * self.layout.slot_bytes)
            slot = (rdata - self.layout.recv_off) // self.layout.slot_bytes % SLOTS
            self.recv_tag[dst][orig][slot] = seq
        self.set_u32(dst, rflag, seq)   # flag after data, same QP
        self.delivered_writes += 1
        return True


class SimProxy:
    """Python twin of proxy_main/post_op as a resumable state machine.

    `diag=False` runs today's relay code path verbatim (and is the default). `diag=True` replaces
    step (2) with the direct diagonal post to rank+2 and forwards nothing -- same recv slot, same
    flag index, same bytes, so the reduced result must be identical (asserted in self_test).
    """

    def __init__(self, fabric: SimFabric, rank: int, diag: bool = False, diag_chunks: int = 0,
                 mutant_swap_rails: bool = False, diag_max_bytes: int = 0) -> None:
        self.f, self.rank, self.L = fabric, rank, fabric.layout
        self.diag = bool(diag)
        self.diag_max_bytes = int(diag_max_bytes)   # twin of the cutoff variant's c->diag_max_bytes
        self.diag_ops = self.relay_ops = 0
        self.dc = diag_chunks or fabric.layout.chunks
        self.dfirst = bool(diag) and diag_chunks != 0
        self.mutant_swap_rails = mutant_swap_rails   # test-only: forward the WRONG rail's slice
        self.last_seq = 0
        self.relay = None   # (seq, phase counter) pending step-2 position (diag mode never sets it)
        self.ops_posted = self.forwards = self.diag_posts = 0
        self.max_pending_seen = 0

    def step(self) -> None:
        f, L, r = self.f, self.L, self.rank
        prev, ctrl = (r - 1) % WORLD, L.ctrl_off
        if self.relay is None:
            seq_now = f.u32(r, ctrl)
            if seq_now == self.last_seq:
                return
            pending = seq_now - self.last_seq
            self.max_pending_seen = max(self.max_pending_seen, pending)
            assert pending <= SLOTS, f"doorbell skipped {pending} ops"
            s = self.last_seq + 1
            nbytes = f.u32(r, ctrl + 16 + (s & 1) * 4)
            slot, total = s & 1, nbytes // 16
            # per-op second hop from the doorbell's nbytes word (the C variant's `use_diag` line);
            # diag_max_bytes == 0 reduces to self.diag, i.e. today's code path for every op
            use_diag = op_uses_diag(nbytes, self.diag, self.diag_max_bytes)
            if use_diag:
                self.diag_ops += 1
            else:
                self.relay_ops += 1
            if use_diag and self.dfirst:
                # twin of the post_op diag-first branch: the marked stripe BEFORE the neighbour writes
                for ch in range(self.dc):
                    for rl in range(L.rails):
                        off, packs = stripe(total, self.dc, L.rails, ch, rl)
                        f.post(r, 2, rl, L.send(slot) + off * 16, packs * 16, L.recv(r, slot) + off * 16,
                               L.flag(r, slot, rl, ch), s, ("send", slot, None))
                        self.diag_posts += 1
            for side in (0, 1):
                for ch in range(L.chunks):
                    for rl in range(L.rails):
                        off, packs = stripe(total, L.chunks, L.rails, ch, rl)
                        f.post(r, side, rl, L.send(slot) + off * 16, packs * 16, L.recv(r, slot) + off * 16,
                               L.flag(r, slot, rl, ch), s, ("send", slot, None))
            self.ops_posted += 1
            if use_diag:
                # twin of post_op's diag branch: my own input straight to the opposite rank (whose
                # DIAG is me), no relay wait, and NO forward -- `forwards` stays 0 in this mode.
                if not self.dfirst:
                    for ch in range(self.dc):
                        for rl in range(L.rails):
                            off, packs = stripe(total, self.dc, L.rails, ch, rl)
                            f.post(r, 2, rl, L.send(slot) + off * 16, packs * 16, L.recv(r, slot) + off * 16,
                                   L.flag(r, slot, rl, ch), s, ("send", slot, None))
                            self.diag_posts += 1
                self.last_seq = s
                return
            self.relay = (s, 0)
        s, k = self.relay
        slot, total = s & 1, f.u32(r, ctrl + 16 + (s & 1) * 4) // 16
        nwait = L.chunks * L.rails if self.dc != L.chunks else 0   # dc != chunks: hoist the step-1 waits
        while True:
            if k < nwait:
                ch, rl = k // L.rails, k % L.rails
                if f.u32(r, L.flag(prev, slot, rl, ch)) != s:
                    self.relay = (s, k)
                    return
                k += 1
                continue
            j = k - nwait
            if j >= self.dc * L.rails:
                self.last_seq = s
                self.relay = None
                return
            ch2, rl = j // L.rails, j % L.rails
            if nwait == 0:   # dc == chunks: today's interleaved wait-then-forward, verbatim
                if f.u32(r, L.flag(prev, slot, rl, ch2)) != s:
                    self.relay = (s, k)
                    return
            off_d, _ = stripe(total, self.dc, L.rails, ch2, rl)
            src_rail = (L.rails - 1 - rl) if self.mutant_swap_rails else rl
            off_s, packs = stripe(total, self.dc, L.rails, ch2, src_rail)
            f.post(r, 0, rl, L.recv(prev, slot) + off_s * 16, packs * 16, L.recv(prev, slot) + off_d * 16,
                   L.flag(prev, slot, rl, ch2), s, ("recv", prev, slot))
            self.forwards += 1
            k += 1


class SimKernel:
    """Python twin of ring_allreduce: stage+doorbell, then (resumable) wait, reduce, epoch."""

    def __init__(self, fabric: SimFabric, rank: int) -> None:
        self.f, self.rank, self.L = fabric, rank, fabric.layout
        self.epoch = 0
        self.inp = None
        self.state = "idle"
        self.out = None

    def launch(self, inp_bits: np.ndarray) -> None:
        assert self.state == "idle"
        f, L, r = self.f, self.L, self.rank
        seq = self.epoch + 1
        slot = seq & 1
        nbytes = inp_bits.nbytes
        self.inp, self.seq = inp_bits, seq
        f.regions[r][L.send(slot):L.send(slot) + nbytes] = inp_bits.view(np.uint8)
        f.send_tag[r][slot] = seq
        f.set_u32(r, L.ctrl_off + 4, nbytes)
        f.set_u32(r, L.ctrl_off + 16 + slot * 4, nbytes)
        f.set_u32(r, L.ctrl_off, seq)
        self.state = "waiting"

    def step(self) -> bool:
        if self.state != "waiting":
            return False
        f, L, r, seq = self.f, self.L, self.rank, self.seq
        slot = seq & 1
        peers = [(r + 1) % WORLD, (r - 1) % WORLD, (r + 2) % WORLD]
        for src in peers:
            nch = (L.diag_chunks or L.chunks) if src == peers[2] else L.chunks
            for rl in range(L.rails):
                for ch in range(nch):
                    if f.u32(r, L.flag(src, slot, rl, ch)) != seq:
                        return False
        nbytes = self.inp.nbytes
        srcs = []
        for src in range(WORLD):
            if src == r:
                srcs.append(self.inp)
            else:
                assert f.recv_tag[r][src][slot] == seq, f"rank {r} reads recv[{src}][{slot}] with stale tag"
                srcs.append(f.regions[r][L.recv(src, slot):L.recv(src, slot) + nbytes].view(np.uint16).copy())
        self.out = reference_sum_bf16(srcs)
        self.epoch = seq
        self.state = "idle"
        return True


class _StubTensor:
    """Duck-typed tensor for the adapter self-test (no torch)."""

    def __init__(self, nbytes: int, dtype: str = "torch.bfloat16", contiguous: bool = True) -> None:
        self._n, self.dtype, self._c = nbytes, dtype, contiguous

    def numel(self) -> int:
        return self._n // self.element_size()

    def element_size(self) -> int:
        return 4 if self.dtype == "torch.float32" else 2

    def is_contiguous(self) -> bool:
        return self._c


class _StubRuntime:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.closed = False
        self.calls = 0
        self.rank = 0

    def should_allreduce(self, t) -> bool:
        return self.ok

    def all_reduce(self, t, out=None, stream=None):
        self.calls += 1
        return out if out is not None else t

    def check_health(self) -> None:
        if not self.ok:
            raise RuntimeError("stub poisoned")

    def stats(self) -> dict:
        return dict(epoch=self.calls)

    def close(self) -> None:
        self.closed = True

    def capture(self, stream=None):
        return _Null()


def _sim_drive(fab: SimFabric, proxies: list, kernels: list, op_inputs: list, rng: random.Random,
               max_steps: int = 200000) -> tuple[dict, int, int]:
    """Random interleaving of kernels / proxies / QP deliveries until every rank ran every op of
    `op_inputs` (a list, per op, of the 4 ranks' bf16 inputs) or `max_steps`. -> (outs, done, steps),
    outs[(rank, op)] = that rank's reduced bits. A run that stops short is a stall: on the device
    the kernel's spin limit poisons the runtime there (fail-stop), it never returns data."""
    nops = len(op_inputs)
    next_op = [0] * WORLD
    outs: dict[tuple[int, int], np.ndarray] = {}
    done = steps = 0
    while done < nops * WORLD and steps < max_steps:
        steps += 1
        actor, r = rng.randrange(3), rng.randrange(WORLD)
        if actor == 0:
            k = kernels[r]
            if k.state == "idle" and next_op[r] < nops:
                k.launch(op_inputs[next_op[r]][r])
            elif k.state == "waiting" and k.step():
                outs[(r, next_op[r])] = k.out
                next_op[r] += 1
                done += 1
        elif actor == 1:
            if rng.random() < 0.7:
                proxies[r].step()
        else:
            fab.deliver_one()
    return outs, done, steps


def self_test(seed: int = 1234, ops: int = 40, verbose: bool = True) -> int:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    failures = 0

    # 1. stripe geometry: every (chunks, rails) partition covers [0, total) exactly once
    for total in (0, 1, 7, 640, 20480, 131071):
        for chunks in (1, 2, 3, 4, 8):
            for rails in (1, 2):
                seen = []
                for ch in range(chunks):
                    for rl in range(rails):
                        off, packs = stripe(total, chunks, rails, ch, rl)
                        seen.extend(range(off, off + packs))
                if seen != list(range(total)):
                    print(f"FAIL stripe total={total} chunks={chunks} rails={rails}")
                    failures += 1

    # 2. bf16 rounding helper against hand-derived RNE cases: 1.00390625 = 0x3F808000 (tie, even
    #    stays 0x3F80), 1.01171875 = 0x3F818000 (tie, odd rounds up to 0x3F82), 3e38 = 0x7F61B1E6
    #    (low half > 0x8000 rounds up), 1e-40 subnormal, NaN canonical
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5, 3.0e38, 1e-40, np.nan], dtype=np.float32)
    got = f32_to_bf16_bits(x)
    want = np.array([0x3F80, 0x3F80, 0x3F82, 0xC020, 0x7F62, 0x0001, 0x7FC0], dtype=np.uint16)
    if not np.array_equal(got, want):
        print(f"FAIL f32_to_bf16_bits got {[hex(v) for v in got]} want {[hex(v) for v in want]}")
        failures += 1

    # 3. protocol simulation, randomized interleavings, several geometries
    for rails, chunks in ((1, 1), (2, 1), (2, 4), (1, 3)):
        L = Layout(16384, rails, chunks)   # holds the 10 KB (5120 bf16) op the sim uses as its largest
        fab = SimFabric(L, rng)
        proxies = [SimProxy(fab, r) for r in range(WORLD)]
        kernels = [SimKernel(fab, r) for r in range(WORLD)]
        pending_launch = [None] * WORLD   # per rank: the next op index to launch
        results = {}
        op_inputs = {}
        next_op = [0] * WORLD
        done = 0
        steps = 0
        while done < ops * WORLD and steps < 200000:
            steps += 1
            actor = rng.randrange(3)
            r = rng.randrange(WORLD)
            if actor == 0:
                k = kernels[r]
                if k.state == "idle" and next_op[r] < ops:
                    op = next_op[r]
                    if op not in op_inputs:
                        n = rng.choice((8, 640, 5120, 4096)) if op else 5120
                        # random bf16 with exponents in [2^-17, 2^8): both signs, no inf/NaN, no overflow
                        op_inputs[op] = [((nrng.integers(110, 136, size=n, dtype=np.uint16) << 7)
                                          | nrng.integers(0, 128, size=n, dtype=np.uint16)
                                          | (nrng.integers(0, 2, size=n, dtype=np.uint16) << 15))
                                         for _ in range(WORLD)]
                    k.launch(op_inputs[op][r])
                    pending_launch[r] = op
                elif k.state == "waiting" and k.step():
                    op = pending_launch[r]
                    results.setdefault(op, {})[r] = k.out
                    next_op[r] += 1
                    done += 1
            elif actor == 1:
                # proxies are sometimes left unscheduled for long stretches (missed-doorbell path)
                if rng.random() < 0.7:
                    proxies[r].step()
            else:
                fab.deliver_one()
        if done != ops * WORLD:
            print(f"FAIL sim rails={rails} chunks={chunks}: only {done}/{ops * WORLD} ops completed in {steps} steps (deadlock?)")
            failures += 1
            continue
        for op in range(ops):
            ref = reference_sum_bf16(op_inputs[op])
            for r in range(WORLD):
                if not np.array_equal(results[op][r], ref):
                    print(f"FAIL sim rails={rails} chunks={chunks} op={op} rank={r}: result differs from fp32-once reference")
                    failures += 1
                    break
        fw = sum(p.forwards for p in proxies)
        exp_fw = ops * WORLD * rails * chunks
        if fw != exp_fw:
            print(f"FAIL sim rails={rails} chunks={chunks}: forwards {fw} != {exp_fw}")
            failures += 1
        maxp = max(p.max_pending_seen for p in proxies)
        if verbose:
            print(f"sim rails={rails} chunks={chunks}: {ops} ops x 4 ranks OK, {fab.delivered_writes} WR pairs, "
                  f"{fw} forwards, max doorbells pending {maxp} (<= {SLOTS}), {steps} scheduler steps")

    # 3b. THE DIAGONAL INVARIANT. Rank R's recv[DIAG]/flag[DIAG] (DIAG=(R+2)%4) carries rank (R+2)'s
    #     bytes in BOTH modes -- relay gets them there by (R+3) forwarding what it took from its prev,
    #     diag has (R+2) write them directly -- so for identical inputs the reduced bf16 output must be
    #     BIT-IDENTICAL between the two paths, and must still equal the fp32-accumulate-once reference.
    #     The operands are distinct per rank AND per element (increasing bf16 bit patterns, all finite),
    #     so any source-index or slot mix-up in either path changes the result. The forward/diag-post
    #     counts prove each mode took its own branch: only the relay path forwards, only diag posts
    #     directly, and with diag off the counts are exactly today's (same code path, same result).
    diag_ops = 8
    for rails, chunks in ((2, 1), (1, 2)):
        n_elem = 5120                                   # 10 KB, the B=1 decode size
        inputs = [np.array([0x3800 + i + r for i in range(n_elem)], dtype=np.uint16) for r in range(WORLD)]
        ref = reference_sum_bf16(inputs)
        last_out: dict[bool, list] = {}
        for diag in (False, True):
            L = Layout(16384, rails, chunks)
            fab = SimFabric(L, random.Random(seed + (7 if diag else 0)), diag=diag)
            proxies = [SimProxy(fab, r, diag=diag) for r in range(WORLD)]
            kernels = [SimKernel(fab, r) for r in range(WORLD)]
            pending_launch = [None] * WORLD
            next_op = [0] * WORLD
            outs: dict[tuple[int, int], np.ndarray] = {}
            done = steps = 0
            while done < diag_ops * WORLD and steps < 200000:
                steps += 1
                actor = rng.randrange(3)
                r = rng.randrange(WORLD)
                if actor == 0:
                    k = kernels[r]
                    if k.state == "idle" and next_op[r] < diag_ops:
                        k.launch(inputs[r])             # identical operands on every op: a mix-up shows
                        pending_launch[r] = next_op[r]
                    elif k.state == "waiting" and k.step():
                        outs[(r, next_op[r])] = k.out
                        next_op[r] += 1
                        done += 1
                elif actor == 1:
                    if rng.random() < 0.7:
                        proxies[r].step()
                else:
                    fab.deliver_one()
            if done != diag_ops * WORLD:
                print(f"FAIL diag={int(diag)} rails={rails} chunks={chunks}: only {done}/{diag_ops * WORLD} "
                      f"ops completed in {steps} steps (deadlock?)")
                failures += 1
                continue
            bad = [(r, op) for op in range(diag_ops) for r in range(WORLD)
                   if not np.array_equal(outs[(r, op)], ref)]
            if bad:
                print(f"FAIL diag={int(diag)} rails={rails} chunks={chunks}: {len(bad)} rank/ops differ from "
                      f"the fp32-once reference, first at rank {bad[0][0]} op {bad[0][1]}")
                failures += 1
            fw, dp = sum(p.forwards for p in proxies), sum(p.diag_posts for p in proxies)
            exp = diag_ops * WORLD * rails * chunks
            if fw != (0 if diag else exp) or dp != (exp if diag else 0):
                print(f"FAIL diag={int(diag)} rails={rails} chunks={chunks}: forwards={fw} diag_posts={dp}, "
                      f"want ({0 if diag else exp}, {exp if diag else 0}) -- wrong branch taken?")
                failures += 1
            maxp = max(p.max_pending_seen for p in proxies)
            if maxp > SLOTS:
                print(f"FAIL diag={int(diag)} rails={rails} chunks={chunks}: doorbells pending {maxp} > {SLOTS}")
                failures += 1
            last_out[diag] = [outs[(r, diag_ops - 1)] for r in range(WORLD)]
            if verbose:
                print(f"diag={int(diag)} rails={rails} chunks={chunks}: {diag_ops} ops x 4 ranks match the "
                      f"reference, {fw} forwards / {dp} direct diag posts, {fab.delivered_writes} WR pairs, "
                      f"max doorbells pending {maxp} (<= {SLOTS}), {steps} scheduler steps")
        if len(last_out) == 2:
            for r in range(WORLD):
                if not np.array_equal(last_out[False][r], last_out[True][r]):
                    print(f"FAIL diag rails={rails} chunks={chunks}: rank {r} diag result differs from the relay "
                          f"result ({int((last_out[False][r] != last_out[True][r]).sum())} of "
                          f"{last_out[False][r].size} elements)")
                    failures += 1
            if verbose:
                print(f"diag invariant rails={rails} chunks={chunks}: relay and diag outputs bit-identical on "
                      f"all 4 ranks, both equal to the fp32-once reference")

    # 3c. DIAG_CHUNKS: step (2) cut into a different chunk count than step (1). The relay forward
    #     and the direct diag post are the two deliveries of the SAME source's stripe, so for any
    #     step-2 geometry the flag layout can index (dc > chunks, dc < chunks, dc == chunks, which
    #     is the diag-first ORDER alone) the relay and diag outputs must stay BIT-IDENTICAL and
    #     equal to the fp32-once reference; the counts prove dc-sized step-2 on both branches.
    for rails, chunks, dc in ((2, 1, 4), (1, 2, 3), (2, 2, 2), (2, 4, 2)):
        n_elem = 5120
        inputs = [np.array([0x3800 + i + r for i in range(n_elem)], dtype=np.uint16) for r in range(WORLD)]
        ref = reference_sum_bf16(inputs)
        outs_by_mode: dict[bool, list] = {}
        for diag in (False, True):
            L = Layout(16384, rails, chunks, dc)
            fab = SimFabric(L, random.Random(seed + (11 if diag else 5)), diag=diag)
            proxies = [SimProxy(fab, r, diag=diag, diag_chunks=dc) for r in range(WORLD)]
            kernels = [SimKernel(fab, r) for r in range(WORLD)]
            pending_launch = [None] * WORLD
            next_op = [0] * WORLD
            outs: dict[tuple[int, int], np.ndarray] = {}
            done = steps = 0
            while done < diag_ops * WORLD and steps < 200000:
                steps += 1
                actor = rng.randrange(3)
                r = rng.randrange(WORLD)
                if actor == 0:
                    k = kernels[r]
                    if k.state == "idle" and next_op[r] < diag_ops:
                        k.launch(inputs[r])
                        pending_launch[r] = next_op[r]
                    elif k.state == "waiting" and k.step():
                        outs[(r, next_op[r])] = k.out
                        next_op[r] += 1
                        done += 1
                elif actor == 1:
                    if rng.random() < 0.7:
                        proxies[r].step()
                else:
                    fab.deliver_one()
            if done != diag_ops * WORLD:
                print(f"FAIL diag_chunks={dc} diag={int(diag)} rails={rails} chunks={chunks}: "
                      f"only {done}/{diag_ops * WORLD} ops completed in {steps} steps (deadlock?)")
                failures += 1
                continue
            bad = [(r, op) for op in range(diag_ops) for r in range(WORLD)
                   if not np.array_equal(outs[(r, op)], ref)]
            if bad:
                print(f"FAIL diag_chunks={dc} diag={int(diag)} rails={rails} chunks={chunks}: "
                      f"{len(bad)} rank/ops differ from the fp32-once reference")
                failures += 1
            exp = diag_ops * WORLD * rails * dc
            fw, dp = sum(p.forwards for p in proxies), sum(p.diag_posts for p in proxies)
            if fw != (0 if diag else exp) or dp != (exp if diag else 0):
                print(f"FAIL diag_chunks={dc} diag={int(diag)} rails={rails} chunks={chunks}: forwards={fw} "
                      f"diag_posts={dp}, want ({0 if diag else exp}, {exp if diag else 0})")
                failures += 1
            outs_by_mode[diag] = [outs[(r, diag_ops - 1)] for r in range(WORLD)]
        if len(outs_by_mode) == 2:
            for r in range(WORLD):
                if not np.array_equal(outs_by_mode[False][r], outs_by_mode[True][r]):
                    print(f"FAIL diag_chunks={dc} rails={rails} chunks={chunks}: rank {r} diag result differs "
                          f"from the relay result")
                    failures += 1
        if verbose:
            print(f"diag_chunks invariant rails={rails} chunks={chunks} dc={dc}: relay and diag outputs "
                  f"bit-identical on all 4 ranks, both equal to the reference")

    # 3d. MUST-FIRE MUTANT. A relay forwarder that hands off the WRONG rail's slice -- bytes covered
    #     by the same flags, landing at the wrong offsets -- must be caught by the 3c compare. If
    #     this arm ever passes clean, the compare is blind, not the protocol correct.
    rails, chunks, dc = 2, 1, 4
    n_elem = 5120
    inputs = [np.array([0x3800 + i + r for i in range(n_elem)], dtype=np.uint16) for r in range(WORLD)]
    ref = reference_sum_bf16(inputs)
    L = Layout(16384, rails, chunks, dc)
    fab = SimFabric(L, random.Random(seed + 17), diag=False)
    proxies = [SimProxy(fab, r, diag=False, diag_chunks=dc, mutant_swap_rails=True) for r in range(WORLD)]
    kernels = [SimKernel(fab, r) for r in range(WORLD)]
    next_op = [0] * WORLD
    outs = {}
    done = steps = 0
    while done < 4 * WORLD and steps < 200000:
        steps += 1
        actor = rng.randrange(3)
        r = rng.randrange(WORLD)
        if actor == 0:
            k = kernels[r]
            if k.state == "idle" and next_op[r] < 4:
                k.launch(inputs[r])
            elif k.state == "waiting" and k.step():
                outs[(r, next_op[r])] = k.out
                next_op[r] += 1
                done += 1
        elif actor == 1:
            proxies[r].step()
        else:
            fab.deliver_one()
    caught = done == 4 * WORLD and any(not np.array_equal(o, ref) for o in outs.values())
    if not caught:
        print("FAIL mutant: rail-swapped relay forwarder was NOT caught (or stalled) -- the "
              "relay==diag compare is blind")
        failures += 1
    elif verbose:
        print("mutant arm fired: rail-swapped relay forwarder caught by the reference compare")

    # 3e. DIAG_MAX_BYTES PER-OP CUTOFF. One runtime, op sizes straddling the cutoff (below, equal,
    #     one pack above, well above): every op's result must equal the fp32-accumulate-once
    #     reference AND be bit-identical to an all-relay run of the same ops; the per-op counts
    #     prove small ops took the diagonal and large ops the relay forward, on every rank. The
    #     geometries include DIAG_CHUNKS (diag-first post order + a different step-2 chunk count).
    if not (op_uses_diag(10240, True, 0) and not op_uses_diag(16, False, 0) and not op_uses_diag(16, False, 4096)
            and op_uses_diag(4096, True, 4096) and not op_uses_diag(4112, True, 4096)):
        print("FAIL op_uses_diag truth table (0 must mean 'diag decides', the cutoff is inclusive)")
        failures += 1
    cut = 4096
    sizes = (16, 2048, 4080, 4096, 4112, 6144, 10240)   # bytes; 4096 is AT the cutoff (diag)
    cut_ops = 24
    for rails, chunks, dc in ((2, 1, 0), (1, 2, 0), (2, 1, 4), (1, 3, 2), (2, 4, 2)):
        for trial in range(2):
            orng = random.Random(seed * 31 + rails * 7 + chunks * 3 + dc + trial)
            op_sizes = [orng.choice(sizes) for _ in range(cut_ops)]
            op_sizes[0], op_sizes[1] = cut, cut + 16   # both sides of the edge in every run
            op_inputs = []
            for nb in op_sizes:
                n = nb // 2
                op_inputs.append([((nrng.integers(110, 136, size=n, dtype=np.uint16) << 7)
                                   | nrng.integers(0, 128, size=n, dtype=np.uint16)
                                   | (nrng.integers(0, 2, size=n, dtype=np.uint16) << 15)) for _ in range(WORLD)])
            runs = {}
            for mode in ("relay", "cutoff"):
                diag = mode == "cutoff"
                L = Layout(16384, rails, chunks, dc)
                fab = SimFabric(L, random.Random(seed + 101 * trial + (13 if diag else 0) + dc), diag=diag)
                proxies = [SimProxy(fab, r, diag=diag, diag_chunks=dc, diag_max_bytes=cut if diag else 0)
                           for r in range(WORLD)]
                kernels = [SimKernel(fab, r) for r in range(WORLD)]
                outs, done, steps = _sim_drive(fab, proxies, kernels, op_inputs, rng)
                if done != cut_ops * WORLD:
                    print(f"FAIL cutoff {mode} rails={rails} chunks={chunks} dc={dc} trial={trial}: only "
                          f"{done}/{cut_ops * WORLD} ops completed in {steps} steps (deadlock?)")
                    failures += 1
                    continue
                bad = [(r, op) for op in range(cut_ops) for r in range(WORLD)
                       if not np.array_equal(outs[(r, op)], reference_sum_bf16(op_inputs[op]))]
                if bad:
                    print(f"FAIL cutoff {mode} rails={rails} chunks={chunks} dc={dc}: {len(bad)} rank/ops differ "
                          f"from the fp32-once reference, first rank {bad[0][0]} op {bad[0][1]} "
                          f"({op_sizes[bad[0][1]]} B)")
                    failures += 1
                n_small = sum(1 for nb in op_sizes if op_uses_diag(nb, diag, cut if diag else 0))
                per_op = rails * (dc or chunks)
                want_dp, want_fw = n_small * WORLD * per_op, (cut_ops - n_small) * WORLD * per_op
                fw, dp = sum(p.forwards for p in proxies), sum(p.diag_posts for p in proxies)
                d_ops, r_ops = [p.diag_ops for p in proxies], [p.relay_ops for p in proxies]
                if (fw, dp) != (want_fw, want_dp) or d_ops != [n_small] * WORLD or r_ops != [cut_ops - n_small] * WORLD:
                    print(f"FAIL cutoff {mode} rails={rails} chunks={chunks} dc={dc}: forwards={fw} diag_posts={dp} "
                          f"diag_ops={d_ops} relay_ops={r_ops}, want ({want_fw}, {want_dp}, {n_small}, "
                          f"{cut_ops - n_small}) per rank -- wrong per-op branch?")
                    failures += 1
                if max(p.max_pending_seen for p in proxies) > SLOTS:
                    print(f"FAIL cutoff {mode} rails={rails} chunks={chunks} dc={dc}: doorbells pending > {SLOTS}")
                    failures += 1
                runs[mode] = (outs, n_small)
            if len(runs) == 2:
                diff = [(r, op) for op in range(cut_ops) for r in range(WORLD)
                        if not np.array_equal(runs["relay"][0][(r, op)], runs["cutoff"][0][(r, op)])]
                if diff:
                    print(f"FAIL cutoff rails={rails} chunks={chunks} dc={dc}: {len(diff)} rank/ops of the cutoff "
                          f"run differ from the all-relay run")
                    failures += 1
                elif verbose and trial == 0:
                    n_small = runs["cutoff"][1]
                    print(f"diag cutoff={cut} B rails={rails} chunks={chunks} dc={dc}: {cut_ops} mixed ops "
                          f"({n_small} diag / {cut_ops - n_small} relay per rank) x 4 ranks equal the reference "
                          f"and are bit-identical to the all-relay run (2 interleavings)")

    # 3f. MUST-FIRE: an INCONSISTENT per-rank cutoff. (i) The all-rank config match -- the same
    #     ring_config/config_mismatches the runtime uses -- must name the odd rank (fail-stop at
    #     setup), and must stay clean for a consistent farm. (ii) Run anyway, the ranks disagree on
    #     who delivers the opposite rank's bytes for ops between the two cutoffs: the run must
    #     STALL (on the device: spin-limit poison, fail-stop), and every op that did complete must
    #     still equal the reference -- a hang or a fail-stop, never silently corrupted bits.
    def _cfg(dmb: int, diag: bool = True) -> dict:
        return ring_config(slot_bytes=16384, rails=2, chunks=1, max_size=16384, threads=512, blocks=8,
                           spin_limit=DEFAULT_SPIN_LIMIT, diag=diag, diag_chunks=0, timing=False, diag_max_bytes=dmb)
    same = config_mismatches([_cfg(cut)] * WORLD)
    odd = config_mismatches([_cfg(cut), _cfg(cut), _cfg(1024), _cfg(cut)])
    off_odd = config_mismatches([_cfg(0, False)] * 3 + [_cfg(cut, False)])
    if same or len(odd) != 1 or not odd[0].startswith("rank 2 ") or len(off_odd) != 1:
        print(f"FAIL must-fire config match: consistent={same} inconsistent={odd} diag-off-inconsistent={off_odd}")
        failures += 1
    L = Layout(16384, 2, 1)
    fab = SimFabric(L, random.Random(seed + 23), diag=True)
    odd_cut = (cut, cut, 1024, cut)
    proxies = [SimProxy(fab, r, diag=True, diag_max_bytes=odd_cut[r]) for r in range(WORLD)]
    kernels = [SimKernel(fab, r) for r in range(WORLD)]
    mixed_inputs = [[np.array([0x3800 + i + r + op for i in range(1024)], dtype=np.uint16) for r in range(WORLD)]
                    for op in range(4)]   # 2048 B: diag under 4096, relay under 1024
    outs, done, steps = _sim_drive(fab, proxies, kernels, mixed_inputs, rng, max_steps=60000)
    wrong = [k for k, o in outs.items() if not np.array_equal(o, reference_sum_bf16(mixed_inputs[k[1]]))]
    if done == 4 * WORLD or wrong:
        print(f"FAIL must-fire inconsistent cutoff: completed {done}/{4 * WORLD}, {len(wrong)} wrong results "
              f"-- want a stall with zero wrong results")
        failures += 1
    elif verbose:
        print(f"must-fire inconsistent cutoff: config match names {odd[0].split(' config')[0]}; run anyway it "
              f"stalls at {done}/{4 * WORLD} ops ({steps} steps) with 0 wrong results (device: spin-limit poison)")

    # 4. adapter: env contract defaults + overrides, routing counters, fail-stop health
    cfg = env_config({})
    if not (cfg["enabled"] is False and cfg["max_size"] == DEFAULT_MAX_SIZE and cfg["rails"] == 2 and cfg["chunks"] == 1
            and cfg["gid_index"] == 1 and cfg["next_hcas"] == DEFAULT_NEXT_HCAS and cfg["prev_hcas"] == DEFAULT_PREV_HCAS
            and cfg["diag"] == 0 and cfg["diag_chunks"] == 0 and cfg["diag_hcas"] == () and cfg["timing"] == 0
            and cfg["diag_max_bytes"] == 0):
        print(f"FAIL env_config defaults: {cfg}")
        failures += 1
    cfg = env_config({"SGLANG_RDMA_RING_ALLREDUCE": "1", "SGLANG_RDMA_RING_MAX_SIZE": "65536", "NCCL_IB_GID_INDEX": "3",
                      "SGLANG_RDMA_RING_RAILS": "1", "SGLANG_RDMA_RING_NEXT_HCA": "a,b", "SGLANG_RDMA_RING_PROXY_CPU": "5",
                      "SGLANG_RDMA_RING_DIAG": "1", "SGLANG_RDMA_RING_DIAG_HCA": "diag0,diag1",
                      "SGLANG_RDMA_RING_TIMING": "1", "SGLANG_RDMA_RING_DIAG_MAX_BYTES": "65536"})
    if not (cfg["enabled"] and cfg["max_size"] == 65536 and cfg["gid_index"] == 3 and cfg["rails"] == 1
            and cfg["next_hcas"] == ("a", "b") and cfg["proxy_cpu"] == 5
            and cfg["diag"] == 1 and cfg["diag_hcas"] == ("diag0", "diag1") and cfg["timing"] == 1
            and cfg["diag_max_bytes"] == 65536):
        print(f"FAIL env_config overrides: {cfg}")
        failures += 1
    # the diagonal's GRH constants are fixed by the underlay, independent of the ring gid_index knob
    if DIAG_GID_INDEX != 3 or DIAG_FLOW_LABEL != 16383:
        print(f"FAIL diag contract: DIAG_GID_INDEX={DIAG_GID_INDEX} DIAG_FLOW_LABEL={DIAG_FLOW_LABEL}")
        failures += 1
    logged = []
    ring = RingAllReduce(_StubRuntime(), dict(cfg, max_size=65536, log_every_s=10.0), logged.append)
    cases = [(_StubTensor(10240), True, None), (_StubTensor(10240), False, "not_sum"),
             (_StubTensor(10240, "torch.int32"), True, "dtype"), (_StubTensor(65552), True, "too_large"),
             (_StubTensor(10240, contiguous=False), True, "not_contiguous"), (_StubTensor(10248), True, "unaligned"),
             (_StubTensor(65536, "torch.float32"), True, None)]
    for t, is_sum, want_reason in cases:
        got_reason = ring.route(t, is_sum)
        if got_reason != want_reason:
            print(f"FAIL route nbytes={t._n} dtype={t.dtype} sum={is_sum}: got {got_reason} want {want_reason}")
            failures += 1
    st = ring.stats()
    if not (st["routed_calls"] == 2 and st["fallback_calls"] == 5 and st["max_nbytes"] == 65536
            and st["fallback_reasons"] == {"not_sum": 1, "dtype": 1, "too_large": 1, "not_contiguous": 1, "unaligned": 1}):
        print(f"FAIL adapter counters: {st}")
        failures += 1
    # cadence is WALL TIME, not calls: 1000 idle-loop calls inside one second log once,
    # the next second logs once more (the call-count cadence flooded the rank log ~5/s
    # while the serve idled, N21 integration round)
    for _ in range(1000):
        check_health_all(now=100.0)
    n_first = len(logged)
    check_health_all(now=109.0)
    check_health_all(now=110.0)
    if n_first != 1 or len(logged) != 2 or "routed_calls=2" not in logged[-1] or "health_calls=1002" not in logged[-1]:
        print(f"FAIL periodic stats log cadence: first={n_first} total={len(logged)} last={logged[-1:]}")
        failures += 1
    ring.close()
    if not ring._rt.closed or ring.stats()["routed_calls"] != 2:
        print("FAIL adapter close")
        failures += 1
    bad = RingAllReduce(_StubRuntime(ok=False), dict(cfg, log_every_s=0), logged.append)
    try:
        check_health_all()
        print("FAIL check_health_all did not raise on a poisoned runtime")
        failures += 1
    except RuntimeError:
        pass
    bad.close()
    if bad.route(_StubTensor(10240), True) is None:
        pass  # a closed adapter is not consulted by the hook (pynccl drops its reference); routing itself still counts
    if verbose:
        print(f"adapter: env contract OK, routing counters OK ({st['routed_calls']} routed / {st['fallback_calls']} fallback), "
              f"fail-stop health OK")

    # 4b. TIMING GATE: off = today's kernel source byte-for-byte (no instrumentation compiled in);
    # on = the stamps spliced in (hash MUST differ -- if it does not, the patch anchors silently
    # stopped matching and the reader would log nothing).
    if build_kernel_source(False) != RING_KERNEL_CU or "timing_ptr" in build_kernel_source(False):
        print("FAIL timing gate off: kernel source is not RING_KERNEL_CU byte-for-byte")
        failures += 1
    if build_kernel_source(True) == RING_KERNEL_CU or "globaltimer" not in build_kernel_source(True):
        print("FAIL timing gate on: kernel source unchanged (patch anchors drifted?)")
        failures += 1
    if verbose:
        print("timing gate: off = source identical, on = spliced (hashes differ)")

    # 4c. DIAG CUTOFF GATE: only DIAG=1 with DIAG_MAX_BYTES > 0 builds the spliced proxy; every
    # other config (DIAG=0 with or without a cutoff, DIAG=1 with cutoff 0) compiles RING_PROXY_C
    # byte-for-byte, so the .so hash is unchanged. Must-fire: the gate-on source differs and carries
    # the per-op choice and the setter (anchor drift would silently leave every op on diag).
    sel = {(d, m): build_proxy_source(proxy_uses_cutoff(d, m)) for d in (False, True) for m in (0, 4096)}
    if any(sel[k] != RING_PROXY_C for k in ((False, 0), (False, 4096), (True, 0))):
        print("FAIL diag-cutoff gate off: proxy source is not RING_PROXY_C byte-for-byte")
        failures += 1
    on = sel[(True, 4096)]
    if (on == RING_PROXY_C or "int use_diag = c->diag && (c->diag_max_bytes == 0" not in on
            or "ring_set_diag_max_bytes" not in on or on.count("use_diag") != 4 or "if (c->diag) {\n        if (!c->diag_first)" in on):
        print("FAIL diag-cutoff gate on: proxy source not spliced as expected (anchors drifted?)")
        failures += 1
    if verbose:
        print(f"diag-cutoff gate: off = proxy source identical (sha {hashlib.sha256(RING_PROXY_C.encode()).hexdigest()[:16]}), "
              f"on = spliced (sha {hashlib.sha256(on.encode()).hexdigest()[:16]}); kernel source independent of it")

    # 5. toolchain checks (SKIP when absent): C shim against libibverbs, kernel through NVRTC
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc and Path("/usr/include/infiniband/verbs.h").exists():
        for variant in (False, True):
            try:
                so = build_proxy(Path(os.environ.get("RING_PROXY_CACHE_DIR",
                                                     os.path.expanduser("~/local/cache/rdma-ring-proxy-selftest"))), variant)
                lib = ctypes.CDLL(str(so))
                lib.ring_abi_version.restype = ctypes.c_int
                ok = lib.ring_abi_version() == RING_ABI and (not variant or hasattr(lib, "ring_set_diag_max_bytes"))
                print(f"proxy shim compile{' +DIAG_CUTOFF' if variant else ''}: {'OK' if ok else 'FAIL'} ({so})")
                failures += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL proxy shim compile{' +DIAG_CUTOFF' if variant else ''}: {exc}")
                failures += 1
    else:
        print("proxy shim compile: SKIP (no gcc or no /usr/include/infiniband/verbs.h on this host)")
    try:
        from cuda.bindings import nvrtc  # noqa: F401

        arms = (("", RING_KERNEL_CU, []),                    # default = the #ifndef DIAG_CHUNKS path
                (" +DIAG_CHUNKS=2", RING_KERNEL_CU, [b"-DDIAG_CHUNKS=2"]),
                (" +TIMING", build_kernel_source(True), [b"-DDIAG_CHUNKS=2"]))  # gate-on must compile too
        for tag, src, extra in arms:
            opts = [b"--gpu-architecture=compute_90", b"--std=c++17", b"-default-device",
                    b"-DRANK=0", b"-DRAILS=2", b"-DCHUNKS=1", b"-DTHREADS=512", b"-DDTYPE_BF16=1"] + extra
            err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"ring.cu", 0, [], [])
            (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                print(f"FAIL kernel NVRTC compile{tag}:\n" + log.decode(errors="replace"))
                failures += 1
            else:
                print(f"kernel NVRTC compile{tag} (compute_90 PTX): OK")
    except ImportError:
        print("kernel NVRTC compile: SKIP (no cuda.bindings on this host)")

    print(f"rdma_ring_allreduce.py --self-test {'OK' if failures == 0 else 'FAIL'} failures={failures}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dump-kernel-source", action="store_true",
                    help="print the NVRTC source the SGLANG_RDMA_RING_TIMING gate selects (hash it to test the gate)")
    ap.add_argument("--dump-proxy-source", action="store_true",
                    help="print the C proxy source SGLANG_RDMA_RING_DIAG / _DIAG_MAX_BYTES select (hash it to test the gate)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.dump_kernel_source:
        sys.stdout.write(build_kernel_source(bool(env_config()["timing"])))
        return 0
    if args.dump_proxy_source:
        cfg = env_config()
        sys.stdout.write(build_proxy_source(proxy_uses_cutoff(bool(cfg["diag"]), cfg["diag_max_bytes"])))
        return 0
    ap.error("this module is imported by sglang (pynccl.py) and by rdma_allreduce_ring_bench.py; use --self-test")
    return 2


# SGLANG_RDMA_RING_CALLSITE_LOG=<path>: at CUDA-graph capture, one line per routed all-reduce
# (capture index, layer, class, function, bytes) to <path>.<pid> -- which model op each graph all-reduce is
_CALLSITE = {"path": None, "n": 0}


def _log_callsite(tensor) -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_current_stream_capturing():
        return
    layer, cls, fn = -1, "?", "?"
    f = sys._getframe(2)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "layer_id"):
            layer, cls, fn = getattr(s, "layer_id"), type(s).__name__, f.f_code.co_name
            break
        f = f.f_back
    with open("%s.%d" % (_CALLSITE["path"], os.getpid()), "a") as out:
        out.write("%d,%s,%s,%s,%d\n" % (_CALLSITE["n"], layer, cls, fn, tensor.numel() * tensor.element_size()))
    _CALLSITE["n"] += 1


if __name__ == "__main__":
    sys.exit(main())
