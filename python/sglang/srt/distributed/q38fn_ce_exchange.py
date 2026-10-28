"""Copy-engine payload exchange for the two-rank FP8 TP all-reduce (SGLANG_TP_AR_FP8_CE=1).

Production's _q38fn_tparfp8 sends each rank's FP8 partial (and its scales) to the peer with NCCL send/recv, an
SM-driven kernel that slows any compute running beside it. Here each rank quantizes straight into a buffer the peer
has opened through CUDA IPC, and the peer pulls it with a device-to-device copy on its copy engine. Readiness is
signalled on the GPU timeline with 64-bit stream memory operations (cuStreamWaitValue64 / cuStreamWriteValue64, run
by the GPU front-end, no kernel) on counters in one page of POSIX shared memory, host-registered and device-mapped in
both processes (a stream write into a peer's IPC-mapped device memory is refused). Same bytes and the same
dequantize-and-add order as the NCCL path: the all-reduce result is bit-identical.

Per call (S = the caller's stream, C = this rank's copy stream; FULL[r] = rank r's send buffer holds an unread payload):
  S: wait FULL[me] == 0          the peer has read my previous payload
  S: quantize into my send buffer (the caller, with output_q / output_s)
  S: write FULL[me] = 1
  C: wait S; wait FULL[peer] == 1; recv <- the peer's send rows; write FULL[peer] = 0
  S: wait C
Every value is a constant, so the sequence means the same thing run eagerly or recorded into a CUDA graph and replayed
any number of times (sglang's prefill capture records it: C joins S's capture through wait_stream). A counter would
bake the capture-time count into the graph. Each flag alternates between its two writers: the owner sets it only while
it reads 0, the peer clears it only while it reads 1.
"""

from __future__ import annotations

import ctypes
import mmap
import os

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor

_G = 128
_PAGE = 4096
_FULL = (0, 8)  # uint64 slot index per rank, 64 bytes apart
_states: dict = {}


def _chk(r, what):
    from cuda.bindings import driver as cu

    err = r[0] if isinstance(r, tuple) else r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"q38fn ce exchange: {what} failed: {err}")
    return r


class _State:
    """Buffers for one (group, hidden) at a row capacity; both ranks build it in lockstep (same rows, same calls)."""

    def __init__(self, group, hidden: int, rows_cap: int):
        from cuda.bindings import driver as cu

        self.cu = cu
        self.me = group.rank_in_group
        self.peer = 1 - self.me
        self.hidden = hidden
        self.groups = hidden // _G
        self.rows_cap = rows_cap
        dev = torch.device("cuda", torch.cuda.current_device())
        self.q = torch.empty(rows_cap * hidden, dtype=torch.uint8, device=dev)
        self.s = torch.empty(rows_cap * self.groups, dtype=torch.float32, device=dev)
        self.rq = torch.empty_like(self.q)
        self.rs = torch.empty_like(self.s)
        self.copy_stream = torch.cuda.Stream(device=dev)
        torch.cuda.synchronize()

        # The flag page: a /dev/shm file opened directly (not multiprocessing.SharedMemory, whose resource tracker is
        # shared by sglang's scheduler processes and double-counts a name both ranks touch).
        path = f"/dev/shm/q38fn-tparce-{os.getpid()}-{hidden}-{rows_cap}" if self.me == 0 else None
        if self.me == 0:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, _PAGE)
        got = [None, None]
        dist.all_gather_object(got, (reduce_tensor(self.q), reduce_tensor(self.s), path), group=group.cpu_group)
        (fq, aq), (fs, as_), _ = got[self.peer]
        self.peer_q = fq(*aq)
        self.peer_s = fs(*as_)
        if self.me == 1:
            path = got[0][2]
            fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(fd, _PAGE)
        os.close(fd)
        self.host_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        _chk(cu.cuMemHostRegister(self.host_addr, _PAGE,
                                  cu.CU_MEMHOSTREGISTER_DEVICEMAP | cu.CU_MEMHOSTREGISTER_PORTABLE), "cuMemHostRegister")
        self.flag_base = int(_chk(cu.cuMemHostGetDevicePointer(self.host_addr, 0), "cuMemHostGetDevicePointer")[1])
        # Both ranks have attached: the name can go now, so a crash leaves nothing in /dev/shm.
        dist.barrier(group=group.cpu_group)
        if self.me == 0:
            os.unlink(path)
        print(f"[q38fn-tparce] copy-engine exchange on: group={group.unique_name} hidden={hidden} "
              f"rows_cap={rows_cap} me={self.me} pid={os.getpid()}", flush=True)

    def _ptr(self, slot):
        return self.cu.CUdeviceptr(self.flag_base + 8 * slot)

    def views(self, rows):
        """(q, s) send views to quantize into, after waiting until the peer has read the previous payload."""
        cu = self.cu
        s = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        _chk(cu.cuStreamWaitValue64(s, self._ptr(_FULL[self.me]), 0,
                                    cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ), "wait FULL[me] == 0")
        q = self.q[: rows * self.hidden].view(torch.float8_e4m3fn).view(rows, self.hidden)
        sc = self.s[: rows * self.groups].view(rows, self.groups)
        return q, sc

    def exchange(self, rows):
        """Signal my payload ready, pull the peer's into the receive buffers; returns (rq, rs) views."""
        cu = self.cu
        S = torch.cuda.current_stream()
        C = self.copy_stream
        _chk(cu.cuStreamWriteValue64(cu.CUstream(S.cuda_stream), self._ptr(_FULL[self.me]), 1, 0), "write FULL[me] = 1")
        C.wait_stream(S)
        c = cu.CUstream(C.cuda_stream)
        _chk(cu.cuStreamWaitValue64(c, self._ptr(_FULL[self.peer]), 1,
                                    cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ), "wait FULL[peer] == 1")
        nq, ns = rows * self.hidden, rows * self.groups
        # cuMemcpyDtoDAsync on C straight from the peer's unified address: a torch cross-device copy_ would also
        # order the peer device's legacy stream, which a CUDA-graph capture in progress elsewhere in the process
        # forbids (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT during prefill graph capture).
        _chk(cu.cuMemcpyDtoDAsync(cu.CUdeviceptr(self.rq.data_ptr()), cu.CUdeviceptr(self.peer_q.data_ptr()), nq, c),
             "copy q")
        _chk(cu.cuMemcpyDtoDAsync(cu.CUdeviceptr(self.rs.data_ptr()), cu.CUdeviceptr(self.peer_s.data_ptr()), 4 * ns, c),
             "copy s")
        _chk(cu.cuStreamWriteValue64(c, self._ptr(_FULL[self.peer]), 0, 0), "write FULL[peer] = 0")
        S.wait_stream(C)
        rq = self.rq[:nq].view(torch.float8_e4m3fn).view(rows, self.hidden)
        rs = self.rs[:ns].view(rows, self.groups)
        return rq, rs


def get_state(group, hidden: int, rows: int) -> _State:
    """The exchange state for (group, hidden) with room for `rows`; rebuilt (in lockstep) when rows outgrow it.

    Building synchronizes the device and registers host memory, which a CUDA-graph capture in progress anywhere in
    the process forbids, so production builds it in preinit() before capture; a build here is the fallback for an
    unforeseen shape at serving time, outside capture."""
    key = (group.unique_name, hidden)
    st = _states.get(key)
    if st is None or rows > st.rows_cap:
        cap = max(rows, 4096 if st is None else 2 * st.rows_cap)
        st = _State(group, hidden, cap)
        _states[key] = st
    return st


def preinit(group, hidden: int, rows: int) -> None:
    """Build the exchange state before any CUDA-graph capture (model_runner.init_cuda_graphs)."""
    if group.world_size != 2:
        return
    get_state(group, hidden, rows)
