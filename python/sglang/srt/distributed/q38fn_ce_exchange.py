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

_G = 128
_PAGE = 4096
_FULL = (0, 8)  # uint64 slot index per rank, 64 bytes apart
_states: dict = {}
_shared_copy_streams: dict = {}


def _chk(r, what, runtime=False):
    from cuda.bindings import driver as cu
    from cuda.bindings import runtime as rt

    err = r[0] if isinstance(r, tuple) else r
    ok = rt.cudaError_t.cudaSuccess if runtime else cu.CUresult.CUDA_SUCCESS
    if err != ok:
        raise RuntimeError(f"q38fn ce exchange: {what} failed: {err}")
    return r


class _State:
    """Buffers for one (group, hidden) at a row capacity; both ranks build it in lockstep (same rows, same calls)."""

    def __init__(self, group, hidden: int, rows_cap: int, slot: int = 0):
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
        # SGLANG_TP_AR_FP8_CE_ONE_STREAM=1: every slot pulls on ONE copy stream, so two exchanges in flight run one
        # after the other at full link bandwidth (in issue order, the same on both ranks) instead of sharing it -- the
        # first one needed finishes in half the time.
        if os.environ.get("SGLANG_TP_AR_FP8_CE_ONE_STREAM") == "1":
            if (dev.index, group.unique_name) not in _shared_copy_streams:
                _shared_copy_streams[(dev.index, group.unique_name)] = torch.cuda.Stream(device=dev)
            self.copy_stream = _shared_copy_streams[(dev.index, group.unique_name)]
            self.done_event = torch.cuda.Event()
        else:
            self.copy_stream = torch.cuda.Stream(device=dev)
            self.done_event = None
        torch.cuda.synchronize()

        # The flag page: a /dev/shm file opened directly (not multiprocessing.SharedMemory, whose resource tracker is
        # shared by sglang's scheduler processes and double-counts a name both ranks touch).
        path = f"/dev/shm/q38fn-tparce-{os.getpid()}-{hidden}-{rows_cap}-{slot}" if self.me == 0 else None
        if self.me == 0:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, _PAGE)
        from cuda.bindings import runtime as rt

        self.rt = rt
        self.dev = dev.index
        got = [None, None]
        dist.all_gather_object(got, (self._ipc(self.q), self._ipc(self.s), dev.index, path), group=group.cpu_group)
        hq, hs, self.pdev, _ = got[self.peer]
        # The peer's send buffers are opened here, with this rank's device current and lazy peer access, so the
        # mapping is a P2P mapping in this rank's context and a copy from it is one copy-engine transfer over PCIe.
        # Opened through torch's IPC rebuild instead (in the peer device's context) every copy was bounced through
        # host memory, a DtoH and an HtoD a transfer, with or without peer access on: served, the GPU sat idle 122 ms
        # of a 286.6 ms prefill step (rule-ce-ar3).
        self.p2p = torch.cuda.can_device_access_peer(dev.index, self.pdev)
        self.peer_q_ptr = self._open(hq)
        self.peer_s_ptr = self._open(hs)
        if self.me == 1:
            path = got[0][3]
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
              f"rows_cap={rows_cap} slot={slot} me={self.me} p2p={int(self.p2p)} pid={os.getpid()}", flush=True)

    def _ipc(self, t):
        """(IPC handle bytes, offset) of a tensor's allocation: the handle names the allocation's base."""
        cu, rt = self.cu, self.rt
        base = int(_chk(cu.cuMemGetAddressRange(cu.CUdeviceptr(t.data_ptr())), "cuMemGetAddressRange")[1])
        h = _chk(rt.cudaIpcGetMemHandle(base), "cudaIpcGetMemHandle", runtime=True)[1]
        return bytes(h.reserved), t.data_ptr() - base

    def _open(self, handle):
        rt = self.rt
        raw, off = handle
        h = rt.cudaIpcMemHandle_t()
        h.reserved = raw
        flags = rt.cudaIpcMemLazyEnablePeerAccess
        base = _chk(rt.cudaIpcOpenMemHandle(h, flags), "cudaIpcOpenMemHandle", runtime=True)[1]
        return int(base) + off

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
        self.start(rows)
        return self.finish(rows)

    def start(self, rows):
        """Signal my payload ready and enqueue the pull of the peer's on the copy stream; the caller's stream does not
        wait. finish() joins it. Both must sit in the same captured segment when recorded into a CUDA graph (the copy
        stream joins the caller's capture here and rejoins it there)."""
        cu = self.cu
        S = torch.cuda.current_stream()
        C = self.copy_stream
        _chk(cu.cuStreamWriteValue64(cu.CUstream(S.cuda_stream), self._ptr(_FULL[self.me]), 1, 0), "write FULL[me] = 1")
        C.wait_stream(S)
        c = cu.CUstream(C.cuda_stream)
        _chk(cu.cuStreamWaitValue64(c, self._ptr(_FULL[self.peer]), 1,
                                    cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ), "wait FULL[peer] == 1")
        nq, ns = rows * self.hidden, rows * self.groups
        # cudaMemcpyAsync on C from the P2P mapping (capturable; cudaMemcpyPeerAsync is not). Not a torch
        # cross-device copy_, which also orders the peer device's legacy stream -- forbidden while a CUDA-graph
        # capture is open elsewhere in the process (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT during prefill graph capture).
        rt = self.rt
        kind = rt.cudaMemcpyKind.cudaMemcpyDefault
        _chk(rt.cudaMemcpyAsync(self.rq.data_ptr(), self.peer_q_ptr, nq, kind, C.cuda_stream), "copy q", runtime=True)
        _chk(rt.cudaMemcpyAsync(self.rs.data_ptr(), self.peer_s_ptr, 4 * ns, kind, C.cuda_stream), "copy s",
             runtime=True)
        _chk(cu.cuStreamWriteValue64(c, self._ptr(_FULL[self.peer]), 0, 0), "write FULL[peer] = 0")
        if self.done_event is not None:
            # A fresh event per pull: one event re-recorded across captured segments is refused inside capture.
            self.done_event = torch.cuda.Event()
            self.done_event.record(C)

    def finish(self, rows):
        """Make the caller's stream wait for start()'s pull; returns the received (rq, rs) views."""
        if self.done_event is not None:
            # A shared copy stream also carries the other slot's later pull; wait for this one only.
            torch.cuda.current_stream().wait_event(self.done_event)
        else:
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        nq, ns = rows * self.hidden, rows * self.groups
        rq = self.rq[:nq].view(torch.float8_e4m3fn).view(rows, self.hidden)
        rs = self.rs[:ns].view(rows, self.groups)
        return rq, rs


def get_state(group, hidden: int, rows: int, slot: int = 0) -> _State:
    """The exchange state for (group, hidden, slot) with room for `rows`; rebuilt (in lockstep) when rows outgrow it.

    A slot is one exchange that may be in flight at a time: two micro-batches each take their own slot (own buffers,
    flags and copy stream), so one's start() can precede the other's finish().

    Building synchronizes the device and registers host memory, which a CUDA-graph capture in progress anywhere in
    the process forbids, so production builds it in preinit() before capture; a build here is the fallback for an
    unforeseen shape at serving time, outside capture."""
    key = (group.unique_name, hidden, slot)
    st = _states.get(key)
    if st is None or rows > st.rows_cap:
        cap = max(rows, 4096 if st is None else 2 * st.rows_cap)
        st = _State(group, hidden, cap, slot)
        _states[key] = st
    return st


def preinit(group, hidden: int, rows: int, slots: int = 1) -> None:
    """Build the exchange state for each slot before any CUDA-graph capture (model_runner.init_cuda_graphs)."""
    if group.world_size != 2:
        return
    for slot in range(slots):
        get_state(group, hidden, rows, slot)
