# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import hashlib
import logging
import multiprocessing as mp
import signal
import threading
import time
from array import array
from collections import OrderedDict
from collections.abc import Callable
from enum import Enum, auto

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    ElasticScaleUpdateReq,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
    sock_send,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.startup_time import aggregate_scheduler_startup_times
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_exec,
    get_observability,
    get_parallel,
    get_serving,
    publish,
)
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

SCHEDULER_PIDS_ARG = "scheduler_pids"


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()
    PREFIX_AFFINITY = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


# A scheduler snapshot this much older than a dispatch cannot have seen it.
PENDING_DISPATCH_GRACE_S = 0.5


class DPBudget:
    def __init__(self, num_dp_ranks: int):
        self.num_dp_ranks = num_dp_ranks
        self.total_requests = [0] * num_dp_ranks
        self.total_tokens = [0] * num_dp_ranks
        # Input tokens still awaiting prefill compute per rank (waiting queue
        # plus the in-flight chunked remainder), from the scheduler snapshot.
        self.prefill_backlog = [0] * num_dp_ranks
        # Dispatches the snapshot cannot have seen yet: (dispatch time, tokens).
        # A refresh re-adds them on top of the snapshot until a snapshot newer
        # than the dispatch (plus grace) arrives; without this the 20 ms refresh
        # wiped the speculative backlog with a stale snapshot between two
        # arrivals and both landed on one rank.
        self.pending_dispatches: list[list[tuple[float, int]]] = [
            [] for _ in range(num_dp_ranks)
        ]
        self.last_timestamp = [0.0] * num_dp_ranks

    def update_budget(self, loads):
        """Update budget from shm snapshots, skipping stale reads."""
        for load in loads:
            if load.timestamp == self.last_timestamp[load.dp_rank]:
                continue
            self.last_timestamp[load.dp_rank] = load.timestamp
            self.total_requests[load.dp_rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self.total_tokens[load.dp_rank] = load.num_total_tokens
            keep = [
                (t, tok)
                for (t, tok) in self.pending_dispatches[load.dp_rank]
                if t + PENDING_DISPATCH_GRACE_S > load.timestamp
            ]
            self.pending_dispatches[load.dp_rank] = keep
            self.prefill_backlog[load.dp_rank] = load.num_waiting_uncached_tokens + sum(
                tok for _, tok in keep
            )
            self.total_requests[load.dp_rank] += len(keep)

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.num_dp_ranks),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        self.total_tokens[target_rank] += estimated_tokens
        return target_rank


class PrefixAffinityIndex:
    """Token-prefix fingerprint -> the DP ranks that have served it.

    Each DP rank owns its own radix cache, so a conversation whose turns are
    spread across ranks re-prefills its whole context on every rank it
    visits. This index approximates the radix caches at the controller: a
    request is fingerprinted with a cumulative hash at every ``chunk`` tokens,
    and for each rank the longest fingerprint that rank has served bounds
    the prefix its cache holds. Bounded LRU; a miss is only a cold prefill,
    never an error.
    """

    def __init__(self, chunk: int, max_entries: int):
        self.chunk = max(1, chunk)
        self.max_entries = max(1, max_entries)
        self._index: OrderedDict[bytes, int] = OrderedDict()   # fp -> rank bitmask

    def fingerprints(self, input_ids) -> list[bytes]:
        n = len(input_ids) // self.chunk
        if n == 0:
            return []
        buf = array("q", input_ids[: n * self.chunk]).tobytes()
        stride = self.chunk * 8
        h = hashlib.blake2b(digest_size=16)
        out = []
        for k in range(n):
            h.update(buf[k * stride : (k + 1) * stride])
            out.append(h.digest())
        return out

    def lookup(self, fps: list[bytes]) -> dict[int, int]:
        """{rank: number of matched chunks} -- the longest known prefix per rank."""
        idx = self._index
        out: dict[int, int] = {}
        for k in range(len(fps) - 1, -1, -1):
            mask = idx.get(fps[k])
            if not mask:
                continue
            r = 0
            while mask:
                if mask & 1 and r not in out:
                    out[r] = k + 1
                mask >>= 1
                r += 1
        return out

    def record(self, fps: list[bytes], rank: int) -> None:
        idx = self._index
        bit = 1 << rank
        for fp in fps:
            if fp in idx:
                idx.move_to_end(fp)
                idx[fp] |= bit
            else:
                idx[fp] = bit
        while len(idx) > self.max_entries:
            idx.popitem(last=False)

    def __len__(self) -> int:
        return len(self._index)


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            get_parallel().load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # Init inter-process communication
        self.context = zmq.Context(1 + get_parallel().num_dp_ranks)
        if get_parallel().node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
            LoadBalanceMethod.PREFIX_AFFINITY: self.prefix_affinity_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
            LoadBalanceMethod.PREFIX_AFFINITY,
        )

        self.launch_dp_size: int = get_parallel().num_dp_ranks
        self.max_dp_size: int = (
            get_parallel().max_ep_size or get_parallel().num_dp_ranks
        )
        assert self.max_dp_size >= self.launch_dp_size, (
            f"--max-ep-size ({self.max_dp_size}) must be >= "
            f"--dp ({self.launch_dp_size})."
        )

        self.dp_active: list[bool] = [True] * self.launch_dp_size + [False] * (
            self.max_dp_size - self.launch_dp_size
        )

        self.dp_budget = DPBudget(get_parallel().num_dp_ranks)
        self.load_snapshot_reader = create_load_snapshot_reader(
            port_args,
            caller="DataParallelController",
        )
        self.prefix_affinity: PrefixAffinityIndex | None = None
        if self.load_balance_method == LoadBalanceMethod.PREFIX_AFFINITY:
            self.prefix_affinity = PrefixAffinityIndex(
                chunk=envs.SGLANG_DP_PREFIX_AFFINITY_CHUNK.get(),
                max_entries=envs.SGLANG_DP_PREFIX_AFFINITY_MAX_ENTRIES.get(),
            )
            self.prefix_affinity_sticky_tokens = (
                envs.SGLANG_DP_PREFIX_AFFINITY_STICKY_TOKENS.get()
            )
            # Clamped at 1.0: a weight below 1 would make the dispatcher PREFER
            # re-prefilling over waiting, which is the defect inverted.
            self.prefix_affinity_reprefill_weight = max(
                1.0, float(envs.SGLANG_DP_PREFIX_AFFINITY_REPREFILL_WEIGHT.get())
            )
            self._prefix_affinity_stats = {"hit": 0, "miss": 0, "spread": 0, "spread_tokens": 0}
            logger.info(
                "DP dispatch: prefix_affinity chunk=%d sticky_tokens=%d max_entries=%d "
                "reprefill_weight=%.2f",
                self.prefix_affinity.chunk,
                self.prefix_affinity_sticky_tokens,
                self.prefix_affinity.max_entries,
                self.prefix_affinity_reprefill_weight,
            )
        self._last_refresh_time = 0.0

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers: list[zmq.Socket | None] = [None] * self.max_dp_size
        self.status: list[bool] = list(self.dp_active)
        self._active_workers: list[int] = list(range(self.launch_dp_size))
        self._active_count_cache: int = self.launch_dp_size

        if get_parallel().attn_dp_enabled:
            self.launch_dp_attention_schedulers(server_args, port_args)
            # When local control broadcast is enabled, send control messages to
            # every DP group leader (attn_tp_rank=0) so each leader broadcasts
            # within its own attn_tp_group instead of the full tp_group.
            # Otherwise fall back to the original behaviour: send to only the
            # first leader, which then broadcasts over the full tp_group.
            local_ctrl = get_parallel().enable_dp_attention_local_control_broadcast
            self.control_message_step = 1 if local_ctrl else get_parallel().tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=get_device().soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if get_observability().enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        for i, worker in enumerate(self.workers):
            if worker is not None and self.status[i]:
                sock_send(worker, obj)

    def send_control_message(self, obj):
        for i in self._active_workers[:: self.control_message_step]:
            worker = self.workers[i]
            if worker is not None:
                sock_send(worker, obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        if get_exec().moe.elastic_ep_backend is not None:
            if len(ranks.status) != self.max_dp_size:
                logger.warning(
                    "[Elastic EP][DPC] active rank status len=%d != max_dp_size=%d; "
                    "ignoring update",
                    len(ranks.status),
                    self.max_dp_size,
                )
                return
            self.status = [
                self.dp_active[i] and bool(ranks.status[i])
                for i in range(self.max_dp_size)
            ]
            self._refresh_active_workers()
            return
        if len(ranks.status) != self.max_dp_size:
            logger.warning(
                "[DPC] update_active_ranks: status len=%d != max_dp_size=%d; "
                "ignoring update",
                len(ranks.status),
                self.max_dp_size,
            )
            return
        self.status = list(ranks.status)

    def add_elastic_workers(self, slot_offset: int, slot_count: int):
        """Activate a range of pre-bound worker slots."""
        end = slot_offset + slot_count
        if end > self.max_dp_size:
            raise ValueError(
                f"[Elastic EP] add_elastic_workers: slot_offset={slot_offset} + "
                f"slot_count={slot_count} exceeds max_dp_size={self.max_dp_size}. "
                f"Restart with a larger --max-ep-size."
            )

        for slot in range(slot_offset, end):
            if self.dp_active[slot]:
                logger.debug(
                    "[Elastic EP] add_elastic_workers: slot %d already active; "
                    "skipping",
                    slot,
                )
                continue
            assert self.workers[slot] is not None, (
                f"[Elastic EP] add_elastic_workers: slot {slot} was not "
                f"pre-bound at launch; expected a primary-bound PUSH socket."
            )
            self.dp_active[slot] = True
            self.status[slot] = True

        self._refresh_active_workers()
        logger.debug(
            "[Elastic EP] DataParallelController activated slots %s "
            "(active=%d / max=%d)",
            list(range(slot_offset, end)),
            self._active_count_cache,
            self.max_dp_size,
        )

    def _refresh_active_workers(self) -> None:
        self._active_workers = [
            i for i, active in enumerate(self.dp_active) if active and self.status[i]
        ]
        self._active_count_cache = len(self._active_workers)

    def refresh_load_budget(self):
        # Throttle to at most once per 20ms.  When a burst of requests
        # arrives, dispatching_with_trace() calls this before every
        # dispatch.  Each call reads the latest scheduler snapshot and
        # overwrites the speculative +1 increments that DPBudget.dispatch()
        # added for previously dispatched requests in this burst.  Without
        # throttling, the budget resets to the (stale) scheduler-reported
        # value on every request, causing the entire burst to land on a
        # single DP rank.  The 20ms interval lets the burst complete
        # using speculative counters, then refreshes from the real
        # scheduler load for the next batch.
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.02:
            return
        self._last_refresh_time = now
        self.dp_budget.update_budget(self.load_snapshot_reader.read_all())

    def dispatching_with_trace(self, req: Req, refresh_load_budget: bool = True):
        if refresh_load_budget and self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()

        time_stats = DPControllerReqTimeStats.new_from_obj(
            unwrap_from_pickle(req.time_stats)
        )

        time_stats.set_dp_dispatch_time()
        req.time_stats = wrap_as_pickle(time_stats)
        self.dispatching(req)
        req.time_stats = time_stats
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (ActiveRanksOutput, self.update_active_ranks),
                (
                    ElasticScaleUpdateReq,
                    lambda msg: self.add_elastic_workers(
                        msg.slot_offset, msg.slot_count
                    ),
                ),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(get_parallel().num_dp_ranks):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            tmp_port_args.instance_id = port_args.instance_id

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                get_parallel().tp_size
                * get_parallel().pp_size
                * get_device().gpu_id_step
            )

            if get_parallel().node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: list[int] | None = None
    ) -> list[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        is_joiner = get_exec().moe.is_ep_scale_joiner
        if get_parallel().dist_init_addr is None or is_joiner:
            na = NetworkAddress(
                get_serving().host or "127.0.0.1",
                get_serving().port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(get_parallel().dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if get_parallel().node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, get_parallel().nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: list[int]
    ) -> list[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = sock_recv(rep_socket)
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                sock_send(rep_socket, wrap_as_pickle(worker_ports))
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            if get_exec().moe.elastic_ep_backend is None:
                rep_socket.close()
            else:
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: list[int]):
        """Background thread: serve the pre-bound worker-port list to
        late-arriving elastic joiners. Publishes port numbers only; the primary
        keeps ownership of every socket."""
        while True:
            try:
                client_rank = sock_recv(rep_socket)
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            # Send worker ports to client
            sock_send(rep_socket, wrap_as_pickle(worker_ports))
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _receive_ports_as_client(self, endpoint: str) -> list[int]:
        """Receive worker ports from the server node."""
        logger.debug("Connecting to node 0 to receive worker ports")
        node_rank = get_parallel().node_rank

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            sock_send(req_socket, wrap_as_pickle(str(node_rank)))

            # Receive worker ports
            worker_ports = sock_recv(req_socket)
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def _joiner_local_tp_span(self, server_args: ServerArgs) -> int:
        return get_parallel().tp_size

    def _joiner_slot_offset(self, server_args: ServerArgs) -> int:
        return get_parallel().ep_join_rank_offset

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        if get_parallel().dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(get_parallel().dist_init_addr).host

        worker_ports = []
        if get_exec().moe.is_ep_scale_joiner:
            # Scale joiners connect to their pre-bound primary worker sockets.
            primary = NetworkAddress.parse(get_parallel().dist_init_addr)
            primary_endpoint = NetworkAddress(
                primary.host, primary.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA
            ).to_tcp()
            all_ports = self._receive_ports_as_client(primary_endpoint)
            offset = self._joiner_slot_offset(server_args)
            local_tp_span = self._joiner_local_tp_span(server_args)
            broadcasted_ports = all_ports[offset : offset + local_tp_span]
        elif get_parallel().node_rank == 0:
            # Elastic primaries reserve sockets for the maximum DP size.
            bind_count = (
                self.max_dp_size
                if get_exec().moe.elastic_ep_backend is not None
                else get_parallel().num_dp_ranks
            )
            for slot in range(bind_count):
                worker_port, worker_socket = get_zmq_socket_on_host(
                    self.context, zmq.PUSH, host=bind_host
                )
                worker_ports.append(worker_port)
                self.workers[slot] = worker_socket
                logger.debug(
                    "Assigned port %s to worker slot %s on host %s",
                    worker_port,
                    slot,
                    bind_host,
                )
            broadcasted_ports = self._broadcast_worker_ports(server_args, worker_ports)
        else:
            broadcasted_ports = self._broadcast_worker_ports(server_args, None)

        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int | None,
        worker_ports: list[int] | None = None,
    ):
        if not get_parallel().attn_dp_enabled:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(get_parallel().pp_size // get_parallel().nnodes, 1)
        nnodes_per_pp_rank = max(get_parallel().nnodes // get_parallel().pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (get_parallel().node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (get_parallel().node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = get_parallel().tp_size // nnodes_per_tp_group
        if get_exec().moe.is_ep_scale_joiner:
            # Scale joiners enumerate their full local TP span.
            tp_rank_range = range(get_parallel().tp_size)
            tp_size_per_node = get_parallel().tp_size
        else:
            tp_rank_range = range(
                tp_size_per_node * (get_parallel().node_rank % nnodes_per_tp_group),
                tp_size_per_node * (get_parallel().node_rank % nnodes_per_tp_group + 1),
            )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if get_parallel().attn_dp_enabled:
                    # dp attention has different sharding logic
                    _, _, dp_rank, _ = compute_dp_attention_world_info(
                        tp_rank,
                        get_parallel().tp_size,
                        get_parallel().attn_dp_size,
                        get_parallel().attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    if get_exec().moe.is_ep_scale_joiner:
                        # Scale-joiner outputs return through the primary tokenizer.
                        primary_addr = NetworkAddress.parse(
                            get_parallel().dist_init_addr
                        )
                        primary_port_base = primary_addr.port + 1
                        rank_port_args.tokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base
                        ).to_tcp()
                        rank_port_args.detokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base + 1
                        ).to_tcp()
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port
                    rank_port_args.instance_id = port_args.instance_id

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    get_device().base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * get_device().gpu_id_step
                )
                # Derive the child's EP rank for its display label.
                moe_ep_rank = (
                    tp_rank
                    % (get_parallel().tp_size // get_parallel().moe_dp_size)
                    // (
                        get_parallel().tp_size
                        // get_parallel().moe_dp_size
                        // get_parallel().ep_size
                    )
                )

                # Scheduler internals use local ranks; logs use global ranks.
                offset = get_parallel().ep_join_rank_offset
                display_tp_rank = tp_rank + offset
                display_moe_ep_rank = moe_ep_rank + offset
                display_dp_rank = dp_rank + offset if dp_rank is not None else None

                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                            display_tp_rank,
                            display_dp_rank,
                            display_moe_ep_rank,
                        ),
                    )
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]
        self.startup_time = aggregate_scheduler_startup_times(
            info.get("startup_time") for info in scheduler_info
        )

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.routed_dp_rank is not None:
            rank = req.routed_dp_rank
            if (
                rank < 0
                or rank >= len(self.workers)
                or rank not in self._active_workers
                or self.workers[rank] is None
            ):
                raise ValueError(f"DP rank {rank} is not active.")
            logger.debug(f"Direct routing to DP rank {rank}")
            sock_send(self.workers[rank], req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        active = self._active_workers
        if not active:
            raise RuntimeError("No active DP workers are available for routing.")
        attempts = 0
        while attempts < len(active):
            slot = active[self.round_robin_counter % len(active)]
            self.round_robin_counter = (self.round_robin_counter + 1) % len(active)
            if self.status[slot]:
                logger.debug(f"Choose worker {slot}")
                sock_send(self.workers[slot], req)
                return
            attempts += 1
        raise RuntimeError(
            f"Cannot route request: all {len(active)} active DP workers "
            "are unavailable."
        )

    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        sock_send(self.workers[target_rank], req)

    def total_requests_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        sock_send(self.workers[target_worker], req)

    def total_tokens_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        estimated_tokens = len(req.input_ids)
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
        )
        sock_send(self.workers[target_worker], req)

    def prefix_affinity_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        index = self.prefix_affinity
        budget = self.dp_budget
        active = [
            i
            for i in self._active_workers
            if self.status[i] and self.workers[i] is not None
        ]
        fps = index.fingerprints(req.input_ids)
        n = len(req.input_ids)
        per_rank = index.lookup(fps)
        matched = {i: per_rank.get(i, 0) * index.chunk for i in active}
        best_match = max(matched.values()) if matched else 0
        # Estimated time-to-first-token in prefill tokens (both ranks prefill at
        # the same rate): wait behind that rank's prefill backlog, then the part
        # of the prompt its cache does not hold. A rank that already holds the
        # shared prompt is a free spread; one that holds nothing costs the whole
        # prompt.
        #
        # 🔴 TTFT alone is the WRONG objective to minimise, which is what this
        # function did before 2026-09-20. A WAIT behind backlog costs this request
        # latency but adds no work -- the machine was going to do it anyway. A
        # SPREAD re-prefills `best_match - matched[i]` tokens another rank already
        # holds: new GPU work that did not need to exist, and under DP attention
        # every rank runs the same forward, so it throttles decode on ALL ranks.
        # Weighting that term (REPREFILL_WEIGHT, default 2.0 = dp_size on the
        # served recipe) makes the dispatcher prefer waiting over re-prefilling.
        # WEIGHT = 1.0 is arithmetically identical to the old behaviour -- the
        # `n - best_match` remainder is constant across ranks, so it cannot move
        # the argmin -- and is therefore the exact rollback. The whole key reduces
        # to `argmin_i backlog[i] - WEIGHT * matched[i]` (the rest is constant
        # across ranks), which is the cleanest way to see both that equivalence
        # and what the weight does.
        #
        # 🔴 THE PRICE, so nobody has to rediscover it: a rank is willing to be
        # WEIGHT * matched[i] tokens MORE backlogged before the request spreads
        # off it. At WEIGHT=2 and a 100k match that is 200k tokens of extra queue
        # -- about 17 s at the measured 11.5k tok/s prefill rate -- accepted to
        # avoid a ~9 s all-rank throttle from re-prefilling that 100k. Mean queue
        # time before this change was 1.65 s, so the tail this can add is real and
        # is the thing to watch if TTFT regresses. Lower WEIGHT toward 1.0 if it
        # does; that is a pure TTFT-vs-waste dial.
        cost = {i: budget.prefill_backlog[i] + (n - matched[i]) for i in active}
        best_rank = min(
            active,
            key=lambda i: (
                cost[i] + (self.prefix_affinity_reprefill_weight - 1.0)
                * (best_match - matched[i]),
                -matched[i],
            ),
        )
        # At an exact backlog tie a SHORT best match -- below STICKY_TOKENS, e.g.
        # only a system prompt shared by every conversation -- alternates among
        # the ranks whose cost is within that much, so quiet-period conversations
        # do not all home on one rank; a longer match (a conversation's own
        # context) sticks.
        #
        # `near` deliberately keeps the UNWEIGHTED cost: it is about this request's
        # own TTFT among equally-backlogged ranks.
        #
        # What alternating can cost, stated rather than waved away: this branch runs
        # only when `best_match < STICKY_TOKENS`, so the waste it can accept is
        # bounded by best_match -- under 16384 tokens at the default, NOT "about
        # zero". That is the deliberate price of not homing every quiet-period
        # conversation onto one rank. And when backlogs are EQUAL -- which `near`
        # requires -- the weighted and unweighted rules select the SAME rank (with
        # backlog equal, both reduce to argmax matched), so the weight cannot move
        # the near set in the case this branch actually fires on.
        near = [
            i
            for i in active
            if budget.prefill_backlog[i] == budget.prefill_backlog[best_rank]
            and cost[i] - cost[best_rank] < self.prefix_affinity_sticky_tokens
        ]
        if best_match < self.prefix_affinity_sticky_tokens and len(near) > 1:
            target = near[self.round_robin_counter % len(near)]
            self.round_robin_counter += 1
        else:
            target = best_rank
        if best_match == 0:
            self._prefix_affinity_stats["miss"] += 1
        elif matched[target] == best_match:
            self._prefix_affinity_stats["hit"] += 1
        else:
            self._prefix_affinity_stats["spread"] += 1
            self._prefix_affinity_stats["spread_tokens"] += best_match - matched[target]
        # Speculative increments until the next snapshot refresh.
        new_tokens = max(0, n - matched[target])
        budget.total_requests[target] += 1
        budget.total_tokens[target] += new_tokens
        budget.prefill_backlog[target] += new_tokens
        budget.pending_dispatches[target].append((time.time(), new_tokens))
        index.record(fps, target)
        st = self._prefix_affinity_stats
        if (st["hit"] + st["miss"] + st["spread"]) % 200 == 0:
            logger.info(
                "DP prefix_affinity: hit=%d miss=%d spread=%d spread_tokens=%d index=%d",
                st["hit"],
                st["miss"],
                st["spread"],
                st["spread_tokens"],
                len(index),
            )
        sock_send(self.workers[target], req)

    def event_loop(self):
        # Wait on the socket with a bounded poll instead of spinning on
        # NOBLOCK: the spin pinned a full host core for the life of the
        # server. The timeout keeps the soft watchdog fed while idle.
        poller = zmq.Poller()
        poller.register(self.recv_from_tokenizer, zmq.POLLIN)
        while True:
            self.soft_watchdog.feed()
            poller.poll(timeout=1000)
            while True:
                try:
                    recv_req = sock_recv(self.recv_from_tokenizer, flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # This process reads the config namespaces before spawning schedulers.
    publish(server_args, role="dp_controller")
    configure_logger(server_args)
    if get_observability().enable_trace:
        process_tracing_init(
            get_observability().otlp_traces_endpoint,
            get_observability().otlp_service_name,
            trace_modules=get_observability().trace_modules,
        )
        thread_label = "DP Controller"
        if get_disagg().disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif get_disagg().disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                "startup_time": controller.startup_time,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        # The primary owns routing for the expanded scheduler set.
        if get_parallel().node_rank == 0 and not get_exec().moe.is_ep_scale_joiner:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
