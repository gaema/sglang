# ds4-transfer T1 -- graph-exec prefetch.
#
# Installed by ai/models/lm/qwen3-8-flash-next/tools/ds4_t1_graph_exec_prefetch.py.
# Concept source: plan/ds4-concept-transfer.md item T1 -- upload a ready graph
# exec ahead of its first launch so the launch finds the graph resident.
#
# Gated OFF by default. With the gate off every entry point returns on the
# module-level boolean below before touching ctypes, CUDA or the graph, so the
# prior path is restored bit-for-bit inside one binary.
from __future__ import annotations

import ctypes
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

GATE = "SGLANG_DS4_T1_GRAPH_EXEC_PREFETCH"
ENABLED = os.environ.get(GATE, "0") == "1"

_fn = None          # the resolved upload entry point
_fn_name = ""       # "cudaGraphUpload" or "cuGraphUpload"
_fn_lib = ""        # the library path it came from
_resolved = False   # resolution attempted (success or failure)
_announced = False
_uploaded = 0
_failed = 0
_seconds = 0.0


def _loaded_library(stem: str) -> Optional[str]:
    """Path of an ALREADY-mapped library whose soname starts with `stem`.

    Resolving out of /proc/self/maps rather than by soname is deliberate: a
    second copy of libcudart in one process keeps its own state, and the exec
    handle we are about to hand over was minted by the copy torch is using.
    """
    try:
        with open("/proc/self/maps", encoding="utf-8") as fh:
            for line in fh:
                path = line.rstrip("\n").split(" ", 5)[-1].strip()
                if not path.startswith("/"):
                    continue
                base = os.path.basename(path)
                if base.startswith(stem):
                    return path
    except OSError:
        return None
    return None


def _resolve() -> None:
    """Resolve an upload entry point. Runtime API first, driver API as fallback."""
    global _fn, _fn_name, _fn_lib, _resolved
    _resolved = True
    for stem, sym in (("libcudart.so", "cudaGraphUpload"), ("libcuda.so", "cuGraphUpload")):
        path = _loaded_library(stem)
        if path is None:
            continue
        try:
            lib = ctypes.CDLL(path)
            fn = getattr(lib, sym)
        except (OSError, AttributeError):
            continue
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _fn, _fn_name, _fn_lib = fn, sym, path
        return
    logger.warning(
        "ds4:T1 graph-exec prefetch DISABLED -- no upload symbol found "
        "(looked for cudaGraphUpload in a mapped libcudart, then cuGraphUpload "
        "in a mapped libcuda). The prior path is unchanged."
    )


def upload(graph: Any, stream: Any = None) -> None:
    """Upload one instantiated graph exec. Never raises into the caller."""
    global _uploaded, _failed, _seconds, _announced
    if not ENABLED:
        return
    if not _resolved:
        _resolve()
    if _fn is None:
        return
    try:
        exec_handle = graph.raw_cuda_graph_exec()
    except Exception as exc:  # not instantiated, or the API moved
        _failed += 1
        logger.warning("ds4:T1 graph-exec prefetch: no exec handle (%s)", exc)
        return
    if stream is None:
        try:
            import torch

            stream = torch.cuda.current_stream()
        except Exception:
            stream = None
    raw_stream = getattr(stream, "cuda_stream", 0) or 0
    t0 = time.perf_counter()
    rc = _fn(ctypes.c_void_p(exec_handle), ctypes.c_void_p(raw_stream))
    _seconds += time.perf_counter() - t0
    if rc != 0:
        _failed += 1
        logger.warning("ds4:T1 graph-exec prefetch: %s returned %d", _fn_name, rc)
        return
    _uploaded += 1
    if not _announced:
        _announced = True
        logger.info(
            "ds4:T1 graph-exec prefetch ENABLED symbol=%s lib=%s", _fn_name, _fn_lib
        )


def log_summary() -> None:
    """One line per capture session. This is the readback the A/B keys on."""
    if not ENABLED:
        return
    logger.info(
        "ds4:T1 graph-exec prefetch SUMMARY uploaded=%d failed=%d upload_ms=%.3f",
        _uploaded,
        _failed,
        _seconds * 1000.0,
    )


def stats() -> dict:
    return {
        "enabled": ENABLED,
        "symbol": _fn_name,
        "lib": _fn_lib,
        "uploaded": _uploaded,
        "failed": _failed,
        "upload_ms": _seconds * 1000.0,
    }
