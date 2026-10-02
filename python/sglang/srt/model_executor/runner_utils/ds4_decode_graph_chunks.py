# ds4-transfer T2 -- chunked decode-graph capture.
#
# Installed by ai/models/lm/qwen3-8-flash-next/tools/ds4_t2_decode_graph_chunks.py.
# Concept source: plan/ds4-concept-transfer.md item T2.
#
# Gated OFF by default (K <= 1). With the gate off capture_target() returns its
# arguments unchanged, active() is a nullcontext and no hook is ever installed,
# so the prior path is restored bit-for-bit inside one binary.
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

GATE = "SGLANG_DS4_T2_DECODE_GRAPH_CHUNKS"


def _read_chunks() -> int:
    raw = os.environ.get(GATE, "1")
    try:
        k = int(raw)
    except ValueError:
        logger.warning("ds4:T2 %s=%r is not an int -- treating as 1 (OFF)", GATE, raw)
        return 1
    return k if k > 1 else 1


CHUNKS = _read_chunks()
ENABLED = CHUNKS > 1

_active = False
_announced = False
_hooks: List[Any] = []
_boundaries: List[int] = []
_stack_name = ""
_stack_depth = 0
_last_segments = 0


def _break_graph():
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        break_graph,
    )

    return break_graph


def boundaries_for(depth: int, chunks: int) -> List[int]:
    """Indices AFTER which a cut falls, for `chunks` near-equal pieces.

    Pure arithmetic, no torch: this is the half the self-test can check without
    a GPU, and the half a wrong K would corrupt silently.
    """
    if chunks <= 1 or depth <= 1:
        return []
    chunks = min(chunks, depth)
    cuts = []
    for i in range(1, chunks):
        idx = (i * depth) // chunks - 1
        if 0 <= idx < depth - 1 and idx not in cuts:
            cuts.append(idx)
    return cuts


def _find_decoder_stack(model: Any) -> Tuple[str, Optional[Any]]:
    """The largest nn.ModuleList in the model -- the decoder stack.

    Chosen by SIZE rather than by name so it does not depend on one model's
    attribute spelling. The pick is logged, so a wrong one is visible in the
    readback rather than silent.
    """
    best_name, best = "", None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > (len(best) if best else 0):
            best_name, best = name, mod
    return best_name, best


def install_hooks(model: Any) -> None:
    """Place K-1 cuts over the decoder stack. Idempotent; no-op when OFF."""
    global _boundaries, _stack_name, _stack_depth
    if not ENABLED or _hooks:
        return
    name, stack = _find_decoder_stack(model)
    if stack is None or len(stack) < 2:
        logger.warning(
            "ds4:T2 chunked decode capture DISABLED -- no decoder ModuleList found; "
            "the prior path is unchanged."
        )
        return
    _stack_name, _stack_depth = name, len(stack)
    _boundaries = boundaries_for(_stack_depth, CHUNKS)
    brk = _break_graph()

    def _hook(_module, _args, _output):
        # Inert unless the FULL decode backend is mid-capture: the prefill
        # backend also sets a breakable capture, and injecting extra cuts there
        # is not this lever.
        if _active:
            brk()
        return None

    for idx in _boundaries:
        _hooks.append(stack[idx].register_forward_hook(_hook))
    logger.info(
        "ds4:T2 chunked decode capture ENABLED chunks=%d stack=%s depth=%d cuts_after=%s",
        CHUNKS,
        _stack_name,
        _stack_depth,
        _boundaries,
    )


def remove_hooks() -> None:
    for h in _hooks:
        try:
            h.remove()
        except Exception:
            pass
    _hooks.clear()


@contextlib.contextmanager
def active():
    """Mark the FULL backend's own capture. nullcontext when the gate is off."""
    global _active
    if not ENABLED:
        yield
        return
    _active = True
    try:
        yield
    finally:
        _active = False


def capture_target(graph: Any, graph_ctx: Callable) -> Tuple[Any, Callable]:
    """Swap in the chunked container + its capture context. Identity when OFF."""
    if not ENABLED or not _hooks:
        return graph, graph_ctx
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
    )

    container = BreakableCUDAGraph()

    def _ctx(cuda_graph=None, pool=None, stream=None, **_kw):
        return BreakableCUDAGraphCapture(cuda_graph, pool=pool, stream=stream)

    return container, _ctx


def note_captured(graph: Any) -> None:
    """Record how many segments a capture actually produced."""
    global _last_segments, _announced
    if not ENABLED:
        return
    segs = len(getattr(graph, "_segments", []) or [])
    _last_segments = segs
    if not _announced:
        _announced = True
        logger.info(
            "ds4:T2 chunked decode capture SUMMARY chunks_requested=%d segments_captured=%d "
            "cuts_after=%s stack=%s depth=%d",
            CHUNKS,
            segs,
            _boundaries,
            _stack_name,
            _stack_depth,
        )
        if segs != CHUNKS:
            logger.warning(
                "ds4:T2 segments_captured=%d != chunks_requested=%d -- the cut count "
                "the device actually got is the first number.",
                segs,
                CHUNKS,
            )


def stats() -> dict:
    return {
        "enabled": ENABLED,
        "chunks": CHUNKS,
        "stack": _stack_name,
        "depth": _stack_depth,
        "boundaries": list(_boundaries),
        "segments": _last_segments,
        "hooks": len(_hooks),
    }
