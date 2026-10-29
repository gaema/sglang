"""Two-micro-batch prefill wavefront for qwen38fn on plain TP (SGLANG_TBO_TP_WAVE=1; overlap plan M3c).

The two TBO children of one extend chunk (A = the first half of the rows, B = the second) run as two Python threads
that pass ONE baton, so exactly one of them runs Python at any time and both enqueue on the parent's CUDA stream: the
GPU order is the baton order. A child gives the baton away inside the copy-engine TP all-reduce, between start() (its
payload is signalled and the peer pull is enqueued on the copy stream) and finish() (the forward stream waits for the
pull). So A's peer copy runs under B's kernels and the reverse, B running half a stage behind A:

    A: attn L, start ->  B: attn L, start ->  A: finish, ffn L, start ->  B: finish, ffn L, start ->  A: finish, attn L+1

B's layer L reads the KV, GDN and PLE conv state A's layer L wrote, which the stream order guarantees. Every TP rank
runs the same schedule, so the ranks' exchanges pair up slot by slot (A on slot 0, B on slot 1).

What the threads must not share, and how this module keeps them apart:
- the forward context is one module-level global: the baton holder re-installs its own on every resume;
- per-module state the layers stash across a yield (Qwen4Exp's _hc_norms / PLE batch / PLE prefetch) is made
  thread-local by `thread_local_attrs` on those classes, installed only when this mode is on.
"""

import os
import threading
from typing import Callable, List, Optional, Sequence

import torch

ENABLED = os.environ.get("SGLANG_TBO_TP_WAVE") == "1"

_tls = threading.local()


class _ThreadLocalAttr:
    """A data descriptor holding one value per (thread, instance); unset reads as None (every attribute it replaces
    is initialised to None)."""

    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        d = getattr(_tls, "attrs", None)
        return None if d is None else d.get((id(obj), self.name))

    def __set__(self, obj, value) -> None:
        d = getattr(_tls, "attrs", None)
        if d is None:
            d = _tls.attrs = {}
        d[(id(obj), self.name)] = value


def thread_local_attrs(cls, names: Sequence[str]):
    """Make `names` thread-local on `cls` when the wave mode is on; otherwise leave the class untouched."""
    if ENABLED:
        for name in names:
            setattr(cls, name, _ThreadLocalAttr(name))
    return cls


class _Baton:
    def __init__(self, set_ctx: Callable, ctxs: Sequence):
        self.cv = threading.Condition()
        self.turn = 0
        self.done = [False, False]
        self.error: List[Optional[BaseException]] = [None, None]
        self.set_ctx = set_ctx
        self.ctxs = ctxs

    def _wait_turn(self, me: int) -> None:
        with self.cv:
            while self.turn != me and not self.done[1 - me]:
                self.cv.wait()
            if self.error[1 - me] is not None:
                raise RuntimeError("q38fn wave: the other micro-batch failed") from self.error[1 - me]
        self.set_ctx(self.ctxs[me])

    def give(self, me: int) -> None:
        """Hand the baton to the other child (if it is still running) and wait for it to come back."""
        with self.cv:
            if self.done[1 - me]:
                return
            self.turn = 1 - me
            self.cv.notify_all()
        self._wait_turn(me)

    def finish(self, me: int, error: Optional[BaseException]) -> None:
        with self.cv:
            self.done[me] = True
            self.error[me] = error
            self.turn = 1 - me
            self.cv.notify_all()


def slot() -> int:
    """The exchange slot of the calling child (0 outside a wave)."""
    return getattr(_tls, "slot", 0)


def role() -> Optional[int]:
    """0 for child A, 1 for child B, None outside a wave."""
    return getattr(_tls, "role", None)


def yield_point() -> None:
    """Called between an all-reduce's start() and finish(): run the other child until its next yield."""
    baton = getattr(_tls, "baton", None)
    if baton is not None:
        baton.give(_tls.role)


def run(fns: Sequence[Callable[[], object]], ctxs: Sequence, set_ctx: Callable) -> list:
    """Run fns[0] (child A) and fns[1] (child B) as the wavefront; returns their results in order."""
    assert len(fns) == 2 and len(ctxs) == 2
    baton = _Baton(set_ctx, ctxs)
    stream = torch.cuda.current_stream()
    device = torch.cuda.current_device()
    grad = torch.is_grad_enabled()
    results = [None, None]
    prev_ctx = set_ctx(None)
    set_ctx(prev_ctx)

    def body(me: int) -> None:
        _tls.role, _tls.slot, _tls.baton = me, me, baton
        err = None
        try:
            torch.cuda.set_device(device)
            with torch.cuda.stream(stream), torch.set_grad_enabled(grad):
                baton._wait_turn(me)
                results[me] = fns[me]()
        except BaseException as e:  # handed to the other child and re-raised by the caller
            err = e
        finally:
            _tls.baton = None
            baton.finish(me, err)

    threads = [threading.Thread(target=body, args=(i,), name=f"q38fn-wave-{'AB'[i]}") for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    set_ctx(prev_ctx)
    for e in baton.error:
        if e is not None:
            raise e
    return results
