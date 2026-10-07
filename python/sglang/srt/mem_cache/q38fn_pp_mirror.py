# _q38fn_schednosync: host mirror of the lazy MTP ping-pong slots (worktree-only
# patch, tools/sched_nosync_patch.py in the qwen3-8-flash-next model dir).
import logging
import os

import torch

ENABLED = os.environ.get("SGLANG_Q38FN_SCHED_NOSYNC", "0") == "1"
CHECK = ENABLED and os.environ.get("SGLANG_Q38FN_SCHED_NOSYNC_CHECK", "0") == "1"
logger = logging.getLogger(__name__)
_mismatch = [0]


def _report(what, mirror, real):
    _mismatch[0] += 1
    if _mismatch[0] <= 20 or _mismatch[0] % 1000 == 0:
        logger.warning(
            "[q38fn-schednosync] MISMATCH %s mirror=%s real=%s n=%d",
            what, mirror, real, _mismatch[0],
        )


def set_all(kv, buf, n):
    if ENABLED:
        kv.mamba_pp_valid_cpu = (buf, [i < n for i in range(int(buf.shape[0]))])


def set_one(kv, idx, value):
    m = kv.mamba_pp_valid_cpu
    if m is not None and m[0] is kv.mamba_ping_pong_track_buffer:
        m[1][idx] = not (isinstance(value, int) and value == -1)


def _mirror(kv):
    m = kv.mamba_pp_valid_cpu
    if ENABLED and m is not None and m[0] is kv.mamba_ping_pong_track_buffer:
        return m[1]
    return None


def valid(kv, idx):
    """True iff kv.mamba_ping_pong_track_buffer[idx] != -1."""
    m = _mirror(kv)
    if m is None:
        return kv.mamba_ping_pong_track_buffer[idx].item() != -1
    if CHECK:
        real = kv.mamba_ping_pong_track_buffer[idx].item() != -1
        if real != m[idx]:
            _report(f"valid[{idx}]", m[idx], real)
            return real
    return m[idx]


def select_valid(kv, row, positions):
    """row[positions] restricted to the allocated (!= -1) slots, without a
    boolean-mask readback; None when the mirror does not describe kv's buffer."""
    m = _mirror(kv)
    if m is None:
        return None
    keep = [p for p in positions if m[p]]
    if not keep:
        out = row[0:0]
    elif keep == list(range(keep[0], keep[-1] + 1)):
        out = row[keep[0] : keep[-1] + 1]
    else:
        return None
    if CHECK:
        real = row[torch.tensor(positions, device=row.device)]
        real = real[real != -1]
        if real.tolist() != out.tolist():
            _report("select_valid", out.tolist(), real.tolist())
            return real
    return out


def check_total(what, host_total, dev_lens):
    if CHECK:
        real = int(dev_lens.sum().item())
        if real != host_total:
            _report(what, host_total, real)
            return False
    return True
