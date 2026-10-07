# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


# _q38fn_fwdnosync: host cu_seqlens registered by the metadata builder for the
# exact device tensor it built (tools/fwd_nosync_patch.py in the model dir).
import logging as _q38fn_logging
import os as _q38fn_os

_Q38FN_ON = _q38fn_os.environ.get("SGLANG_Q38FN_FWD_NOSYNC", "0") == "1"
_Q38FN_CHECK = _Q38FN_ON and _q38fn_os.environ.get("SGLANG_Q38FN_FWD_NOSYNC_CHECK", "0") == "1"
_q38fn_hint = [None, None]
_q38fn_log = _q38fn_logging.getLogger(__name__)
_q38fn_mismatch = [0]


def q38fn_set_host_cu_seqlens(cu_seqlens, host):
    if _Q38FN_ON:
        _q38fn_hint[0] = cu_seqlens
        _q38fn_hint[1] = host


def q38fn_report(what, mine, real):
    _q38fn_mismatch[0] += 1
    if _q38fn_mismatch[0] <= 20 or _q38fn_mismatch[0] % 1000 == 0:
        _q38fn_log.warning(
            "[q38fn-fwdnosync] MISMATCH %s mine=%s real=%s n=%d",
            what, str(mine)[:200], str(real)[:200], _q38fn_mismatch[0],
        )


_q38fn_checks = [0]


def q38fn_checked(what):
    _q38fn_checks[0] += 1
    if _q38fn_checks[0] % 500 == 0:
        _q38fn_log.info(
            "[q38fn-fwdnosync] checks=%d mismatches=%d last=%s",
            _q38fn_checks[0], _q38fn_mismatch[0], what,
        )


def _q38fn_host_lens(cu_seqlens):
    if _Q38FN_ON and _q38fn_hint[0] is cu_seqlens:
        h = _q38fn_hint[1]
        return [int(b) - int(a) for a, b in zip(h[:-1], h[1:])]
    return None


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    _q38fn_lens = _q38fn_host_lens(cu_seqlens)  # _q38fn_fwdnosync
    if _q38fn_lens is not None:
        _q38fn_i = torch.cat(
            [torch.arange((n + chunk_size - 1) // chunk_size) for n in _q38fn_lens]
        )
        _q38fn_out = (
            torch.stack([_q38fn_i.eq(0).cumsum(0) - 1, _q38fn_i], 1)
            .to(cu_seqlens.dtype)
            .pin_memory()
            .to(cu_seqlens.device, non_blocking=True)
        )
        if not _Q38FN_CHECK:
            return _q38fn_out
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    _q38fn_real = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    if _q38fn_lens is not None:
        q38fn_checked("chunk_indices")
        if not torch.equal(_q38fn_out, _q38fn_real):
            q38fn_report("chunk_indices", _q38fn_out.tolist(), _q38fn_real.tolist())
    return _q38fn_real


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    _q38fn_lens = _q38fn_host_lens(cu_seqlens)  # _q38fn_fwdnosync
    if _q38fn_lens is not None:
        _q38fn_acc = [0]
        for n in _q38fn_lens:
            _q38fn_acc.append(_q38fn_acc[-1] + (n + chunk_size - 1) // chunk_size)
        _q38fn_out = torch.tensor(_q38fn_acc, dtype=torch.int64, pin_memory=True).to(
            cu_seqlens.device, non_blocking=True
        )
        if not _Q38FN_CHECK:
            return _q38fn_out
    _q38fn_real = torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
    if _q38fn_lens is not None:
        q38fn_checked("chunk_offsets")
        if not (
            _q38fn_out.dtype == _q38fn_real.dtype
            and torch.equal(_q38fn_out, _q38fn_real)
        ):
            q38fn_report("chunk_offsets", _q38fn_out.tolist(), _q38fn_real.tolist())
    return _q38fn_real
