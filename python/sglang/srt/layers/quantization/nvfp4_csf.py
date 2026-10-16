"""Load-time decoding of NVFP4 compressed scale factors ("csf").

Some ModelOpt NVFP4 checkpoints (e.g. nvidia/Qwen3.8-Flash-Next-FP4-Local) mark
a layer with ``"weight_scale_encoding": "csf"`` and store each projection's
E4M3 block scales not as ``<prefix>.weight_scale`` but as two tensors:

* ``<prefix>.weight_scale.nvfp4_csf_fixed`` -- uint8, shape
  ``[rows / 16, 16 * (1 + columns / 2)]``. Each row of it is one 16-row slab:
  16 per-row base bytes (0..240), then the slab's 16 rows of packed 4-bit
  offsets, two per byte, low nibble first.
* ``<prefix>.weight_scale.nvfp4_csf_exceptions`` -- uint32, one word per block
  whose scale does not fit base + offset: the low 24 bits are the flat
  position ``row * columns + column`` (sorted, unique) and the high 8 bits the
  scale byte itself.

``columns`` is ``in_features / 16``, one E4M3 scale per 16-element group. The
encoding is lossless: base + offset (or the exception byte) is exactly the
standard ModelOpt ``weight_scale`` byte. Decoding here, in the weight iterator,
hands every downstream NVFP4 method the standard ``weight_scale`` it already
loads, so no kernel or quant method needs to know the encoding exists.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple

import torch

CSF_FIXED_SUFFIX = ".nvfp4_csf_fixed"
CSF_EXCEPTIONS_SUFFIX = ".nvfp4_csf_exceptions"
CSF_SCALE_ENCODING = "csf"

_SLAB_ROWS = 16
_MAX_BASE = 240
_POSITION_MASK = 0xFFFFFF


def decode_nvfp4_csf_scale(
    fixed: torch.Tensor, exceptions: torch.Tensor
) -> torch.Tensor:
    """Return the ``[rows, columns]`` float8_e4m3fn scale a csf pair encodes."""
    if fixed.dtype != torch.uint8 or fixed.dim() != 2:
        raise ValueError(
            f"NVFP4-CSF fixed plane must be a 2-D uint8 tensor, got "
            f"{fixed.dtype} {tuple(fixed.shape)}"
        )
    if exceptions.dim() != 1 or exceptions.element_size() != 4 or (
        exceptions.dtype.is_floating_point
    ):
        raise ValueError(
            f"NVFP4-CSF exceptions must be a 1-D 32-bit integer tensor, got "
            f"{exceptions.dtype} {tuple(exceptions.shape)}"
        )
    slabs, width = fixed.shape
    if slabs == 0 or width % _SLAB_ROWS or width < 2 * _SLAB_ROWS:
        raise ValueError(
            f"NVFP4-CSF fixed plane shape {tuple(fixed.shape)} is not "
            f"[rows/16, 16 * (1 + columns/2)]"
        )
    rows = slabs * _SLAB_ROWS
    packed_columns = width // _SLAB_ROWS - 1
    columns = 2 * packed_columns

    fixed = fixed.contiguous()
    bases = fixed[:, :_SLAB_ROWS].reshape(rows, 1)
    if int(bases.max()) > _MAX_BASE:
        raise ValueError(f"NVFP4-CSF row bases must be in 0..{_MAX_BASE}")
    packed = fixed[:, _SLAB_ROWS:].reshape(rows, packed_columns)
    offsets = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, columns)
    # base <= 240 and offset <= 15, so the uint8 sum cannot wrap.
    logical = bases + offsets

    if exceptions.numel():
        words = exceptions.contiguous().view(torch.int32).to(torch.int64)
        words = words & 0xFFFFFFFF
        positions = words & _POSITION_MASK
        if int(positions[-1]) >= rows * columns or (
            positions.numel() > 1 and bool((positions[1:] <= positions[:-1]).any())
        ):
            raise ValueError(
                "NVFP4-CSF exception positions must be sorted, unique and inside "
                f"the {rows}x{columns} scale plane"
            )
        logical.view(-1)[positions] = (words >> 24).to(torch.uint8)

    return logical.view(torch.float8_e4m3fn)


def decode_nvfp4_csf_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Pass a checkpoint's tensors through, replacing every csf pair with the
    ``weight_scale`` it encodes.

    The two halves of a pair may arrive in either order and from different
    shards. A checkpoint without csf tensors passes through unchanged. An
    unpaired half at the end is an incomplete checkpoint, and raises.
    """
    pending: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        if name.endswith(CSF_FIXED_SUFFIX):
            suffix = CSF_FIXED_SUFFIX
        elif name.endswith(CSF_EXCEPTIONS_SUFFIX):
            suffix = CSF_EXCEPTIONS_SUFFIX
        else:
            yield name, tensor
            continue
        base = name[: -len(suffix)]
        parts = pending.setdefault(base, {})
        if suffix in parts:
            raise ValueError(f"NVFP4-CSF tensor {name} appears twice")
        if len(parts) == 0:
            # Hold a private copy: an mmap-backed tensor may not outlive its
            # shard, and the other half can come from a later one.
            parts[suffix] = tensor.clone()
            continue
        parts[suffix] = tensor
        del pending[base]
        yield base, decode_nvfp4_csf_scale(
            parts[CSF_FIXED_SUFFIX], parts[CSF_EXCEPTIONS_SUFFIX]
        )
    if pending:
        missing = sorted(pending)
        raise ValueError(
            f"NVFP4-CSF checkpoint is missing the other half of {len(missing)} "
            f"scale plane(s), e.g. {missing[0]}"
        )
