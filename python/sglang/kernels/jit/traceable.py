"""Route DECLARED tvm_ffi JIT entry points through torch custom ops.

R2d (fn:N286, gate ``SGLANG_JIT_TRACEABLE_OPS``, default OFF): every JIT
kernel in ``sglang.kernels.ops`` is invoked as a raw ``tvm_ffi`` ``Function``
(``module.<export>(...)``), and Dynamo refuses that call inside a traced
region (``Dynamo does not know how to trace method `__call__` of class
`Function```). ``fn:N285`` fixed one instance by hand and hit the next one a
boot later; the fork has >100 such call sites, so the fix lives here, at the
layer that hands the ``Function`` out.

Mechanism. :func:`wrap_module` is called by ``load_jit`` on the module it is
about to return. With the gate OFF it returns the module untouched -- the
shipped path is byte-for-byte the same. With the gate ON, and only for a
module whose family (``module_args[0]``) has at least one export declared in
:data:`SPECS`, it returns a :class:`TraceableJitModule` proxy whose DECLARED
exports are ``torch.ops.sglang.<op>`` custom ops (eager impl = the raw
``Function``; fake impl = ``None``, since every declared kernel is
out-parameter style and returns nothing) and whose UNDECLARED exports are
the raw ``Function`` objects, unchanged. So the gate cannot silently alter a
kernel nobody declared: an undeclared kernel on a traced path still refuses
exactly as before, which is the signal to declare it.

A spec is the POSITIONAL calling convention of one export, as the wrapper in
``sglang.kernels.ops`` passes it to ``module.<export>(...)``:

    ``T`` Tensor input  ``O`` Tensor output (mutated in place)  ``i`` int
    ``f`` float  ``b`` bool

One custom op is registered per (module variant, export) -- the op name
carries the ``load_jit`` module name (template arguments included), so two
dtype/shape instantiations of one kernel are two ops, each bound to its own
``Function``. Registration is idempotent through ``direct_register_custom_op``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs

_KIND_ANNOTATION = {
    "T": "torch.Tensor",
    "O": "torch.Tensor",
    "i": "int",
    "f": "float",
    "b": "bool",
}


@dataclass(frozen=True)
class JitOpSpec:
    """Positional calling convention of one JIT export (see module doc)."""

    kinds: Tuple[str, ...]

    def __post_init__(self) -> None:
        bad = [k for k in self.kinds if k not in _KIND_ANNOTATION]
        if bad:
            raise ValueError(f"JitOpSpec: unknown arg kinds {bad}")
        if "O" not in self.kinds:
            raise ValueError(
                "JitOpSpec: an out-parameter kernel must declare at least one "
                "'O' (mutated output) argument"
            )


# (module family = load_jit's first marker arg, export name) -> spec.
# Declare ONLY kernels whose call site sits in a traced region; every
# undeclared export keeps the raw Function (see module doc).
SPECS: Dict[Tuple[str, str], JitOpSpec] = {
    # kernels/ops/layernorm/grouped_gemma_rmsnorm.py:79
    #   module.grouped_gemma_rmsnorm(x, weight, out, eps)
    ("grouped_gemma_rmsnorm", "grouped_gemma_rmsnorm"): JitOpSpec(("T", "T", "O", "f")),
    # kernels/ops/elementwise/hc_combine.py:92
    #   module.hc_combine(y, r, n, inject_weight, out)
    ("hc_combine", "hc_combine"): JitOpSpec(("T", "T", "T", "T", "O")),
    # kernels/ops/elementwise/hc_combine.py:130 (rows <= 32)
    #   module.hc_combine_split(y, r, n, inject_weight, out, partials)
    ("hc_combine", "hc_combine_split"): JitOpSpec(("T", "T", "T", "T", "O", "O")),
}

_FORCED = False


def enable() -> None:
    """Turn the gate on for this process (a model opting in calls this)."""
    global _FORCED
    _FORCED = True


def is_enabled() -> bool:
    return _FORCED or bool(envs.SGLANG_JIT_TRACEABLE_OPS.get())


def op_name_for(module_name: str, export: str) -> str:
    return "jit_" + re.sub(r"[^0-9A-Za-z_]", "_", f"{module_name}__{export}")


def _make_op_func(fn: Callable[..., Any], spec: JitOpSpec) -> Callable[..., None]:
    """A Python function with the export's positional signature, annotated so
    ``torch.library.infer_schema`` can derive the schema, forwarding to the raw
    ``Function``. Returns None: every declared kernel writes its ``O`` args."""
    names = [f"a{i}" for i in range(len(spec.kinds))]
    params = ", ".join(f"{n}: {_KIND_ANNOTATION[k]}" for n, k in zip(names, spec.kinds))
    src = f"def _op({params}) -> None:\n    _fn({', '.join(names)})\n"
    ns: Dict[str, Any] = {"torch": torch, "_fn": fn}
    exec(src, ns)  # noqa: S102 -- internal, spec-derived source only
    return ns["_op"]


def mutated_arg_names(spec: JitOpSpec) -> list:
    return [f"a{i}" for i, k in enumerate(spec.kinds) if k == "O"]


def _fake_impl(*args: Any, **kwargs: Any) -> None:
    return None


def register_op(module_name: str, export: str, fn: Callable[..., Any], spec: JitOpSpec):
    """Register (idempotently) and return ``torch.ops.sglang.<op>`` for one export."""
    from sglang.srt.utils.common import direct_register_custom_op

    name = op_name_for(module_name, export)
    if not hasattr(torch.ops.sglang, name):
        direct_register_custom_op(
            op_name=name,
            op_func=_make_op_func(fn, spec),
            mutates_args=mutated_arg_names(spec),
            fake_impl=_fake_impl,
        )
    return getattr(torch.ops.sglang, name)


class TraceableJitModule:
    """Proxy for one loaded JIT module (gate ON, family declared).

    Declared exports are plain instance attributes holding the custom op;
    undeclared exports are plain instance attributes holding the raw
    ``Function`` (bound at construction from the build's wrapper list), so
    attribute access in a traced region is an ordinary instance-dict lookup.
    Anything else falls through to the wrapped module.
    """

    def __init__(
        self,
        module: Any,
        module_name: str,
        family: str,
        exports: Sequence[str],
    ) -> None:
        self._module = module
        self._module_name = module_name
        self._family = family
        self.declared: Tuple[str, ...] = ()
        declared = []
        for export in exports:
            fn = getattr(module, export)
            spec = SPECS.get((family, export))
            if spec is None:
                setattr(self, export, fn)
            else:
                setattr(self, export, register_op(module_name, export, fn, spec))
                declared.append(export)
        self.declared = tuple(declared)

    def __getattr__(self, name: str) -> Any:  # only reached for missing attrs
        return getattr(self._module, name)

    def __repr__(self) -> str:
        return (
            f"TraceableJitModule({self._module_name}, declared={self.declared})"
        )


def wrap_module(
    module: Any,
    *,
    module_name: str,
    module_args: Sequence[str],
    exports: Sequence[str],
) -> Any:
    """``load_jit``'s hook: the module itself unless the gate is ON and at
    least one of this build's exports is declared for its family."""
    if not module_args or not is_enabled():
        return module
    family = str(module_args[0])
    if not any((family, e) in SPECS for e in exports):
        return module
    return TraceableJitModule(module, module_name, family, exports)
