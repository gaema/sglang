# Copyright 2023-2026 SGLang Team
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
from __future__ import annotations

import contextlib
import datetime
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_model,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
)
from sglang.srt.utils import empty_context, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS = frozenset()


# --- fn:N129 memory probe (gated, default OFF) -------------------------------
_N129_PROBE_ENV = "SGLANG_N129_MEM_PROBE"


def _n129_mem_probe(tag: str) -> None:
    if tag.startswith("P0"):
        _n129_history_start()

    """Log driver-level and PyTorch-level GPU memory at one point.

    The three numbers are chosen so their differences separate a POOL from a
    LEAK: `torch_reserved - torch_alloc` is cached-but-free (empty_cache()
    reclaims it), `driver_used - torch_reserved` is everything PyTorch's
    allocator does not own -- flashinfer/TVM-FFI tensors, cuBLAS/cutlass
    workspaces, and CUDA modules loaded by JIT'd tactics.
    """
    if not os.environ.get(_N129_PROBE_ENV):
        return
    try:
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        alloc = torch.cuda.memory_allocated()
        used = total - free
        logger.info(
            "N129MEM %s driver_used=%.4f free=%.4f torch_reserved=%.4f "
            "torch_alloc=%.4f pooled_free=%.4f non_torch=%.4f (GiB)",
            tag,
            used / 2**30,
            free / 2**30,
            reserved / 2**30,
            alloc / 2**30,
            (reserved - alloc) / 2**30,
            (used - reserved) / 2**30,
        )
    except Exception as exc:  # a probe must never change an outcome
        logger.info("N129MEM %s probe failed: %r", tag, exc)


def _n129_drop_moe_runners(tag: str) -> None:
    """Drop flashinfer's process-wide cached MoE runner objects.

    `MoERunner.runner_dict` is keyed by dtype/feature tuple and holds the C++
    runner returned by `module.init(...)`.  The profiling path
    (`run_gemm_profile`) allocates inside that C++ object, so the only way to
    return the memory is to drop the last Python reference to the runner.
    """
    if not os.environ.get("SGLANG_N129_DROP_MOE_RUNNERS"):
        return
    import gc

    n_cls = 0
    n_entries = 0
    n_skipped = 0
    try:
        gc.collect()
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, type) or obj.__name__ != "MoERunner":
                    continue
                d = obj.__dict__.get("runner_dict")
            except Exception:
                # a dead weak proxy, or any object that raises on attribute
                # access, must not abandon the sweep
                n_skipped += 1
                continue
            if isinstance(d, dict) and d:
                n_entries += len(d)
                d.clear()
                n_cls += 1
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            "N129DROP %s cleared MoERunner classes=%d runner entries=%d "
            "skipped_objects=%d",
            tag, n_cls, n_entries, n_skipped,
        )
        _n129_mem_probe(tag + "-after-drop")
    except Exception as exc:
        logger.info("N129DROP %s failed: %r", tag, exc)


def _n129_history_start() -> None:
    if not os.environ.get("SGLANG_N129_MEM_HISTORY"):
        return
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=400000
        )
        logger.info("N129HIST recording started")
    except Exception as exc:
        logger.info("N129HIST start failed: %r", exc)


def _n129_history_report(tag: str, top: int = 25) -> None:
    """Aggregate every STILL-ACTIVE allocator block by its allocation frame."""
    if not os.environ.get("SGLANG_N129_MEM_HISTORY"):
        return
    import collections

    try:
        snap = torch.cuda.memory_snapshot()
        by_site = collections.Counter()
        n_site = collections.Counter()
        active_total = 0
        for seg in snap:
            for blk in seg.get("blocks", ()):
                if blk.get("state") != "active_allocated":
                    continue
                sz = blk.get("size", 0)
                active_total += sz
                frames = blk.get("frames") or []
                sig = []
                for fr in frames:
                    fn = fr.get("filename", "?") or "?"
                    nm = fr.get("name", "?") or "?"
                    if fn.startswith("??") or fn == "?" or fn.endswith(".cpp"):
                        continue
                    if ("CachingAllocator" in nm or "unwind" in nm
                            or "CapturedTraceback" in nm or nm.startswith("c10::")
                            or nm.startswith("at::detail::")):
                        continue
                    for cut in ("site-packages/", "python/sglang/", "/torch/"):
                        if cut in fn:
                            fn = fn.split(cut)[-1]
                            break
                    sig.append(f"{fn}:{fr.get('line','?')}:{nm}")
                    if len(sig) >= 5:
                        break
                key = " <- ".join(sig) if sig else "<no-python-frames>"
                by_site[key] += sz
                n_site[key] += 1
        logger.info(
            "N129HIST %s active_blocks=%d active_total=%.4f GiB sites=%d",
            tag, sum(n_site.values()), active_total / 2**30, len(by_site),
        )
        for key, nb in by_site.most_common(top):
            logger.info(
                "N129HIST %s %.6f %d %s", tag, nb / 2**30, n_site[key], key
            )
    except Exception as exc:
        logger.info("N129HIST %s report failed: %r", tag, exc)


def _n129_dump_cuda_tensors(tag: str, top: int = 20) -> None:
    """Per-shape-signature deduped CUDA storage totals + owners for the top ones."""
    if not os.environ.get("SGLANG_N129_MEM_DUMP"):
        return
    import collections
    import gc

    def _describe(obj, depth=0):
        rn = type(obj).__name__
        mod = getattr(type(obj), "__module__", "")
        if rn in ("dict", "list", "tuple", "cell") and depth < 2:
            inner = set()
            for r in gc.get_referrers(obj):
                if r is obj:
                    continue
                inner.add(_describe(r, depth + 1))
                if len(inner) >= 3:
                    break
            return f"{rn}<-{'|'.join(sorted(inner))}"
        if rn == "function":
            return f"fn:{getattr(obj, '__qualname__', '?')}"
        if rn == "module":
            return f"mod:{getattr(obj, '__name__', '?')}"
        return f"{mod}.{rn}" if mod else rn

    try:
        stores = {}  # ptr -> [nbytes, signature, representative tensor]
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor) or not obj.is_cuda:
                    continue
                st = obj.untyped_storage()
                ptr, nb = st.data_ptr(), st.nbytes()
            except Exception:
                continue
            sig = f"{tuple(obj.shape)}x{obj.dtype}"
            e = stores.get(ptr)
            if e is None:
                stores[ptr] = [nb, sig, obj]
            elif sig < e[1]:  # deterministic representative signature
                e[1], e[2] = sig, obj

        by_sig_bytes = collections.Counter()
        by_sig_n = collections.Counter()
        rep = {}
        for ptr, (nb, sig, obj) in stores.items():
            by_sig_bytes[sig] += nb
            by_sig_n[sig] += 1
            rep.setdefault(sig, obj)

        total = sum(v[0] for v in stores.values())
        logger.info(
            "N129SIG %s TOTAL storages=%d dedup_total=%.4f GiB signatures=%d",
            tag, len(stores), total / 2**30, len(by_sig_bytes),
        )
        ordered = by_sig_bytes.most_common()
        for sig, nb in ordered:
            if nb < 16 * 2**20:
                break
            logger.info(
                "N129SIG %s %.6f %d %s", tag, nb / 2**30, by_sig_n[sig], sig
            )
        for sig, nb in ordered[:top]:
            owners = collections.Counter()
            try:
                for r in gc.get_referrers(rep[sig]):
                    owners[_describe(r)] += 1
            except Exception as exc:
                owners[f"walk-failed:{exc!r}"] += 1
            logger.info("N129OWN %s %.6f %s :: %s", tag, nb / 2**30, sig, dict(owners))
    except Exception as exc:
        logger.info("N129SIG %s failed: %r", tag, exc)


def _n129_deep_release(tag: str) -> None:
    """Second-stage release: Python garbage, then the allocator arena again.

    Separates "a Python reference was still holding it" from "PyTorch never owned
    it".  Gated by the same env var; a no-op when the probe is off.
    """
    if not os.environ.get(_N129_PROBE_ENV):
        return
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    _n129_mem_probe(tag)
    _n129_dump_cuda_tensors(tag)
    _n129_history_report(tag)
# --- end fn:N129 memory probe ------------------------------------------------


def get_flashinfer_autotune_skip_ops(model_runner: ModelRunner) -> set[str]:
    skip_ops = set(get_exec().kernel.flashinfer_autotune_skip_ops or ())
    skip_ops.update(FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS)
    return skip_ops


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    mr = model_runner
    if mr.device != "cuda":
        return False
    if get_exec().kernel.disable_flashinfer_autotune:
        return False
    if get_exec().deterministic.enable_deterministic_inference:
        # Tuned configs are per problem shape, so the reduction order would follow
        # the batch shape.
        return False

    if for_speculative_draft:
        backend_str = (
            get_spec().speculative_moe_runner_backend
            or get_exec().moe.moe_runner_backend
        )
        a2a_backend_str = (
            get_spec().speculative_moe_a2a_backend or get_exec().moe.moe_a2a_backend
        )
    else:
        backend_str = get_exec().moe.moe_runner_backend
        a2a_backend_str = get_exec().moe.moe_a2a_backend

    # Autotune can run before the MoE backend globals are initialized, so read
    # the configured backends -- the draft leaves (`get_spec()`) or the target
    # leaves (`get_exec().moe`) above. CuteDSL v1 bypasses MoeRunner, and its
    # dummy dispatch can exceed DeepEP low-latency's token limit.
    if backend_str == "flashinfer_cutedsl" and a2a_backend_str == "deepep":
        return False

    # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

    moe_needs_autotune = backend_str in [
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
    ]

    from sglang.srt.layers.quantization.fp4_utils import (
        get_fp4_gemm_runner_backend,
    )

    model_quantization = mr.model_config.quantization
    model_uses_fp4 = model_quantization in (
        "modelopt_fp4",
        "modelopt_mixed",
    )
    fp4_gemm_needs_autotune = model_uses_fp4 and (
        get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
        or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
    )

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_per_tensor_fp8_supported,
        resolve_mxfp8_dense_gemm_backend,
    )

    if model_quantization == "mxfp8":
        fp8_gemm_needs_autotune = resolve_mxfp8_dense_gemm_backend().is_flashinfer()
    elif model_quantization in ("modelopt", "modelopt_fp8", "modelopt_mixed"):
        fp8_gemm_needs_autotune = flashinfer_per_tensor_fp8_supported()
    else:
        fp8_gemm_needs_autotune = False

    if not (moe_needs_autotune or fp4_gemm_needs_autotune or fp8_gemm_needs_autotune):
        return False

    if torch.cuda.get_device_capability()[0] < 9:
        return False

    if mr.spec_algorithm.is_speculative():
        return mr.is_draft_worker if for_speculative_draft else not mr.is_draft_worker

    return True


def flashinfer_autotune_cache_path(model_runner: ModelRunner) -> Path:
    import flashinfer

    mr = model_runner
    major, minor = torch.cuda.get_device_capability(mr.device)
    arch = f"sm{major}{minor}"
    flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    model_key_parts = [
        str(get_model().model_path),
        str(mr.dtype),
        str(get_model().quantization),
        str(get_exec().moe.moe_runner_backend),
        str(mr.ps.tp_size),
        str(mr.ps.pp_size),
        str(mr.ps.attn_dp_size),
        str(mr.ps.moe_ep_size),
        str(mr.model_config.hf_config.__class__.__name__),
    ]
    # A different skip policy must not reuse previously tuned tactics.
    skip_ops = get_flashinfer_autotune_skip_ops(mr)
    model_key_parts.append("skip_ops=" + ",".join(sorted(skip_ops)))
    if mr.is_draft_worker:
        model_key_parts.append(f"draft_quant={mr.model_config.quantization}")
    model_key = "|".join(model_key_parts)
    cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
    cache_dir = (
        Path(envs.SGLANG_CACHE_DIR.get())
        / "flashinfer"
        / "autotune"
        / flashinfer_version
        / arch
        / cache_key
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir
        / f"rank_tp{mr.ps.tp_rank}_pp{mr.ps.pp_rank}_dp{mr.ps.dp_rank or 0}.json"
    )


def _autotune_tactic_sync_group(
    tp_group: GroupCoordinator,
) -> Optional[torch.distributed.ProcessGroup]:
    """CPU group over the ranks that must agree on the tuned tactics.

    Per-rank timing noise alone makes each rank's ``argmin`` pick a different
    tactic for the same shape. FlashInfer all-reduces the timings over this
    group so every rank minimizes over the same numbers. TP is the scope: those
    ranks run the same dummy forward, and PP stages are already separate groups.
    """
    if tp_group.world_size <= 1:
        return None
    # The CPU group keeps the reduction of these scalars off the profiled stream.
    return tp_group.cpu_group


@contextlib.contextmanager
def _autotune_process_group(group: Optional[torch.distributed.ProcessGroup]):
    """Set FlashInfer's timing-reduction group, restoring the previous one after."""
    from flashinfer.autotuner import (
        get_autotune_process_group,
        set_autotune_process_group,
    )

    previous = get_autotune_process_group()
    set_autotune_process_group(group)
    try:
        yield
    finally:
        set_autotune_process_group(previous)


def _autotune_cache_digest(cache_path: Path, env: dict[str, str]) -> str:
    """Hash of what this rank would load from ``cache_path`` ("" for nothing).

    Includes the environment: ``load_configs`` ignores the whole file when its
    ``_metadata`` stamp disagrees with the environment reading it, so equal
    tactics alone do not mean two ranks load the same thing.
    """
    if not cache_path.is_file():
        return ""
    try:
        configs = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        return ""
    if not isinstance(configs, dict):
        return ""
    payload = {"file": configs, "env": env}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _drop_diverged_autotune_cache(
    cache_path: Path, group: torch.distributed.ProcessGroup, env: dict[str, str]
) -> None:
    """Enter tuning with the same cache on every rank, or with none at all.

    A cache hit skips a profile, so caches that disagree desync the reduction.
    """
    digests: list[str] = [""] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(
        digests, _autotune_cache_digest(cache_path, env), group=group
    )
    if len(set(digests)) == 1:
        return
    log_info_on_rank0(
        logger,
        "FlashInfer autotune: per-rank caches disagree, discarding them and "
        "tuning from scratch so all ranks agree on the tactics.",
    )
    cache_path.unlink(missing_ok=True)


@contextlib.contextmanager
def flashinfer_autotune_context(model_runner: ModelRunner, *, run_lm_head: bool):
    # The gate below decides on the same inputs load_configs does.
    from flashinfer.autotuner import _collect_metadata, autotune

    mr = model_runner
    cache_path = flashinfer_autotune_cache_path(mr)
    sync_group = _autotune_tactic_sync_group(mr.tp_group)
    if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
        autotune_cache = cache_path
        if sync_group is not None:
            _drop_diverged_autotune_cache(cache_path, sync_group, _collect_metadata())
        logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = cache_path.parent / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        autotune_cache = runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
        logger.info(
            "Running FlashInfer autotune (cache reuse DISABLED via "
            "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
            autotune_cache,
        )

    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
    mr.forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(mr.device).stream(mr.forward_stream):
        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        skip_ops = get_flashinfer_autotune_skip_ops(mr)
        with (
            _autotune_process_group(sync_group),
            autotune(
                True,
                cache=str(autotune_cache),
                skip_ops=skip_ops,
            ),
            autotune_dummy_run_mode(run_lm_head=run_lm_head),
        ):
            yield
    torch.cuda.current_stream().wait_stream(mr.forward_stream)
    logger.info("FlashInfer autotune completed.")


def run_flashinfer_autotune_forward(
    model_runner: ModelRunner, forward_fn: Callable[[], None], *, run_lm_head: bool
) -> None:
    """Run flashinfer autotune forward."""
    with flashinfer_autotune_context(model_runner, run_lm_head=run_lm_head):
        forward_fn()


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    run_lm_head: bool = True,
) -> None:
    """Run speculative draft flashinfer autotune."""
    mr = runner.model_runner
    phase_key = f"{runner.__class__.__module__}.{runner.__class__.__qualname__}"
    tuned_phases = getattr(mr, "_flashinfer_spec_draft_autotuned_phases", None)
    if tuned_phases is None:
        tuned_phases = set()
        mr._flashinfer_spec_draft_autotuned_phases = tuned_phases
    if phase_key in tuned_phases:
        return
    if (
        not mr.spec_algorithm.is_speculative()
        or not mr.is_draft_worker
        or not should_run_flashinfer_autotune(mr, for_speculative_draft=True)
    ):
        return

    def run_and_reset():
        forward_fn()
        if post_warmup_hook is not None:
            post_warmup_hook()

    run_flashinfer_autotune_forward(mr, run_and_reset, run_lm_head=run_lm_head)
    tuned_phases.add(phase_key)


def mm_dummy_extend_is_safe(model_runner: ModelRunner) -> bool:
    """Whether a MULTIMODAL model's EXTEND prefill tolerates `mm_inputs=None`.

    `model_config.is_multimodal` is a **config** property: it is True when the
    architecture is registered multimodal, or merely when the checkpoint's
    config carries a `vision_config` subconfig. The hazard the caller guards
    against is a **code** property: several multimodal prefill paths iterate
    `forward_batch.mm_inputs` unconditionally (`mllama`, `moss_vl`,
    `gemma3_mm`, `gemma4_mm`) and raise on the autotune dummy's `None`.

    Models whose multimodal entry point is `general_mm_embed_routine` do not:
    that routine's mm branch is gated on `ForwardBatch.contains_mm_inputs()`,
    which short-circuits to False when `mm_inputs is None`, so the dummy takes
    the plain text-only extend path.

    Keying the skip on the config property therefore denies the extend
    autotune to every such model, including one whose language model is the
    only thing a deployment ever prefills. A model class opts back in by
    setting `mm_dummy_extend_safe = True`; the default is the historical skip,
    so an un-annotated multimodal model is unaffected by this function.
    """
    if not envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND_MULTIMODAL.get():
        return False
    return bool(getattr(getattr(model_runner, "model", None), "mm_dummy_extend_safe", False))


def maybe_flashinfer_autotune_extend(
    runner: BaseRunner, *, decode_num_tokens: int
) -> None:
    """Also autotune one EXTEND-shaped dummy forward.

    The decode-shaped autotune only covers token counts up to the decode
    batch size, so larger prefill/extend batches fall outside the tuned
    buckets and run flashinfer's default heuristic — which can be far
    slower than the tuned tactic (e.g. trtllm-gen fp4 MoE is ~30% slower
    untuned at >=8k tokens on sm100). One extra forward at the largest
    per-rank extend token count tunes all buckets up to it.
    """
    if not envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        return
    mr = runner.model_runner
    # Prefer the per-rank scheduler buffer while preserving the legacy ceiling
    # when chunked prefill is disabled.
    num_tokens = max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens
    if num_tokens <= (decode_num_tokens or 0):
        return  # decode-shaped autotune already covered these buckets
    is_pd_prefill_target = (
        get_disagg().disaggregation_mode == "prefill" and not mr.is_draft_worker
    )
    if not mr.is_generation or (
        mr.spec_algorithm.is_speculative() and not is_pd_prefill_target
    ):
        # Ordinary speculative runners force TARGET_VERIFY; PD prefill targets
        # have no draft-side state and preserve the requested EXTEND mode.
        return
    if mr.model_config.is_multimodal and not mm_dummy_extend_is_safe(mr):
        # LOCAL MODIFICATION (gaema): the dummy runs mm_inputs=None, which
        # multimodal prefill paths iterate. A model class whose prefill is
        # None-safe opts back in -- see mm_dummy_extend_is_safe() for why the
        # config property is the wrong key.
        return
    # Multimodal generation wrappers can still run this text-only EXTEND dummy;
    # an incompatible model should fail the explicit opt-in visibly.

    if mr.attn_backend.extend_dummy_seqs_capped_by_req_pool:
        pool_size = mr.req_to_token_pool.size
        num_tokens_per_req = (num_tokens + pool_size - 1) // pool_size
    else:
        # Packed dummies tune measurably worse tactics for the same token
        # bucket, so pack only where the backend would otherwise crash. None
        # (not 1) keeps the backend's own seq_len_fill_value in _dummy_run.
        num_tokens_per_req = None
    per_req = num_tokens_per_req or 1
    batch_size = (num_tokens + per_req - 1) // per_req
    num_tokens = batch_size * per_req

    _n129_mem_probe("P1-extend-pre-alloc")
    buffers = runner._alloc_dummy_decode_buffers(
        batch_size,
        num_tokens_per_req=per_req,
        allocate_logits_buffer=False,
    )
    _n129_mem_probe("P2-extend-post-alloc")
    canary_run_ctx = (
        c.with_active_single_forward_manager(0)
        if (c := mr.canary_manager) is not None
        else empty_context()
    )

    forward_fn = functools.partial(
        runner._dummy_run,
        batch_size=batch_size,
        buffers=buffers,
        run_ctx=canary_run_ctx,
        forward_mode_override=ForwardMode.EXTEND,
        extend_num_tokens_per_req=num_tokens_per_req,
    )

    log_info_on_rank0(
        logger,
        f"FlashInfer autotune: extra EXTEND pass at {num_tokens} tokens "
        f"({batch_size} seqs x {per_req} tokens).",
    )
    try:
        run_flashinfer_autotune_forward(mr, forward_fn, run_lm_head=False)
    except (MemoryError, torch.OutOfMemoryError) as exc:
        if _autotune_tactic_sync_group(mr.tp_group) is not None:
            # Tuning is collective: this rank has stopped reducing while its
            # peers wait on the next tactic, so skipping the pass would hang
            # them. Fail instead of degrading alone.
            raise
        # The pass is an optimization; without headroom for the extend-shaped
        # forward, fall back to untuned extend buckets instead of failing.
        #
        # 🔴 `torch.OutOfMemoryError` ALONE DOES NOT CATCH THIS PATH, so the
        # documented fallback could not fire and the "optimization" killed the
        # server instead of degrading.  Measured on ai100 (sm_120a, TP2,
        # Qwen3.8-Flash-Next-NVFP4, mem-fraction-static 0.94, census clean):
        # the allocation that fails is inside flashinfer's own MoE runner and
        # comes back through TVM-FFI, so it surfaces as a plain
        #   MemoryError: CUDA out of memory. Tried to allocate 1.64 GiB ...
        #     File "flashinfer/fused_moe/core.py", line 636, in cutlass_fused_moe
        #     File "<unknown>", line 0, in TVMFFIEnvTensorAlloc
        # on BOTH ranks -- not a torch.OutOfMemoryError.  `MemoryError` is the
        # 🔴 CORRECTED fn:N132 -- that sentence was FALSE and it cost a server.
        # `torch.OutOfMemoryError` derives from RuntimeError, NOT MemoryError:
        #   mro = (OutOfMemoryError, RuntimeError, Exception, BaseException)
        #   issubclass(torch.OutOfMemoryError, MemoryError) -> False
        # on torch 2.13.0+cu130.  So `except MemoryError` did not widen the
        # original guard, it MOVED it, and the two OOM sites in this pass raise
        # DIFFERENT classes:
        #   mfs 0.94 -> flashinfer MoE runner, via TVM-FFI  -> MemoryError
        #   mfs 0.92 -> the model's own PLE short-conv       -> torch.OutOfMemoryError
        #               (qwen4_exp.py:1031, torch.cat 360 MiB)
        # The second escaped to Scheduler.__init__ and KILLED THE SERVER AT
        # STARTUP, while 0.91 and 0.94 both degraded -- i.e. the failure was
        # NON-MONOTONE in mfs and a recipe interpolated between two working
        # points was not safe.  Catch both; the recovered state is the one the
        # 0.94 cell already reaches on its own fallback (15/15 arms, all of
        # which served), because the `finally` below is what cleans up and it
        # already ran on the escaping path.
        log_info_on_rank0(
            logger,
            "FlashInfer extend autotune skipped: not enough free memory "
            f"for a {num_tokens}-token dummy forward ({type(exc).__name__}).",
        )
    finally:
        _n129_mem_probe("P3-extend-post-pass")
        # release dummy buffers before capture measures free memory
        del forward_fn, buffers
        torch.cuda.empty_cache()
        _n129_mem_probe("P4-extend-post-empty-cache")
        _n129_deep_release("P5-extend-post-gc-empty-cache")
        _n129_drop_moe_runners("P5b-extend")
