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
"""Logits processing."""

import dataclasses
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.kernels.ops.activation.softcap import (
    softcap_inplace_logits as fused_softcap,
)
from sglang.srt.beam_search.logits_capture import BeamLogitsCapture
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStates,
    pack_aux_hidden_states,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    attn_tp_all_gather,
    attn_tp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_scatter,
    get_dp_device,
    get_dp_dtype,
    get_dp_hidden_size,
)
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    LogprobStage,
    get_token_ids_logprobs_raw,
    get_top_logprobs_raw,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.sampling.sampling_observer import DeviceAuxiliaryOutput
from sglang.srt.utils.common import (
    is_cpu,
    is_npu,
    is_pin_memory_available,
    use_intel_amx_backend,
)

logger = logging.getLogger(__name__)

_is_npu = is_npu()
_is_cpu = is_cpu()

_UNQUANTIZED_LM_HEAD_METHODS = {
    "UnquantizedEmbeddingMethod",
    "UnquantizedLinearMethod",
    "PackWeightMethod",
}

# None outside a FlashInfer autotune pass; inside one, whether that pass runs the
# LM head. Not-None means the forward's output is discarded -- attention backends
# read that via get_in_autotune_dummy_run() to skip a cross-node exchange.
# Skipping the LM head skips its [batch * dp_size, vocab] all-gather, which OOMs
# under DP attention with a tight mem_fraction_static.
_autotune_run_lm_head: Optional[bool] = None


def _trace_e2e_logits(stage: str, **fields) -> None:
    if not envs.SGLANG_TRACE_LOGITS_E2E.get():
        return
    try:
        parallel = get_parallel()
        rank = f"dp={parallel.attn_dp_rank} tp={parallel.tp_rank}"
    except Exception:
        rank = "rank=unknown"
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"SGLANG_TRACE_LOGITS_E2E {rank} stage={stage} {details}", flush=True)


def _has_lm_head_runtime_attrs(lm_head, attr_names: Tuple[str, ...]) -> bool:
    return all(hasattr(lm_head, attr_name) for attr_name in attr_names)


def should_apply_lm_head_quant_method(lm_head, quant_method) -> bool:
    if (
        quant_method is None
        or not hasattr(lm_head, "weight")
        or not callable(getattr(quant_method, "apply", None))
    ):
        return False

    method_name = type(quant_method).__name__
    if method_name in _UNQUANTIZED_LM_HEAD_METHODS:
        return False

    # A shared target lm_head can retain the draft's stale ModelOpt method; use it
    # only when the runtime tensor layout matches that method.
    if method_name == "ModelOptFp4LinearMethod":
        if lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        ):
            return True
        return lm_head.weight.dtype == torch.uint8 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale_interleaved",
                "alpha",
                "input_scale_inv",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptNvFp4A16LinearMethod":
        return lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptFp8LinearMethod":
        return (
            lm_head.weight.dtype == torch.float8_e4m3fn
            and _has_lm_head_runtime_attrs(lm_head, ("weight_scale", "input_scale"))
        )

    return True


# FlashInfer autotune skips the unprofiled LM-head all-gather; its
# [batch * dp_size, vocab] output can OOM under tight DP-attention memory.
_in_autotune_dummy_run = False


def get_in_autotune_dummy_run() -> bool:
    return _autotune_run_lm_head is not None


@contextmanager
def autotune_dummy_run_mode(*, run_lm_head: bool):
    global _autotune_run_lm_head
    _autotune_run_lm_head = run_lm_head
    try:
        yield
    finally:
        _autotune_run_lm_head = None


@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    # Can be None for certain prefill-only requests (e.g., multi-item scoring) that don't need next token generation
    next_token_logits: Optional[torch.Tensor]
    # Used by speculative decoding (EAGLE)
    # The last hidden layers
    hidden_states: Optional[torch.Tensor] = None

    ## Part 2: This part will be assigned in python/sglang/srt/layers/sampler.py::Sampler
    # he log probs of output tokens, if SGLANG_RETURN_ORIGINAL_LOGPROB = True, will get the log probs before applying temperature. If False, will get the log probs before applying temperature.
    next_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    # Can contain either lists or GPU tensors (for delayed copy optimization in prefill-only requests)
    next_token_token_ids_logprobs_val: Optional[
        List[Union[List[float], torch.Tensor]]
    ] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None
    # Sparse top-k/top-p/min-p support ids and selected-token logprob after
    # truncation/renormalization. Only populated when requested.
    next_token_sampling_mask_idx: Optional[List[Optional[List[int]]]] = None
    next_token_sampling_logprobs: Optional[List[Optional[float]]] = None

    ## Part 3: Prefill-only. This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logprobs of input tokens.        shape: [#token]
    input_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    # The logprobs and ids of the requested token ids in input positions. shape: [#seq, n] (n is the number of requested token ids)
    # Can contain either lists or GPU tensors (for delayed GPU-to-CPU transfer optimization)
    input_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    input_token_ids_logprobs_idx: Optional[List] = None

    ## Part 4: Diffusion LLM only.
    full_logits: Optional[torch.Tensor] = None

    # Beam search only: raw pre-sample logits for the scheduler-side joint
    # selection; see beam_search.logits_capture.
    beam: Optional[BeamLogitsCapture] = None

    ## Part 5: Customized Info
    customized_info: Optional[Dict[str, List[Any]]] = None

    ## Part 6: Temporary variables
    # FIXME: These fields are not logits-related but are passed through here as a
    # workaround since ForwardBatch is local to forward_batch_generation().
    # They should be moved to GenerationBatchResult to keep this class clean.
    mm_input_embeds: Optional[torch.Tensor] = None

    # Scheduler-local output copied alongside the ordinary generation result.
    auxiliary_device_output: Optional[DeviceAuxiliaryOutput] = None

    ## Part 7: fn:N94 LOCAL MODIFICATION (gaema).  GRAPH-OUTPUT SHARD EXPORT.
    # The rank-local, PRE-all-gather lm_head shard, shape [#seq, vocab/tp].
    # It is the output of the lm_head GEMM that the captured decode graph
    # already computes and today discards after the collective; exporting it
    # as a graph output lets a POST-graph all_gather rebuild the exact dense
    # [#seq, vocab] rows for requests the shard-top-k -1e30 floor is not exact
    # for (sampling, logprobs, penalties, logit_bias, grammar).
    #
    # Populated ONLY when SGLANG_LOGITS_SHARD_TOPK=<k> AND
    # SGLANG_LOGITS_SHARD_EXPORT=1 and the batch is a decode batch whose rows
    # map 1:1 onto next_token_logits.  None on every other path, and never
    # read unless the export mode is on.
    local_vocab_shard: Optional[torch.Tensor] = None


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    next_token_logits_buffer: Optional[torch.Tensor] = None

    extend_return_logprob: bool = False
    extend_return_top_logprob: bool = False
    extend_token_ids_logprob: bool = False
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None
    top_logprobs_nums: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # logits and logprobs post processing
    temperature: torch.Tensor = None
    top_p: torch.Tensor = None

    # DP attention metadata. Not needed when DP attention is not used.
    # Number of tokens in the request.
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    # The start position of local hidden states.
    dp_local_start_pos: Optional[torch.Tensor] = None
    dp_local_num_tokens: Optional[torch.Tensor] = None
    global_dp_buffer_len: Optional[int] = None
    # Number of tokens to sample per DP rank
    global_num_tokens_for_logprob_cpu: Optional[torch.Tensor] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None
    # The gather mode for DP attention
    dp_padding_mode: Optional[DpPaddingMode] = None

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # Carried from ForwardBatch so logits pruning can reconstruct the SP gather.
    attn_tp_sequence_sharded: bool = False

    mm_input_embeds: Optional[torch.Tensor] = None

    # DRAFT_EXTEND_V2: when set, lm_head and LAST hidden capture use only these
    # rows (see EagleDraftExtendInput.select_index).
    draft_extend_select_index: Optional[torch.Tensor] = None

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.return_logprob
            and not forward_batch.forward_mode.is_target_verify()
        ):
            extend_return_top_logprob = any(
                x > 0 for x in forward_batch.top_logprobs_nums
            )
            extend_token_ids_logprob = any(
                x is not None for x in forward_batch.token_ids_logprobs
            )
            extend_return_logprob = False
            extend_logprob_pruned_lens_cpu = []
            for extend_len, start_len in zip(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            ):
                if extend_len - start_len > 0:
                    extend_return_logprob = True
                extend_logprob_pruned_lens_cpu.append(extend_len - start_len)
        else:
            extend_return_logprob = extend_return_top_logprob = (
                extend_token_ids_logprob
            ) = extend_logprob_pruned_lens_cpu = False

        if forward_batch.forward_mode.is_draft_extend_v2():
            draft_extend_select_index = forward_batch.spec_info.select_index
        else:
            draft_extend_select_index = None

        return cls(
            forward_mode=forward_batch.forward_mode,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            next_token_logits_buffer=forward_batch.next_token_logits_buffer,
            extend_return_logprob=extend_return_logprob,
            extend_return_top_logprob=extend_return_top_logprob,
            extend_token_ids_logprob=extend_token_ids_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            token_ids_logprobs=forward_batch.token_ids_logprobs,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            is_prefill_only=forward_batch.is_prefill_only,
            attn_tp_sequence_sharded=forward_batch.attn_tp_sequence_sharded,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            dp_local_start_pos=forward_batch.dp_local_start_pos,
            dp_local_num_tokens=forward_batch.dp_local_num_tokens,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            mm_input_embeds=forward_batch.mm_input_embeds,
            draft_extend_select_index=draft_extend_select_index,
        )

    def compute_dp_attention_metadata(self):
        cumtokens = torch.cumsum(self.global_num_tokens_for_logprob_gpu, dim=0)
        dp_rank = get_parallel().attn_dp_rank
        if dp_rank == 0:
            dp_local_start_pos = torch.zeros_like(
                self.global_num_tokens_for_logprob_gpu[0]
            )
        else:
            dp_local_start_pos = cumtokens[dp_rank - 1]

        self.dp_local_start_pos = dp_local_start_pos
        self.dp_local_num_tokens = self.global_num_tokens_for_logprob_gpu[dp_rank]

        hidden_size = get_dp_hidden_size()
        dtype = get_dp_dtype()
        device = get_dp_device()

        if self.global_num_tokens_for_logprob_cpu is not None:
            # create a smaller buffer to reduce peak memory usage
            self.global_dp_buffer_len = sum(self.global_num_tokens_for_logprob_cpu)
        else:
            self.global_dp_buffer_len = self.global_dp_buffer_len

        self.gathered_buffer = torch.empty(
            (
                self.global_dp_buffer_len,
                hidden_size,
            ),
            dtype=dtype,
            device=device,
        )


class LogitsProcessor(nn.Module):
    def __init__(
        self,
        config,
        skip_all_gather: bool = False,
        logit_scale: Optional[float] = None,
        return_full_logits: bool = False,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.logit_scale = logit_scale
        self.use_attn_tp_group = get_parallel().enable_dp_lm_head
        self.use_tp_lm_head_all_to_all = get_parallel().enable_tp_lm_head_all_to_all
        self.use_fp32_lm_head = get_exec().features.enable_fp32_lm_head or getattr(
            config, "enable_lm_head_fp32", False
        )
        if self.use_attn_tp_group:
            self.attn_tp_size = get_parallel().attn_tp_size
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and self.attn_tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = False
        else:
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and get_parallel().tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = (
                self.do_tensor_parallel_all_gather and get_parallel().attn_dp_size != 1
            )
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        self.return_full_logits = return_full_logits
        self.enable_mis = get_exec().features.enable_mis
        self.rl_on_policy_target = get_exec().deterministic.rl_on_policy_target

        self._logits_gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=triton_symm_mem_ag.recommended_max_tokens(
                include_prefill=False, floor=128
            ),
            enabled=self.do_tensor_parallel_all_gather and not self.use_attn_tp_group,
            skip_entry_sync=True,
        )

        # --- fn:N89 LOCAL MODIFICATION (gaema) ------------------------------
        # Opt-in shard-top-k logits gather, replacing the vocab-parallel
        # lm_head logits AllGather ([B, V/tp] -> [B, V], 52.4 MB/rank/step at
        # the c210 cell) with a per-shard top-k plus a [B, 2k] gather that is
        # scattered back into a -1e30-filled [B, V].  The [B, V] shape
        # contract is preserved, so nothing downstream of this module changes.
        #
        # EXACT for greedy/argmax: the global argmax is the argmax of some
        # rank's shard, hence a member of that rank's top-k for every k >= 1,
        # and it is carried back with its exact value (bf16 -> fp32 -> bf16).
        # NOT EXACT for temperature / top-p / top-k / min-p sampling, for
        # logprobs, for repetition penalties or for grammar masks: those need
        # mass or ordering from the whole vocabulary, which the -1e30 floor
        # destroys.  Hence: env-gated, default OFF, greedy-only.
        #
        #   SGLANG_LOGITS_SHARD_TOPK=<k>   0 or unset = upstream path
        self.shard_topk_k = int(os.environ.get("SGLANG_LOGITS_SHARD_TOPK", "0") or 0)
        self.use_shard_topk_gather = (
            self.shard_topk_k > 0
            and self.do_tensor_parallel_all_gather
            and not self.use_attn_tp_group
            and not self.do_tensor_parallel_all_gather_dp_attn
        )
        if self.shard_topk_k > 0:
            logger.warning(
                "fn:N89 shard-top-k logits gather: k=%d enabled=%s "
                "(greedy/argmax exact; sampling, logprobs, penalties and "
                "grammar masks are NOT exact)",
                self.shard_topk_k,
                self.use_shard_topk_gather,
            )
        # fn:N90 DIAGNOSTIC, default OFF.  With SGLANG_LOGITS_SHARD_TOPK_DEBUG=1
        # the shard-top-k path ALSO computes the upstream dense all-gather and
        # reports every row where the two disagree on argmax, or where the dense
        # top-2 logits are tied or near-tied.  Costs a second all-gather per
        # step: a correctness instrument, never a rate configuration.
        self.shard_topk_debug = os.environ.get(
            "SGLANG_LOGITS_SHARD_TOPK_DEBUG", "0"
        ) not in ("", "0")
        self._n90_step = 0
        if self.shard_topk_debug and self.use_shard_topk_gather:
            logger.warning("fn:N90 shard-top-k DEBUG compare: enabled=True")
        # fn:N92 NULL PERTURBATION, default OFF.  Issues the fn:N89 arm's
        # extra top-k + [B, 2k] all-gather + [B, V] fill + scatter and
        # DISCARDS the result, returning the UPSTREAM DENSE all-gather.  The
        # returned tensor is bit-identical to the control's BY CONSTRUCTION
        # (it is the same call on the same input), so this arm perturbs the
        # process without perturbing the arithmetic.  Used to bisect the
        # per-PROCESS latch fn:N90 located upstream of the logits gather.
        self.shard_topk_null_k = int(
            os.environ.get("SGLANG_LOGITS_SHARD_TOPK_NULL", "0") or 0
        )
        self.use_shard_topk_null = (
            self.shard_topk_null_k > 0
            and not self.use_shard_topk_gather
            and self.do_tensor_parallel_all_gather
            and not self.use_attn_tp_group
            and not self.do_tensor_parallel_all_gather_dp_attn
        )
        if self.shard_topk_null_k > 0:
            logger.warning(
                "fn:N92 NULL perturbation: k=%d enabled=%s "
                "(extra collective issued and DISCARDED; the dense gather "
                "is what is returned, so the output is unchanged)",
                self.shard_topk_null_k,
                self.use_shard_topk_null,
            )
            if self.use_shard_topk_null:
                # _gather_shard_topk_logits reads self.shard_topk_k; give it
                # the null arm's k without enabling the real arm.
                self.shard_topk_k = self.shard_topk_null_k
        # --- end fn:N89 -----------------------------------------------------

        # --- fn:N93 ROUTE PROBE (gaema), default OFF ------------------------
        # Answers ONE question, and it is the question that decides whether the
        # fn:N89 shard-top-k gather can be ROUTED (fast path only for request
        # shapes where it is semantically exact) instead of applied
        # universally: is the Python-level gather branch in _get_logits
        # evaluated once per DECODE STEP, or once per CAPTURED CUDA-graph
        # shape?  A route that reads per-batch state (is_all_greedy,
        # return_logprob, grammar, logit_bias) exists only if that branch runs
        # per step; if it is baked at capture, no per-batch route is reachable
        # from inside this module at all.
        #
        #   SGLANG_LOGITS_ROUTE_PROBE=1
        #
        # The channel is two-sided by construction: capturing=True must appear
        # (graph capture happens at startup) and capturing=False must appear
        # (prefill runs eager), so a probe stuck at either value is visible.
        self.route_probe = os.environ.get("SGLANG_LOGITS_ROUTE_PROBE", "0") not in (
            "",
            "0",
        )
        self._n93_cap = 0
        self._n93_eager = 0
        self._n93_cap_decode = 0
        self._n93_eager_decode = 0
        if self.route_probe:
            logger.warning("fn:N93 route-probe: enabled=True")
        # --- end fn:N93 -----------------------------------------------------

        # --- fn:N94 GRAPH-OUTPUT SHARD EXPORT (gaema), default OFF ----------
        # fn:N93 MEASURED that the gather branch below is a CUDA-graph
        # CAPTURE-TIME constant (<=99 Python executions against >=975 decode
        # steps; 31 captured shapes; LogitsMetadata carries no sampling_info),
        # so no per-batch route is reachable from inside this module.  The
        # route it CAN take is post-graph, and the material it needs is the
        # rank-local [B, vocab/tp] shard the graph already computes.  This mode
        # exports that shard as a graph output; ModelRunner.sample() then
        # all_gathers the rows that need full-vocab exactness and writes them
        # into the floored [B, vocab] the fast path produced.
        #
        #   SGLANG_LOGITS_SHARD_EXPORT=1   (requires SGLANG_LOGITS_SHARD_TOPK)
        #
        # Two invariants make it shippable where the bare fn:N89 arm was not:
        #   1. Sampler.forward and ModelRunner._preprocess_logits are UNTOUCHED
        #      -- they still receive one [B, vocab] fp32 tensor.
        #   2. The reconstructed rows are bitwise the dense path's own values
        #      BY CONSTRUCTION: the same all_gather over the same bf16 shard,
        #      then the same bf16->fp32 widening _copy_logits_to_buffer does.
        #
        # With the export on, the fast path is restricted to DECODE batches:
        # extend/prefill rows feed input-logprob work that reads the whole
        # vocabulary inside this module, upstream of any post-graph repair.
        self.shard_export = os.environ.get("SGLANG_LOGITS_SHARD_EXPORT", "0") not in (
            "",
            "0",
        )
        self.use_shard_export = (
            self.shard_export
            and self.use_shard_topk_gather
            # Softcapping is applied to the [B, vocab] tensor AFTER the gather
            # (see _get_logits), so a post-graph write would bypass it.
            and self.final_logit_softcapping is None
        )
        self._n94_pending_shard: Optional[torch.Tensor] = None
        if self.shard_export and not self.use_shard_export:
            # Asked for the routed build and it cannot be delivered here.  Fail
            # SAFE: turn the fast path off rather than run it unrepaired, which
            # is the one configuration that would be silently wrong.
            self.use_shard_topk_gather = False
        if self.shard_export:
            logger.warning(
                "fn:N94 graph-output shard export: enabled=%s "
                "(shard_topk=%d softcap=%s fastpath=%s) -- decode-only fast "
                "path, post-graph exact reconstruction in ModelRunner.sample()",
                self.use_shard_export,
                self.shard_topk_k,
                self.final_logit_softcapping,
                self.use_shard_topk_gather,
            )
        # --- end fn:N94 -----------------------------------------------------

        self.input_logprob_processor = InputLogprobProcessor()

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        aux_hidden_states: Optional[AuxHiddenStates] = None,
        hidden_states_before_norm: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        # Extract MIS indices before ForwardBatch → LogitsMetadata conversion
        multi_item_delimiter_indices = None
        if isinstance(logits_metadata, ForwardBatch):
            multi_item_delimiter_indices = logits_metadata.multi_item_delimiter_indices
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

        # Autotune dummy run discards this output. `is False` not `not`: None
        # means no autotune pass, which must not skip. Placed before the MIS /
        # DLLM / common dispatch so all three LM-head paths are skipped.
        if _autotune_run_lm_head is False:
            return LogitsProcessorOutput(next_token_logits=None)

        # Under LayerNorm SP the decoder loop leaves these sequence-sharded; undo
        # that before the LM head, which must not participate.
        hidden_states, hidden_states_before_norm = layernorm_sp.maybe_exit_gather(
            hidden_states=hidden_states,
            hidden_states_before_norm=hidden_states_before_norm,
            input_ids=input_ids,
            forward_mode=logits_metadata.forward_mode,
        )

        # Multi-item scoring only for prefill-only requests with pre-computed indices.
        if multi_item_delimiter_indices is not None and logits_metadata.is_prefill_only:
            return self.compute_logprobs_for_multi_item_scoring(
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                multi_item_delimiter_indices,
            )

        # Diffusion LLM only.
        if logits_metadata.forward_mode.is_dllm_extend():
            return self._get_dllm_logits(hidden_states, lm_head, logits_metadata)

        # Get the last hidden states and last logits for the next token prediction
        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        ) = self._get_pruned_states(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            logits_metadata,
        )

        hidden_states_to_store = self._get_hidden_states_to_store(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            logits_metadata,
        )
        del hidden_states

        if not logits_metadata.extend_return_logprob:
            # Compute logits for both input and sampled tokens.
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)
            sampled_logits = (
                logits[sample_indices] if sample_indices is not None else logits
            )

            # fn:N94 (gaema): attach the exported pre-gather shard.  Only when
            # sample_indices is None, i.e. the rows of next_token_logits map
            # 1:1 onto the shard's rows -- true for decode, which is the only
            # mode the export's fast path runs in.
            local_vocab_shard = None
            if self._n94_pending_shard is not None:
                if sample_indices is None:
                    local_vocab_shard = self._n94_pending_shard
                self._n94_pending_shard = None

            # Decode mode or extend mode without return_logprob.
            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
                mm_input_embeds=logits_metadata.mm_input_embeds,
                local_vocab_shard=local_vocab_shard,
            )

        logprobs_result, sampled_logits = self.input_logprob_processor.forward(
            pruned_states=pruned_states,
            sample_indices=sample_indices,
            input_logprob_indices=input_logprob_indices,
            token_to_seq_idx=token_to_seq_idx,
            lm_head=lm_head,
            get_logits_fn=self._get_logits,
            logits_metadata=logits_metadata,
            skip_chunking_for_dp_attn=self.do_tensor_parallel_all_gather_dp_attn,
        )

        # fn:N94 (gaema): this branch is extend-with-logprobs, where the fast
        # path is disabled outright, so there is nothing to export.  Drop any
        # reference defensively so a stale shard can never reach a later step.
        self._n94_pending_shard = None

        logits_output = LogitsProcessorOutput(
            next_token_logits=sampled_logits,
            hidden_states=hidden_states_to_store,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )
        logprobs_result.write_input_to(logits_output)
        return logits_output

    def _get_pruned_states(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[AuxHiddenStates],
        logits_metadata: LogitsMetadata,
    ):
        pruned_states_before_norm: Optional[torch.Tensor] = None
        aux_pruned_states = None
        token_to_seq_idx = []

        if (
            logits_metadata.forward_mode.is_decode_or_idle()
            or logits_metadata.forward_mode.is_target_verify()
            or logits_metadata.forward_mode.is_draft_extend_v2()
        ):
            draft_extend_select_index = logits_metadata.draft_extend_select_index
            if draft_extend_select_index is not None:
                # The draft-extend graph returns LAST hidden states alongside
                # selected logits. Build selected variants for every hidden-state
                # representation; FULL capture below still uses the original
                # unpruned tensors.
                pruned_states = hidden_states[draft_extend_select_index]
                pruned_states_before_norm = (
                    hidden_states_before_norm[draft_extend_select_index]
                    if hidden_states_before_norm is not None
                    else None
                )
            else:
                pruned_states = hidden_states
                pruned_states_before_norm = hidden_states_before_norm
            if aux_hidden_states is not None:
                if draft_extend_select_index is not None:
                    aux_pruned_states = (
                        aux_hidden_states[draft_extend_select_index]
                        if isinstance(aux_hidden_states, torch.Tensor)
                        else [
                            hidden[draft_extend_select_index]
                            for hidden in aux_hidden_states
                        ]
                    )
                else:
                    aux_pruned_states = (
                        aux_hidden_states
                        if isinstance(aux_hidden_states, torch.Tensor)
                        else [hidden for hidden in aux_hidden_states]
                    )
            sample_indices = None
            input_logprob_indices = None

        elif (
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):
            # Prefill without input logprobs.
            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            pruned_states = hidden_states[last_index]
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = hidden_states_before_norm[last_index]
            if aux_hidden_states is not None:
                aux_pruned_states = (
                    aux_hidden_states[last_index]
                    if isinstance(aux_hidden_states, torch.Tensor)
                    else [hidden[last_index] for hidden in aux_hidden_states]
                )
            sample_indices = None
            input_logprob_indices = None
        else:
            # Prefill with input logprobs.
            # Find 4 different indices.
            # 1. pruned_states: hidden states that we want logprobs from.
            # 2. sample_indices: Indices that have sampled tokens.
            # 3. input_logprob_indices: Indices that have input logprob tokens.
            # 4. token_to_seq_idx: map each token to its sequence index
            #
            # Example
            # -------
            # Suppose a batch (flattened by sequence):
            # [t00, t01, t02, t03, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t25]
            # extend_seq_lens_cpu           = [4, 5, 6]
            # extend_logprob_start_lens_cpu = [0, 5, 3]
            #
            # Then, the indices are:
            # pruned_states         -> [t00, t01, t02, t03, t14, t23, t24, t25]
            # sample_indices        -> [3, 4, 7]
            # input_logprob_indices -> [0, 1, 2, 3, 5, 6, 7]
            # token_to_seq_idx      -> [0, 0, 0, 0, 1, 2, 2, 2]
            #
            # If chunk is enabled and chunk_size = 3, the chunks will be computed in a chunked manner:
            # [t00, t01, t02], [t03, t14, t23], [t24, t25]

            sample_index_pt = -1
            sample_indices = []
            input_logprob_indices_pt = 0
            input_logprob_indices = []
            pt, pruned_states_list, pruned_states_before_norm_list = 0, [], []
            is_packed_aux_hidden_states = isinstance(aux_hidden_states, torch.Tensor)
            aux_pruned_states_lists = None
            if aux_hidden_states is not None:
                aux_pruned_states_lists = (
                    []
                    if is_packed_aux_hidden_states
                    else [[] for _ in aux_hidden_states]
                )

            for idx, (extend_logprob_start_len, extend_len) in enumerate(
                zip(
                    logits_metadata.extend_logprob_start_lens_cpu,
                    logits_metadata.extend_seq_lens_cpu,
                )
            ):
                # It can happen in chunked prefill. We still need to sample 1 token,
                # But we don't want to include it in input logprob.
                if extend_len == extend_logprob_start_len:
                    start_len = extend_logprob_start_len - 1
                else:
                    start_len = extend_logprob_start_len

                # We always need at least 1 token to sample because that's required
                # by a caller.
                assert extend_len > start_len
                pruned_states_list.append(
                    hidden_states[pt + start_len : pt + extend_len]
                )
                if hidden_states_before_norm is not None:
                    pruned_states_before_norm_list.append(
                        hidden_states_before_norm[pt + start_len : pt + extend_len]
                    )
                if aux_pruned_states_lists is not None:
                    if is_packed_aux_hidden_states:
                        aux_pruned_states_lists.append(
                            aux_hidden_states[pt + start_len : pt + extend_len]
                        )
                    else:
                        for j, hidden in enumerate(aux_hidden_states):
                            aux_pruned_states_lists[j].append(
                                hidden[pt + start_len : pt + extend_len]
                            )
                # Map each token to its sequence index, for chunked computation
                # of input logprobs
                token_to_seq_idx.extend([idx] * (extend_len - start_len))
                pt += extend_len
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                input_logprob_indices.extend(
                    [
                        input_logprob_indices_pt + i
                        for i in range(extend_len - extend_logprob_start_len)
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len

            pruned_states = torch.cat(pruned_states_list)
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = torch.cat(pruned_states_before_norm_list)
            if aux_pruned_states_lists is not None:
                aux_pruned_states = (
                    torch.cat(aux_pruned_states_lists)
                    if is_packed_aux_hidden_states
                    else [torch.cat(lst) for lst in aux_pruned_states_lists]
                )

            # Build the index tensors via pinned host memory + non-blocking H2D
            # so the small copy doesn't drain the stream.
            sample_indices = torch.tensor(
                sample_indices,
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(pruned_states.device, non_blocking=True)
            input_logprob_indices = torch.tensor(
                input_logprob_indices,
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(pruned_states.device, non_blocking=True)

        return (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        )

    def _get_hidden_states_to_store(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[AuxHiddenStates],
        pruned_states: torch.Tensor,
        pruned_states_before_norm: Optional[torch.Tensor],
        aux_pruned_states: Optional[AuxHiddenStates],
        sample_indices: Optional[torch.Tensor],
        logits_metadata: LogitsMetadata,
    ) -> Optional[torch.Tensor]:
        hidden_states_to_store: Optional[torch.Tensor] = None
        hidden_states_to_store_before_norm: Optional[torch.Tensor] = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                if aux_hidden_states is not None:
                    hidden_states_to_store = pack_aux_hidden_states(aux_hidden_states)
                else:
                    hidden_states_to_store = hidden_states
                hidden_states_to_store_before_norm = hidden_states_before_norm
            elif logits_metadata.capture_hidden_mode.is_last():
                # Get the last token hidden states. If sample_indices is None,
                # pruned states only contain the last tokens already.
                if aux_hidden_states is not None:
                    assert aux_pruned_states is not None
                    aux_pruned_states = pack_aux_hidden_states(aux_pruned_states)
                    hidden_states_to_store = (
                        aux_pruned_states[sample_indices]
                        if sample_indices is not None
                        else aux_pruned_states
                    )
                else:
                    hidden_states_to_store = (
                        pruned_states[sample_indices]
                        if sample_indices is not None
                        else pruned_states
                    )
                    if hidden_states_before_norm is not None:
                        hidden_states_to_store_before_norm = (
                            pruned_states_before_norm[sample_indices]
                            if sample_indices is not None
                            else pruned_states_before_norm
                        )
            else:
                assert False, "Should never reach"

        if hidden_states_to_store_before_norm is not None:
            # NOTE: when hidden_states_before_norm is provided, we always
            # prefer to return it.
            hidden_states_to_store = hidden_states_to_store_before_norm

        return hidden_states_to_store

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
        use_logits_buffer: bool = True,
    ) -> torch.Tensor:
        """Get logits from hidden_states.

        If sampled_logits_only is True, it means hidden_states only contain the
        last position (e.g., extend without input logprobs). The caller should
        guarantee the given hidden_states follow this constraint.
        """
        _trace_e2e_logits(
            "get_logits_enter",
            hidden_shape=tuple(hidden_states.shape),
            dp_gather=self.do_tensor_parallel_all_gather_dp_attn,
            tp_gather=self.do_tensor_parallel_all_gather,
        )
        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
            hidden_states, logits_metadata
        )
        _trace_e2e_logits(
            "dp_hidden_gather_returned",
            global_shape=tuple(hidden_states.shape),
            local_shape=tuple(local_hidden_states.shape),
        )

        if envs.SGLANG_TRACE_LOGITS_E2E_SYNC.get():
            _trace_e2e_logits("pre_lm_head_sync_enter")
            torch.cuda.synchronize()
            _trace_e2e_logits("pre_lm_head_sync_returned")

        _trace_e2e_logits("lm_head_enter", hidden_shape=tuple(hidden_states.shape))
        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)
        _trace_e2e_logits("lm_head_returned", logits_shape=tuple(logits.shape))
        if envs.SGLANG_TRACE_LOGITS_E2E_SYNC.get():
            _trace_e2e_logits("post_lm_head_sync_enter")
            torch.cuda.synchronize()
            _trace_e2e_logits("post_lm_head_sync_returned")

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        used_tp_lm_head_all_to_all = False

        if self.route_probe:
            self._n93_route_probe(logits_metadata)

        if self.do_tensor_parallel_all_gather:
            _trace_e2e_logits(
                "tp_logits_gather_enter", logits_shape=tuple(logits.shape)
            )
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            elif self.use_shard_topk_gather and self._n94_fast_path_allowed(
                logits_metadata
            ):
                # fn:N94: hand the PRE-gather local shard to forward(), which
                # attaches it to the LogitsProcessorOutput.  Inside a capture
                # this reference is what makes the shard a graph OUTPUT: the
                # backend stores the output object per shape, so the tensor
                # stays alive and every replay refreshes it in place.
                if self.use_shard_export:
                    self._n94_pending_shard = logits
                logits = self._gather_shard_topk_logits(logits)
            elif self.use_shard_topk_null:
                # fn:N92: pay the arm's cost, keep the control's answer.
                _ = self._gather_shard_topk_logits(logits)
                logits = self._logits_gatherer(logits)
            elif self._can_use_tp_lm_head_all_to_all(
                logits, local_hidden_states, lm_head, logits_metadata
            ):
                logits = self._tp_lm_head_all_to_all(logits)
                used_tp_lm_head_all_to_all = True
            else:
                logits = self._logits_gatherer(logits)
            _trace_e2e_logits(
                "tp_logits_gather_returned", logits_shape=tuple(logits.shape)
            )

        if not used_tp_lm_head_all_to_all:
            _trace_e2e_logits(
                "dp_logits_scatter_enter", logits_shape=tuple(logits.shape)
            )
            logits = self._scatter_dp_attn_logits(
                logits, local_hidden_states, logits_metadata
            )
            _trace_e2e_logits(
                "dp_logits_scatter_returned", logits_shape=tuple(logits.shape)
            )

        logits = self._copy_logits_to_buffer(
            logits, logits_metadata, use_buffer=use_logits_buffer
        )

        if self.final_logit_softcapping:
            if not (_is_npu or _is_cpu):
                fused_softcap(logits, self.final_logit_softcapping)
            else:
                logits = self.final_logit_softcapping * torch.tanh(
                    logits / self.final_logit_softcapping
                )

        return logits

    def _compute_lm_head(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        quant_method = getattr(lm_head, "quant_method", None)
        if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
            # This is a LoRA-wrapped module, use its forward method
            logits = lm_head(hidden_states)
        elif should_apply_lm_head_quant_method(lm_head, quant_method):
            logits = quant_method.apply(lm_head, hidden_states, embedding_bias)
        elif hasattr(lm_head, "weight"):
            # Normal linear layer
            if self.use_fp32_lm_head:
                # Avoid materializing FP32 copies for same-dtype CUDA FP16/BF16
                # inputs. Retain explicit FP32 casts for unsupported devices or
                # dtype combinations.
                use_mm_out_dtype = (
                    hidden_states.is_cuda
                    and hidden_states.dtype == lm_head.weight.dtype
                    and hidden_states.dtype in (torch.float16, torch.bfloat16)
                )
                if use_mm_out_dtype:
                    logits = torch.mm(
                        hidden_states,
                        lm_head.weight.T,
                        out_dtype=torch.float32,
                    )
                else:
                    logits = torch.matmul(
                        hidden_states.to(torch.float32),
                        lm_head.weight.to(torch.float32).T,
                    )
            elif use_intel_amx_backend(lm_head):
                logits = torch.ops.sgl_kernel.weight_packed_linear(
                    hidden_states.to(lm_head.weight.dtype),
                    lm_head.weight,
                    None,  # bias
                    True,  # is_vnni
                )
            elif self.rl_on_policy_target is not None:
                # Due to tie-weight, we may not be able to change lm_head's weight dtype
                logits = torch.matmul(
                    hidden_states.bfloat16(), lm_head.weight.T.bfloat16()
                )
            else:
                logits = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:
            # GGUF models
            # TODO: use weight_packed_linear for GGUF models
            if self.use_fp32_lm_head:
                with torch.cuda.amp.autocast(enabled=False):
                    logits = lm_head.quant_method.apply(
                        lm_head, hidden_states.to(torch.float32), embedding_bias
                    )
            else:
                logits = lm_head.quant_method.apply(
                    lm_head, hidden_states, embedding_bias
                )
        return logits

    def _gather_dp_attn_hidden_states(
        self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.do_tensor_parallel_all_gather_dp_attn:
            _trace_e2e_logits(
                "dp_metadata_enter",
                local_shape=tuple(hidden_states.shape),
                global_counts_cpu=logits_metadata.global_num_tokens_for_logprob_cpu,
            )
            logits_metadata.compute_dp_attention_metadata()
            _trace_e2e_logits(
                "dp_metadata_returned",
                buffer_shape=tuple(logits_metadata.gathered_buffer.shape),
                local_start=logits_metadata.dp_local_start_pos,
                local_tokens=logits_metadata.dp_local_num_tokens,
            )
            local_hidden_states = hidden_states
            hidden_states = logits_metadata.gathered_buffer
            _trace_e2e_logits(
                "dp_hidden_gather_enter",
                global_shape=tuple(hidden_states.shape),
                local_shape=tuple(local_hidden_states.shape),
            )
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)
            _trace_e2e_logits("dp_hidden_gather_collective_returned")
            return hidden_states, local_hidden_states
        return hidden_states, hidden_states

    def _gather_attn_tp_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.vocab_size % self.attn_tp_size == 0:
            global_logits = torch.empty(
                (
                    self.attn_tp_size,
                    logits.shape[0],
                    self.vocab_size // self.attn_tp_size,
                ),
                device=logits.device,
                dtype=logits.dtype,
            )
            attn_tp_all_gather_into_tensor(global_logits, logits)
            global_logits = global_logits.permute(1, 0, 2).reshape(
                logits.shape[0], self.vocab_size
            )
        else:
            global_logits = torch.empty(
                (self.vocab_size, logits.shape[0]),
                device=logits.device,
                dtype=logits.dtype,
            )
            global_logits = global_logits.T
            attn_tp_all_gather(
                list(global_logits.tensor_split(self.attn_tp_size, dim=-1)),
                logits,
            )
        return global_logits

    def _can_use_tp_lm_head_all_to_all(
        self,
        logits: torch.Tensor,
        local_hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> bool:
        if not self.use_tp_lm_head_all_to_all:
            return False

        tp_size = get_parallel().tp_size
        base_lm_head = getattr(lm_head, "base_layer", lm_head)
        if getattr(base_lm_head, "tp_size", None) != tp_size:
            # Tied embeddings may be replicated across DP ranks (tp_size=1),
            # even though the logits processor runs in a larger global TP
            # group. Such logits are full-vocabulary rather than TP shards and
            # therefore do not satisfy the all-to-all layout contract.
            return False

        # Every participant must make the same collective choice. Decode CUDA
        # graphs omit CPU counts and fill every GPU count with the same padded
        # bucket size. Eager batches carry the same global CPU count list on
        # every rank, so they are also safe when all entries are equal.
        global_counts_cpu = logits_metadata.global_num_tokens_for_logprob_cpu
        is_equal_padded_graph_layout = global_counts_cpu is None and (
            logits_metadata.global_num_tokens_for_logprob_gpu is not None
        )
        is_equal_eager_layout = (
            global_counts_cpu is not None
            and len(global_counts_cpu) == tp_size
            and len(global_counts_cpu) > 0
            and all(count == global_counts_cpu[0] for count in global_counts_cpu)
        )
        if not (is_equal_padded_graph_layout or is_equal_eager_layout):
            return False

        local_rows = local_hidden_states.shape[0]
        return local_rows > 0 and logits.shape[0] == local_rows * tp_size

    def _tp_lm_head_all_to_all(self, logits: torch.Tensor) -> torch.Tensor:
        """Exchange only the row block owned by each destination DP rank."""
        logits = logits.contiguous()
        all_to_all_output = torch.empty_like(logits)
        get_tp_group().all_to_all_single(all_to_all_output.view(-1), logits.view(-1))
        return _reassemble_tp_lm_head_all_to_all_output(
            all_to_all_output, get_parallel().tp_size
        )

    def _n93_route_probe(self, logits_metadata: LogitsMetadata):
        """fn:N93 ROUTE PROBE (gaema).  Inert unless SGLANG_LOGITS_ROUTE_PROBE=1.

        Records, for every Python-level execution of the gather branch,
        whether the current stream is capturing a CUDA graph and whether the
        batch is a decode batch, and reports the running totals.  See __init__
        for what the reading decides.
        """
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except Exception:  # pragma: no cover - non-CUDA devices
            capturing = False
        is_decode = bool(logits_metadata.forward_mode.is_decode_or_idle())
        if capturing:
            self._n93_cap += 1
            self._n93_cap_decode += int(is_decode)
        else:
            self._n93_eager += 1
            self._n93_eager_decode += int(is_decode)
        n = self._n93_cap + self._n93_eager
        if n <= 6 or n % 100 == 0:
            logger.warning(
                "fn:N93 route-probe n=%d capturing=%s decode=%s rows=%s | "
                "cap=%d eager=%d cap_decode=%d eager_decode=%d | "
                "metadata_has_sampling_info=%s",
                n,
                capturing,
                is_decode,
                getattr(logits_metadata, "extend_seq_lens_cpu", None) is None,
                self._n93_cap,
                self._n93_eager,
                self._n93_cap_decode,
                self._n93_eager_decode,
                hasattr(logits_metadata, "sampling_info"),
            )

    def _n94_fast_path_allowed(self, logits_metadata: LogitsMetadata) -> bool:
        """fn:N94 LOCAL MODIFICATION (gaema).

        Without the export mode this is a constant True, so fn:N89's arm is
        bit-for-bit the behaviour it had before this round.  WITH the export
        mode it restricts the -1e30 fast path to DECODE batches: an extend
        batch's input-logprob work reads the whole vocabulary inside this
        module (InputLogprobProcessor), upstream of any post-graph repair, so
        the floor would be read before it could be undone.

        For a captured decode graph this predicate is evaluated at CAPTURE
        (fn:N93 D1) and is True there; for eager prefill it is evaluated per
        batch and is False.  Both are the correct answers, which is why the
        capture-baking that refuted fn:N93's route is harmless here.
        """
        if not self.use_shard_export:
            return True
        return bool(logits_metadata.forward_mode.is_decode_or_idle())

    def _gather_shard_topk_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """fn:N89 LOCAL MODIFICATION (gaema).  See __init__ for the semantics
        and for why this is greedy-only.

        Every op has a static shape, so this is CUDA-graph capturable in the
        same position the full all-gather occupies today.
        """
        # Lazy import, mirroring MultimemAllGatherer.__call__'s own fallback.
        from sglang.srt.distributed import tensor_model_parallel_all_gather

        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        rows, shard_width = logits.shape
        k = min(self.shard_topk_k, shard_width)

        vals, idx = torch.topk(logits, k, dim=-1)
        idx = idx + tp_rank * shard_width
        # fp32 carries a bf16 value losslessly, and every vocab id here is
        # < 2**24, so the id survives the float round-trip exactly.
        packed = torch.cat([vals.to(torch.float32), idx.to(torch.float32)], dim=-1)

        # all_gather(dim=-1) is concat-style in rank order, so each row is
        # [r0_vals(k), r0_idx(k), r1_vals(k), r1_idx(k), ...].
        gathered = tensor_model_parallel_all_gather(packed, dim=-1)
        gathered = gathered.reshape(rows, tp_size, 2, k)
        all_vals = gathered[:, :, 0, :].reshape(rows, tp_size * k)
        all_idx = gathered[:, :, 1, :].reshape(rows, tp_size * k)

        # -1e30 is the same floor sanitize_nan_logits already writes for -inf,
        # and it is below every real logit, so argmax is unaffected.
        out = logits.new_full((rows, tp_size * shard_width), -1e30)
        out.scatter_(1, all_idx.to(torch.int64), all_vals.to(logits.dtype))
        if self.shard_topk_debug:
            self._n90_debug_compare(logits, out)
        return out

    def _n90_debug_compare(self, local_logits: torch.Tensor, out: torch.Tensor):
        """fn:N90 DIAGNOSTIC (gaema).  Inert unless
        SGLANG_LOGITS_SHARD_TOPK_DEBUG=1.

        Recomputes the UPSTREAM dense all-gather beside the shard-top-k output
        and reports, per row, (a) whether the two disagree on argmax and
        (b) whether the dense top-2 logits are exactly tied or near-tied.
        This is the direct test of the fn:N89 tie-break hypothesis: if the
        prompt-0 divergence is a tie, TIED=True must appear at that step.
        """
        from sglang.srt.distributed import tensor_model_parallel_all_gather

        with torch.no_grad():
            dense = tensor_model_parallel_all_gather(local_logits, dim=-1)
            df = dense.float()
            # The greedy sampler dispatches is_all_greedy -> torch.argmax
            # (layers/sampler.py:126), so argmax is the reduction under test on
            # BOTH sides.  v1 used topk on the dense side and was comparing two
            # different tie-breaks.
            d_arg = df.argmax(dim=-1)
            s_arg = out.float().argmax(dim=-1)
            d_max = df.amax(dim=-1, keepdim=True)
            # Tie MULTIPLICITY: how many vocab entries attain the dense maximum.
            mult = (df == d_max).sum(dim=-1)
            differ = d_arg != s_arg
            tied = mult > 1
            self._n90_step += 1
            hit = torch.nonzero(differ | tied).flatten()
            if hit.numel() == 0:
                return
            # One host transfer for the whole step, not one per row.
            rows = hit.tolist()
            d_a = d_arg[hit].tolist()
            s_a = s_arg[hit].tolist()
            m_c = mult[hit].tolist()
            v_m = d_max.squeeze(-1)[hit].tolist()
            for r, da, sa, mc, vm in zip(rows, d_a, s_a, m_c, v_m):
                logger.warning(
                    "fn:N90 step=%d row=%d DIFFER=%s TIEMULT=%d "
                    "dense_argmax=%d sparse_argmax=%d max=%.9g",
                    self._n90_step, r, da != sa, mc, da, sa, vm,
                )

    def _scatter_dp_attn_logits(
        self,
        logits: torch.Tensor,
        local_hidden_states: torch.Tensor,
        logits_metadata: LogitsMetadata,
    ) -> torch.Tensor:
        if self.do_tensor_parallel_all_gather_dp_attn:
            global_logits = logits
            logits = torch.empty(
                (local_hidden_states.shape[0], global_logits.shape[1]),
                device=global_logits.device,
                dtype=global_logits.dtype,
            )
            dp_scatter(logits, global_logits, logits_metadata)
        return logits

    def _copy_logits_to_buffer(
        self,
        logits: torch.Tensor,
        logits_metadata: LogitsMetadata,
        use_buffer: bool = True,
    ) -> torch.Tensor:
        logits_buffer = logits_metadata.next_token_logits_buffer if use_buffer else None
        if logits.shape[-1] > self.vocab_size:
            logits = logits[:, : self.vocab_size]
        logits_width = logits.shape[-1]
        # The shared logits buffer is keyed by vocab width and rows; skip it
        # when this batch has a different logits shape than the graph buffer.
        if logits_buffer is not None and tuple(logits_buffer.shape) == tuple(
            logits.shape
        ):
            assert logits_buffer.dtype == torch.float
            logits_buffer.copy_(logits)
            logits = logits_buffer
        else:
            logits = logits.float()
        return logits

    def _get_dllm_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> LogitsProcessorOutput:
        assert self.return_full_logits
        full_logits = self._get_logits(hidden_states, lm_head, logits_metadata)
        return LogitsProcessorOutput(
            full_logits=full_logits,
            next_token_logits=None,
        )

    def compute_logprobs_for_multi_item_scoring(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        multi_item_delimiter_indices: List[torch.Tensor],
    ):
        """
        Compute logprobs for multi-item scoring using pre-computed delimiter indices.

        Sequence format: Query<delimiter>Item1<delimiter>Item2<delimiter>...
        Scoring positions: Extracts logprobs at positions before each <delimiter>

        Args:
            input_ids: Input token IDs. Shape: [total_sequence_length].
            hidden_states: Hidden states from the model. Shape: [sequence_length, hidden_dim].
            lm_head: Language model head for computing logits.
            logits_metadata: Metadata containing batch info and logprob specs.
            multi_item_delimiter_indices: Pre-computed delimiter positions per request (CPU tensors).
        """
        # Compute positions just before each delimiter.
        # Build offset-adjusted indices on CPU, then do a single CPU→GPU transfer.
        device = input_ids.device
        all_tensors = []
        if logits_metadata.extend_seq_lens_cpu is not None:
            offset = 0
            for req_seq_len, indices_tensor in zip(
                logits_metadata.extend_seq_lens_cpu, multi_item_delimiter_indices
            ):
                if len(indices_tensor) > 0:
                    # Note: if the first delimiter is at position 0 (empty query),
                    # indices - 1 wraps to -1. This is harmless — the first
                    # delimiter entry is always discarded by
                    # _process_multi_item_scoring_results.
                    all_tensors.append(indices_tensor + (offset - 1))
                offset += req_seq_len
        else:
            all_tensors.append(multi_item_delimiter_indices[0] - 1)
        multi_item_indices = torch.cat(all_tensors).to(device, non_blocking=True)

        # Extract hidden states at delimiter positions for multi-item scoring
        sliced_hidden = hidden_states[multi_item_indices]

        sliced_logits = self._get_logits(sliced_hidden, lm_head, logits_metadata)
        sliced_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1)

        # Initialize return values
        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []
        input_top_logprobs_val = None
        input_top_logprobs_idx = None

        # Recalculate extend_logprob_pruned_lens_cpu to match delimiter counts per request
        if (
            logits_metadata.token_ids_logprobs
            or logits_metadata.extend_return_top_logprob
        ):
            logits_metadata.extend_logprob_pruned_lens_cpu = [
                len(t) for t in multi_item_delimiter_indices
            ]

        # Get the logprobs of specified token ids
        if logits_metadata.extend_token_ids_logprob:
            (
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_raw(
                sliced_logprobs,
                logits_metadata.token_ids_logprobs,
                stage=LogprobStage.PREFILL,
                extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
                no_copy_to_cpu=True,
            )

        # Get the logprob of top-k tokens
        if logits_metadata.extend_return_top_logprob:
            (
                input_top_logprobs_val,
                input_top_logprobs_idx,
            ) = get_top_logprobs_raw(
                sliced_logprobs,
                logits_metadata.top_logprobs_nums,
                stage=LogprobStage.PREFILL,
                extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
            )

        # MIS scores come from input_token_ids_logprobs_val (label-token logprobs),
        # not from per-position input_token_logprobs. However, the shared logprob
        # pipeline (add_input_logprob_return_values) asserts input_token_logprobs is
        # non-None, converts it to a tuple, slices it, and validates its length —
        # all before score_request() ever sees the result. We can't set it to None
        # without changing those shared asserts, so we fill with zeros to satisfy
        # the pipeline. score_request() ignores this field entirely.
        input_token_logprobs = torch.zeros(multi_item_indices.shape[0], device=device)

        return LogitsProcessorOutput(
            next_token_logits=None,
            input_token_logprobs=input_token_logprobs,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )


def _reassemble_tp_lm_head_all_to_all_output(
    all_to_all_output: torch.Tensor, tp_size: int
) -> torch.Tensor:
    """Convert source-major all-to-all output to row-major full-vocab logits.

    Each source TP rank contributes ``[local_rows, vocab_shard]`` for this
    destination DP rank. ``all_to_all_single`` concatenates those contributions
    along dim 0, while the sampler expects the vocab shards concatenated along
    dim 1.
    """
    assert all_to_all_output.shape[0] % tp_size == 0
    local_rows = all_to_all_output.shape[0] // tp_size
    vocab_shard = all_to_all_output.shape[1]
    return (
        all_to_all_output.view(tp_size, local_rows, vocab_shard)
        .permute(1, 0, 2)
        .reshape(local_rows, tp_size * vocab_shard)
    )


# --- fn:N94 GRAPH-OUTPUT SHARD EXPORT (gaema) ---------------------------------
# The post-graph half of the export.  Called from ModelRunner.sample() (and
# compute_logprobs_only) BEFORE _preprocess_logits, so both that function and
# Sampler.forward are untouched: they still receive exactly one [B, vocab]
# fp32 tensor with the same contract it has always had.
#
# ROUTE (host-side, batch-level, NO device synchronization):
#   fast  -- every request in this batch is argmax-safe against the -1e30
#            floor, so the fast path's tensor is used as-is and the dense
#            collective is never issued.
#   exact -- at least one request needs full-vocab mass or ordering, so the
#            exported shard is all_gathered and written over the floored
#            tensor.  The result is bitwise the dense path's own values.
#
# The route is deliberately batch-level rather than per-row: `is_all_greedy`
# and every logprob/penalty/bias/grammar flag are already HOST values, while a
# per-row predicate lives in `sampling_info.top_ks` on the DEVICE and would
# need a D2H `.item()` per decode step to shape the collective -- a CPU/GPU
# pipeline stall that costs more than the 2.599 ms/step it is trying to save.
_N94_ROUTE_COUNTS = {"fast": 0, "exact": 0, "skip": 0}

# fn:N94 VERIFY, default OFF.  The IN-PROCESS, BITWISE, TWO-SIDED control on the
# reconstruction.  A cross-process comparison cannot settle this question at
# this configuration: fn:N92 measured the stock stack's own per-process
# nondeterminism (2/23 lever-OFF, on the pristine upstream file), and this round
# reproduced it on logprobs -- control C1 and export A1 disagree on the FIRST
# generated token's logprobs by 0.026 nats, and that token comes from PREFILL,
# where the export's fast path is disabled outright and both arms run byte-
# identical code.  So the divergence is upstream of anything this lever touches
# and a control-vs-arm comparison is measuring the latch, not the lever.
#
# What CAN be settled, in one process with no collective of its own, is the
# relationship between the two tensors the exact route holds at that instant:
#
#   must-accept  at the <=2k positions the fast path's scatter FILLED, the
#                reconstruction must agree BITWISE -- those are the same values
#                carried by two different routes, so any mismatch is a real bug;
#   must-reject  at the -1e30 FLOORED positions the reconstruction must CHANGE
#                the value -- a repair that changes nothing is a dead lever, and
#                would read as a pass to every downstream check.
#
# SGLANG_LOGITS_EXPORT_VERIFY=1
_N94_VERIFY = os.environ.get("SGLANG_LOGITS_EXPORT_VERIFY", "0") not in ("", "0")
_N94_VERIFY_CALLS = 0
_N94_VERIFY_ROWS = 4      # bound the cost: compare the first few rows only
_N94_VERIFY_MAX = 8       # ...on the first few exact-route steps only


def _n94_verify(dst, dense) -> None:
    """fn:N94 (gaema).  Inert unless SGLANG_LOGITS_EXPORT_VERIFY=1."""
    global _N94_VERIFY_CALLS
    if _N94_VERIFY_CALLS >= _N94_VERIFY_MAX:
        return
    _N94_VERIFY_CALLS += 1
    try:
        with torch.no_grad():
            r = min(_N94_VERIFY_ROWS, dst.shape[0])
            floored = dst[:r]
            rebuilt = dense[:r].to(dst.dtype)
            filled = floored > -1e29
            n_fill = int(filled.sum().item())
            n_floor = int(filled.numel() - n_fill)
            if n_fill:
                mism = int((floored[filled] != rebuilt[filled]).sum().item())
                mx = float((floored[filled] - rebuilt[filled]).abs().max().item())
            else:
                mism, mx = -1, float("nan")
            changed = (
                int((rebuilt[~filled] != floored[~filled]).sum().item())
                if n_floor
                else -1
            )
            logger.warning(
                "fn:N94 VERIFY call=%d rows=%d | MUST-ACCEPT filled=%d "
                "mismatch=%d max_abs_diff=%.9g | MUST-REJECT floored=%d "
                "changed=%d (%.4f%%)",
                _N94_VERIFY_CALLS, r, n_fill, mism, mx, n_floor, changed,
                100.0 * changed / n_floor if n_floor else float("nan"),
            )
    except Exception as e:  # pragma: no cover - an instrument must not kill a run
        logger.warning("fn:N94 VERIFY raised %r -- reading is UNEVALUABLE", e)


def n94_batch_needs_exact_logits(forward_batch) -> bool:
    """fn:N94 (gaema).  True when ANY request in this batch reads mass or
    ordering the -1e30 floor destroys.  Conservative by construction: every
    unknown resolves to True, so an unrecognized consumer gets the dense
    reconstruction rather than the floored tensor."""
    si = getattr(forward_batch, "sampling_info", None)
    if si is None:
        return True
    # Non-greedy: temperature / top-p / top-k / min-p all renormalize over the
    # surviving mass, which the floor has changed.
    if not getattr(si, "is_all_greedy", False):
        return True
    # log_softmax over a floored row is wrong even when the argmax is right.
    if getattr(forward_batch, "return_logprob", False):
        return True
    if getattr(forward_batch, "token_ids_logprobs", None):
        return True
    if any(getattr(forward_batch, "top_logprobs_nums", None) or []):
        return True
    # Additive/multiplicative modifications of a floored entry are meaningless.
    if getattr(si, "logit_bias", None) is not None:
        return True
    if getattr(si, "grammar_mask", None) is not None:
        return True
    if getattr(si, "grammars", None):
        return True
    if getattr(si, "has_custom_logit_processor", False):
        return True
    if getattr(si, "acc_additive_penalties", None) is not None:
        return True
    if getattr(si, "acc_scaling_penalties", None) is not None:
        return True
    po = getattr(si, "penalizer_orchestrator", None)
    if po is not None and getattr(po, "is_required", False):
        return True
    if any(getattr(si, "return_sampling_masks", None) or []):
        return True
    return False


def n94_restore_exact_logits(logits_output, forward_batch) -> Optional[str]:
    """fn:N94 (gaema).  Reconstruct the exact dense logits for a batch that
    needs them, from the shard the captured graph exported.

    Returns the route taken ("fast" / "exact" / "skip"), or None when this
    output carries no exported shard (every path but the export mode's decode
    fast path), in which case it is a no-op.
    """
    shard = getattr(logits_output, "local_vocab_shard", None)
    if shard is None:
        return None
    # One step, one use.  Never let a reference outlive the step that made it.
    logits_output.local_vocab_shard = None

    if not n94_batch_needs_exact_logits(forward_batch):
        _N94_ROUTE_COUNTS["fast"] += 1
        _n94_maybe_log()
        return "fast"

    dst = logits_output.next_token_logits
    if dst is None or dst.shape[0] != shard.shape[0]:
        # Rows do not correspond; refuse rather than write the wrong rows.
        # Unreachable for decode, where forward() only attaches the shard when
        # sample_indices is None.
        _N94_ROUTE_COUNTS["skip"] += 1
        logger.warning(
            "fn:N94 export: row mismatch dst=%s shard=%s -- NOT reconstructing",
            None if dst is None else tuple(dst.shape),
            tuple(shard.shape),
        )
        return "skip"

    from sglang.srt.distributed import tensor_model_parallel_all_gather

    dense = tensor_model_parallel_all_gather(shard, dim=-1)
    if dense.shape[-1] > dst.shape[-1]:
        # Same truncation _copy_logits_to_buffer applies to a padded vocab.
        dense = dense[:, : dst.shape[-1]]
    if _N94_VERIFY:
        _n94_verify(dst, dense)
    # Same bf16 -> fp32 widening _copy_logits_to_buffer performs, so these rows
    # are bit-for-bit what the dense path would have left here.
    dst.copy_(dense)
    _N94_ROUTE_COUNTS["exact"] += 1
    _n94_maybe_log()
    return "exact"


def _n94_maybe_log() -> None:
    """fn:N94 (gaema) LIVENESS.  The route counters ARE the instrument: a run
    that never prints an `exact` is claiming every batch was argmax-safe, and a
    run that never prints a `fast` is claiming the lever bought nothing.  Both
    are falsifiable readings, and a run with neither has a dead instrument.
    First six calls, then every 200th."""
    n = _N94_ROUTE_COUNTS["fast"] + _N94_ROUTE_COUNTS["exact"] + _N94_ROUTE_COUNTS["skip"]
    if n <= 6 or n % 200 == 0:
        logger.warning(
            "fn:N94 export route n=%d fast=%d exact=%d skip=%d",
            n,
            _N94_ROUTE_COUNTS["fast"],
            _N94_ROUTE_COUNTS["exact"],
            _N94_ROUTE_COUNTS["skip"],
        )


def n94_route_counts() -> dict:
    """fn:N94 (gaema) liveness readback.  A run whose counters are all zero has
    a DEAD instrument, not a fast batch mix."""
    return dict(_N94_ROUTE_COUNTS)


# --- end fn:N94 ---------------------------------------------------------------


def _has_lm_head_runtime_attrs(lm_head, attr_names: Tuple[str, ...]) -> bool:
    return all(hasattr(lm_head, attr_name) for attr_name in attr_names)


def should_apply_lm_head_quant_method(lm_head, quant_method) -> bool:
    if (
        quant_method is None
        or not hasattr(lm_head, "weight")
        or not callable(getattr(quant_method, "apply", None))
    ):
        return False

    method_name = type(quant_method).__name__
    if method_name in _UNQUANTIZED_LM_HEAD_METHODS:
        return False

    # Some draft models share an unquantized target lm_head tensor while still
    # carrying the draft model's stale ModelOpt quant_method. Only use the
    # ModelOpt lm_head kernel when the runtime quantization state matches it.
    if method_name == "ModelOptFp4LinearMethod":
        if quant_method.quant_mode == "w4a16":
            return lm_head.weight.dtype == torch.uint8 and _has_lm_head_runtime_attrs(
                lm_head,
                (
                    "weight_scale_interleaved",
                    "alpha",
                    "input_size_per_partition",
                    "output_size_per_partition",
                ),
            )
        if lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        ):
            return True
        return lm_head.weight.dtype == torch.uint8 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale_interleaved",
                "alpha",
                "input_scale_inv",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptNvFp4A16LinearMethod":
        return lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptFp8LinearMethod":
        return (
            lm_head.weight.dtype == torch.float8_e4m3fn
            and _has_lm_head_runtime_attrs(lm_head, ("weight_scale", "input_scale"))
        )

    return True
