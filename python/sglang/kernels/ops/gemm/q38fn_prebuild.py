# _q38fn_prebuild: build the q38fn FP8 weight copies right after weight load, before CUDA graph capture.
# SGLANG_Q38FN_LMHEAD_FP8 and SGLANG_Q38FN_MIX_FP8 build their copies lazily, on their first call outside graph
# capture. A boot whose first such calls all happen inside capture never builds them, and its graphs record the
# bf16 path for good (seen on the FP4-Local checkpoint, whose boot logged neither build line). Building here makes
# the copies exist before the first capture, with the same functions and the same attributes the lazy paths read.
import os

import torch


def q38fn_prebuild_fp8_copies(model) -> dict:
    counts = {"lm_head": 0, "hc_mix": 0}
    if os.environ.get("SGLANG_Q38FN_LMHEAD_FP8", "0") == "1":
        from sglang.kernels.ops.gemm.q38fn_lm_head_fp8 import q38fn_quant_rows_e4m3

        for m in model.modules():
            w = getattr(m, "weight", None)
            if (
                type(m).__name__ == "ParallelLMHead"
                and isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == torch.bfloat16
                and w.is_contiguous() and getattr(m, "_q38fn_fp8", None) is None
            ):
                m._q38fn_fp8 = q38fn_quant_rows_e4m3(w)
                counts["lm_head"] += 1
    if os.environ.get("SGLANG_Q38FN_MIX_FP8", "0") == "1":
        from sglang.kernels.ops.gemm.hc_mix import q38fn_quant_rows_fp8

        for m in model.modules():
            down = getattr(m, "input_mix_weight_down", None)
            up = getattr(m, "input_mix_weight_up", None)
            if (
                type(m).__name__ == "GatedResidual" and down is not None and up is not None
                and down.weight.is_cuda and getattr(m, "_q38fn_fp8w", None) is None
            ):
                m._q38fn_fp8w = (*q38fn_quant_rows_fp8(down.weight), *q38fn_quant_rows_fp8(up.weight))
                counts["hc_mix"] += 1
    if any(counts.values()):
        print(f"[q38fn-prebuild] fp8 copies built after load: lm_head={counts['lm_head']} "
              f"hc_mix={counts['hc_mix']}", flush=True)
    return counts
