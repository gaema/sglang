"""CPU unit tests for the Qwen4-Exp PLE-offload default with an NVFP4 n-gram table.

The pinned-host PLE offload gathers bf16 / fp8 rows only, and qwen4_exp refuses
it for an NVFP4 table, so the default (on for BF16 checkpoints on CUDA) must be
off when the checkpoint's n-gram table is NVFP4 (nvidia/Qwen3.8-Flash-Next-FP4-Local).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.arg_groups.model_overrides.qwen4_exp import hf_ple_table_is_nvfp4
from sglang.test.test_utils import CustomTestCase

NGRAM = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def _cfg(layers=None, ple_dtype=None):
    text = types.SimpleNamespace(ple_embedding_dtype=ple_dtype)
    q = None if layers is None else {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                     "quantized_layers": layers}
    return types.SimpleNamespace(text_config=text, quantization_config=q)


class TestPleOffloadDefault(CustomTestCase):
    def test_nvfp4_table_is_detected(self):
        self.assertTrue(hf_ple_table_is_nvfp4(_cfg({NGRAM: {"quant_algo": "NVFP4"}})))
        self.assertTrue(hf_ple_table_is_nvfp4(_cfg(None, "nvfp4")))

    def test_other_tables_keep_the_default(self):
        self.assertFalse(hf_ple_table_is_nvfp4(_cfg(None)))
        self.assertFalse(hf_ple_table_is_nvfp4(_cfg({NGRAM: {"quant_algo": "FP8"}})))
        self.assertFalse(hf_ple_table_is_nvfp4(
            _cfg({"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"}})))
        self.assertFalse(hf_ple_table_is_nvfp4(_cfg(None, "float8_e4m3fn")))


if __name__ == "__main__":
    unittest.main()
