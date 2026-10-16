"""CPU unit tests for choosing an NVFP4 PLE (n-gram) table in qwen4_exp.

nvidia/Qwen3.8-Flash-Next-FP4-Local stores its 51B-parameter n-gram table as
NVFP4 (ModelOpt MIXED_PRECISION entry ``...ple.ple_embedding.ngram_embedding``,
``ple_embedding_dtype="nvfp4"``). The table must then be gathered through
ModelOptNvFp4EmbeddingMethod; the bf16 / fp8 tables of other checkpoints must
keep their existing path.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.models.qwen4_exp import _ple_table_is_fp8, _ple_table_is_nvfp4
from sglang.test.test_utils import CustomTestCase

PREFIX = "model.layers.1.ple.ple_embedding.ngram_embedding"


def _mixed(ngram_algo):
    layers = {
        "model.language_model.layers.0.mlp.experts": {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
    }
    if ngram_algo:
        layers["model.language_model.layers.1.ple.ple_embedding.ngram_embedding"] = {
            "quant_algo": ngram_algo,
            "group_size": 16,
        }
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "ignore": [],
            "quantized_layers": layers,
        }
    )


def _cfg(dtype=None):
    return types.SimpleNamespace(ple_embedding_dtype=dtype)


class TestNvFp4PleSelection(CustomTestCase):
    def test_nvfp4_entry_selects_nvfp4_and_not_fp8(self):
        q = _mixed("NVFP4")
        self.assertTrue(_ple_table_is_nvfp4(_cfg("nvfp4"), q, PREFIX))
        self.assertTrue(_ple_table_is_nvfp4(_cfg(None), q, PREFIX))
        self.assertFalse(_ple_table_is_fp8(_cfg("nvfp4"), q, PREFIX))

    def test_other_tables_keep_their_path(self):
        self.assertFalse(_ple_table_is_nvfp4(_cfg(None), None, PREFIX))
        self.assertFalse(_ple_table_is_nvfp4(_cfg("float8_e4m3fn"), None, PREFIX))
        self.assertFalse(_ple_table_is_nvfp4(_cfg(None), _mixed(None), PREFIX))
        self.assertTrue(_ple_table_is_fp8(_cfg(None), _mixed("FP8"), PREFIX))
        self.assertFalse(_ple_table_is_nvfp4(_cfg(None), _mixed("FP8"), PREFIX))

    def test_declared_nvfp4_without_a_quant_entry_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nvfp4"):
            _ple_table_is_nvfp4(_cfg("nvfp4"), None, PREFIX)
        with self.assertRaisesRegex(ValueError, "nvfp4"):
            _ple_table_is_nvfp4(_cfg("nvfp4"), _mixed(None), PREFIX)


if __name__ == "__main__":
    unittest.main()
