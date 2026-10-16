"""CPU unit tests for serving a small MXFP8 linear from a bf16 copy.

FlashInfer's MXFP8 dense GEMMs (CUTLASS, CuTe-DSL) need n >= 128 and k >= 128;
nvidia/Qwen3.8-Flash-Next-FP4-Local's linear-attention in_proj_ba is n = 96.
Such a layer is dequantized once: every element is an e4m3 value times a power
of two, so the bf16 copy is exact.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.layers.quantization.fp8 import (
    dequantize_mxfp8_weight,
    mxfp8_below_flashinfer_min,
)
from sglang.test.test_utils import CustomTestCase


def _backend(name):
    return types.SimpleNamespace(is_flashinfer_cutlass=lambda: name == "cutlass",
                                 is_flashinfer_cutedsl=lambda: name == "cutedsl")


class TestMxfp8SmallLinear(CustomTestCase):
    def test_dequant_matches_the_per_block_reference_exactly(self):
        g = torch.Generator().manual_seed(0)
        n, k = 96, 2560
        w = (torch.randn(n, k, generator=g) * 4).to(torch.float8_e4m3fn)
        s = torch.randint(120, 135, (n, k // 32), generator=g, dtype=torch.uint8)
        got = dequantize_mxfp8_weight(w, s)
        self.assertEqual(got.dtype, torch.bfloat16)
        ref = torch.empty(n, k, dtype=torch.float64)
        for b in range(k // 32):
            ref[:, b * 32:(b + 1) * 32] = (w[:, b * 32:(b + 1) * 32].double()
                                           * torch.pow(2.0, s[:, b:b + 1].double() - 127))
        self.assertTrue(torch.equal(got.double(), ref))

    def test_only_small_shapes_on_flashinfer_backends_take_the_copy(self):
        for name in ("cutlass", "cutedsl"):
            self.assertTrue(mxfp8_below_flashinfer_min((96, 2560), _backend(name)))
            self.assertTrue(mxfp8_below_flashinfer_min((2560, 96), _backend(name)))
            self.assertFalse(mxfp8_below_flashinfer_min((128, 128), _backend(name)))
        self.assertFalse(mxfp8_below_flashinfer_min((96, 2560), _backend("trtllm")))
        self.assertFalse(mxfp8_below_flashinfer_min((96, 2560), None))


if __name__ == "__main__":
    unittest.main()
