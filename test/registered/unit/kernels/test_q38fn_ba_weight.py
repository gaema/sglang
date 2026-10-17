"""The GDN in_proj_ba small-M GEMM reads the exact bf16 copy of an MXFP8 layer below FlashInfer's minimum, and the
layer's own weight otherwise; on a GPU its output matches F.linear on that bf16 weight."""

import types
import unittest

import torch

from sglang.kernels.ops.gemm.q38fn_ba_gemm import q38fn_ba_gemm, q38fn_ba_gemm_ok, q38fn_ba_weight


class TestQ38fnBaWeight(unittest.TestCase):
    def test_prefers_mxfp8_bf16_copy(self):
        copy = torch.randn(96, 2560, dtype=torch.bfloat16)
        layer = types.SimpleNamespace(weight=torch.zeros(96, 2560, dtype=torch.float8_e4m3fn),
                                      mxfp8_dequant_weight=copy, bias=None)
        self.assertIs(q38fn_ba_weight(layer), copy)

    def test_falls_back_to_layer_weight(self):
        w = torch.randn(96, 2560, dtype=torch.bfloat16)
        layer = types.SimpleNamespace(weight=w, bias=None)
        self.assertIs(q38fn_ba_weight(layer), w)

    def test_mxfp8_layer_without_copy_is_refused(self):
        layer = types.SimpleNamespace(weight=torch.zeros(96, 2560, dtype=torch.float8_e4m3fn), bias=None)
        x = torch.randn(4, 2560, dtype=torch.bfloat16)
        self.assertFalse(q38fn_ba_gemm_ok(x, layer))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_gemm_on_copy_matches_linear(self):
        copy = torch.randn(96, 2560, dtype=torch.bfloat16, device="cuda")
        layer = types.SimpleNamespace(weight=torch.zeros(96, 2560, dtype=torch.float8_e4m3fn, device="cuda"),
                                      mxfp8_dequant_weight=copy, bias=None)
        for m in (1, 3, 9, 64):
            x = torch.randn(m, 2560, dtype=torch.bfloat16, device="cuda")
            self.assertTrue(q38fn_ba_gemm_ok(x, layer))
            got = q38fn_ba_gemm(x, q38fn_ba_weight(layer))
            want = torch.nn.functional.linear(x.float(), copy.float()).to(torch.bfloat16)
            # outputs are ~sqrt(2560) = 50 in size, where one bf16 step is 0.25: allow two bf16 steps
            torch.testing.assert_close(got.float(), want.float(), rtol=2 * 2.0 ** -7, atol=0.05,
                                       msg=f"m={m}")


if __name__ == "__main__":
    unittest.main()
