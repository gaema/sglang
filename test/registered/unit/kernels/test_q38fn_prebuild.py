"""q38fn_prebuild_fp8_copies builds the FP8 lm_head and HC-mix copies after load, byte-identical to what the lazy
paths build, only when each lever is on, and never twice."""

import os
import unittest

import torch

from sglang.kernels.ops.gemm.hc_mix import q38fn_quant_rows_fp8
from sglang.kernels.ops.gemm.q38fn_lm_head_fp8 import q38fn_quant_rows_e4m3
from sglang.kernels.ops.gemm.q38fn_prebuild import q38fn_prebuild_fp8_copies


class ParallelLMHead(torch.nn.Module):  # name-matched stand-in
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(512, 2560, dtype=torch.bfloat16, device="cuda"),
                                         requires_grad=False)


class GatedResidual(torch.nn.Module):  # name-matched stand-in
    def __init__(self):
        super().__init__()
        self.input_mix_weight_down = torch.nn.Linear(10240, 320, bias=False, dtype=torch.bfloat16, device="cuda")
        self.input_mix_weight_up = torch.nn.Linear(320, 10240, bias=False, dtype=torch.bfloat16, device="cuda")


def _model():
    m = torch.nn.Module()
    m.lm_head = ParallelLMHead()
    m.layers = torch.nn.ModuleList([GatedResidual(), GatedResidual()])
    return m


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestQ38fnPrebuild(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("SGLANG_Q38FN_LMHEAD_FP8", "SGLANG_Q38FN_MIX_FP8")}

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_builds_copies_equal_to_the_lazy_ones(self):
        os.environ["SGLANG_Q38FN_LMHEAD_FP8"] = "1"
        os.environ["SGLANG_Q38FN_MIX_FP8"] = "1"
        m = _model()
        counts = q38fn_prebuild_fp8_copies(m)
        self.assertEqual(counts, {"lm_head": 1, "hc_mix": 2})
        self.assertTrue(_same(m.lm_head._q38fn_fp8, q38fn_quant_rows_e4m3(m.lm_head.weight)))
        g = m.layers[0]
        lazy = (*q38fn_quant_rows_fp8(g.input_mix_weight_down.weight),
                *q38fn_quant_rows_fp8(g.input_mix_weight_up.weight))
        self.assertTrue(_same(g._q38fn_fp8w, lazy))

    def test_off_levers_build_nothing(self):
        os.environ["SGLANG_Q38FN_LMHEAD_FP8"] = "0"
        os.environ["SGLANG_Q38FN_MIX_FP8"] = "0"
        m = _model()
        self.assertEqual(q38fn_prebuild_fp8_copies(m), {"lm_head": 0, "hc_mix": 0})
        self.assertIsNone(getattr(m.lm_head, "_q38fn_fp8", None))

    def test_second_call_builds_nothing(self):
        os.environ["SGLANG_Q38FN_LMHEAD_FP8"] = "1"
        os.environ["SGLANG_Q38FN_MIX_FP8"] = "1"
        m = _model()
        q38fn_prebuild_fp8_copies(m)
        self.assertEqual(q38fn_prebuild_fp8_copies(m), {"lm_head": 0, "hc_mix": 0})


if __name__ == "__main__":
    unittest.main()
