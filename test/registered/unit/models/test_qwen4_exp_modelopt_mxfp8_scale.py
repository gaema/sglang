"""CPU unit tests for loading ModelOpt MXFP8 block scales into qwen4_exp.

ModelOpt names an MXFP8 layer's UE8M0 block scale ``weight_scale`` (uint8);
sglang's block-FP8 linear method registers it as ``weight_scale_inv``. The
loader must take the ``_inv`` parameter, also through a fused (stacked)
mapping, and leave NVFP4 scales (float8) and unknown names alone.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.models.qwen4_exp import _modelopt_mxfp8_scale_name
from sglang.test.test_utils import CustomTestCase

STACKED = [("in_proj_ba.", "in_proj_b.", 0), ("in_proj_ba.", "in_proj_a.", 1),
           ("gate_up_proj", "gate_proj", 0)]
U8 = torch.zeros(2, 2, dtype=torch.uint8)


class TestModelOptMxfp8ScaleName(CustomTestCase):
    def test_uint8_scale_maps_to_the_inv_parameter(self):
        params = {"model.layers.0.linear_attn.out_proj.weight_scale_inv": 0,
                  "model.layers.0.linear_attn.in_proj_ba.weight_scale_inv": 0,
                  "model.layers.0.mlp.shared_expert.gate_up_proj.weight_scale_inv": 0}
        for name in ("model.layers.0.linear_attn.out_proj.weight_scale",
                     "model.layers.0.linear_attn.in_proj_a.weight_scale",
                     "model.layers.0.mlp.shared_expert.gate_proj.weight_scale"):
            self.assertEqual(_modelopt_mxfp8_scale_name(name, U8, params, STACKED), name + "_inv")

    def test_other_scales_and_names_are_unchanged(self):
        params = {"model.layers.0.linear_attn.out_proj.weight_scale_inv": 0,
                  "model.layers.0.mlp.experts.w13_weight_scale": 0}
        fp8 = torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
        name = "model.layers.0.linear_attn.out_proj.weight_scale"
        self.assertEqual(_modelopt_mxfp8_scale_name(name, fp8, params, STACKED), name)
        other = "model.layers.0.self_attn.q_proj.weight_scale"
        self.assertEqual(_modelopt_mxfp8_scale_name(other, U8, params, STACKED), other)
        w = "model.layers.0.linear_attn.out_proj.weight"
        self.assertEqual(_modelopt_mxfp8_scale_name(w, U8, params, STACKED), w)


if __name__ == "__main__":
    unittest.main()
