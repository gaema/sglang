"""CPU unit tests for load-time NVFP4 compressed-scale ("csf") decoding.

The decoder must reproduce, byte for byte, the E4M3 block scales the csf pair
encodes, and must refuse any pair it cannot decode exactly -- a wrong scale
byte is silently wrong output, never a crash. The encoder below is written from
the format description, independently of the decoder, so the round trip can
fail. Shapes follow the Qwen3.8-Flash-Next-FP4-Local experts (hidden 2560,
moe_intermediate 640).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.nvfp4_csf import (
    CSF_EXCEPTIONS_SUFFIX,
    CSF_FIXED_SUFFIX,
    decode_nvfp4_csf_scale,
    decode_nvfp4_csf_weights,
)
from sglang.test.test_utils import CustomTestCase

# (rows, in_features) of the three expert projections: gate/up, down.
EXPERT_SHAPES = [(640, 2560), (2560, 640)]


def _scales(rows: int, columns: int, seed: int) -> torch.Tensor:
    """Scale bytes shaped like real NVFP4 block scales: rows cluster around a
    per-row level (offsets 0..15 fit), with a sprinkling of outliers far above
    their row (exceptions) and a few rows whose minimum is above 240 (the base
    clamps, so every block in them is an exception)."""
    g = torch.Generator().manual_seed(seed)
    level = torch.randint(20, 200, (rows, 1), generator=g)
    spread = torch.randint(0, 16, (rows, columns), generator=g)
    v = level + spread
    outlier = torch.rand((rows, columns), generator=g) < 0.01
    v = torch.where(outlier, torch.randint(0, 126, (rows, columns), generator=g), v)
    v[3] = torch.randint(241, 255, (columns,), generator=g)
    return v.clamp(0, 254).to(torch.uint8)


def _encode(scales: torch.Tensor) -> tuple:
    """Encode [rows, columns] scale bytes as a csf (fixed, exceptions) pair."""
    rows, columns = scales.shape
    v = scales.to(torch.int64)
    base = v.min(dim=1, keepdim=True).values.clamp(max=240)
    off = v - base
    exc = off > 15
    nib = torch.where(exc, torch.zeros_like(off), off)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat(
        (base.to(torch.uint8).reshape(rows // 16, 16), packed.reshape(rows // 16, -1)),
        dim=1,
    )
    pos = torch.nonzero(exc.reshape(-1)).reshape(-1)
    words = pos | (v.reshape(-1)[pos] << 24)
    exceptions = words.to(torch.int64).to(torch.int32)  # wraps like a uint32 view
    exceptions = exceptions.view(torch.uint32)
    return fixed, exceptions


class TestDecodeScale(CustomTestCase):
    def test_round_trip_expert_shapes(self):
        for i, (rows, in_features) in enumerate(EXPERT_SHAPES):
            scales = _scales(rows, in_features // 16, seed=i)
            fixed, exceptions = _encode(scales)
            self.assertEqual(
                tuple(fixed.shape), (rows // 16, 16 * (1 + in_features // 32))
            )
            self.assertGreater(exceptions.numel(), 0)
            out = decode_nvfp4_csf_scale(fixed, exceptions)
            self.assertEqual(out.dtype, torch.float8_e4m3fn)
            self.assertEqual(tuple(out.shape), (rows, in_features // 16))
            self.assertTrue(torch.equal(out.view(torch.uint8), scales))

    def test_no_exceptions(self):
        scales = torch.arange(16 * 8, dtype=torch.int64).reshape(16, 8) % 16 + 100
        fixed, exceptions = _encode(scales.to(torch.uint8))
        self.assertEqual(exceptions.numel(), 0)
        out = decode_nvfp4_csf_scale(fixed, exceptions)
        self.assertTrue(torch.equal(out.view(torch.uint8), scales.to(torch.uint8)))

    def test_nibble_order_low_first(self):
        # One slab, 2 columns: row r has base 50 and offsets (1, 9).
        fixed = torch.zeros((1, 32), dtype=torch.uint8)
        fixed[0, :16] = 50
        fixed[0, 16:] = 1 | (9 << 4)
        out = decode_nvfp4_csf_scale(fixed, torch.zeros(0, dtype=torch.uint32))
        self.assertTrue(torch.all(out.view(torch.uint8)[:, 0] == 51))
        self.assertTrue(torch.all(out.view(torch.uint8)[:, 1] == 59))

    def test_exception_high_byte_and_position(self):
        fixed = torch.zeros((1, 32), dtype=torch.uint8)
        fixed[0, :16] = 50
        word = torch.tensor([(255 << 24) | (5 * 2 + 1)], dtype=torch.int64)
        out = decode_nvfp4_csf_scale(fixed, word.to(torch.int32).view(torch.uint32))
        u = out.view(torch.uint8)
        self.assertEqual(int(u[5, 1]), 255)
        self.assertEqual(int((u != 50).sum()), 1)

    def test_rejects_base_above_240(self):
        fixed, exceptions = _encode(_scales(16, 8, seed=7))
        fixed[0, 2] = 241
        with self.assertRaisesRegex(ValueError, "bases"):
            decode_nvfp4_csf_scale(fixed, exceptions)

    def test_rejects_unsorted_duplicate_or_out_of_range_positions(self):
        fixed = torch.full((1, 32), 60, dtype=torch.uint8)
        for positions in ([3, 1], [2, 2], [32]):
            words = torch.tensor(positions, dtype=torch.int32).view(torch.uint32)
            with self.assertRaisesRegex(ValueError, "sorted"):
                decode_nvfp4_csf_scale(fixed, words)

    def test_rejects_bad_dtypes_and_shapes(self):
        good = torch.zeros(0, dtype=torch.uint32)
        for fixed in (
            torch.zeros((1, 32), dtype=torch.int8),
            torch.zeros(32, dtype=torch.uint8),
            torch.zeros((1, 24), dtype=torch.uint8),
            torch.zeros((1, 16), dtype=torch.uint8),
        ):
            with self.assertRaises(ValueError):
                decode_nvfp4_csf_scale(fixed, good)
        with self.assertRaises(ValueError):
            decode_nvfp4_csf_scale(
                torch.zeros((1, 32), dtype=torch.uint8), torch.zeros(0, dtype=torch.float32)
            )


class TestDecodeWeights(CustomTestCase):
    def _pair(self, prefix, seed):
        scales = _scales(640, 160, seed)
        fixed, exceptions = _encode(scales)
        return scales, [
            (prefix + ".weight_scale" + CSF_FIXED_SUFFIX, fixed),
            (prefix + ".weight_scale" + CSF_EXCEPTIONS_SUFFIX, exceptions),
        ]

    def test_pairs_decode_in_either_order_across_shards_and_others_pass(self):
        s0, p0 = self._pair("model.layers.0.mlp.experts.0.gate_proj", 1)
        s1, p1 = self._pair("model.layers.0.mlp.experts.1.up_proj", 2)
        other = ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.ones(2))
        stream = [p0[1], other, p1[0], ("x.weight_scale_2", torch.ones(1)), p0[0], p1[1]]
        out = list(decode_nvfp4_csf_weights(iter(stream)))
        names = [n for n, _ in out]
        self.assertEqual(
            names,
            [
                other[0],
                "x.weight_scale_2",
                "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
                "model.layers.0.mlp.experts.1.up_proj.weight_scale",
            ],
        )
        self.assertIs(out[0][1], other[1])
        self.assertTrue(torch.equal(out[2][1].view(torch.uint8), s0))
        self.assertTrue(torch.equal(out[3][1].view(torch.uint8), s1))

    def test_buffered_half_survives_its_source_being_overwritten(self):
        scales, pair = self._pair("e.down_proj", 3)
        source = pair[0][1].clone()

        def stream():
            yield pair[0][0], source
            source.zero_()  # the shard's buffer is reused after the yield
            yield pair[1]

        out = list(decode_nvfp4_csf_weights(stream()))
        self.assertTrue(torch.equal(out[0][1].view(torch.uint8), scales))

    def test_checkpoint_without_csf_passes_unchanged(self):
        stream = [("a.weight", torch.zeros(1)), ("a.weight_scale", torch.ones(1))]
        out = list(decode_nvfp4_csf_weights(iter(stream)))
        self.assertEqual([n for n, _ in out], ["a.weight", "a.weight_scale"])

    def test_unpaired_half_raises(self):
        _, pair = self._pair("e.gate_proj", 4)
        with self.assertRaisesRegex(ValueError, "missing the other half"):
            list(decode_nvfp4_csf_weights(iter(pair[:1])))

    def test_duplicate_half_raises(self):
        _, pair = self._pair("e.gate_proj", 5)
        with self.assertRaisesRegex(ValueError, "appears twice"):
            list(decode_nvfp4_csf_weights(iter([pair[0], pair[0]])))


class TestMixedPrecisionConfigEncoding(CustomTestCase):
    def _config(self, **expert_extra):
        experts = {"quant_algo": "NVFP4", "group_size": 16, **expert_extra}
        return {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "ignore": ["lm_head"],
            "quantized_layers": {
                "model.language_model.layers.0.mlp.experts": experts,
                "model.language_model.layers.0.self_attn.q_proj": {
                    "quant_algo": "MXFP8",
                    "group_size": 32,
                },
            },
        }

    def test_csf_on_nvfp4_is_accepted(self):
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptMixedPrecisionConfig,
        )

        cfg = ModelOptMixedPrecisionConfig.from_config(
            self._config(weight_scale_encoding="csf")
        )
        self.assertEqual(
            cfg.resolve_quant_algo("model.language_model.layers.0.mlp.experts"), "NVFP4"
        )

    def test_unknown_encoding_or_csf_off_nvfp4_is_refused(self):
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptMixedPrecisionConfig,
        )

        with self.assertRaisesRegex(ValueError, "weight_scale_encoding"):
            ModelOptMixedPrecisionConfig.from_config(
                self._config(weight_scale_encoding="zstd")
            )
        bad = self._config()
        bad["quantized_layers"]["model.language_model.layers.0.self_attn.q_proj"][
            "weight_scale_encoding"
        ] = "csf"
        with self.assertRaisesRegex(ValueError, "MXFP8"):
            ModelOptMixedPrecisionConfig.from_config(bad)


if __name__ == "__main__":
    unittest.main()
