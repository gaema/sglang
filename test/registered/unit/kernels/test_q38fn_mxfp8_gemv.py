"""The small-M MXFP8 GEMV matches F.linear on the exactly dequantized weight, is run-to-run repeatable, refuses what it
cannot serve, and replays under a CUDA graph."""

import unittest

import torch

from sglang.kernels.ops.gemm.q38fn_mxfp8_gemv import MAX_M, q38fn_mxfp8_gemv, q38fn_mxfp8_gemv_ok
from sglang.srt.layers.quantization.fp8 import dequantize_mxfp8_weight

SHAPES = ((3072, 2560), (2560, 3072), (6144, 2560), (2560, 1024), (512, 2560))


def _layer(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    s = torch.randint(118, 128, (n, k // 32), device="cuda", generator=g, dtype=torch.int32).to(torch.uint8)
    return w, s


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestQ38fnMxfp8Gemv(unittest.TestCase):
    def test_matches_linear_on_dequantized_weight(self):
        for i, (n, k) in enumerate(SHAPES):
            w, s = _layer(n, k, i)
            wd = dequantize_mxfp8_weight(w, s)
            for m in (m for m in (1, 3, 6, 9, 12, 16, 32) if m <= MAX_M):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(q38fn_mxfp8_gemv_ok(x, w, s))
                got = q38fn_mxfp8_gemv(x, w, s).float()
                want = torch.nn.functional.linear(x.float(), wd.float())
                torch.testing.assert_close(got, want.to(torch.bfloat16).float(), rtol=2 * 2.0 ** -7,
                                           atol=1e-2 * want.abs().max().item(), msg=f"n={n} k={k} m={m}")

    def test_repeatable(self):
        w, s = _layer(3072, 2560, 7)
        x = torch.randn(6, 2560, device="cuda", dtype=torch.bfloat16)
        a = q38fn_mxfp8_gemv(x, w, s)
        self.assertTrue(all(torch.equal(a, q38fn_mxfp8_gemv(x, w, s)) for _ in range(5)))

    def test_refuses(self):
        w, s = _layer(512, 2560, 3)
        self.assertFalse(q38fn_mxfp8_gemv_ok(torch.randn(MAX_M + 1, 2560, device="cuda", dtype=torch.bfloat16), w, s))
        self.assertFalse(q38fn_mxfp8_gemv_ok(torch.randn(4, 2560, device="cuda", dtype=torch.float16), w, s))
        self.assertFalse(q38fn_mxfp8_gemv_ok(torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16),
                                             w.to(torch.bfloat16), s))

    def test_cuda_graph_replay(self):
        w, s = _layer(2560, 3072, 5)
        x = torch.randn(3, 3072, device="cuda", dtype=torch.bfloat16)
        q38fn_mxfp8_gemv(x, w, s)  # compile outside capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = q38fn_mxfp8_gemv(x, w, s)
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, q38fn_mxfp8_gemv(x, w, s)))


if __name__ == "__main__":
    unittest.main()
