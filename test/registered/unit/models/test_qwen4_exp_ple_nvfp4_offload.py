"""The pinned-host NVFP4 PLE gather returns rows bit-identical to the on-device ModelOptNvFp4EmbeddingMethod gather,
zero rows for ids outside the rank's vocab shard, and works under CUDA graph capture."""

import types
import unittest

import torch

from sglang.srt.layers.quantization.modelopt_quant import _E2M1_LUT, ModelOptNvFp4EmbeddingMethod
from sglang.srt.models.qwen4_exp import _gather_ple_nvfp4_from_pinned_kernel

ROWS, DIM, GROUP = 4096, 160, 16


def _tables(seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (ROWS, DIM // 2), generator=g, dtype=torch.uint8)
    scale = (torch.rand(ROWS, DIM // GROUP, generator=g) * 4 + 0.01).to(torch.float8_e4m3fn)
    return packed, scale, torch.tensor([0.0123], dtype=torch.float32)


def _device_reference(packed, scale, scale_2, ids_local):
    layer = types.SimpleNamespace(weight=packed.cuda(), weight_scale=scale.cuda(), weight_scale_2=scale_2.cuda(),
                                  e2m1_lut=torch.tensor(_E2M1_LUT, dtype=torch.float32, device="cuda"))
    method = ModelOptNvFp4EmbeddingMethod.__new__(ModelOptNvFp4EmbeddingMethod)
    method.quant_config = types.SimpleNamespace(group_size=GROUP)
    method.params_dtype = torch.bfloat16
    return method.embedding(layer, ids_local.cuda())


_LUT = {}


def _lut():
    if "cuda" not in _LUT:  # built once, outside any graph capture
        _LUT["cuda"] = torch.tensor(_E2M1_LUT, dtype=torch.float32, device="cuda")
    return _LUT["cuda"]


def _offload_gather(packed_host, scale_host, scale_2, ids, vocab_start, vocab_end, out):
    _gather_ple_nvfp4_from_pinned_kernel[(ids.numel(),)](
        packed_host.data_ptr(), scale_host.data_ptr(), scale_2, _lut(),
        ids, out, embedding_dim=DIM, tp_vocab_start=vocab_start, tp_vocab_end=vocab_end,
        GROUP=GROUP, BLOCK_D=256)
    return out


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestPleNvfp4Offload(unittest.TestCase):
    def test_bit_identical_to_device_gather(self):
        packed, scale, scale_2 = _tables(1)
        packed_host, scale_host = packed.pin_memory(), scale.pin_memory()
        vocab_start = 10000
        ids_local = torch.randint(0, ROWS, (777,))
        ids = (ids_local + vocab_start).cuda()
        out = torch.empty(777, DIM, dtype=torch.bfloat16, device="cuda")
        got = _offload_gather(packed_host, scale_host, scale_2.cuda(), ids, vocab_start, vocab_start + ROWS, out)
        want = _device_reference(packed, scale, scale_2, ids_local)
        self.assertTrue(torch.equal(got.view(torch.int16), want.view(torch.int16)))

    def test_out_of_shard_ids_are_zero(self):
        packed, scale, scale_2 = _tables(2)
        packed_host, scale_host = packed.pin_memory(), scale.pin_memory()
        ids = torch.tensor([0, 9999, 10000 + ROWS, 10005], device="cuda")
        out = torch.full((4, DIM), 7.0, dtype=torch.bfloat16, device="cuda")
        got = _offload_gather(packed_host, scale_host, scale_2.cuda(), ids, 10000, 10000 + ROWS, out)
        self.assertTrue(torch.all(got[:3] == 0))
        want = _device_reference(packed, scale, scale_2, torch.tensor([5]))
        self.assertTrue(torch.equal(got[3:].view(torch.int16), want.view(torch.int16)))

    def test_replays_under_cuda_graph(self):
        packed, scale, scale_2 = _tables(3)
        packed_host, scale_host = packed.pin_memory(), scale.pin_memory()
        ids = torch.zeros(64, dtype=torch.long, device="cuda")
        out = torch.empty(64, DIM, dtype=torch.bfloat16, device="cuda")
        s2 = scale_2.cuda()
        _offload_gather(packed_host, scale_host, s2, ids, 0, ROWS, out)  # compile outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _offload_gather(packed_host, scale_host, s2, ids, 0, ROWS, out)
        new_local = torch.randint(0, ROWS, (64,))
        ids.copy_(new_local.cuda())
        graph.replay()
        torch.cuda.synchronize()
        want = _device_reference(packed, scale, scale_2, new_local)
        self.assertTrue(torch.equal(out.view(torch.int16), want.view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
