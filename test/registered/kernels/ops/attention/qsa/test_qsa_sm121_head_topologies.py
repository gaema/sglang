"""SM121 packed-QSA decode at every tensor-parallel head split of Qwen3.8-Flash-Next.

The model has 24 query heads over 2 KV heads (12:1 GQA). TP1 runs 24Q/2KV, TP2 12Q/1KV, and TP4
6Q/1KV (the two KV heads are replicated). The kernel groups the query heads that share a KV head in
one CTA and masks rows past NUM_Q_HEADS // NUM_KV_HEADS, so every split is checked here against a
float32 reference over the same packed varlen rows, including the multi-split merge.
"""

import unittest

import torch

from sglang.kernels.kda_kernels.qwen38_qsa_sm121 import can_use_qwen38_qsa_sm121
from sglang.kernels.ops.attention import qwen38_qsa_sm121_varlen

HEAD_DIM = 256


def _reference(q, k, v, cu_k, scale):
    out = torch.empty_like(q)
    group = q.shape[1] // k.shape[1]
    for b in range(q.shape[0]):
        lo, hi = int(cu_k[b]), int(cu_k[b + 1])
        keys = k[lo:hi].float().repeat_interleave(group, dim=1)  # [n, Hq, D]
        vals = v[lo:hi].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[b].float(), keys) * scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hn,nhd->hd", probs, vals).to(q.dtype)
    return out


def _case(num_q, num_kv, lengths, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(lengths)
    q = torch.randn(batch, num_q, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=gen)
    total = sum(lengths)
    k = torch.randn(total, num_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(total, num_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=gen)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    cu_k = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_k[1:] = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    return q, k, v, cu_q, cu_k, max(lengths)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1),
    "SM121-only kernel",
)
class TestQsaSm121HeadTopologies(unittest.TestCase):
    # batch 1 at 2051 rows takes 8 splits; batch 2 at 600 rows 4 splits; batch 16 one split.
    LENGTHS = ([2051], [600, 37], [1] + [129] * 7 + [2000] * 8, list(range(1, 129)))

    def _check(self, num_q, num_kv):
        scale = HEAD_DIM ** -0.5
        for i, lengths in enumerate(self.LENGTHS):
            q, k, v, cu_q, cu_k, max_k = _case(num_q, num_kv, lengths, 1000 + i)
            self.assertTrue(can_use_qwen38_qsa_sm121(q, k, v, cu_q, cu_k, max_k))
            got = qwen38_qsa_sm121_varlen(q, k, v, cu_q, cu_k, max_seqlen_q=1,
                                          max_seqlen_k=max_k, softmax_scale=scale)
            want = _reference(q, k, v, cu_k, scale)
            err = (got.float() - want.float()).abs().max().item()
            self.assertLess(err, 2e-2, f"{num_q}Q/{num_kv}KV lengths#{i} max abs err {err}")

    def test_tp1_24q_2kv(self):
        self._check(24, 2)

    def test_tp2_12q_1kv(self):
        self._check(12, 1)

    def test_tp4_6q_1kv(self):
        self._check(6, 1)

    def test_unqualified_split_still_refused(self):
        q, k, v, cu_q, cu_k, max_k = _case(8, 1, [16], 7)
        self.assertFalse(can_use_qwen38_qsa_sm121(q, k, v, cu_q, cu_k, max_k))


if __name__ == "__main__":
    unittest.main()
