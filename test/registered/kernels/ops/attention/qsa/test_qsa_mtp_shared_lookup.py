"""QSAMTPSharedSparseIndices.lookup returns each row's valid entries first, the tail included.

The draft decode steps' consumer (_qsa_fused_kv_prepare's non-expanded path + trtllm decode) reads a row's first
COUNTS columns, COUNTS = its valid entries, so a -1 ahead of a valid entry drops that entry from the attention.
Two-sided: the must-fire arm checks the old layout (tail after the -1 padding) would lose the tail.
"""

import unittest

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QSAMTPSharedSparseIndices


def _prefix_read(row):
    """The consumer's read: the first COUNTS columns, COUNTS = the valid entries."""
    cnt = int((row >= 0).sum())
    return row[:cnt].tolist()


class TestMTPSharedLookup(unittest.TestCase):
    def _state(self, token_topk=10, tail_width=4):
        return QSAMTPSharedSparseIndices(
            layer_ids=[0], num_requests=2, token_topk=token_topk, tail_width=tail_width, device="cpu"
        )

    def test_valid_first_with_tail(self):
        st = self._state()
        # captured: blocks 0 (tokens 0-3), padding, then the anchor's tail 4, 5 at the expansion's last columns
        cap = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1, 4, 5]], dtype=torch.int32)
        st.capture(cap, torch.tensor([1]), torch.tensor([6]), 0)
        out = st.lookup(torch.tensor([1]), torch.tensor([7]), 0)[0]
        self.assertEqual(_prefix_read(out), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(bool((out[8:] == -1).all()))

    def test_full_row_unchanged_order(self):
        st = self._state(token_topk=4, tail_width=2)
        cap = torch.tensor([[9, 3, 7, 1]], dtype=torch.int32)
        st.capture(cap, torch.tensor([1]), torch.tensor([10]), 0)
        out = st.lookup(torch.tensor([1]), torch.tensor([10]), 0)[0]
        self.assertEqual(out.tolist(), [9, 3, 7, 1, 10, -1])

    def test_must_fire_old_layout_loses_tail(self):
        # the layout before the fix: the tail at the row's end, after the padding
        old = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1])
        self.assertNotIn(6, _prefix_read(old))


if __name__ == "__main__":
    unittest.main()
