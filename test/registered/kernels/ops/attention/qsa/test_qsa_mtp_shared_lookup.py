"""QSAMTPSharedSparseIndices.lookup returns each row's valid entries first, the tail included.

The draft decode steps' consumer (_qsa_fused_kv_prepare's non-expanded path + trtllm decode) reads a row's first
COUNTS columns, COUNTS = its valid entries, so a -1 ahead of a valid entry drops that entry from the attention.
Two-sided: the must-fire arm checks the old layout (tail after the -1 padding) would lose the tail.
"""

import unittest
import unittest.mock

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

    def test_no_tail_mode(self):
        # SGLANG_QSA_MTP_DRAFT_TAIL=0: the captured selection only, compacted, no drafted positions
        with unittest.mock.patch.dict("os.environ", {"SGLANG_QSA_MTP_DRAFT_TAIL": "0"}):
            st = self._state()
        cap = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1, 4, 5]], dtype=torch.int32)
        st.capture(cap, torch.tensor([1]), torch.tensor([6]), 0)
        out = st.lookup(torch.tensor([1]), torch.tensor([7]), 0)[0]
        self.assertEqual(_prefix_read(out), [0, 1, 2, 3, 4, 5])
        self.assertTrue(bool((out[6:] == -1).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_triton_valid_first_equals_stable_argsort(self):
        # SGLANG_QSA_VALID_FIRST_TRITON: the Triton compaction against the stable-argsort path, byte-equal, on the
        # served width (512 + 3 + 4) and odd widths, rows 1..9, hole rates 0..100 %.
        import sglang.srt.layers.attention.qwen_sparse_attn_backend as be

        g = torch.Generator(device="cpu").manual_seed(0)
        for width in (519, 10, 513, 1024, 1):
            for rows in (1, 2, 4, 9):
                for p_hole in (0.0, 0.3, 0.9, 1.0):
                    x = torch.randint(0, 1 << 20, (rows, width), generator=g, dtype=torch.int32)
                    x[torch.rand((rows, width), generator=g) < p_hole] = -1
                    x = x.cuda()
                    with unittest.mock.patch.object(be, "_QSA_VALID_FIRST_TRITON", False):
                        ref = be.qsa_valid_first(x)
                    with unittest.mock.patch.object(be, "_QSA_VALID_FIRST_TRITON", True):
                        got = be.qsa_valid_first(x)
                    self.assertTrue(torch.equal(ref, got), f"width {width} rows {rows} p_hole {p_hole}")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_must_fire_triton_compare_sees_order(self):
        # the equality above can fail: an order-scrambled compaction (valid entries reversed) differs from it
        import sglang.srt.layers.attention.qwen_sparse_attn_backend as be

        x = torch.tensor([[5, -1, 7, 9, -1, 2]], dtype=torch.int32, device="cuda")
        with unittest.mock.patch.object(be, "_QSA_VALID_FIRST_TRITON", True):
            got = be.qsa_valid_first(x)
        self.assertEqual(got.tolist(), [[5, 7, 9, 2, -1, -1]])
        self.assertFalse(torch.equal(got, torch.tensor([[2, 9, 7, 5, -1, -1]], dtype=torch.int32, device="cuda")))

    def test_must_fire_old_layout_loses_tail(self):
        # the layout before the fix: the tail at the row's end, after the padding
        old = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1])
        self.assertNotIn(6, _prefix_read(old))


if __name__ == "__main__":
    unittest.main()
