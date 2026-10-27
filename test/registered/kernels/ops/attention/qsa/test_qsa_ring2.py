"""The QSA pending index-key ring through many verify forwards: every completed group's compressed key is built
from its own four accepted members.

A verify window of W rows starting at position L completes the groups ending inside it; with W = 5 and L % 4 == 3
that is two groups in one forward. The test replays a request's ring with the real slot builders
(build_pending_ring_slots / build_pending_store_slots / build_group_ring_slots) over many forwards with random
acceptance -- rejected rows carry wrong keys, so a group compressed from them must be compressed again from the
accepted ones -- and checks each group's last compress read exactly its members' accepted keys.

Two-sided: a one-group ring (span 4) corrupts groups at W = 5 AND at W = 4, production's shape. The rows a window
writes past a group close are re-stored over open-group members, and a rejected closing row makes the next forward
compress that group again from a rejected key. A two-group ring (SGLANG_QSA_RING2=1, span 8) corrupts none at W = 4
or 5. Further arms: the CUDA-graph row kernel writes the host rule's slots, and the fork's own
QSAIndexer.update_key_state_and_compress shows the two-forward failure on the real code.
"""

import os
import random
import unittest
import unittest.mock
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    build_pending_store_slots,
    qsa_max_verify_rows,
    qsa_ring_span,
)

RATIO = 4


def _replay(window, ring2, steps=400, n_req=3, seed=0):
    """Returns the number of groups whose final compressed members differ from their accepted keys."""
    rnd = random.Random(seed)
    env = {"SGLANG_QSA_RING2": "1" if ring2 else "0"}
    with unittest.mock.patch.dict(os.environ, env):
        span = qsa_ring_span(RATIO)
        ring = torch.full(((n_req + 1) * span,), -1, dtype=torch.long)  # request slot 0 = the inert dump
        L = [rnd.randrange(0, 4) for _ in range(n_req)]  # each request's committed length (positions < L final)
        last = {}  # (request, group) -> the member keys of its last compress
        truth = lambda p: 10 * p + 1  # noqa: E731  the accepted key of position p
        for r in range(n_req):  # the prefill stored the open group's members before L
            for p in range(RATIO * (L[r] // RATIO), L[r]):
                ring[(r + 1) * span + p % span] = truth(p)
        for step in range(steps):
            reqs = torch.arange(1, n_req + 1)
            tok = torch.arange(n_req).repeat_interleave(window)
            pos = torch.tensor([L[r] + j for r in range(n_req) for j in range(window)])
            acc = [rnd.randrange(1, window + 1) for _ in range(n_req)]
            keys = torch.tensor([truth(L[r] + j) if j < acc[r] else -(1000 * step + 10 * (L[r] + j) + 7)
                                 for r in range(n_req) for j in range(window)])
            lens = pos + 1  # a paged verify row's length
            ring_slots = build_pending_ring_slots(token_to_batch_idx=tok, req_pool_indices=reqs,
                                                  sequence_lengths=lens, logical_positions=pos,
                                                  compress_ratio=RATIO, is_extend=False)
            store = build_pending_store_slots(pending_ring_slots=ring_slots, token_to_batch_idx=tok,
                                              req_pool_indices=reqs, logical_positions=pos, compress_ratio=RATIO)
            ring[store] = keys
            ends = (lens % RATIO == 0).nonzero().flatten()  # rows completing a group (the paged write plan)
            if ends.numel():
                locs = build_group_ring_slots(req_pool_indices=reqs, group_end_positions=pos[ends],
                                              sequence_ids=tok[ends], compress_ratio=RATIO)
                for e, row in zip(ends.tolist(), locs):
                    last[(int(tok[e]), int(pos[e]) // RATIO)] = ring[row].tolist()
            ring[ring_slots] = keys  # the re-store after the compress
            for r in range(n_req):
                L[r] += acc[r]
        bad = 0
        for (r, g), members in last.items():
            if (g + 1) * RATIO <= L[r]:  # a group whose four positions are all accepted
                bad += int(members != [truth(RATIO * g + k) for k in range(RATIO)])
        return bad, len(last)


class TestQSARing2(unittest.TestCase):
    def test_limits(self):
        with unittest.mock.patch.dict(os.environ, {"SGLANG_QSA_RING2": "0"}):
            self.assertEqual((qsa_ring_span(RATIO), qsa_max_verify_rows(RATIO)), (4, 4))
        with unittest.mock.patch.dict(os.environ, {"SGLANG_QSA_RING2": "1"}):
            self.assertEqual((qsa_ring_span(RATIO), qsa_max_verify_rows(RATIO)), (8, 5))

    def test_one_group_ring_corrupts_four_rows(self):
        """Measured 2026-10-27: the one-group ring corrupts groups at production's own 4-row window too. A row
        past a close is parked during the compress and then re-stored at its ring slot, which is the slot of an
        open-group member (position p + 4 shares p's). When the closing row was rejected, the next forward
        compresses that group again and reads the rejected row's key. Must-fire; the two-group ring removes it."""
        bad, n = _replay(window=4, ring2=False)
        self.assertGreater(n, 100)
        self.assertGreater(bad, 0)

    def test_one_group_ring_breaks_at_five_rows(self):  # the must-fire arm
        bad, n = _replay(window=5, ring2=False)
        self.assertGreater(bad, 0, "a one-group ring must corrupt a 5-row window's second group")

    def test_two_group_ring_holds_five_rows(self):
        for seed in range(5):
            bad, n = _replay(window=5, ring2=True, seed=seed)
            self.assertGreater(n, 100)
            self.assertEqual(bad, 0, f"seed {seed}")

    def test_two_group_ring_holds_four_rows(self):
        bad, n = _replay(window=4, ring2=True)
        self.assertEqual(bad, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA")
    def test_graph_kernel_matches_host_rule(self):
        """The decode-graph row kernel (MODE 1, the verify layout) writes the host rule's ring / store / member
        slots, for both spans."""
        import triton

        from sglang.srt.layers.attention.qsa.graph_metadata import _qsa_graph_row_metadata_kernel

        for ring2, window in ((False, 4), (True, 5)):
            with unittest.mock.patch.dict(os.environ, {"SGLANG_QSA_RING2": "1" if ring2 else "0"}):
                span = qsa_ring_span(RATIO)
                bs, dev, page = 3, "cuda", 64
                base = torch.tensor([7, 12, 33], dtype=torch.int32, device=dev)
                reqs = torch.tensor([1, 2, 3], dtype=torch.int32, device=dev)
                rows = bs * window
                r2t = torch.arange(4 * 256, dtype=torch.int32, device=dev).view(4, 256)
                z32 = lambda n: torch.zeros(n, dtype=torch.int32, device=dev)  # noqa: E731
                z64 = lambda n: torch.zeros(n, dtype=torch.int64, device=dev)  # noqa: E731
                row_len, row_req, comp, wl = z32(rows), z32(rows), z32(rows), z32(rows)
                ptab = torch.zeros(rows, 4, dtype=torch.int32, device=dev)
                lpos, state, store, ring_locs, prefix = z32(rows), z64(rows), z64(rows), z32(rows * RATIO), z32(rows)
                _qsa_graph_row_metadata_kernel[(rows, 1)](
                    row_len, row_req, comp, wl, ptab, lpos, state, store, ring_locs, r2t, r2t.stride(0), 4,
                    RATIO=RATIO, FULL_PAGE=page, PAGE_BLOCK=128, SPAN=span, seq_lens_ptr=base, req_pool_ptr=reqs,
                    row_prefix_lens_ptr=prefix, bs=bs, num_padding=0, extend_len=window, seq_offset=0, MODE=1,
                    num_warps=1)
                tok = torch.arange(bs, device=dev).repeat_interleave(window)
                pos = lpos.long()
                host_ring = build_pending_ring_slots(token_to_batch_idx=tok, req_pool_indices=reqs.long(),
                                                     sequence_lengths=row_len.long(), logical_positions=pos,
                                                     compress_ratio=RATIO, is_extend=False)
                host_store = build_pending_store_slots(pending_ring_slots=host_ring, token_to_batch_idx=tok,
                                                       req_pool_indices=reqs.long(), logical_positions=pos,
                                                       compress_ratio=RATIO)
                host_members = build_group_ring_slots(req_pool_indices=reqs.long(), group_end_positions=pos,
                                                      sequence_ids=tok, compress_ratio=RATIO)
                self.assertTrue(torch.equal(state, host_ring), f"ring2={ring2}")
                self.assertTrue(torch.equal(store, host_store), f"ring2={ring2}")
                self.assertTrue(torch.equal(ring_locs.view(rows, RATIO).long(), host_members), f"ring2={ring2}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class TestQSARing2RealIndexer(unittest.TestCase):
    """The fork's own QSAIndexer.update_key_state_and_compress (test_qsa_indexer's fixtures), across TWO verify
    forwards: forward 1 rows 41..44 with only 41, 42 accepted (43, 44 carry rejected keys); forward 2 rows 43..46,
    whose row 43 closes group 40..43 again. Group 40..43's compressed key must equal the compress of its four
    accepted keys. One-group ring: forward 1's re-store puts rejected row 44 at position 40's slot (44 % 4 = 40 %
    4), so forward 2 compresses a rejected key (the must-fire arm, expected to differ); two-group ring: equal."""

    def setUp(self):
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def _run(self, ring2):
        import sys

        try:
            import pytest  # noqa: F401
        except ImportError:  # test_qsa_indexer's module scope uses only these decorators (its fixtures, not here)
            stub = SimpleNamespace(fixture=lambda *a, **k: (lambda f: f),
                                   mark=SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f)))
            sys.modules["pytest"] = stub
        import test_qsa_indexer as ti

        with unittest.mock.patch.dict(os.environ, {"SGLANG_QSA_RING2": "1" if ring2 else "0"}):
            span = qsa_ring_span(RATIO)
            dev, dt = torch.device("cuda"), torch.bfloat16
            indexer = ti._make_indexer(ti._make_rotary([24, 20, 20], True, dev, dt), dev, dt)
            req, write_loc = 2, 37
            torch.manual_seed(0)
            true_k = {p: torch.randn(1, ti.HEAD_DIM, device=dev, dtype=dt) for p in range(40, 48)}
            bogus = {p: torch.randn(1, ti.HEAD_DIM, device=dev, dtype=dt) for p in range(40, 48)}
            pool = ti.FakePool((req + 1) * span, 64, dev, dt)
            pool.set_qsa_key_state_buffer(0, torch.tensor([req * span + 40 % span], device=dev), true_k[40])
            pool.set_qsa_rope_position_buffer(torch.tensor([req * span + 40 % span], device=dev),
                                              torch.tensor([40], device=dev))

            def forward(start, accepted):
                pos = torch.arange(start, start + 4, device=dev)
                keys = torch.cat([true_k[p] if p < start + accepted else bogus[p] for p in pos.tolist()])
                keys = keys.view(4, 1, ti.HEAD_DIM)
                tok = torch.zeros(4, dtype=torch.int32, device=dev)
                reqs = torch.tensor([req], dtype=torch.int32, device=dev)
                ring = build_pending_ring_slots(token_to_batch_idx=tok, req_pool_indices=reqs, sequence_lengths=pos + 1,
                                                logical_positions=pos, compress_ratio=RATIO, is_extend=False)
                store = build_pending_store_slots(pending_ring_slots=ring, token_to_batch_idx=tok,
                                                  req_pool_indices=reqs, logical_positions=pos, compress_ratio=RATIO)
                group_locs = build_group_ring_slots(req_pool_indices=reqs, group_end_positions=pos,
                                                    sequence_ids=tok.long(), compress_ratio=RATIO).to(torch.int32)
                write_locs = torch.where(pos % RATIO == RATIO - 1, torch.full_like(pos, write_loc),
                                         torch.zeros_like(pos)).to(torch.int32)
                meta = SimpleNamespace(token_to_kv_pool=pool, compress_member_rows=None, is_cuda_graph=True,
                                       graph_write_locs=write_locs, graph_ring_group_locs=group_locs)
                indexer.update_key_state_and_compress(keys, pos, pos.unsqueeze(0).expand(3, -1).contiguous(), meta,
                                                      state_slots=ring, state_stored=False, store_slots=store)

            forward(41, 2)  # rows 41, 42 accepted; 43, 44 rejected
            forward(43, 4)  # row 43 now carries the accepted key and closes group 40..43 again
            got = pool.compressed[write_loc].clone()
            ref = ti.FakePool((req + 1) * span, 64, dev, dt)
            slots = torch.tensor([req * span + p % span for p in range(40, 44)], device=dev)
            ref.set_qsa_key_state_buffer(0, slots, torch.cat([true_k[p] for p in range(40, 44)]).view(4, 1, -1))
            ref.set_qsa_rope_position_buffer(slots, torch.arange(40, 44, device=dev))
            ti._eager_compress_reference(indexer, ref, slots.view(1, RATIO).to(torch.int32),
                                         torch.tensor([write_loc], dtype=torch.int32, device=dev))
            return (got.float() - ref.compressed[write_loc].float()).abs().max().item()

    def test_one_group_ring_compresses_a_rejected_key(self):  # must-fire: the production ring
        self.assertGreater(self._run(False), 0.05)

    def test_two_group_ring_compresses_the_accepted_keys(self):
        self.assertLess(self._run(True), 0.02)


if __name__ == "__main__":
    unittest.main()
