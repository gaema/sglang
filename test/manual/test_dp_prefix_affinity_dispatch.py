"""Unit tests for the DP `prefix_affinity` dispatcher's spread-vs-wait trade.

No GPU, no sockets, no server: the real `DataParallelController.prefix_affinity_scheduler`
is driven with a duck-typed `self` and a patched `sock_send`, so these exercise the
shipped decision code rather than a re-implementation of it.

The property under test (added 2026-09-20): a WAIT behind a rank's prefill backlog and
a SPREAD onto a rank that must re-prefill are NOT the same cost. The wait is work the
machine already committed to; the spread is new work, and under DP attention it throttles
decode on every rank. `SGLANG_DP_PREFIX_AFFINITY_REPREFILL_WEIGHT` scales the spread term.

    python3 test_dp_prefix_affinity_dispatch.py

🔴 The legacy-equivalence arm is the load-bearing one: WEIGHT=1.0 must reproduce the old
chooser EXACTLY, because that is the documented rollback. It is checked against an
INDEPENDENT re-implementation of the pre-change formula, not against the new code.
"""
import random
import sys
import unittest
from unittest import mock

from sglang.srt.managers import data_parallel_controller as dpc


class FakeIndex:
    def __init__(self, per_rank, chunk=2048):
        self.chunk = chunk
        self.max_entries = 65536
        self._per_rank = per_rank
        self.recorded = []

    def fingerprints(self, input_ids):
        return [b"fp"] * (len(input_ids) // self.chunk)

    def lookup(self, fps):
        return dict(self._per_rank)

    def record(self, fps, rank):
        self.recorded.append(rank)

    def __len__(self):
        return len(self._per_rank)


class FakeBudget:
    def __init__(self, backlog):
        n = len(backlog)
        self.prefill_backlog = dict(backlog)
        self.total_requests = {i: 0 for i in range(n)}
        self.total_tokens = {i: 0 for i in range(n)}
        self.pending_dispatches = {i: [] for i in range(n)}


class FakeReq:
    def __init__(self, n):
        self.input_ids = list(range(n))


def dispatch(matched_tokens, backlog, n, weight, chunk=2048, sticky=16384, rr=0):
    """Run the REAL scheduler; return (target_rank, stats)."""
    ranks = sorted(matched_tokens)
    per_rank = {i: matched_tokens[i] // chunk for i in ranks}
    ctl = mock.Mock()
    ctl.maybe_external_dp_rank_routing = lambda req: False
    ctl.prefix_affinity = FakeIndex(per_rank, chunk=chunk)
    ctl.dp_budget = FakeBudget(backlog)
    ctl._active_workers = ranks
    ctl.status = {i: True for i in ranks}
    ctl.workers = {i: object() for i in ranks}
    ctl.prefix_affinity_sticky_tokens = sticky
    ctl.prefix_affinity_reprefill_weight = weight
    ctl.round_robin_counter = rr
    ctl._prefix_affinity_stats = {"hit": 0, "miss": 0, "spread": 0, "spread_tokens": 0}

    sent = {}
    with mock.patch.object(dpc, "sock_send", lambda w, r: sent.setdefault("w", w)):
        dpc.DataParallelController.prefix_affinity_scheduler(ctl, FakeReq(n))
    target = next(i for i in ranks if ctl.workers[i] is sent["w"])
    return target, ctl._prefix_affinity_stats


def legacy_choice(matched, backlog, n, sticky=16384, rr=0):
    """INDEPENDENT re-implementation of the pre-2026-09-20 chooser."""
    active = sorted(matched)
    best_match = max(matched.values()) if matched else 0
    cost = {i: backlog[i] + (n - matched[i]) for i in active}
    best_rank = min(active, key=lambda i: (cost[i], -matched[i]))
    near = [
        i for i in active
        if backlog[i] == backlog[best_rank] and cost[i] - cost[best_rank] < sticky
    ]
    if best_match < sticky and len(near) > 1:
        return near[rr % len(near)]
    return best_rank


class LegacyEquivalence(unittest.TestCase):
    def test_weight_one_reproduces_the_old_chooser(self):
        """MUST-ACCEPT: WEIGHT=1.0 is the documented exact rollback."""
        rng = random.Random(20260920)
        for _ in range(400):
            chunk = 2048
            matched = {i: rng.randrange(0, 30) * chunk for i in range(2)}
            backlog = {i: rng.randrange(0, 200000) for i in range(2)}
            n = max(max(matched.values()), 1) + rng.randrange(0, 200000)
            rr = rng.randrange(0, 4)
            got, _ = dispatch(matched, backlog, n, weight=1.0, rr=rr)
            want = legacy_choice(matched, backlog, n, rr=rr)
            self.assertEqual(got, want, f"matched={matched} backlog={backlog} n={n}")

    def test_the_harness_can_see_a_difference(self):
        """MUST-FIRE: if no scenario ever separated the two weights, the suite above
        would pass on a no-op change. At least one random draw must diverge."""
        rng = random.Random(7)
        diverged = 0
        for _ in range(400):
            chunk = 2048
            matched = {i: rng.randrange(0, 30) * chunk for i in range(2)}
            backlog = {i: rng.randrange(0, 200000) for i in range(2)}
            n = max(max(matched.values()), 1) + rng.randrange(0, 200000)
            if dispatch(matched, backlog, n, 1.0)[0] != dispatch(matched, backlog, n, 4.0)[0]:
                diverged += 1
        self.assertGreater(diverged, 0, "weight has no effect on any scenario -- test is blind")


class SpreadIsPricedAgainstWait(unittest.TestCase):
    def test_costly_spread_is_refused_when_weighted(self):
        """rank0 holds a 100k prefix behind a 150k backlog; rank1 holds nothing and is
        idle. Old rule: waiting 150k > re-prefilling 100k, so it spreads and burns 100k
        tokens of prefill. Weighted: that waste counts double and the wait wins."""
        matched = {0: 100352, 1: 0}          # 49 chunks x 2048
        backlog = {0: 150000, 1: 0}
        n = 120000
        self.assertEqual(dispatch(matched, backlog, n, weight=1.0)[0], 1)
        self.assertEqual(dispatch(matched, backlog, n, weight=2.0)[0], 0)

    def test_a_spread_still_happens_when_the_wait_is_absurd(self):
        """The weight is a price, not a ban: a big enough backlog still outweighs it."""
        matched = {0: 100352, 1: 0}
        backlog = {0: 5_000_000, 1: 0}
        self.assertEqual(dispatch(matched, backlog, 120000, weight=2.0)[0], 1)

    def test_free_spread_is_unaffected(self):
        """Both ranks hold the same prefix: the waste term is 0 either way, so the
        idler must win at any weight -- the fix must not make the dispatcher sticky
        when stickiness buys nothing."""
        matched = {0: 49152, 1: 49152}
        backlog = {0: 90000, 1: 10000}
        for w in (1.0, 2.0, 8.0):
            self.assertEqual(dispatch(matched, backlog, 120000, weight=w)[0], 1, f"w={w}")

    def test_stats_record_the_spread_it_did_take(self):
        _, stats = dispatch({0: 100352, 1: 0}, {0: 5_000_000, 1: 0}, 120000, weight=2.0)
        self.assertEqual(stats["spread"], 1)
        self.assertEqual(stats["spread_tokens"], 100352)

    def test_hit_is_counted_when_the_best_cached_rank_wins(self):
        _, stats = dispatch({0: 100352, 1: 0}, {0: 150000, 1: 0}, 120000, weight=2.0)
        self.assertEqual(stats["hit"], 1)
        self.assertEqual(stats["spread"], 0)


class StickyBehaviourUnchanged(unittest.TestCase):
    def test_short_match_still_alternates_at_equal_backlog(self):
        """A bare shared system prompt (below STICKY_TOKENS) must keep round-robining
        at equal backlog, at any weight -- otherwise quiet-period conversations all
        home on one rank, which is the defect the sticky rule exists to prevent."""
        matched = {0: 4096, 1: 4096}
        backlog = {0: 0, 1: 0}
        for w in (1.0, 2.0):
            seen = {dispatch(matched, backlog, 50000, weight=w, rr=r)[0] for r in range(2)}
            self.assertEqual(seen, {0, 1}, f"w={w} did not alternate")

    def test_long_match_sticks_at_equal_backlog(self):
        matched = {0: 100352, 1: 0}
        backlog = {0: 0, 1: 0}
        for w in (1.0, 2.0):
            self.assertEqual(dispatch(matched, backlog, 120000, weight=w, rr=1)[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
