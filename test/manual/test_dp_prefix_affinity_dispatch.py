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
    def __init__(self, n, rid="rid-test"):
        self.input_ids = list(range(n))
        self.rid = rid


def dispatch(
    matched_tokens,
    backlog,
    n,
    weight,
    chunk=2048,
    sticky=16384,
    rr=0,
    observe=0,
    rid="rid-test",
    capture_logs=False,
):
    """Run the REAL scheduler; return (target_rank, stats), or
    (target, stats, lines) when `capture_logs` -- only the instrument arms need
    the emitted text. `observe` defaults to 0 (the shipped default, instrument
    OFF), so every decision arm also asserts the instrument stays silent."""
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
    ctl.prefix_affinity_observe_tokens = observe
    ctl._prefix_affinity_stats = {"hit": 0, "miss": 0, "spread": 0, "spread_tokens": 0}

    sent = {}
    lines = []
    fake_logger = mock.Mock()
    fake_logger.info = lambda fmt, *args: lines.append(fmt % args)
    with mock.patch.object(dpc, "sock_send", lambda w, r: sent.setdefault("w", w)), \
            mock.patch.object(dpc, "logger", fake_logger):
        dpc.DataParallelController.prefix_affinity_scheduler(ctl, FakeReq(n, rid=rid))
    target = next(i for i in ranks if ctl.workers[i] is sent["w"])
    dispatch_lines = [l for l in lines if l.startswith("DP prefix_affinity dispatch:")]
    if capture_logs:
        return target, ctl._prefix_affinity_stats, dispatch_lines
    assert not dispatch_lines, f"instrument fired with observe={observe}: {dispatch_lines}"
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

    def test_the_documented_tolerance_bound_is_the_real_one(self):
        """The comment promises a rank is tolerated up to WEIGHT * matched[i] tokens
        MORE backlogged before the request spreads off it. That bound is what an
        operator would reason about when TTFT regresses, so assert it rather than
        trusting the prose: at match 100352 and WEIGHT=2 the crossover must sit at
        ~200704 tokens of extra backlog, not somewhere else."""
        matched = {0: 100352, 1: 0}
        just_under = 2 * 100352 - 2048
        just_over = 2 * 100352 + 2048
        self.assertEqual(dispatch(matched, {0: just_under, 1: 0}, 120000, 2.0)[0], 0)
        self.assertEqual(dispatch(matched, {0: just_over, 1: 0}, 120000, 2.0)[0], 1)

    def test_the_weight_does_not_disable_anti_homing_at_three_ranks(self):
        """🔴 REGRESSION ARM for a real bug, REPLACING a test that could not fail.

        The old arm asserted `seen.issubset({0,1})` over targets drawn from ranks
        {0,1} -- a tautology -- and its own scenario homed anyway, so it passed while
        its docstring claimed the opposite. Both defects found by review 2026-09-20.

        The bug it now covers: `near` was anchored on the WEIGHTED best_rank, so once
        the weight moved the anchor to a rank at a different backlog level the set
        collapsed to a singleton and the round-robin silently stopped -- the weight
        disabling the anti-homing rule. Invisible at dp_size=2 (a non-singleton
        `near` needs exact backlog equality, where both rules agree), live at 3.
        Measured before the fix: W=1.0 alternated [1,2,1,2], W=2.0 homed [0,0,0,0].

        The invariant: a SHORT best match alternates identically at any weight."""
        matched = {0: 8192, 1: 0, 2: 0}
        backlog = {0: 10000, 1: 0, 2: 0}
        legacy = [dispatch(matched, backlog, 50000, weight=1.0, rr=r)[0] for r in range(4)]
        weighted = [dispatch(matched, backlog, 50000, weight=2.0, rr=r)[0] for r in range(4)]
        self.assertGreater(len(set(legacy)), 1,
                           "fixture no longer exercises alternation -- test is blind")
        self.assertEqual(weighted, legacy,
                         f"weight changed the anti-homing branch: {weighted} vs {legacy}")


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


class EnvWiring(unittest.TestCase):
    """The one production line the decision tests cannot reach: the constructor's
    read of the env field. If the field were missing or not float-able, every test
    above would still pass (they set the attribute directly on the fake)."""

    def test_field_exists_and_defaults_to_the_documented_value(self):
        from sglang.srt.environ import envs
        self.assertEqual(float(envs.SGLANG_DP_PREFIX_AFFINITY_REPREFILL_WEIGHT.get()), 2.0)

    def test_clamp_refuses_a_weight_below_one(self):
        """Below 1.0 the dispatcher would PREFER re-prefilling over waiting -- the
        defect inverted -- so the value is clamped.

        🔴 This arm previously asserted `max(1.0, float(raw)) == want`: a COPY of the
        expression, not the code, which would have passed with the constructor line
        deleted. Commit e0c4233c1a claimed it closed this gap; it did not. The clamp
        now lives in `resolve_reprefill_weight` -- the function the constructor calls
        -- so this exercises the shipped path."""
        for raw, want in ((0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (2, 2.0), (3.5, 3.5)):
            self.assertEqual(dpc.resolve_reprefill_weight(raw), want, f"raw={raw!r}")

    def test_a_mistyped_weight_degrades_to_the_legacy_chooser(self):
        """A non-numeric env value must fall back to 1.0 -- the exact old behaviour --
        rather than to an arbitrary one or a crash at startup."""
        for raw in ("", "two", None, "abc"):
            self.assertEqual(dpc.resolve_reprefill_weight(raw), 1.0, f"raw={raw!r}")


class EvictionInstrument(unittest.TestCase):
    """`SGLANG_DP_PREFIX_AFFINITY_OBSERVE_TOKENS` logs `best_match` / `matched`,
    which the dispatcher otherwise computes and throws away. Its purpose is to
    make one currently-unanswerable question answerable: is a >100k-uncached-token
    prefill a re-prefill of an EVICTED conversation, or a genuinely new prompt?

    🔴 It must change no decision and must be silent at its shipped default --
    the endpoint runs ~0.5 req/s with 200-dispatch summary logging, so a
    per-request line by default is not affordable."""

    BIG = {0: 100352, 1: 0}          # 49 chunks x 2048 on rank 0

    def test_off_by_default_emits_nothing(self):
        """MUST-NOT-FIRE: observe=0 (the shipped default) is silent even on a
        request far larger than any threshold an operator would set."""
        _, _, lines = dispatch(
            self.BIG, {0: 0, 1: 0}, 500000, weight=2.0, observe=0, capture_logs=True
        )
        self.assertEqual(lines, [])

    def test_below_the_threshold_emits_nothing(self):
        _, _, lines = dispatch(
            {0: 0, 1: 0}, {0: 0, 1: 0}, 99999, weight=2.0, observe=100000,
            capture_logs=True,
        )
        self.assertEqual(lines, [])

    def test_at_and_above_the_threshold_emits_exactly_one_line(self):
        """MUST-FIRE: the pair with the arm above is what proves the threshold is
        the real gate and not an always-off / always-on knob."""
        for n in (100000, 400000):
            _, _, lines = dispatch(
                {0: 0, 1: 0}, {0: 0, 1: 0}, n, weight=2.0, observe=100000,
                capture_logs=True,
            )
            self.assertEqual(len(lines), 1, f"n={n}: {lines}")

    def test_the_line_carries_every_operand_the_join_needs(self):
        """The join is against that rank's next `Prefill batch` line, which carries
        no rid -- so the line must name the RANK (which log stream) and `new` (what
        that rank's `#new-token` should be), plus `best_match` and `matched` (the
        eviction discriminator) and `n`. Assert the values, not just the keys."""
        target, _, lines = dispatch(
            self.BIG, {0: 0, 1: 0}, 120000, weight=2.0, observe=100000,
            rid="req-abc123", capture_logs=True,
        )
        self.assertEqual(target, 0)          # long match sticks at equal backlog
        line = lines[0]
        for operand in (
            "rid=req-abc123",
            "rank=0",
            "n=120000",
            "best_match=100352",
            "matched=100352",
            "new=19648",                     # 120000 - 100352, the expected #new-token
        ):
            self.assertIn(operand, line, f"{operand!r} missing from {line!r}")

    def test_it_discriminates_an_eviction_from_a_new_prompt(self):
        """The whole point. Same prompt length, same backlogs, same threshold:
        a conversation the index has seen reads `best_match>0` (so a following
        `#cached-token: 0` convicts the cache of evicting it); a prompt no rank
        was ever sent reads `best_match=0` and cannot be mistaken for one."""
        _, _, seen = dispatch(
            self.BIG, {0: 0, 1: 0}, 120000, weight=2.0, observe=100000,
            capture_logs=True,
        )
        _, _, fresh = dispatch(
            {0: 0, 1: 0}, {0: 0, 1: 0}, 120000, weight=2.0, observe=100000,
            capture_logs=True,
        )
        self.assertIn("best_match=100352", seen[0])
        self.assertIn("best_match=0", fresh[0])
        self.assertIn("new=120000", fresh[0])

    def test_the_instrument_does_not_move_the_decision(self):
        """An instrument that changes dispatch is not an instrument. Every
        scenario the decision suite above cares about must pick the same rank
        with the knob on as with it off."""
        cases = [
            ({0: 100352, 1: 0}, {0: 150000, 1: 0}, 120000),
            ({0: 100352, 1: 0}, {0: 5_000_000, 1: 0}, 120000),
            ({0: 49152, 1: 49152}, {0: 90000, 1: 10000}, 120000),
            ({0: 4096, 1: 4096}, {0: 0, 1: 0}, 50000),
        ]
        for matched, backlog, n in cases:
            off, _ = dispatch(matched, backlog, n, weight=2.0, observe=0)
            on, _, _ = dispatch(
                matched, backlog, n, weight=2.0, observe=1, capture_logs=True
            )
            self.assertEqual(off, on, f"matched={matched} backlog={backlog} n={n}")

    def test_env_field_exists_and_defaults_to_off(self):
        from sglang.srt.environ import envs
        self.assertEqual(int(envs.SGLANG_DP_PREFIX_AFFINITY_OBSERVE_TOKENS.get()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
