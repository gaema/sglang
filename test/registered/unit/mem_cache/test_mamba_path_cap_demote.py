from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import os
import unittest
from collections import defaultdict
from unittest import mock

import torch

from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

from test_mamba_path_state_cap import _build_unified_chain


def _walk(component, tail):
    device_frees = defaultdict(list)
    host_frees = defaultdict(list)
    cache_actions = []
    component._evict_excess_path_states(tail, device_frees, host_frees, cache_actions)
    return device_frees, host_frees, cache_actions


class TestMambaPathCapDemote(unittest.TestCase):
    """SGLANG_MAMBA_PATH_CAP_DEMOTE=1: the cap orders eviction instead of freeing."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"SGLANG_MAMBA_PATH_CAP_DEMOTE": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_excess_states_stay_and_move_to_the_lru_end_shallowest_last(self):
        component, nodes, core, _ = _build_unified_chain(cap=2, length=4)
        lru = core.lru_lists[ComponentType.MAMBA]

        device_frees, host_frees, cache_actions = _walk(component, nodes[-1])

        self.assertEqual(core.evicted, [])
        self.assertEqual(device_frees[ComponentType.MAMBA], [])
        self.assertEqual(host_frees[ComponentType.MAMBA], [])
        self.assertEqual(cache_actions, [])
        self.assertTrue(
            all(n.component_data[ComponentType.MAMBA].value is not None for n in nodes)
        )
        self.assertEqual(core.component_evictable_size_[ComponentType.MAMBA], 4)
        # Least-recently-used first: the shallowest excess state, then the next.
        first = lru.get_lru_no_lock()
        self.assertIs(first, nodes[0])
        self.assertIs(lru.get_prev_no_lock(first), nodes[1])
        # The two deepest (within the cap) keep their order ahead of them.
        self.assertIs(lru.get_prev_no_lock(nodes[1]), nodes[2])
        self.assertIs(lru.get_prev_no_lock(nodes[2]), nodes[3])

    def test_within_cap_changes_nothing(self):
        component, nodes, core, _ = _build_unified_chain(cap=4, length=3)
        lru = core.lru_lists[ComponentType.MAMBA]
        before = lru.get_lru_no_lock()
        _walk(component, nodes[-1])
        self.assertEqual(core.evicted, [])
        self.assertIs(lru.get_lru_no_lock(), before)

    def test_locked_and_fork_nodes_are_not_demoted(self):
        component, nodes, core, _ = _build_unified_chain(cap=1, length=4)
        lru = core.lru_lists[ComponentType.MAMBA]
        nodes[0].component_data[ComponentType.MAMBA].lock_ref = 1
        nodes[1].children["fork"] = object()
        _walk(component, nodes[-1])
        self.assertEqual(core.evicted, [])
        # Only nodes[2] was eligible: it is now the least recently used unlocked node.
        self.assertIs(lru.get_lru_no_lock(), nodes[2])

    def test_demoted_states_keep_lru_order_across_paths(self):
        component, nodes, core, _ = _build_unified_chain(cap=1, length=3)
        lru = core.lru_lists[ComponentType.MAMBA]
        _walk(component, nodes[-1])  # demotes nodes[0], nodes[1]
        # A second, later path from the root.
        parent, branch = core.root_node, []
        for index in range(2):
            node = UnifiedTreeNode(core.tree_components)
            node.parent = parent
            node.component_data[ComponentType.FULL].value = torch.tensor([200 + index])
            node.component_data[ComponentType.MAMBA].value = torch.tensor([10 + index])
            parent.children[f"m{index}"] = node
            core.component_evictable_size_[ComponentType.MAMBA] += 1
            lru.insert_mru(node)
            branch.append(node)
            parent = node
        _walk(component, branch[-1])  # demotes branch[0]
        self.assertEqual(core.evicted, [])
        # Oldest demotion first: the first path's shallowest, its next, then the later path's.
        order = [lru.get_lru_no_lock()]
        for _ in range(2):
            order.append(lru.get_prev_no_lock(order[-1]))
        self.assertEqual(order, [nodes[0], nodes[1], branch[0]])
        # Then the undemoted tails, newest last.
        self.assertIs(lru.get_prev_no_lock(branch[0]), nodes[2])

    def test_a_matched_demoted_state_returns_to_mru(self):
        component, nodes, core, _ = _build_unified_chain(cap=1, length=3)
        lru = core.lru_lists[ComponentType.MAMBA]
        _walk(component, nodes[-1])
        lru.reset_node_mru(nodes[0])
        self.assertIs(lru.get_lru_no_lock(), nodes[1])

    def test_off_keeps_the_eager_free(self):
        with mock.patch.dict(os.environ, {"SGLANG_MAMBA_PATH_CAP_DEMOTE": "0"}):
            component, nodes, core, _ = _build_unified_chain(cap=2, length=3)
            device_frees, _, _ = _walk(component, nodes[-1])
        self.assertEqual(core.evicted, [nodes[0]])
        self.assertEqual([v.item() for v in device_frees[ComponentType.MAMBA]], [0])


if __name__ == "__main__":
    unittest.main()
