from __future__ import annotations

import unittest

import numpy as np

from gscbba.cbba import Harvester, Task, make_rows, square_field


class DiminishingGainTests(unittest.TestCase):
    def test_raw_marginal_cost_is_not_monotone_from_the_gate(self):
        tasks = make_rows(square_field(60.0), n_rows=12, heading_deg=90.0)
        task_map = {task.id: task for task in tasks}
        harvester = Harvester(0, np.array([5.0, 1.0]))
        route = []
        marginal_costs = []
        for task in tasks:
            before = harvester._route_length(route, task_map)
            route.append(task.id)
            marginal_costs.append(harvester._route_length(route, task_map) - before)

        self.assertGreater(marginal_costs[0], marginal_costs[1])

    def test_append_only_finish_bids_are_non_increasing(self):
        tasks = make_rows(square_field(60.0), n_rows=12, heading_deg=90.0)
        task_map = {task.id: task for task in tasks}
        harvester = Harvester(0, np.array([5.0, 1.0]))
        route = []
        bids = []
        for task in tasks:
            route.append(task.id)
            bids.append(-harvester._route_length(route, task_map) / harvester.speed)
        self.assertTrue(
            all(later <= earlier + 1e-12 for earlier, later in zip(bids, bids[1:]))
        )

    def test_sorted_insertion_can_increase_stored_bid(self):
        ends = [(1.0, 9.0), (8.0, 16.0), (-6.0, 5.0), (-2.0, -1.0)]
        tasks = {
            i: Task(i, np.array([float(i), a]), np.array([float(i), b]))
            for i, (a, b) in enumerate(ends)
        }
        harvester = Harvester(0, np.array([3.0, 6.0]), speed=1.0)
        harvester.build_bundle(list(tasks), tasks)
        bids = [harvester.y[tid] for tid in harvester.bundle]
        self.assertTrue(any(later > earlier for earlier, later in zip(bids, bids[1:])))

    def test_proximity_admission_does_not_guarantee_adjacency(self):
        tasks = {
            i: Task(i, np.array([float(i), 0.0]), np.array([float(i), 10.0]))
            for i in range(4)
        }
        harvester = Harvester(0, np.array([3.0, 0.0]), speed=1.0)
        harvester.build_bundle([0, 3], tasks)
        self.assertCountEqual(harvester.path, [0, 3])


if __name__ == "__main__":
    unittest.main()
