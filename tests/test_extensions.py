from __future__ import annotations

import unittest

import numpy as np

from gscbba.baselines import optimal_contiguous_partition
from gscbba.cbba import (
    BASE_ID,
    Harvester,
    Simulator,
    Task,
    make_harvesters,
    make_rows,
    square_field,
)
from gscbba.standard_cbba import StandardCBBA, standard_cbba


def _field(seed: int = 3, n: int = 3, rows: int = 12, extent: float = 60.0):
    rng = np.random.default_rng(seed)
    polygon = square_field(extent)
    tasks = make_rows(polygon, n_rows=rows, heading_deg=90.0)
    harvesters = make_harvesters(n, polygon, rng, max_bundle=rows)
    return tasks, harvesters


class StandardCBBATests(unittest.TestCase):
    def test_assigns_every_row_once_and_converges(self):
        tasks, harvesters = _field()
        run = standard_cbba(tasks, harvesters, discount=0.9, bundle_limit=8)
        self.assertTrue(run.converged)
        self.assertEqual(run.conflicts, 0)
        self.assertEqual(run.unassigned, 0)
        owned = [tid for route in run.result.assignment.values() for tid in route]
        self.assertEqual(sorted(owned), sorted(t.id for t in tasks))

    def test_agents_agree_on_winners_after_convergence(self):
        tasks, harvesters = _field(seed=5)
        solver = StandardCBBA(tasks, harvesters, discount=0.9, bundle_limit=8)
        solver.solve()
        reference = solver.agents[0].z
        for agent in solver.agents[1:]:
            self.assertTrue(np.array_equal(agent.z, reference))

    def test_reset_rule_releases_tasks_after_a_lost_one(self):
        tasks, harvesters = _field()
        solver = StandardCBBA(tasks, harvesters, discount=0.9, bundle_limit=4)
        agent = solver.agents[0]
        solver._build(agent)
        lost = agent.bundle[0]
        later = agent.bundle[1:]
        agent.z[lost] = 1
        solver._truncate(agent)
        self.assertEqual(agent.bundle, [])
        for j in later:
            self.assertEqual(agent.z[j], -1)

    def test_rejects_invalid_discount(self):
        tasks, harvesters = _field()
        with self.assertRaises(ValueError):
            StandardCBBA(tasks, harvesters, discount=1.0)


class LocalInformationTests(unittest.TestCase):
    def test_local_mode_ignores_global_completion_state(self):
        tasks, harvesters = _field()
        simulator = Simulator(tasks, harvesters, comm_range=1e9, information="local")
        simulator.done.add(tasks[0].id)
        self.assertIn(tasks[0].id, simulator._avail_all(harvesters[0]))
        shared = Simulator(tasks, _field()[1], comm_range=1e9)
        shared.done.add(tasks[0].id)
        self.assertNotIn(tasks[0].id, shared._avail_all(shared.harvesters[0]))

    def test_heartbeat_detects_failure_and_recovers_rows(self):
        tasks, harvesters = _field(rows=18)
        result = Simulator(
            tasks,
            harvesters,
            comm_range=1e9,
            hops=2,
            spatial=True,
            information="local",
            component_balancing=True,
            idle_help=True,
            loss_events=[(60.0, 1)],
            heartbeat_timeout=6.0,
        ).run()
        self.assertTrue(result.converged)
        self.assertEqual(result.coverage_fraction, 1.0)
        self.assertEqual(result.false_failure_detections, 0)
        self.assertEqual(len(result.detection_delays), 1)
        self.assertGreater(result.detection_delays[0], 6.0)

    def test_packet_loss_is_counted_and_coverage_completes(self):
        tasks, harvesters = _field(rows=18)
        result = Simulator(
            tasks,
            harvesters,
            comm_range=1e9,
            hops=2,
            spatial=True,
            information="local",
            component_balancing=True,
            idle_help=True,
            packet_loss=0.3,
            round_latency=0.1,
            comm_seed=4,
        ).run()
        self.assertTrue(result.converged)
        self.assertGreater(result.msg_dropped, 0)
        self.assertGreater(result.planning_delay, 0.0)

    def test_predicted_position_follows_the_reported_plan(self):
        state = (np.array([0.0, 0.0]), 1.0, [np.array([0.0, 10.0])], 0.0)
        position, finished = Simulator._predict(state, 4.0)
        self.assertTrue(np.allclose(position, [0.0, 4.0]))
        self.assertFalse(finished)
        position, finished = Simulator._predict(state, 20.0)
        self.assertTrue(np.allclose(position, [0.0, 10.0]))
        self.assertTrue(finished)


class BalancingTests(unittest.TestCase):
    def test_component_balancing_rejects_a_worse_split(self):
        tasks = [
            Task(i, np.array([2.0 * i, 0.0]), np.array([2.0 * i, 50.0]))
            for i in range(4)
        ]
        left = Harvester(0, np.array([0.0, 0.0]), speed=1.0)
        right = Harvester(1, np.array([6.0, 0.0]), speed=1.0)
        simulator = Simulator(tasks, [left, right], comm_range=1e9)
        left.path, right.path = [0, 1], [2, 3]
        self.assertFalse(
            simulator._balanced_partition([left, right], [0, 1, 2, 3], True)
        )
        self.assertEqual(left.path, [0, 1])

    def test_component_balancing_accepts_a_better_split(self):
        tasks = [
            Task(i, np.array([2.0 * i, 0.0]), np.array([2.0 * i, 50.0]))
            for i in range(4)
        ]
        left = Harvester(0, np.array([0.0, 0.0]), speed=1.0)
        right = Harvester(1, np.array([6.0, 0.0]), speed=1.0)
        simulator = Simulator(tasks, [left, right], comm_range=1e9)
        left.path, right.path = [0, 1, 2], [3]
        self.assertTrue(
            simulator._balanced_partition([left, right], [0, 1, 2, 3], True)
        )
        self.assertEqual(sorted(left.path), [0, 1])
        self.assertEqual(left.z[2], 1)


class BaseStationTests(unittest.TestCase):
    def test_base_station_only_replans_reachable_machines(self):
        tasks, harvesters = _field(rows=12)
        plan = optimal_contiguous_partition(tasks, harvesters)
        simulator = Simulator(
            tasks,
            harvesters,
            comm_range=15.0,
            allocation_policy="centralized_base",
            initial_assignment=plan.assignment,
            base_position=np.array([-500.0, -500.0]),
        )
        simulator._apply_assignment(plan.assignment)
        simulator._base_plan = {h.hid: list(h.path) for h in harvesters}
        simulator._update_base_reach()
        self.assertEqual(simulator._base_reach, set())
        self.assertFalse(simulator._base_replan(force=True))

    def test_base_station_run_completes_every_row(self):
        tasks, harvesters = _field(rows=12)
        plan = optimal_contiguous_partition(tasks, harvesters)
        result = Simulator(
            tasks,
            harvesters,
            comm_range=40.0,
            allocation_policy="centralized_base",
            initial_assignment=plan.assignment,
            base_position=np.array([30.0, 0.5]),
            loss_events=[(30.0, 2)],
        ).run()
        self.assertTrue(result.converged)
        self.assertEqual(result.duplicate_services, 0)
        self.assertGreaterEqual(result.planner_calls, 2)

    def test_base_identifier_is_not_a_harvester(self):
        _, harvesters = _field()
        self.assertNotIn(BASE_ID, [h.hid for h in harvesters])

    def test_base_policy_requires_a_position(self):
        tasks, harvesters = _field()
        with self.assertRaises(ValueError):
            Simulator(
                tasks, harvesters, comm_range=10.0, allocation_policy="centralized_base"
            )


if __name__ == "__main__":
    unittest.main()
