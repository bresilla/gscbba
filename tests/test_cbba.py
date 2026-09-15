from __future__ import annotations

import unittest

import numpy as np

from gscbba.baselines import optimal_contiguous_partition
from gscbba.cbba import (
    Harvester,
    Simulator,
    Task,
    make_harvesters,
    make_rows,
    square_field,
)


class SimulatorTests(unittest.TestCase):
    def test_central_replan_keeps_partly_harvested_row(self):
        tasks = [
            Task(i, np.array([float(i), 0.0]), np.array([float(i), 100.0]))
            for i in range(3)
        ]
        harvester = Harvester(0, np.array([0.0, 40.0]), speed=1.0)
        simulator = Simulator(
            tasks, [harvester], comm_range=1e9, allocation_policy="centralized"
        )
        harvester.path = [0, 1, 2]
        harvester.cur_task = 0
        harvester.cur_entry, harvester.cur_exit = tasks[0].a, tasks[0].b
        harvester.substate = "to_exit"
        simulator._centralized_allocation()
        self.assertEqual(harvester.cur_task, 0)
        self.assertEqual(harvester.path[0], 0)
        self.assertEqual(harvester.substate, "to_exit")
        simulator._move(harvester)
        self.assertAlmostEqual(harvester.pos[1], 40.5)

    def test_coverage_is_not_auction_stability(self):
        result = Simulator(
            self.tasks,
            self.harvesters(),
            comm_range=1e9,
            spatial=True,
            query_k=1,
            alloc_max_rounds=1,
        ).run()
        self.assertTrue(result.converged)
        self.assertEqual(result.allocation_stable, [False])

    def test_static_planner_metrics_use_simulated_execution(self):
        from gscbba.experiments import _planner_metrics

        machines = self.harvesters()
        plan = optimal_contiguous_partition(self.tasks, machines)
        metrics = _planner_metrics(plan, self.tasks, machines)
        replay = Simulator(
            self.tasks,
            self.harvesters(),
            comm_range=1e9,
            allocation_policy="frozen",
            initial_assignment=plan.assignment,
        ).run()
        self.assertEqual(metrics["makespan"], replay.completion_time)
        self.assertEqual(metrics["analytic_makespan"], plan.makespan)

    def setUp(self):
        self.polygon = square_field(60.0)
        self.tasks = make_rows(self.polygon, n_rows=12, heading_deg=90.0)

    def harvesters(self, n=3):
        rng = np.random.default_rng(11)
        return make_harvesters(n, self.polygon, rng, max_bundle=len(self.tasks))

    def test_all_primary_modes_cover_the_field(self):
        results = (
            Simulator(
                self.tasks,
                self.harvesters(),
                comm_range=1.0e9,
                hops=2,
                mode="marginal",
            ).run(),
            Simulator(
                self.tasks,
                self.harvesters(),
                comm_range=1.0e9,
                hops=2,
                balancing=False,
            ).run(),
            Simulator(
                self.tasks,
                self.harvesters(),
                comm_range=1.0e9,
                hops=2,
                balancing=True,
            ).run(),
        )
        for result in results:
            with self.subTest(result=result):
                self.assertTrue(result.converged)
                self.assertEqual(sum(result.per_harvester_tasks), len(self.tasks))
                self.assertEqual(result.duplicate_services, 0)

    def test_turn_distance_is_used_and_accounted_for(self):
        result = Simulator(
            self.tasks,
            self.harvesters(),
            comm_range=1.0e9,
            hops=2,
            turn_distance=20.0,
        ).run()
        expected_turns = sum(max(0, count - 1) for count in result.per_harvester_tasks)
        self.assertAlmostEqual(result.turn_distance, 20.0 * expected_turns)
        self.assertAlmostEqual(
            result.total_distance,
            result.row_distance + result.deadhead_distance + result.turn_distance,
            places=8,
        )

    def test_multiple_losses_reallocate_orphaned_rows(self):
        result = Simulator(
            self.tasks,
            self.harvesters(4),
            comm_range=1.0e9,
            hops=2,
            loss_events=[(20.0, 1), (70.0, 2)],
            loss_detect_delay=2.0,
            t_max=2000.0,
        ).run()
        self.assertTrue(result.converged)
        self.assertEqual(len(result.reconvergence_rounds), 2)
        self.assertEqual(
            [event["kind"] for event in result.event_log].count("loss_detected"),
            2,
        )

    def test_join_and_speed_change_trigger_replanning(self):
        joining = Harvester(hid=3, pos=np.array([30.0, 1.0]), max_bundle=12)
        result = Simulator(
            self.tasks,
            self.harvesters(),
            comm_range=1.0e9,
            hops=2,
            join_events=[(20.0, joining)],
            speed_events=[(50.0, 0, 0.7)],
            t_max=2000.0,
        ).run()
        self.assertTrue(result.converged)
        self.assertEqual(len(result.per_harvester_tasks), 4)
        self.assertGreaterEqual(len(result.reconvergence_rounds), 2)

    def test_frozen_plan_reports_incomplete_coverage_after_loss(self):
        harvesters = self.harvesters()
        plan = optimal_contiguous_partition(self.tasks, harvesters).assignment
        result = Simulator(
            self.tasks,
            harvesters,
            comm_range=1.0e9,
            allocation_policy="frozen",
            initial_assignment=plan,
            loss_events=[(20.0, 1)],
            loss_detect_delay=2.0,
            t_max=2000.0,
        ).run()
        self.assertFalse(result.converged)
        self.assertLess(result.coverage_fraction, 1.0)

    def test_message_telemetry_separates_auction_and_balancing(self):
        full = Simulator(
            self.tasks,
            self.harvesters(),
            comm_range=1.0e9,
            hops=2,
        ).run()
        auction_only = Simulator(
            self.tasks,
            self.harvesters(),
            comm_range=1.0e9,
            hops=2,
            balancing=False,
        ).run()
        self.assertGreater(full.msg_records_total, 0)
        self.assertGreater(full.msg_balancing_total, 0)
        self.assertEqual(auction_only.msg_balancing_total, 0)


if __name__ == "__main__":
    unittest.main()
