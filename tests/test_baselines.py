from __future__ import annotations

import importlib.util
import itertools
import unittest
from unittest.mock import patch

import numpy as np

from gscbba.baselines import (
    evaluate_assignment,
    milp_assignment,
    optimal_contiguous_partition,
    sorted_harvesters,
    sorted_rows,
    static_speed_split,
)
from gscbba.cbba import Harvester, Task, make_harvesters, make_rows, square_field


def brute_force_contiguous(tasks, harvesters, turn_distance=0.0, turn_time=0.0):
    rows = sorted_rows(tasks)
    machines = sorted_harvesters(tasks, harvesters)
    best = None
    for cuts in itertools.product(range(len(rows) + 1), repeat=len(machines) - 1):
        if tuple(sorted(cuts)) != cuts:
            continue
        bounds = (0, *cuts, len(rows))
        assignment = {
            machine.hid: rows[bounds[i] : bounds[i + 1]]
            for i, machine in enumerate(machines)
        }
        result = evaluate_assignment(
            tasks,
            harvesters,
            assignment,
            turn_distance=turn_distance,
            turn_time=turn_time,
        )
        if best is None or result.makespan < best.makespan:
            best = result
    return best


class ContiguousPartitionTests(unittest.TestCase):
    def test_ready_time_and_initial_turn_accounting(self):
        tasks = [Task(0, np.array([0.0, 0.0]), np.array([0.0, 10.0]))]
        machines = [Harvester(0, np.array([0.0, 0.0]), speed=2.0)]
        result = optimal_contiguous_partition(
            tasks,
            machines,
            ready_times={0: 7.0},
            initial_turns=[0],
            turn_distance=4.0,
            turn_time=3.0,
        )
        self.assertAlmostEqual(result.makespan, 17.0)
        self.assertAlmostEqual(result.total_distance, 14.0)
        self.assertAlmostEqual(result.turn_distance, 4.0)
        empty = optimal_contiguous_partition([], machines, ready_times={0: 7.0})
        self.assertAlmostEqual(empty.makespan, 7.0)

    def test_fixed_sweep_does_not_claim_best_reversal(self):
        tasks = [
            Task(i, np.array([x, 0.0]), np.array([x, 1.0]))
            for i, x in enumerate((0.0, 10.0))
        ]
        machines = [Harvester(0, np.array([0.0, 0.0]), speed=1.0)]
        result = optimal_contiguous_partition(tasks, machines)
        reverse = evaluate_assignment(
            tasks, machines, {0: list(reversed(result.assignment[0]))}
        )
        self.assertAlmostEqual(result.makespan, 22.0)
        self.assertAlmostEqual(reverse.makespan, 12.0)

    def test_ready_times_match_all_two_machine_cuts(self):
        tasks = make_rows(square_field(20.0), n_rows=4, heading_deg=90.0)
        machines = [
            Harvester(0, np.array([0.0, 1.0]), speed=1.0),
            Harvester(1, np.array([19.0, 1.0]), speed=2.0),
        ]
        rows = sorted_rows(tasks)
        ordered = sorted_harvesters(tasks, machines)
        for ready in ({0: 25.0, 1: 3.0}, {0: 0.0, 1: 150.0}):
            exact = optimal_contiguous_partition(
                tasks,
                machines,
                ready_times=ready,
                initial_turns=[0],
                turn_distance=4.0,
                turn_time=3.0,
            )
            costs = []
            for cut in range(5):
                assignment = {ordered[0].hid: rows[:cut], ordered[1].hid: rows[cut:]}
                base = evaluate_assignment(
                    tasks, machines, assignment, turn_distance=4.0, turn_time=3.0
                )
                costs.append(
                    max(
                        base.finish_times[h.hid]
                        + ready[h.hid]
                        + (
                            4.0 / h.speed + 3.0
                            if h.hid == 0 and assignment[h.hid]
                            else 0.0
                        )
                        for h in machines
                    )
                )
            self.assertAlmostEqual(exact.makespan, min(costs), places=9)

    def test_dynamic_program_matches_brute_force(self):
        polygon = square_field(80.0)
        for n_machines in (2, 3):
            for n_rows in range(2, 9):
                with self.subTest(n_machines=n_machines, n_rows=n_rows):
                    rng = np.random.default_rng(700 + 11 * n_machines + n_rows)
                    tasks = make_rows(polygon, n_rows=n_rows, heading_deg=87.0)
                    harvesters = make_harvesters(
                        n_machines,
                        polygon,
                        rng,
                        speeds=np.linspace(1.0, 2.0, n_machines),
                        max_bundle=n_rows,
                    )
                    exact = optimal_contiguous_partition(
                        tasks,
                        harvesters,
                        turn_distance=20.0,
                        turn_time=3.0,
                    )
                    brute = brute_force_contiguous(
                        tasks,
                        harvesters,
                        turn_distance=20.0,
                        turn_time=3.0,
                    )
                    assert brute is not None
                    self.assertAlmostEqual(exact.makespan, brute.makespan, places=9)
                    assigned = [
                        tid for tids in exact.assignment.values() for tid in tids
                    ]
                    self.assertCountEqual(assigned, [task.id for task in tasks])

    def test_static_split_covers_each_row_once(self):
        polygon = square_field(80.0)
        rng = np.random.default_rng(19)
        tasks = make_rows(polygon, n_rows=8, heading_deg=90.0)
        harvesters = make_harvesters(
            3,
            polygon,
            rng,
            speeds=[1.0, 1.5, 2.0],
            max_bundle=8,
        )
        result = static_speed_split(tasks, harvesters)
        assigned = [tid for tids in result.assignment.values() for tid in tids]
        self.assertCountEqual(assigned, [task.id for task in tasks])
        for tids in result.assignment.values():
            indices = sorted(sorted_rows(tasks).index(tid) for tid in tids)
            if indices:
                self.assertEqual(indices, list(range(indices[0], indices[-1] + 1)))


@unittest.skipUnless(
    importlib.util.find_spec("pulp"), "optional PuLP solver is not installed"
)
class MilpTests(unittest.TestCase):
    def test_two_machine_solution_matches_all_orders_and_cuts(self):
        tasks = [
            Task(
                i,
                np.array([float(i * 3), float(i % 2)]),
                np.array([float(i * 3), 8.0 + i]),
            )
            for i in range(4)
        ]
        machines = [
            Harvester(0, np.array([2.0, 4.0]), speed=1.0),
            Harvester(1, np.array([8.0, 7.0]), speed=1.5),
        ]
        result = milp_assignment(tasks, machines, turn_distance=2.0, turn_time=1.0)
        brute = min(
            evaluate_assignment(
                tasks,
                machines,
                {0: order[:cut], 1: order[cut:]},
                turn_distance=2.0,
                turn_time=1.0,
            ).makespan
            for order in itertools.permutations(range(4))
            for cut in range(5)
        )
        self.assertAlmostEqual(result.makespan, brute, places=7)

    def test_tiny_routes_match_exhaustive_nearest_endpoint_replay(self):
        for n in (2, 3, 4):
            tasks = [
                Task(
                    i,
                    np.array([float(i * 3), float(i % 2)]),
                    np.array([float(i * 3), 8.0 + i]),
                )
                for i in range(n)
            ]
            machines = [Harvester(0, np.array([2.0, 4.0]), speed=1.0)]
            result = milp_assignment(tasks, machines, turn_distance=2.0, turn_time=1.0)
            brute = min(
                evaluate_assignment(
                    tasks, machines, {0: route}, turn_distance=2.0, turn_time=1.0
                ).makespan
                for route in itertools.permutations(range(n))
            )
            self.assertAlmostEqual(result.makespan, brute, places=7)
            self.assertCountEqual(result.assignment[0], range(n))

    def test_timeout_incumbent_is_not_certified(self):
        import pulp

        def timed_out(problem, solver):
            problem.status = pulp.LpStatusOptimal
            problem.sol_status = pulp.LpSolutionIntegerFeasible

        tasks = [Task(0, np.array([0.0, 0.0]), np.array([0.0, 10.0]))]
        machines = [Harvester(0, np.zeros(2))]
        with patch.object(pulp.LpProblem, "solve", timed_out):
            with self.assertRaisesRegex(RuntimeError, "did not certify"):
                milp_assignment(tasks, machines)


if __name__ == "__main__":
    unittest.main()
