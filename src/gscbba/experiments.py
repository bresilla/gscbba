from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from .baselines import (
    AllocationResult,
    milp_assignment,
    optimal_contiguous_partition,
    rows_differ,
    static_speed_split,
)
from .cbba import (
    Harvester,
    Simulator,
    make_harvesters,
    make_rows,
    rect_field,
    square_field,
)
from .standard_cbba import standard_cbba

RESULTS_DIR = os.path.abspath("results")
INF_RANGE = 1.0e9
METHOD = {"component_balancing": True, "idle_help": True}
CBBA_DISCOUNT = 0.9
CBBA_BUNDLE_FACTOR = 1.5


def _stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    if len(a) == 0:
        return {"mean": None, "std": None, "ci95": [None, None], "values": []}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, len(a) - 1) * std / np.sqrt(len(a)))
        if len(a) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
        "values": [float(x) for x in a],
    }


def _paired_stats(values: list[float], reference: list[float]) -> dict:
    if len(values) != len(reference):
        raise ValueError("paired samples must have the same length")
    delta = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
    summary = _stats(delta.tolist())
    sd = summary["std"] or 0.0
    summary["standardized_effect"] = float(summary["mean"] / sd) if sd > 1e-12 else 0.0
    summary["n"] = len(values)
    return summary


def _scenario(
    seed: int,
    n_harvesters: int,
    n_rows: int,
    extent: float,
    *,
    speed_range=None,
):
    rng = np.random.default_rng(seed)
    polygon = square_field(extent)
    heading = rng.uniform(80.0, 100.0)
    tasks = make_rows(polygon, n_rows=n_rows, heading_deg=heading)
    harvesters = make_harvesters(
        n_harvesters,
        polygon,
        rng,
        layout="headland",
        speed_range=speed_range,
        max_bundle=n_rows,
    )
    return polygon, tasks, harvesters


def _finish_spread(values: list[float]) -> float:
    used = [value for value in values if value > 0.0]
    return max(used) - min(used) if used else 0.0


def _sim_metrics(result, reference_assignment=None) -> dict[str, float]:
    nonproductive = result.deadhead_distance + result.turn_distance
    return {
        "makespan": result.completion_time,
        "total_distance": result.total_distance,
        "deadhead_share": nonproductive / result.total_distance
        if result.total_distance
        else 0.0,
        "finish_spread": _finish_spread(result.per_harvester_finish),
        "compute_time": result.greedy_time + result.balance_time,
        "coverage_fraction": result.coverage_fraction,
        "duplicate_services": float(result.duplicate_services),
        "capped_allocation_cycles": float(result.allocation_stable.count(False)),
        "rows_differ_from_dp": float(
            rows_differ(result.initial_assignment, reference_assignment)
        )
        if reference_assignment is not None
        else 0.0,
        "auction_rounds": float(result.alloc_rounds),
        "records": float(result.msg_records_total + result.msg_balancing_total),
    }


def _planner_metrics(
    result: AllocationResult,
    tasks,
    harvesters,
    reference_assignment=None,
    *,
    turn_distance=0.0,
    turn_time=0.0,
) -> dict[str, float]:
    replay = Simulator(
        tasks,
        [replace(h, pos=h.pos.copy()) for h in harvesters],
        comm_range=INF_RANGE,
        allocation_policy="frozen",
        initial_assignment=result.assignment,
        turn_distance=turn_distance,
        turn_time=turn_time,
    ).run()
    metrics = _sim_metrics(replay, reference_assignment)
    metrics["compute_time"] = result.solve_time
    metrics["analytic_makespan"] = result.makespan
    return metrics


def _summarize_methods(raw: dict[str, list[dict]]) -> dict:
    summary = {}
    dp_values = [row["makespan"] for row in raw["dp"]]
    lower_bound = [row["makespan"] for row in raw["lower_bound"]]
    for method, rows in raw.items():
        metrics = {key: _stats([float(row[key]) for row in rows]) for key in rows[0]}
        makespans = [row["makespan"] for row in rows]
        metrics["paired_makespan_vs_dp"] = _paired_stats(makespans, dp_values)
        metrics["ratio_to_dp"] = _stats(
            [value / ref for value, ref in zip(makespans, dp_values)]
        )
        metrics["ratio_to_lower_bound"] = _stats(
            [value / ref for value, ref in zip(makespans, lower_bound)]
        )
        summary[method] = metrics
    return summary


def run_scalability(trials: int) -> dict:
    Ns = [2, 4, 6, 8, 10]
    M, extent = 100, 200.0
    data = {}
    for N in Ns:
        ct, td, rounds, imbalance, finish_spread = [], [], [], [], []
        auction_records, balancing_records = [], []
        auction_peak, balancing_peak = [], []
        capped, coverage = [], []
        poly = square_field(extent)
        for tr in range(trials):
            rng = np.random.default_rng(10_000 + 137 * N + tr)
            heading = rng.uniform(80.0, 100.0)
            tasks = make_rows(poly, n_rows=M, heading_deg=heading)
            harv = make_harvesters(N, poly, rng, layout="headland", max_bundle=M)
            r = Simulator(
                tasks, harv, comm_range=INF_RANGE, hops=2, spatial=True, **METHOD
            ).run()
            ct.append(r.completion_time)
            td.append(r.total_distance)
            rounds.append(r.alloc_rounds / max(1, r.alloc_cycles))
            imbalance.append(max(r.per_harvester_tasks) - min(r.per_harvester_tasks))
            finish_spread.append(_finish_spread(r.per_harvester_finish))
            auction_records.append(r.msg_records_total / max(1, r.alloc_cycles))
            balancing_records.append(r.msg_balancing_total / max(1, r.alloc_cycles))
            auction_peak.append(r.msg_records_peak_round)
            balancing_peak.append(r.msg_balancing_peak_round)
            capped.append(float(r.allocation_stable.count(False)))
            coverage.append(r.coverage_fraction)
        data[str(N)] = {
            "completion_time": _stats(ct),
            "total_distance": _stats(td),
            "consensus_rounds": _stats(rounds),
            "imbalance": _stats(imbalance),
            "finish_spread": _stats(finish_spread),
            "auction_records_per_cycle": _stats(auction_records),
            "balancing_records_per_cycle": _stats(balancing_records),
            "auction_records_peak_round": _stats(auction_peak),
            "balancing_records_peak_round": _stats(balancing_peak),
            "capped_allocation_cycles": _stats(capped),
            "coverage_fraction": _stats(coverage),
        }
        print(
            f"  [scal] N={N:2d}  t={data[str(N)]['completion_time']['mean']:7.1f}s "
            f"dist={data[str(N)]['total_distance']['mean']:7.0f}m "
            f"rounds={data[str(N)]['consensus_rounds']['mean']:.2f} "
            f"imbalance={data[str(N)]['imbalance']['mean']:.1f}"
        )
    return {"Ns": Ns, "M": M, "extent": extent, "trials": trials, "data": data}


def _run_benchmark_case(
    n_harvesters: int,
    n_rows: int,
    extent: float,
    trials: int,
    *,
    seed_base: int,
    speed_range=None,
) -> dict:
    raw: dict[str, list[dict]] = defaultdict(list)
    for tr in range(trials):
        seed = seed_base + 137 * n_harvesters + tr
        _, tasks, harvesters = _scenario(
            seed,
            n_harvesters,
            n_rows,
            extent,
            speed_range=speed_range,
        )
        dp = optimal_contiguous_partition(tasks, harvesters)
        static = static_speed_split(tasks, harvesters)
        total_row_length = sum(task.length for task in tasks)
        lower_bound = total_row_length / sum(h.speed for h in harvesters)
        raw["lower_bound"].append(
            {
                "makespan": lower_bound,
                "total_distance": total_row_length,
                "deadhead_share": 0.0,
                "finish_spread": 0.0,
                "compute_time": 0.0,
                "coverage_fraction": 1.0,
                "duplicate_services": 0.0,
                "rows_differ_from_dp": 0.0,
                "auction_rounds": 0.0,
                "records": 0.0,
            }
        )
        raw["dp"].append(_planner_metrics(dp, tasks, harvesters, dp.assignment))
        raw["static"].append(_planner_metrics(static, tasks, harvesters, dp.assignment))
        cbba = standard_cbba(
            tasks,
            harvesters,
            discount=CBBA_DISCOUNT,
            bundle_limit=math.ceil(CBBA_BUNDLE_FACTOR * n_rows / n_harvesters),
        )
        cbba_metrics = _planner_metrics(cbba.result, tasks, harvesters, dp.assignment)
        cbba_metrics["auction_rounds"] = float(cbba.iterations)
        cbba_metrics["records"] = float(cbba.messages)
        cbba_metrics["coverage_fraction"] = 1.0 - cbba.unassigned / len(tasks)
        cbba_metrics["converged"] = float(cbba.converged)
        cbba_metrics["conflicts"] = float(cbba.conflicts)
        raw["standard_cbba"].append(cbba_metrics)

        for method, mode, spatial, balancing in (
            ("auction_only", "gscbba", True, False),
            ("full", "gscbba", True, True),
        ):
            _, sim_tasks, sim_harvesters = _scenario(
                seed,
                n_harvesters,
                n_rows,
                extent,
                speed_range=speed_range,
            )
            result = Simulator(
                sim_tasks,
                sim_harvesters,
                comm_range=INF_RANGE,
                hops=2,
                mode=mode,
                spatial=spatial,
                balancing=balancing,
                **METHOD,
            ).run()
            raw[method].append(_sim_metrics(result, dp.assignment))

        print(
            f"  [bench] N={n_harvesters:2d} M={n_rows:3d} trial={tr:2d} "
            f"DP={dp.makespan:7.1f}s full={raw['full'][-1]['makespan']:7.1f}s "
            f"auction={raw['auction_only'][-1]['makespan']:7.1f}s "
            f"cbba={raw['standard_cbba'][-1]['makespan']:7.1f}s"
        )

    return {
        "N": n_harvesters,
        "M": n_rows,
        "extent": extent,
        "trials": trials,
        "speed_range": list(speed_range) if speed_range is not None else None,
        "methods": _summarize_methods(dict(raw)),
    }


def _run_milp_certification(trials: int) -> dict:
    records = []
    try:
        import_module("pulp")
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "PuLP is not installed",
            "trials": 0,
            "records": [],
        }

    for tr in range(trials):
        seed = 15_000 + tr
        _, tasks, harvesters = _scenario(seed, 3, 6, 80.0, speed_range=(1.0, 2.0))
        dp = optimal_contiguous_partition(tasks, harvesters)
        milp = milp_assignment(tasks, harvesters, time_limit=120.0)
        records.append(
            {
                "seed": seed,
                "dp_makespan": dp.makespan,
                "milp_makespan": milp.makespan,
                "dp_gap_fraction": dp.makespan / milp.makespan - 1.0,
                "milp_solve_time": milp.solve_time,
                "status": milp.status,
            }
        )
    return {
        "status": "complete",
        "trials": trials,
        "N": 3,
        "M": 6,
        "cost_model": "continuous nearest-endpoint routes",
        "records": records,
    }


def run_benchmark(trials: int, heterogeneous_trials: int = 40) -> dict:
    scalability = {}
    for n_harvesters in (2, 4, 6, 8, 10):
        scalability[str(n_harvesters)] = _run_benchmark_case(
            n_harvesters,
            100,
            200.0,
            trials,
            seed_base=10_000,
        )
    heterogeneous = _run_benchmark_case(
        4,
        100,
        200.0,
        heterogeneous_trials,
        seed_base=50_000,
        speed_range=(1.0, 2.0),
    )
    return {
        "scalability": scalability,
        "heterogeneous": heterogeneous,
        "standard_cbba": {
            "discount": CBBA_DISCOUNT,
            "bundle_limit_factor": CBBA_BUNDLE_FACTOR,
            "tuning_seeds": [900, 901, 902],
        },
        "milp_small": _run_milp_certification(min(3, trials)),
    }


def run_commrange(trials: int) -> dict:
    N, M, extent = 6, 100, 200.0
    Rcs = [25, 50, 75, 100, 150, 300]
    data = {"1": {}, "2": {}}
    poly = square_field(extent)
    for hops in (1, 2):
        for Rc in Rcs:
            ct, td, dup, coverage, capped = [], [], [], [], []
            for tr in range(trials):
                rng = np.random.default_rng(20_000 + 91 * int(Rc) + tr)
                heading = rng.uniform(82.0, 98.0)
                tasks = make_rows(poly, n_rows=M, heading_deg=heading)
                harv = make_harvesters(N, poly, rng, layout="headland", max_bundle=M)
                r = Simulator(
                    tasks,
                    harv,
                    comm_range=float(Rc),
                    hops=hops,
                    spatial=True,
                    **METHOD,
                ).run()
                ct.append(r.completion_time)
                td.append(r.total_distance)
                dup.append(r.duplicate_services)
                coverage.append(r.coverage_fraction)
                capped.append(float(r.allocation_stable.count(False)))
            data[str(hops)][str(Rc)] = {
                "completion_time": _stats(ct),
                "total_distance": _stats(td),
                "duplicate_services": _stats(dup),
                "coverage_fraction": _stats(coverage),
                "capped_allocation_cycles": _stats(capped),
            }
            print(
                f"  [comm] hops={hops} Rc={Rc:3d}  "
                f"t={data[str(hops)][str(Rc)]['completion_time']['mean']:7.1f}s "
                f"dist={data[str(hops)][str(Rc)]['total_distance']['mean']:6.0f}m "
                f"dup={data[str(hops)][str(Rc)]['duplicate_services']['mean']:.2f}"
            )
    return {
        "N": N,
        "M": M,
        "extent": extent,
        "Rcs": Rcs,
        "trials": trials,
        "data": data,
    }


def run_loss(trials: int) -> dict:
    n_harvesters, n_rows, extent = 6, 100, 200.0
    ranges = [50.0, 100.0, INF_RANGE]
    failure_fractions = [0.10, 0.30, 0.60]
    failure_counts = [1, 2]
    data: dict[str, Any] = {}

    for comm_range in ranges:
        range_key = "inf" if comm_range == INF_RANGE else str(int(comm_range))
        data[range_key] = {}
        for fraction in failure_fractions:
            data[range_key][str(fraction)] = {}
            for failure_count in failure_counts:
                metrics = defaultdict(list)
                for tr in range(trials):
                    seed = 30_000 + tr
                    _, tasks, harvesters = _scenario(seed, n_harvesters, n_rows, extent)
                    baseline = Simulator(
                        tasks,
                        harvesters,
                        comm_range=comm_range,
                        hops=2,
                        spatial=True,
                        **METHOD,
                    ).run()
                    event_times = [fraction * baseline.completion_time]
                    event_ids = [n_harvesters // 2]
                    if failure_count == 2:
                        event_times.append(
                            min(0.90, fraction + 0.30) * baseline.completion_time
                        )
                        event_ids.append(n_harvesters // 2 + 1)

                    _, loss_tasks, loss_harvesters = _scenario(
                        seed, n_harvesters, n_rows, extent
                    )
                    result = Simulator(
                        loss_tasks,
                        loss_harvesters,
                        comm_range=comm_range,
                        hops=2,
                        spatial=True,
                        loss_events=list(zip(event_times, event_ids)),
                        **METHOD,
                    ).run()
                    metrics["baseline_makespan"].append(baseline.completion_time)
                    metrics["makespan"].append(result.completion_time)
                    metrics["overhead_fraction"].append(
                        result.completion_time / baseline.completion_time - 1.0
                    )
                    metrics["reconvergence_rounds"].append(
                        float(sum(result.reconvergence_rounds))
                    )
                    metrics["reconvergence_time"].append(
                        float(sum(result.reconvergence_time))
                    )
                    metrics["rows_rebid"].append(float(result.rows_rebid))
                    metrics["duplicate_services"].append(
                        float(result.duplicate_services)
                    )
                    metrics["coverage_fraction"].append(result.coverage_fraction)
                    metrics["capped_allocation_cycles"].append(
                        float(result.allocation_stable.count(False))
                    )
                data[range_key][str(fraction)][str(failure_count)] = {
                    key: _stats(values) for key, values in metrics.items()
                }
                print(
                    f"  [loss] Rc={range_key:>3s} at={fraction:.0%} "
                    f"count={failure_count} overhead="
                    f"{100 * np.mean(metrics['overhead_fraction']):5.1f}%"
                )

    return {
        "N": n_harvesters,
        "M": n_rows,
        "extent": extent,
        "trials": trials,
        "comm_ranges": [50, 100, "inf"],
        "failure_fractions": failure_fractions,
        "failure_counts": failure_counts,
        "loss_detect_delay": 4.0,
        "data": data,
    }


def _event_options(name: str, baseline_time: float, extent: float) -> dict:
    joining = Harvester(
        hid=6,
        pos=np.array([0.5 * extent, 0.5]),
        speed=1.5,
        max_bundle=100,
    )
    if name == "one_loss":
        return {"loss_events": [(0.30 * baseline_time, 3)]}
    if name == "two_losses":
        return {
            "loss_events": [
                (0.20 * baseline_time, 2),
                (0.60 * baseline_time, 4),
            ]
        }
    if name == "one_join":
        return {"join_events": [(0.30 * baseline_time, joining)]}
    if name == "loss_then_join":
        return {
            "loss_events": [(0.20 * baseline_time, 3)],
            "join_events": [(0.60 * baseline_time, joining)],
        }
    if name == "speed_drop":
        return {"speed_events": [(0.30 * baseline_time, 3, 0.60)]}
    raise ValueError(f"unknown event scenario {name!r}")


EVENT_SCENARIOS = [
    "one_loss",
    "two_losses",
    "one_join",
    "loss_then_join",
    "speed_drop",
]


def _event_row(result) -> dict:
    return {
        "completion_time": result.completion_time,
        "coverage_fraction": result.coverage_fraction,
        "completed": float(result.converged),
        "duplicate_services": float(result.duplicate_services),
        "capped_allocation_cycles": float(result.allocation_stable.count(False)),
        "rows_rebid": float(result.rows_rebid),
        "reconvergence_rounds": float(sum(result.reconvergence_rounds)),
        "reconvergence_time": float(sum(result.reconvergence_time)),
        "idle_time": float(sum(result.per_harvester_idle)),
        "false_failure_detections": float(result.false_failure_detections),
        "detection_delay": float(np.mean(result.detection_delays))
        if result.detection_delays
        else 0.0,
        "msg_dropped": float(result.msg_dropped),
        "planning_delay": float(result.planning_delay),
        "planner_calls": float(result.planner_calls),
    }


def _summarize_rows(rows: list[dict], reference: list[float]) -> dict:
    summary: dict[str, Any] = {
        key: _stats([row[key] for row in rows]) for key in rows[0]
    }
    if all(row["completed"] for row in rows):
        summary["paired_completion_vs_centralized"] = _paired_stats(
            [row["completion_time"] for row in rows], reference
        )
    else:
        summary["paired_completion_vs_centralized"] = None
    return summary


def _base_position(extent: float) -> np.ndarray:
    return np.array([0.5 * extent, 0.5])


def _run_policy(
    seed: int,
    n_harvesters: int,
    n_rows: int,
    extent: float,
    scenario: str,
    baseline_time: float,
    initial: dict,
    comm_range: float,
    policy: str,
    information: str,
    **network,
):
    _, tasks, harvesters = _scenario(seed, n_harvesters, n_rows, extent)
    options: dict[str, Any] = dict(_event_options(scenario, baseline_time, extent))
    if policy == "decentralized":
        options.update(METHOD)
    else:
        options["initial_assignment"] = initial
    if policy == "centralized_base":
        options["base_position"] = _base_position(extent)
    return Simulator(
        tasks,
        harvesters,
        comm_range=comm_range,
        hops=2,
        spatial=True,
        allocation_policy=policy,
        information=information,
        t_max=3.0 * baseline_time,
        comm_seed=seed,
        **options,
        **network,
    ).run()


def run_events(trials: int) -> dict:
    n_harvesters, n_rows, extent = 6, 100, 200.0
    ranges = {"full": INF_RANGE, "100": 100.0, "50": 50.0}
    informations = ("shared", "local")
    methods = ("frozen", "centralized_resolve", "centralized_base", "decentralized")
    policy_of = {
        "frozen": "frozen",
        "centralized_resolve": "centralized",
        "centralized_base": "centralized_base",
        "decentralized": "decentralized",
    }
    raw = {
        info: {
            range_name: {
                scenario: {method: [] for method in methods}
                for scenario in EVENT_SCENARIOS
            }
            for range_name in ranges
        }
        for info in informations
    }
    for tr in range(trials):
        seed = 60_000 + tr
        _, tasks, harvesters = _scenario(seed, n_harvesters, n_rows, extent)
        initial = optimal_contiguous_partition(tasks, harvesters)
        baseline_time = initial.makespan
        for range_name, comm_range in ranges.items():
            for scenario in EVENT_SCENARIOS:
                shared_rows = {}
                for method in ("frozen", "centralized_resolve"):
                    result = _run_policy(
                        seed,
                        n_harvesters,
                        n_rows,
                        extent,
                        scenario,
                        baseline_time,
                        initial.assignment,
                        comm_range,
                        policy_of[method],
                        "shared",
                    )
                    shared_rows[method] = _event_row(result)
                for info in informations:
                    for method in methods:
                        if method in shared_rows:
                            raw[info][range_name][scenario][method].append(
                                dict(shared_rows[method])
                            )
                            continue
                        result = _run_policy(
                            seed,
                            n_harvesters,
                            n_rows,
                            extent,
                            scenario,
                            baseline_time,
                            initial.assignment,
                            comm_range,
                            policy_of[method],
                            info,
                        )
                        raw[info][range_name][scenario][method].append(
                            _event_row(result)
                        )
                print(
                    f"  [event] trial={tr:2d} Rc={range_name:>4s} scenario={scenario} "
                    + " ".join(
                        f"{info[:3]}:dec="
                        f"{raw[info][range_name][scenario]['decentralized'][-1]['completion_time']:.0f}"
                        f"/base="
                        f"{raw[info][range_name][scenario]['centralized_base'][-1]['completion_time']:.0f}"
                        for info in informations
                    )
                )

    data: dict[str, Any] = {}
    for info in informations:
        data[info] = {}
        for range_name in ranges:
            data[info][range_name] = {}
            for scenario in EVENT_SCENARIOS:
                cell = raw[info][range_name][scenario]
                reference = [
                    row["completion_time"] for row in cell["centralized_resolve"]
                ]
                data[info][range_name][scenario] = {
                    method: _summarize_rows(cell[method], reference)
                    for method in methods
                }
                base_times = [
                    row["completion_time"] for row in cell["centralized_base"]
                ]
                if all(row["completed"] for row in cell["decentralized"]) and all(
                    row["completed"] for row in cell["centralized_base"]
                ):
                    data[info][range_name][scenario]["decentralized"][
                        "paired_completion_vs_base"
                    ] = _paired_stats(
                        [row["completion_time"] for row in cell["decentralized"]],
                        base_times,
                    )

    return {
        "N": n_harvesters,
        "M": n_rows,
        "extent": extent,
        "trials": trials,
        "scenarios": EVENT_SCENARIOS,
        "informations": list(informations),
        "communication_ranges": {"full": "inf", "100": 100, "50": 50},
        "methods": list(methods),
        "base_position": _base_position(extent).tolist(),
        "heartbeat_timeout": 6.0,
        "claim_ttl": 240.0,
        "data": data,
    }


def run_network(trials: int) -> dict:
    n_harvesters, n_rows, extent = 6, 100, 200.0
    comm_range = 100.0
    conditions = {
        "ideal": (0.0, 0.0),
        "loss10": (0.1, 0.0),
        "loss20": (0.2, 0.0),
        "loss30": (0.3, 0.0),
        "lat50ms": (0.0, 0.05),
        "lat250ms": (0.0, 0.25),
        "lat1s": (0.0, 1.0),
        "loss20_lat250ms": (0.2, 0.25),
    }
    scenarios = ["one_loss", "two_losses", "speed_drop"]
    methods = ("centralized_base", "decentralized")
    raw = {
        name: {scenario: {method: [] for method in methods} for scenario in scenarios}
        for name in conditions
    }
    references: dict[str, dict[int, float]] = {scenario: {} for scenario in scenarios}
    for tr in range(trials):
        seed = 90_000 + tr
        _, tasks, harvesters = _scenario(seed, n_harvesters, n_rows, extent)
        initial = optimal_contiguous_partition(tasks, harvesters)
        baseline_time = initial.makespan
        for scenario in scenarios:
            oracle = _run_policy(
                seed,
                n_harvesters,
                n_rows,
                extent,
                scenario,
                baseline_time,
                initial.assignment,
                comm_range,
                "centralized",
                "shared",
            )
            references[scenario][tr] = oracle.completion_time
            for name, (loss, latency) in conditions.items():
                for method in methods:
                    result = _run_policy(
                        seed,
                        n_harvesters,
                        n_rows,
                        extent,
                        scenario,
                        baseline_time,
                        initial.assignment,
                        comm_range,
                        method,
                        "local",
                        packet_loss=loss,
                        round_latency=latency,
                        latency_jitter=latency,
                    )
                    raw[name][scenario][method].append(_event_row(result))
            print(f"  [network] trial={tr:2d} scenario={scenario}")
    data: dict[str, Any] = {}
    for name in conditions:
        data[name] = {}
        for scenario in scenarios:
            reference = [references[scenario][tr] for tr in range(trials)]
            data[name][scenario] = {
                method: _summarize_rows(raw[name][scenario][method], reference)
                for method in methods
            }
    return {
        "N": n_harvesters,
        "M": n_rows,
        "extent": extent,
        "trials": trials,
        "comm_range": comm_range,
        "information": "local",
        "conditions": {
            name: {"packet_loss": loss, "round_latency": latency, "jitter": latency}
            for name, (loss, latency) in conditions.items()
        },
        "scenarios": scenarios,
        "methods": list(methods),
        "oracle_completion": {
            scenario: _stats([references[scenario][tr] for tr in range(trials)])
            for scenario in scenarios
        },
        "data": data,
    }


def run_turncost(trials: int) -> dict:
    n_harvesters, n_rows, extent = 6, 100, 200.0
    turn_distances = [0, 10, 20, 30, 40]
    raw = {str(turn): defaultdict(list) for turn in turn_distances}
    zero_partitions = {}
    for tr in range(trials):
        seed = 70_000 + tr
        for turn_distance in turn_distances:
            _, tasks, harvesters = _scenario(seed, n_harvesters, n_rows, extent)
            dp = optimal_contiguous_partition(
                tasks, harvesters, turn_distance=float(turn_distance)
            )
            total_row_length = sum(task.length for task in tasks)
            lower_bound = total_row_length / sum(h.speed for h in harvesters)
            dp_replay = _planner_metrics(
                dp, tasks, harvesters, turn_distance=float(turn_distance)
            )
            result = Simulator(
                tasks,
                harvesters,
                comm_range=INF_RANGE,
                hops=2,
                spatial=True,
                turn_distance=float(turn_distance),
                **METHOD,
            ).run()
            if turn_distance == 0:
                zero_partitions[tr] = result.initial_assignment
            values = raw[str(turn_distance)]
            values["makespan"].append(result.completion_time)
            values["deadhead_share"].append(
                (result.deadhead_distance + result.turn_distance)
                / result.total_distance
            )
            values["gap_to_dp"].append(
                result.completion_time / dp_replay["makespan"] - 1.0
            )
            values["gap_to_lower_bound"].append(
                result.completion_time / lower_bound - 1.0
            )
            values["partition_changed"].append(
                float(
                    turn_distance > 0
                    and rows_differ(result.initial_assignment, zero_partitions[tr]) > 0
                )
            )
        print(f"  [turn] trial={tr:2d} complete")
    return {
        "N": n_harvesters,
        "M": n_rows,
        "extent": extent,
        "turn_distances": turn_distances,
        "trials": trials,
        "data": {
            turn: {key: _stats(values) for key, values in metrics.items()}
            for turn, metrics in raw.items()
        },
    }


def run_ksweep(trials: int) -> dict:
    n_harvesters, n_rows, extent = 6, 100, 200.0
    candidate_counts = [8, 12, 16, 24, 32, 48, n_rows]
    ranges = {"full": INF_RANGE, "75": 75.0}
    data = {name: {} for name in ranges}
    for range_name, comm_range in ranges.items():
        for candidate_count in candidate_counts:
            metrics = defaultdict(list)
            for tr in range(trials):
                seed = 80_000 + tr
                _, tasks, harvesters = _scenario(seed, n_harvesters, n_rows, extent)
                result = Simulator(
                    tasks,
                    harvesters,
                    comm_range=comm_range,
                    hops=2,
                    spatial=candidate_count < n_rows,
                    query_k=candidate_count,
                    **METHOD,
                ).run()
                metrics["makespan"].append(result.completion_time)
                metrics["rows_scored"].append(float(result.greedy_rows_scored))
                metrics["auction_ms"].append(1000.0 * result.greedy_time)
                metrics["duplicate_services"].append(float(result.duplicate_services))
                metrics["coverage_gaps"].append(
                    float(result.n_tasks) * (1.0 - result.coverage_fraction)
                )
            key = "all" if candidate_count == n_rows else str(candidate_count)
            data[range_name][key] = {
                metric: _stats(values) for metric, values in metrics.items()
            }
            print(
                f"  [ksweep] Rc={range_name:>4s} k={key:>3s} "
                f"makespan={np.mean(metrics['makespan']):7.1f}s"
            )
    return {
        "N": n_harvesters,
        "M": n_rows,
        "extent": extent,
        "candidate_counts": [8, 12, 16, 24, 32, 48, "all"],
        "trials": trials,
        "data": data,
    }


def run_ablation(trials: int) -> dict:
    Ns = [2, 4, 6, 8, 10]
    M, extent, k = 100, 200.0, 24
    poly = square_field(extent)
    data = {}
    for N in Ns:
        ct_off, ct_on, eff = [], [], []
        rounds_off, rounds_on = [], []
        rows_off, rows_on, ms_off, ms_on = [], [], [], []
        dup_off, dup_on = [], []
        for tr in range(trials):
            seed = 10_000 + 137 * N + tr
            for spatial, ct, rd, rw, ms, dp in (
                (False, ct_off, rounds_off, rows_off, ms_off, dup_off),
                (True, ct_on, rounds_on, rows_on, ms_on, dup_on),
            ):
                rng = np.random.default_rng(seed)
                heading = rng.uniform(80.0, 100.0)
                tasks = make_rows(poly, n_rows=M, heading_deg=heading)
                harv = make_harvesters(N, poly, rng, layout="headland", max_bundle=M)
                r = Simulator(
                    tasks,
                    harv,
                    comm_range=INF_RANGE,
                    hops=2,
                    spatial=spatial,
                    query_k=k,
                    **METHOD,
                ).run()
                ct.append(r.completion_time)
                rd.append(r.alloc_rounds / max(1, r.alloc_cycles))
                rw.append(r.greedy_rows_scored)
                ms.append(1000.0 * r.greedy_time)
                dp.append(r.duplicate_services)
                if spatial is False:
                    L = sum(t.length for t in tasks)
                    c_lb = L / sum(h.speed for h in harv)
                    eff.append(c_lb / r.completion_time)
        data[str(N)] = {
            "completion_off": _stats(ct_off),
            "completion_on": _stats(ct_on),
            "efficiency": _stats(eff),
            "rounds_off": _stats(rounds_off),
            "rounds_on": _stats(rounds_on),
            "rows_scored_off": _stats(rows_off),
            "rows_scored_on": _stats(rows_on),
            "greedy_ms_off": _stats(ms_off),
            "greedy_ms_on": _stats(ms_on),
            "duplicate_off": _stats(dup_off),
            "duplicate_on": _stats(dup_on),
        }
        d = data[str(N)]
        speedup = d["rows_scored_off"]["mean"] / max(1e-9, d["rows_scored_on"]["mean"])
        print(
            f"  [abla] N={N:2d}  makespan off/on={d['completion_off']['mean']:7.1f}/"
            f"{d['completion_on']['mean']:7.1f}s  "
            f"rows off/on={d['rows_scored_off']['mean']:6.0f}/{d['rows_scored_on']['mean']:5.0f} "
            f"({speedup:4.1f}x)  ms off/on={d['greedy_ms_off']['mean']:6.1f}/{d['greedy_ms_on']['mean']:5.1f}"
        )
    return {
        "Ns": Ns,
        "M": M,
        "extent": extent,
        "query_k": k,
        "trials": trials,
        "data": data,
    }


def run_timing(trials: int) -> dict:
    N, row_length, spacing, k = 6, 200.0, 2.0, 24
    Ms = [50, 100, 200, 400, 800, 1600]
    data = {}
    for M in Ms:
        width = spacing * M
        poly = rect_field(width, row_length)
        rows_off, rows_on, ms_off, ms_on = [], [], [], []
        stable_off, stable_on = [], []
        for tr in range(trials):
            tasks = make_rows(poly, n_rows=M, heading_deg=90.0)
            for spatial, rw, ms in (
                (False, rows_off, ms_off),
                (True, rows_on, ms_on),
            ):
                rng2 = np.random.default_rng(40_000 + 7 * M + tr)
                harv = make_harvesters(N, poly, rng2, layout="headland", max_bundle=M)
                sim = Simulator(
                    tasks,
                    harv,
                    comm_range=INF_RANGE,
                    hops=2,
                    spatial=spatial,
                    query_k=k,
                    alloc_max_rounds=12,
                )
                sim._allocation_cycle()
                (stable_on if spatial else stable_off).append(
                    float(all(sim.allocation_stable))
                )
                rw.append(sum(h.rows_scored for h in sim.harvesters))
                ms.append(1000.0 * sim.greedy_time)
        data[str(M)] = {
            "rows_scored_off": _stats(rows_off),
            "rows_scored_on": _stats(rows_on),
            "greedy_ms_off": _stats(ms_off),
            "greedy_ms_on": _stats(ms_on),
            "stable_off": _stats(stable_off),
            "stable_on": _stats(stable_on),
        }
        d = data[str(M)]
        print(
            f"  [time] M={M:5d}  rows off/on={d['rows_scored_off']['mean']:8.0f}/"
            f"{d['rows_scored_on']['mean']:6.0f}  "
            f"ms off/on={d['greedy_ms_off']['mean']:8.2f}/{d['greedy_ms_on']['mean']:6.2f}"
        )
    return {
        "N": N,
        "Ms": Ms,
        "row_length": row_length,
        "row_spacing": spacing,
        "query_k": k,
        "trials": trials,
        "data": data,
    }


def run_diagnostics(trials: int, speed_trials: int = 40) -> dict:
    extent, n_rows = 200.0, 100
    polygon = square_field(extent)

    rng = np.random.default_rng(6)
    tasks = make_rows(polygon, n_rows=n_rows, heading_deg=90.0)
    harvesters = make_harvesters(6, polygon, rng, max_bundle=n_rows)
    consensus_result = Simulator(
        tasks, harvesters, comm_range=INF_RANGE, hops=2, spatial=True
    ).run()
    consensus = {
        "per_harvester_tasks": consensus_result.per_harvester_tasks,
        "per_harvester_finish": consensus_result.per_harvester_finish,
    }

    convergence = {}
    for n_harvesters in (2, 4, 6, 10):
        traces = []
        for tr in range(trials):
            seed = 321 + 13 * n_harvesters + tr
            rng = np.random.default_rng(seed)
            tasks = make_rows(
                polygon,
                n_rows=n_rows,
                heading_deg=rng.uniform(80.0, 100.0),
            )
            harvesters = make_harvesters(n_harvesters, polygon, rng, max_bundle=n_rows)
            simulator = Simulator(
                tasks,
                harvesters,
                comm_range=INF_RANGE,
                hops=2,
                spatial=True,
            )
            simulator._allocation_cycle()
            traces.append(simulator.alloc_trace)
        convergence[str(n_harvesters)] = traces

    speeds = [1.6, 2.0, 1.0, 1.3]
    rng = np.random.default_rng(0)
    speed_tasks = make_rows(polygon, n_rows=80, heading_deg=90.0)
    speed_harvesters = make_harvesters(
        4,
        polygon,
        rng,
        speeds=speeds,
        max_bundle=80,
    )
    starts = [h.start_pos.tolist() for h in speed_harvesters]
    speed_simulator = Simulator(
        speed_tasks,
        speed_harvesters,
        comm_range=INF_RANGE,
        hops=2,
        spatial=True,
    )
    speed_simulator.run()
    rows_by_speed = defaultdict(list)
    for tr in range(speed_trials):
        trial_rng = np.random.default_rng(1000 + tr)
        trial_speeds = list(speeds)
        trial_rng.shuffle(trial_speeds)
        trial_tasks = make_rows(
            polygon,
            n_rows=80,
            heading_deg=trial_rng.uniform(80.0, 100.0),
        )
        trial_harvesters = make_harvesters(
            4,
            polygon,
            trial_rng,
            speeds=trial_speeds,
            max_bundle=80,
        )
        result = Simulator(
            trial_tasks,
            trial_harvesters,
            comm_range=INF_RANGE,
            hops=2,
            spatial=True,
        ).run()
        for speed, count in zip(result.per_harvester_speed, result.per_harvester_tasks):
            rows_by_speed[f"{speed:.1f}"].append(count)
    speed = {
        "M": 80,
        "N": 4,
        "speeds": speeds,
        "trials": speed_trials,
        "starts": starts,
        "tasks": [
            {"id": task.id, "a": task.a.tolist(), "b": task.b.tolist()}
            for task in speed_tasks
        ],
        "owner": {str(tid): hid for tid, hid in speed_simulator.completed_by.items()},
        "rows_by_speed": {
            value: _stats(counts) for value, counts in rows_by_speed.items()
        },
    }
    return {
        "extent": extent,
        "M": n_rows,
        "trials": trials,
        "consensus": consensus,
        "convergence": convergence,
        "speed": speed,
    }


def main(argv: list[str] | None = None) -> None:
    global RESULTS_DIR
    from . import __version__

    ap = argparse.ArgumentParser(description="Simulate multi-harvester row allocation.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--quick", action="store_true", help="fewer trials")
    ap.add_argument(
        "--output-dir", help="result directory (quick runs use quick-results)"
    )
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument(
        "--only",
        choices=[
            "scalability",
            "benchmark",
            "commrange",
            "events",
            "turncost",
            "loss",
            "ksweep",
            "ablation",
            "timing",
            "diagnostics",
            "network",
        ],
        default=None,
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="regenerate every result and figure",
    )
    selection.add_argument(
        "--figures-only", action="store_true", help="plot the cached results"
    )
    args = ap.parse_args(argv)

    if not (args.all or args.only or args.figures_only):
        ap.print_help()
        return

    if args.output_dir:
        RESULTS_DIR = os.path.abspath(args.output_dir)
    elif args.quick:
        RESULTS_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "quick-results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trials = 2 if args.quick else 10
    speed_trials = 4 if args.quick else 40

    jobs = {
        "scalability": lambda: run_scalability(trials),
        "benchmark": lambda: run_benchmark(trials, speed_trials),
        "commrange": lambda: run_commrange(trials),
        "events": lambda: run_events(trials),
        "turncost": lambda: run_turncost(trials),
        "loss": lambda: run_loss(trials),
        "ksweep": lambda: run_ksweep(trials),
        "ablation": lambda: run_ablation(trials),
        "timing": lambda: run_timing(trials),
        "diagnostics": lambda: run_diagnostics(trials, speed_trials),
        "network": lambda: run_network(trials),
    }
    if args.figures_only:
        jobs = {}
    elif args.only:
        jobs = {args.only: jobs[args.only]}

    for name, fn in jobs.items():
        print(f"== {name} ==")
        t0 = time.time()
        result = fn()
        result["metadata"] = {
            "schema_version": 2,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                name: version(name) for name in ("numpy", "scipy", "matplotlib")
            },
            "quick": args.quick,
            "assumptions": "shared completion state, global event announcements, and cost-free synchronous rounds unless an experiment sets information='local' or network latency and loss",
        }
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"   -> {path}  ({time.time() - t0:.1f}s)\n")

    if args.all or args.figures_only:
        from . import make_figures

        make_figures.RESULTS = RESULTS_DIR
        if args.quick or args.output_dir:
            make_figures.FIGDIR = os.path.join(RESULTS_DIR, "figures")
            os.makedirs(make_figures.FIGDIR, exist_ok=True)
        else:
            make_figures.FIGDIR = os.path.abspath("figures")
        make_figures.main()


if __name__ == "__main__":
    main()
