from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from .cbba import Harvester, Task


@dataclass(frozen=True)
class RouteMetrics:
    task_ids: tuple[int, ...]
    finish_time: float
    total_distance: float
    row_distance: float
    deadhead_distance: float
    turn_distance: float


@dataclass(frozen=True)
class AllocationResult:
    assignment: dict[int, list[int]]
    makespan: float
    finish_times: dict[int, float]
    distances: dict[int, float]
    total_distance: float
    row_distance: float
    deadhead_distance: float
    turn_distance: float
    solve_time: float = 0.0
    status: str = "optimal"


def _task_map(tasks: Sequence[Task] | Mapping[int, Task]) -> dict[int, Task]:
    if isinstance(tasks, Mapping):
        return dict(tasks)
    return {task.id: task for task in tasks}


def _axes(tasks: Mapping[int, Task]) -> tuple[np.ndarray, np.ndarray]:
    first = next(iter(tasks.values()))
    along = first.b - first.a
    norm = float(np.linalg.norm(along)) or 1.0
    along = along / norm
    across = np.array([-along[1], along[0]])
    return along, across


def sorted_rows(tasks: Sequence[Task] | Mapping[int, Task]) -> list[int]:
    task_map = _task_map(tasks)
    _, across = _axes(task_map)
    return sorted(task_map, key=lambda tid: float(task_map[tid].centroid @ across))


def sorted_harvesters(
    tasks: Sequence[Task] | Mapping[int, Task], harvesters: Sequence[Harvester]
) -> list[Harvester]:
    task_map = _task_map(tasks)
    _, across = _axes(task_map)
    return sorted(harvesters, key=lambda h: float(h.pos @ across))


def route_metrics(
    tasks: Sequence[Task] | Mapping[int, Task],
    harvester: Harvester,
    task_ids: Iterable[int],
    *,
    turn_distance: float = 0.0,
    turn_time: float = 0.0,
) -> RouteMetrics:
    task_map = _task_map(tasks)
    ids = tuple(task_ids)
    pos = np.asarray(harvester.pos, dtype=float)
    deadhead = 0.0
    row_distance = 0.0
    for tid in ids:
        task = task_map[tid]
        da = float(np.linalg.norm(pos - task.a))
        db = float(np.linalg.norm(pos - task.b))
        if da <= db:
            deadhead += da
            pos = task.b
        else:
            deadhead += db
            pos = task.a
        row_distance += task.length
    transitions = max(0, len(ids) - 1)
    turns = transitions * turn_distance
    distance = deadhead + row_distance + turns
    finish = distance / harvester.speed + transitions * turn_time
    return RouteMetrics(
        task_ids=ids,
        finish_time=finish,
        total_distance=distance,
        row_distance=row_distance,
        deadhead_distance=deadhead,
        turn_distance=turns,
    )


def evaluate_assignment(
    tasks: Sequence[Task] | Mapping[int, Task],
    harvesters: Sequence[Harvester],
    assignment: Mapping[int, Sequence[int]],
    *,
    turn_distance: float = 0.0,
    turn_time: float = 0.0,
    solve_time: float = 0.0,
    status: str = "optimal",
) -> AllocationResult:
    task_map = _task_map(tasks)
    finish: dict[int, float] = {}
    distances: dict[int, float] = {}
    row_distance = deadhead = turns = 0.0
    normalized: dict[int, list[int]] = {}
    for harvester in harvesters:
        tids = list(assignment.get(harvester.hid, []))
        metrics = route_metrics(
            task_map,
            harvester,
            tids,
            turn_distance=turn_distance,
            turn_time=turn_time,
        )
        normalized[harvester.hid] = tids
        finish[harvester.hid] = metrics.finish_time
        distances[harvester.hid] = metrics.total_distance
        row_distance += metrics.row_distance
        deadhead += metrics.deadhead_distance
        turns += metrics.turn_distance
    return AllocationResult(
        assignment=normalized,
        makespan=max(finish.values(), default=0.0),
        finish_times=finish,
        distances=distances,
        total_distance=row_distance + deadhead + turns,
        row_distance=row_distance,
        deadhead_distance=deadhead,
        turn_distance=turns,
        solve_time=solve_time,
        status=status,
    )


def optimal_contiguous_partition(
    tasks: Sequence[Task] | Mapping[int, Task],
    harvesters: Sequence[Harvester],
    *,
    turn_distance: float = 0.0,
    turn_time: float = 0.0,
    ready_times: Mapping[int, float] | None = None,
    initial_turns: Iterable[int] = (),
) -> AllocationResult:
    import time

    started = time.perf_counter()
    task_map = _task_map(tasks)
    ready = dict(ready_times or {})
    turning = set(initial_turns)
    if any(delay < 0 for delay in ready.values()):
        raise ValueError("ready times cannot be negative")
    if not task_map:
        result = evaluate_assignment(tasks, harvesters, {}, solve_time=0.0)
        finish = {h.hid: ready.get(h.hid, 0.0) for h in harvesters}
        return replace(
            result, finish_times=finish, makespan=max(finish.values(), default=0.0)
        )
    rows = sorted_rows(task_map)
    machines = sorted_harvesters(task_map, harvesters)
    n, m = len(machines), len(rows)
    if n == 0:
        raise ValueError("at least one harvester is required")

    costs = np.zeros((n, m + 1, m + 1), dtype=float)
    for i, harvester in enumerate(machines):
        delay = ready.get(harvester.hid, 0.0)
        if delay < 0:
            raise ValueError("ready times cannot be negative")
        costs[i, :, :] = delay
        for a in range(m):
            pos = np.asarray(harvester.pos, dtype=float)
            distance = turn_distance if harvester.hid in turning else 0.0
            for b in range(a, m):
                task = task_map[rows[b]]
                da = float(np.linalg.norm(pos - task.a))
                db = float(np.linalg.norm(pos - task.b))
                if b > a:
                    distance += turn_distance
                if da <= db:
                    distance += da + task.length
                    pos = task.b
                else:
                    distance += db + task.length
                    pos = task.a
                transitions = b - a + int(harvester.hid in turning)
                costs[i, a, b + 1] = (
                    delay + distance / harvester.speed + transitions * turn_time
                )

    inf = float("inf")
    dp = np.full((n + 1, m + 1), inf, dtype=float)
    split = np.full((n + 1, m + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for end in range(m + 1):
            best = inf
            best_start = -1
            for start in range(end + 1):
                candidate = max(dp[i - 1, start], costs[i - 1, start, end])
                if candidate < best - 1e-12:
                    best = candidate
                    best_start = start
            dp[i, end] = best
            split[i, end] = best_start

    assignment: dict[int, list[int]] = {h.hid: [] for h in machines}
    end = m
    for i in range(n, 0, -1):
        start = int(split[i, end])
        if start < 0:
            raise RuntimeError("contiguous partition reconstruction failed")
        assignment[machines[i - 1].hid] = rows[start:end]
        end = start

    result = evaluate_assignment(
        task_map,
        harvesters,
        assignment,
        turn_distance=turn_distance,
        turn_time=turn_time,
        solve_time=time.perf_counter() - started,
    )
    finish_times = {
        h.hid: result.finish_times[h.hid]
        + ready.get(h.hid, 0.0)
        + (
            turn_distance / h.speed + turn_time
            if h.hid in turning and assignment[h.hid]
            else 0.0
        )
        for h in harvesters
    }
    extra_turns = {
        h.hid: turn_distance if h.hid in turning and assignment[h.hid] else 0.0
        for h in harvesters
    }
    return replace(
        result,
        finish_times=finish_times,
        makespan=max(finish_times.values()),
        distances={
            hid: distance + extra_turns[hid]
            for hid, distance in result.distances.items()
        },
        total_distance=result.total_distance + sum(extra_turns.values()),
        turn_distance=result.turn_distance + sum(extra_turns.values()),
    )


def static_speed_split(
    tasks: Sequence[Task] | Mapping[int, Task],
    harvesters: Sequence[Harvester],
    *,
    turn_distance: float = 0.0,
    turn_time: float = 0.0,
) -> AllocationResult:
    import time

    started = time.perf_counter()
    task_map = _task_map(tasks)
    if not task_map:
        return evaluate_assignment(tasks, harvesters, {}, solve_time=0.0)
    rows = sorted_rows(task_map)
    machines = sorted_harvesters(task_map, harvesters)
    if not machines:
        raise ValueError("at least one harvester is required")
    total_work = sum(task_map[tid].length for tid in rows)
    total_speed = sum(h.speed for h in machines)
    if total_speed <= 0:
        raise ValueError("harvester speeds must be positive")

    bounds: list[float] = []
    cumulative_speed = 0.0
    for harvester in machines:
        cumulative_speed += harvester.speed
        bounds.append(total_work * cumulative_speed / total_speed)

    assignment: dict[int, list[int]] = {h.hid: [] for h in machines}
    cumulative_work = 0.0
    machine_index = 0
    for tid in rows:
        assignment[machines[machine_index].hid].append(tid)
        cumulative_work += task_map[tid].length
        while (
            machine_index < len(machines) - 1
            and cumulative_work >= bounds[machine_index] - 1e-12
        ):
            machine_index += 1

    return evaluate_assignment(
        task_map,
        harvesters,
        assignment,
        turn_distance=turn_distance,
        turn_time=turn_time,
        solve_time=time.perf_counter() - started,
        status="static",
    )


def milp_assignment(
    tasks: Sequence[Task] | Mapping[int, Task],
    harvesters: Sequence[Harvester],
    *,
    contiguity: bool = False,
    turn_distance: float = 0.0,
    turn_time: float = 0.0,
    time_limit: float = 120.0,
) -> AllocationResult:
    if contiguity:
        return optimal_contiguous_partition(
            tasks,
            harvesters,
            turn_distance=turn_distance,
            turn_time=turn_time,
        )
    try:
        import pulp
    except ImportError as exc:
        raise RuntimeError("milp_assignment requires the 'pulp' package") from exc

    import time

    started = time.perf_counter()
    task_map = _task_map(tasks)
    rows = list(task_map)
    machines = list(harvesters)
    if len(rows) > 30 or len(machines) > 4:
        raise ValueError("the exact MILP is limited to 30 rows and four harvesters")
    if not machines:
        raise ValueError("at least one harvester is required")
    if not rows:
        return evaluate_assignment(tasks, harvesters, {}, solve_time=0.0)

    oriented = [(tid, direction) for tid in rows for direction in (0, 1)]

    def entry(node):
        task = task_map[node[0]]
        return task.a if node[1] == 0 else task.b

    def exit_point(node):
        task = task_map[node[0]]
        return task.b if node[1] == 0 else task.a

    def nearest_direction(pos, tid):
        task = task_map[tid]
        return 0 if np.linalg.norm(pos - task.a) <= np.linalg.norm(pos - task.b) else 1

    problem = pulp.LpProblem("noncontiguous_row_allocation", pulp.LpMinimize)
    makespan = pulp.LpVariable("makespan", lowBound=0.0)
    visit = {
        (h.hid, node): pulp.LpVariable(
            f"visit_{h.hid}_{node[0]}_{node[1]}", cat="Binary"
        )
        for h in machines
        for node in oriented
    }
    use = {h.hid: pulp.LpVariable(f"use_{h.hid}", cat="Binary") for h in machines}
    arc = {}
    for h in machines:
        hid = h.hid
        for node in oriented:
            arc[(hid, "start", node)] = pulp.LpVariable(
                f"arc_{hid}_start_{node[0]}_{node[1]}", cat="Binary"
            )
            arc[(hid, node, "end")] = pulp.LpVariable(
                f"arc_{hid}_{node[0]}_{node[1]}_end", cat="Binary"
            )
            if node[1] != nearest_direction(h.pos, node[0]):
                problem += arc[(hid, "start", node)] == 0
        for left in oriented:
            for right in oriented:
                if left[0] == right[0]:
                    continue
                arc[(hid, left, right)] = pulp.LpVariable(
                    f"arc_{hid}_{left[0]}_{left[1]}_{right[0]}_{right[1]}",
                    cat="Binary",
                )
                if right[1] != nearest_direction(exit_point(left), right[0]):
                    problem += arc[(hid, left, right)] == 0

    for tid in rows:
        problem += (
            pulp.lpSum(visit[(h.hid, (tid, d))] for h in machines for d in (0, 1)) == 1
        )

    order = {
        (h.hid, node): pulp.LpVariable(
            f"order_{h.hid}_{node[0]}_{node[1]}", lowBound=0, upBound=len(rows)
        )
        for h in machines
        for node in oriented
    }
    for h in machines:
        hid = h.hid
        problem += pulp.lpSum(arc[(hid, "start", q)] for q in oriented) == use[hid]
        problem += pulp.lpSum(arc[(hid, p, "end")] for p in oriented) == use[hid]
        for node in oriented:
            incoming = [arc[(hid, "start", node)]] + [
                arc[(hid, left, node)] for left in oriented if left[0] != node[0]
            ]
            outgoing = [arc[(hid, node, "end")]] + [
                arc[(hid, node, right)] for right in oriented if right[0] != node[0]
            ]
            problem += pulp.lpSum(incoming) == visit[(hid, node)]
            problem += pulp.lpSum(outgoing) == visit[(hid, node)]
            problem += order[(hid, node)] >= visit[(hid, node)]
            problem += order[(hid, node)] <= len(rows) * visit[(hid, node)]
        for left in oriented:
            for right in oriented:
                if left[0] == right[0]:
                    continue
                problem += order[(hid, right)] >= order[(hid, left)] + 1 - (
                    len(rows) + 1
                ) * (1 - arc[(hid, left, right)])

        route_terms = []
        for node in oriented:
            first_leg = (
                float(np.linalg.norm(h.pos - entry(node))) + task_map[node[0]].length
            )
            route_terms.append(first_leg / h.speed * arc[(hid, "start", node)])
        for left in oriented:
            for right in oriented:
                if left[0] == right[0]:
                    continue
                leg = (
                    float(np.linalg.norm(exit_point(left) - entry(right)))
                    + turn_distance
                    + task_map[right[0]].length
                ) / h.speed + turn_time
                route_terms.append(leg * arc[(hid, left, right)])
        problem += makespan >= pulp.lpSum(route_terms)

    problem += makespan
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status].lower()
    if status != "optimal" or problem.sol_status != pulp.LpSolutionOptimal:
        solution_status = pulp.LpSolution.get(problem.sol_status, "unknown")
        raise RuntimeError(
            f"MILP solver did not certify an optimum: {status!r}; {solution_status}"
        )

    assignment: dict[int, list[int]] = {h.hid: [] for h in machines}
    oriented_assignment: dict[int, list[tuple[int, int]]] = {
        h.hid: [] for h in machines
    }
    for h in machines:
        hid = h.hid
        starts = [
            node for node in oriented if pulp.value(arc[(hid, "start", node)]) > 0.5
        ]
        if not starts:
            continue
        current = starts[0]
        seen: set[int] = set()
        while current != "end":
            tid = current[0]
            if tid in seen:
                raise RuntimeError("MILP route extraction found a cycle")
            seen.add(tid)
            assignment[hid].append(tid)
            oriented_assignment[hid].append(current)
            successors = [
                node
                for node in oriented
                if node[0] != current[0] and pulp.value(arc[(hid, current, node)]) > 0.5
            ]
            if pulp.value(arc[(hid, current, "end")]) > 0.5:
                current = "end"
            elif successors:
                current = successors[0]
            else:
                raise RuntimeError("MILP route extraction found an open route")

    finish_times: dict[int, float] = {}
    distances: dict[int, float] = {}
    total_rows = deadhead = turns = 0.0
    for harvester in machines:
        route = oriented_assignment[harvester.hid]
        if route:
            route_deadhead = float(np.linalg.norm(harvester.pos - entry(route[0])))
            route_deadhead += sum(
                float(np.linalg.norm(exit_point(left) - entry(right)))
                for left, right in zip(route, route[1:])
            )
        else:
            route_deadhead = 0.0
        route_rows = sum(task_map[node[0]].length for node in route)
        route_turns = max(0, len(route) - 1) * turn_distance
        route_distance = route_deadhead + route_rows + route_turns
        finish_times[harvester.hid] = (
            route_distance / harvester.speed + max(0, len(route) - 1) * turn_time
        )
        distances[harvester.hid] = route_distance
        total_rows += route_rows
        deadhead += route_deadhead
        turns += route_turns
    result = AllocationResult(
        assignment=assignment,
        makespan=max(finish_times.values(), default=0.0),
        finish_times=finish_times,
        distances=distances,
        total_distance=total_rows + deadhead + turns,
        row_distance=total_rows,
        deadhead_distance=deadhead,
        turn_distance=turns,
        solve_time=time.perf_counter() - started,
        status=status,
    )
    replay = evaluate_assignment(
        task_map,
        machines,
        assignment,
        turn_distance=turn_distance,
        turn_time=turn_time,
    )
    if not np.isclose(result.makespan, replay.makespan, rtol=1e-8, atol=1e-8):
        raise RuntimeError("MILP route cost does not match nearest-endpoint replay")
    return result


def rows_differ(
    left: Mapping[int, Sequence[int]], right: Mapping[int, Sequence[int]]
) -> int:
    owner_left = {tid: hid for hid, tids in left.items() for tid in tids}
    owner_right = {tid: hid for hid, tids in right.items() for tid in tids}
    return sum(
        owner_left.get(tid) != owner_right.get(tid)
        for tid in owner_left.keys() | owner_right.keys()
    )
