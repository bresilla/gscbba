from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .baselines import AllocationResult, evaluate_assignment
from .cbba import Harvester, Task

LEAVE, UPDATE, RESET = 0, 1, 2
EPS = 1e-12


@dataclass
class CBBARun:
    result: AllocationResult
    iterations: int
    converged: bool
    messages: int
    conflicts: int
    unassigned: int
    bundle_limit: int
    discount: float
    history: list[int] = field(default_factory=list)


class _Agent:
    def __init__(self, index: int, harvester: Harvester, m: int, n: int) -> None:
        self.index = index
        self.hid = harvester.hid
        self.pos = np.asarray(harvester.pos, dtype=float)
        self.speed = float(harvester.speed)
        self.bundle: list[int] = []
        self.path: list[int] = []
        self.y = np.zeros(m)
        self.z = np.full(m, -1, dtype=int)
        self.s = np.zeros(n)


class StandardCBBA:
    def __init__(
        self,
        tasks: Sequence[Task] | Mapping[int, Task],
        harvesters: Sequence[Harvester],
        *,
        discount: float = 0.9,
        time_scale: float | None = None,
        bundle_limit: int | None = None,
        comm_range: float = float("inf"),
        turn_distance: float = 0.0,
        turn_time: float = 0.0,
        max_iterations: int = 2000,
    ) -> None:
        if not 0.0 < discount < 1.0:
            raise ValueError("the discount factor must lie in (0, 1)")
        task_map = (
            dict(tasks) if isinstance(tasks, Mapping) else {t.id: t for t in tasks}
        )
        self.task_map = task_map
        self.task_ids = list(task_map)
        self.A = np.array([task_map[tid].a for tid in self.task_ids], dtype=float)
        self.B = np.array([task_map[tid].b for tid in self.task_ids], dtype=float)
        self.L = np.array([task_map[tid].length for tid in self.task_ids], dtype=float)
        self.harvesters = list(harvesters)
        m, n = len(self.task_ids), len(self.harvesters)
        mean_speed = float(np.mean([h.speed for h in self.harvesters]))
        self.time_scale = (
            float(time_scale)
            if time_scale is not None
            else float(self.L.mean()) / mean_speed
        )
        self.log_discount = math.log(discount)
        self.discount = discount
        self.bundle_limit = int(bundle_limit) if bundle_limit is not None else m
        self.turn_distance = float(turn_distance)
        self.turn_time = float(turn_time)
        self.max_iterations = max_iterations
        self.agents = [_Agent(i, h, m, n) for i, h in enumerate(self.harvesters)]
        positions = np.array([a.pos for a in self.agents])
        self.neighbors: list[list[int]] = [[] for _ in self.agents]
        if n > 1:
            limit = comm_range if math.isfinite(comm_range) else 1e18
            for i, j in cKDTree(positions).query_pairs(limit):
                self.neighbors[i].append(j)
                self.neighbors[j].append(i)

    def _reward(self, times):
        return np.exp(self.log_discount * np.asarray(times) / self.time_scale)

    def _turn(self, agent: _Agent) -> float:
        return self.turn_distance / agent.speed + self.turn_time

    def _prefix(self, agent: _Agent):
        pos = agent.pos
        t = 0.0
        score = 0.0
        positions, times, scores = [pos], [0.0], [0.0]
        turn = self._turn(agent)
        for k, j in enumerate(agent.path):
            if k:
                t += turn
            da = math.hypot(*(pos - self.A[j]))
            db = math.hypot(*(pos - self.B[j]))
            if da <= db:
                t += (da + self.L[j]) / agent.speed
                pos = self.B[j]
            else:
                t += (db + self.L[j]) / agent.speed
                pos = self.A[j]
            score += math.exp(self.log_discount * t / self.time_scale)
            positions.append(pos)
            times.append(t)
            scores.append(score)
        return positions, times, scores

    def path_score(self, agent: _Agent) -> float:
        return self._prefix(agent)[2][-1]

    def _insertions(self, agent: _Agent, cand: np.ndarray):
        positions, times, scores = self._prefix(agent)
        base = scores[-1]
        path = agent.path
        n = len(path)
        v = agent.speed
        turn = self._turn(agent)
        Ac, Bc, Lc = self.A[cand], self.B[cand], self.L[cand]
        best = np.full(len(cand), -np.inf)
        where = np.zeros(len(cand), dtype=int)
        for ins in range(n + 1):
            p = positions[ins]
            da = np.hypot(Ac[:, 0] - p[0], Ac[:, 1] - p[1])
            db = np.hypot(Bc[:, 0] - p[0], Bc[:, 1] - p[1])
            near_a = da <= db
            tt = times[ins] + (turn if ins else 0.0) + (np.minimum(da, db) + Lc) / v
            total = scores[ins] + self._reward(tt)
            ex = np.where(near_a[:, None], Bc, Ac)
            for k in range(ins, n):
                j = path[k]
                ea = np.hypot(ex[:, 0] - self.A[j, 0], ex[:, 1] - self.A[j, 1])
                eb = np.hypot(ex[:, 0] - self.B[j, 0], ex[:, 1] - self.B[j, 1])
                first_a = ea <= eb
                tt = tt + turn + (np.minimum(ea, eb) + self.L[j]) / v
                total = total + self._reward(tt)
                ex = np.where(first_a[:, None], self.B[j], self.A[j])
            gain = total - base
            better = gain > best
            best[better] = gain[better]
            where[better] = ins
        return best, where

    @staticmethod
    def _outbids(y1: float, a1: int, y2: float, a2: int) -> bool:
        return y1 > y2 + EPS or (abs(y1 - y2) <= EPS and a1 < a2)

    def _build(self, agent: _Agent) -> bool:
        changed = False
        while len(agent.bundle) < self.bundle_limit:
            owned = set(agent.bundle)
            cand = np.array(
                [j for j in range(len(self.task_ids)) if j not in owned], dtype=int
            )
            if cand.size == 0:
                break
            gain, where = self._insertions(agent, cand)
            yv = agent.y[cand]
            zv = agent.z[cand]
            valid = (gain > yv + EPS) | (
                (np.abs(gain - yv) <= EPS) & ((zv == -1) | (agent.index < zv))
            )
            valid &= gain > EPS
            if not valid.any():
                break
            pick = int(np.argmax(np.where(valid, gain, -np.inf)))
            j = int(cand[pick])
            agent.bundle.append(j)
            agent.path.insert(int(where[pick]), j)
            agent.y[j] = float(gain[pick])
            agent.z[j] = agent.index
            changed = True
        return changed

    def _receive(self, agent: _Agent, sender: int, yk, zk, sk, now: float) -> None:
        i, k = agent.index, sender
        yi, zi, si = agent.y, agent.z, agent.s
        for j in range(len(self.task_ids)):
            zkj, zij = int(zk[j]), int(zi[j])
            ykj, yij = float(yk[j]), float(yi[j])
            action = LEAVE
            if zkj == k:
                if zij == i:
                    if self._outbids(ykj, k, yij, i):
                        action = UPDATE
                elif zij == k or zij == -1:
                    action = UPDATE
                elif sk[zij] > si[zij] or self._outbids(ykj, k, yij, zij):
                    action = UPDATE
            elif zkj == i:
                if zij == k:
                    action = RESET
                elif zij not in (i, -1) and sk[zij] > si[zij]:
                    action = RESET
            elif zkj == -1:
                if zij == k:
                    action = UPDATE
                elif zij not in (i, -1) and sk[zij] > si[zij]:
                    action = UPDATE
            else:
                m = zkj
                if zij == i:
                    if sk[m] > si[m] and self._outbids(ykj, m, yij, i):
                        action = UPDATE
                elif zij == k:
                    action = UPDATE if sk[m] > si[k] else RESET
                elif zij == m or zij == -1:
                    if sk[m] > si[m]:
                        action = UPDATE
                else:
                    n = zij
                    if sk[m] > si[m] and sk[n] > si[n]:
                        action = UPDATE
                    elif sk[m] > si[m] and self._outbids(ykj, m, yij, n):
                        action = UPDATE
                    elif sk[n] > si[n] and si[m] > sk[m]:
                        action = RESET
            if action == UPDATE:
                yi[j], zi[j] = ykj, zkj
            elif action == RESET:
                yi[j], zi[j] = 0.0, -1
        for m in range(len(self.agents)):
            if m != i and m != k:
                si[m] = max(si[m], sk[m])
        si[k] = now

    def _truncate(self, agent: _Agent) -> None:
        cut = next(
            (n for n, j in enumerate(agent.bundle) if agent.z[j] != agent.index), None
        )
        if cut is None:
            return
        removed = agent.bundle[cut:]
        for j in removed[1:]:
            if agent.z[j] == agent.index:
                agent.y[j] = 0.0
                agent.z[j] = -1
        gone = set(removed)
        agent.bundle = agent.bundle[:cut]
        agent.path = [j for j in agent.path if j not in gone]

    def solve(self) -> CBBARun:
        started = time.perf_counter()
        messages = 0
        quiet = 0
        converged = False
        history: list[int] = []
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            before = [(tuple(a.bundle), a.y.copy(), a.z.copy()) for a in self.agents]
            for agent in self.agents:
                self._build(agent)
            snapshot = [(a.y.copy(), a.z.copy(), a.s.copy()) for a in self.agents]
            for agent in self.agents:
                for k in self.neighbors[agent.index]:
                    yk, zk, sk = snapshot[k]
                    self._receive(agent, k, yk, zk, sk, float(iteration))
                    messages += len(self.task_ids)
            for agent in self.agents:
                self._truncate(agent)
            same = all(
                tuple(a.bundle) == b
                and np.allclose(a.y, y, atol=EPS, rtol=0.0)
                and np.array_equal(a.z, z)
                for a, (b, y, z) in zip(self.agents, before)
            )
            history.append(sum(len(a.bundle) for a in self.agents))
            quiet = quiet + 1 if same else 0
            if quiet >= 2:
                converged = True
                break
        owner: dict[int, list[int]] = {}
        for agent in self.agents:
            for j in agent.bundle:
                owner.setdefault(j, []).append(agent.index)
        conflicts = sum(len(v) > 1 for v in owner.values())
        assignment: dict[int, list[int]] = {a.hid: [] for a in self.agents}
        taken: set[int] = set()
        for agent in self.agents:
            route = []
            for j in agent.path:
                if j in taken:
                    continue
                taken.add(j)
                route.append(self.task_ids[j])
            assignment[agent.hid] = route
        unassigned = len(self.task_ids) - len(taken)
        result = evaluate_assignment(
            self.task_map,
            self.harvesters,
            assignment,
            turn_distance=self.turn_distance,
            turn_time=self.turn_time,
            solve_time=time.perf_counter() - started,
            status="converged" if converged else "iteration_cap",
        )
        return CBBARun(
            result=result,
            iterations=iteration,
            converged=converged,
            messages=messages,
            conflicts=conflicts,
            unassigned=unassigned,
            bundle_limit=self.bundle_limit,
            discount=self.discount,
            history=history,
        )


def standard_cbba(
    tasks: Sequence[Task] | Mapping[int, Task],
    harvesters: Sequence[Harvester],
    **options,
) -> CBBARun:
    return StandardCBBA(tasks, harvesters, **options).solve()
