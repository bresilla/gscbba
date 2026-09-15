from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

NEG_INF = -1e18
LOCK_BID = 1e9


@dataclass
class Task:
    id: int
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.length = float(np.linalg.norm(self.b - self.a))
        self.centroid = 0.5 * (self.a + self.b)


def _service_cost(pos: np.ndarray, t: Task) -> tuple[float, np.ndarray]:
    da = float(np.hypot(*(pos - t.a)))
    db = float(np.hypot(*(pos - t.b)))
    if da <= db:
        return da + t.length, t.b
    return db + t.length, t.a


@dataclass
class Harvester:
    hid: int
    pos: np.ndarray
    speed: float = 1.5
    max_bundle: int = 40
    query_k: int = 20
    turn_distance: float = 0.0
    turn_time: float = 0.0
    contiguous_admission: bool = True
    bid_mode: str = "finish_time"

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float)
        self.start_pos = self.pos.copy()
        self.bundle: list[int] = []
        self.path: list[int] = []
        self.y: dict[int, float] = {}
        self.z: dict[int, int] = {}
        self.t: dict[int, int] = {}
        self._now: int = 0
        self.known_done: set[int] = set()
        self.distance: float = 0.0
        self.row_distance: float = 0.0
        self.deadhead_distance: float = 0.0
        self.turn_distance_travelled: float = 0.0
        self.active: bool = True
        self.cur_task: Optional[int] = None
        self.cur_entry: Optional[np.ndarray] = None
        self.cur_exit: Optional[np.ndarray] = None
        self.substate: str = "to_entry"
        self.turn_distance_remaining: float = 0.0
        self.turn_time_remaining: float = 0.0
        self._turn_due: bool = False
        self.failed: bool = False
        self.idle_time: float = 0.0
        self.needs_build: bool = True
        self.n_releases: int = 0
        self.rows_scored: int = 0
        self.home_lo: Optional[float] = None
        self.home_hi: Optional[float] = None
        self.unrestricted: bool = False
        self.last_heard: dict[int, float] = {}
        self.peer_state: dict[int, tuple] = {}
        self.failed_view: set[int] = set()
        self.hold_until: float = 0.0

    def _route_length(self, seq: list[int], tasks: dict[int, Task]) -> float:
        pos = self.pos
        total = 0.0
        for index, tid in enumerate(seq):
            t = tasks[tid]
            if index:
                total += self.turn_distance + self.speed * self.turn_time
            da = float(np.hypot(*(pos - t.a)))
            db = float(np.hypot(*(pos - t.b)))
            if da <= db:
                total += da + t.length
                pos = t.b
            else:
                total += db + t.length
                pos = t.a
        return total

    def build_bundle(self, candidate_ids: list[int], tasks: dict[int, Task]) -> bool:
        added = False
        self.rows_scored += len(candidate_ids)
        inpath = set(self.path)
        scored = []
        for j in candidate_ids:
            if j in inpath or j in self.known_done:
                continue
            t = tasks[j]
            d = min(
                float(np.hypot(*(self.pos - t.a))),
                float(np.hypot(*(self.pos - t.b))),
            )
            scored.append((d, j))
        if not scored:
            if self.contiguous_admission:
                self._order_serpentine(tasks)
            return False
        scored.sort()
        ref_id = self.path[0] if self.path else scored[0][1]
        off = self._offset_key(tasks, ref_id)
        offs = sorted(off(j) for _, j in scored)
        diffs = [b - a for a, b in zip(offs, offs[1:]) if b - a > 1e-6]
        spacing = float(np.median(diffs)) if diffs else 1.0
        reach = 4.0 * spacing
        block_off = sorted(off(j) for j in self.path)
        lo = block_off[0] if block_off else None
        hi = block_off[-1] if block_off else None
        if self.bid_mode == "marginal":
            if len(self.bundle) >= self.max_bundle:
                return False
            old_cost = self._route_length(self.path, tasks)
            best = None
            previous_bid = self.y.get(self.bundle[-1], NEG_INF) if self.bundle else None
            for _, j in scored:
                choices = [self.path + [j]]
                if self.cur_task is None:
                    choices.append([j] + self.path)
                seq = min(choices, key=lambda route: self._route_length(route, tasks))
                raw_bid = -(self._route_length(seq, tasks) - old_cost) / self.speed
                bid = (
                    min(raw_bid, previous_bid - 1e-6)
                    if previous_bid is not None
                    else raw_bid
                )
                if bid <= self.y.get(j, NEG_INF) + 1e-9:
                    continue
                candidate = (bid, -j, j, seq)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if best is None:
                return False
            bid, _, j, seq = best
            self.path = seq
            self.bundle.append(j)
            self.y[j] = bid
            self.z[j] = self.hid
            self.t[j] = self._now
            return True

        for d, j in scored:
            if len(self.bundle) >= self.max_bundle:
                break
            oj = off(j)
            if (
                self.contiguous_admission
                and not self.unrestricted
                and self.home_lo is not None
                and self.home_hi is not None
                and not (self.home_lo <= oj <= self.home_hi)
            ):
                continue
            if self.contiguous_admission and lo is None:
                if d > 6.0 * spacing and not self.unrestricted:
                    continue
            elif self.contiguous_admission:
                assert lo is not None and hi is not None
                if oj < lo - reach or oj > hi + reach:
                    continue
            seq = sorted(self.path + [j], key=off)
            bid = -self._route_length(seq, tasks) / self.speed
            if bid > self.y.get(j, NEG_INF) + 1e-9:
                self.path = seq
                self.bundle.append(j)
                self.y[j] = bid
                self.z[j] = self.hid
                self.t[j] = self._now
                lo = oj if lo is None else min(lo, oj)
                hi = oj if hi is None else max(hi, oj)
                added = True
        if self.contiguous_admission:
            self._order_serpentine(tasks)
        return added

    def _offset_key(self, tasks: dict[int, Task], ref_id):
        if ref_id is None:
            return lambda j: 0.0
        dv = tasks[ref_id].b - tasks[ref_id].a
        nrm = float(np.hypot(dv[0], dv[1])) or 1.0
        px, py = -dv[1] / nrm, dv[0] / nrm
        return lambda j: float(tasks[j].centroid[0] * px + tasks[j].centroid[1] * py)

    def _order_serpentine(self, tasks: dict[int, Task]) -> None:
        if len(self.path) <= 1:
            return
        ref = tasks[self.path[0]]
        dv = ref.b - ref.a
        nrm = float(np.hypot(dv[0], dv[1]))
        if nrm < 1e-9:
            return
        px, py = -dv[1] / nrm, dv[0] / nrm
        cur = self.cur_task if self.cur_task in self.path else None
        rest = [j for j in self.path if j != cur]
        rest.sort(
            key=lambda j: float(tasks[j].centroid[0] * px + tasks[j].centroid[1] * py)
        )
        self.path = ([cur] if cur is not None else []) + rest

    def receive(
        self,
        other_y: dict[int, float],
        other_z: dict[int, int],
        other_t: dict[int, int],
    ) -> int:
        changes = 0
        for j, tk in other_t.items():
            ti = self.t.get(j, -1)
            yk = other_y.get(j, NEG_INF)
            zk = other_z.get(j, -1)
            if tk > ti:
                adopt = True
            elif tk == ti:
                yi = self.y.get(j, NEG_INF)
                zi = self.z.get(j, -1)
                adopt = (yk > yi + 1e-9) or (
                    abs(yk - yi) <= 1e-9 and zk != -1 and (zi == -1 or zk < zi)
                )
            else:
                adopt = False
            if adopt and (
                self.z.get(j, -1) != zk
                or self.t.get(j, -1) != tk
                or abs(self.y.get(j, NEG_INF) - yk) > 1e-9
            ):
                self.y[j] = yk
                self.z[j] = zk
                self.t[j] = tk
                changes += 1
        if changes:
            self._update_bundle_after_consensus()
        return changes

    def _update_bundle_after_consensus(self) -> None:
        rel_idx = None
        for idx, j in enumerate(self.bundle):
            if self.z.get(j, -1) != self.hid:
                rel_idx = idx
                break
        if rel_idx is None:
            return
        released = self.bundle[rel_idx:]
        rel_set = set(released)
        self.bundle = self.bundle[:rel_idx]
        self.path = [j for j in self.path if j not in rel_set]
        self.n_releases += len(released)
        for j in released:
            if self.z.get(j, -1) == self.hid:
                self.y[j] = NEG_INF
                self.z[j] = -1
                self.t[j] = self._now
        if self.cur_task in rel_set:
            self.cur_task = None
        self.needs_build = True

    def clear_stale_claims(self) -> bool:
        in_bundle = set(self.bundle)
        changed = False
        for j, zz in list(self.z.items()):
            if zz == self.hid and j not in in_bundle and j not in self.known_done:
                self.y[j] = NEG_INF
                self.z[j] = -1
                self.t[j] = self._now
                changed = True
        return changed

    def drop_done(self) -> None:
        if not self.known_done:
            return
        before = len(self.path)
        self.path = [j for j in self.path if j not in self.known_done]
        self.bundle = [j for j in self.bundle if j not in self.known_done]
        if len(self.path) != before:
            self.needs_build = True
        if self.cur_task in self.known_done:
            self.cur_task = None


@dataclass
class SimResult:
    completion_time: float
    total_distance: float
    row_distance: float
    deadhead_distance: float
    turn_distance: float
    per_harvester_tasks: list[int]
    per_harvester_distance: list[float]
    per_harvester_finish: list[float]
    per_harvester_speed: list[float]
    per_harvester_idle: list[float]
    consensus_changes: int
    total_releases: int
    alloc_cycles: int
    alloc_rounds: int
    n_tasks: int
    converged: bool
    allocation_stable: list[bool]
    duplicate_services: int
    greedy_rows_scored: int
    greedy_time: float
    balance_time: float
    partition_changed_by_balancing: list[bool]
    msg_records_total: int
    msg_records_by_harvester: dict[int, int]
    msg_records_peak_round: int
    msg_balancing_total: int
    msg_balancing_by_harvester: dict[int, int]
    msg_balancing_peak_round: int
    rows_rebid: int
    reconvergence_rounds: list[int]
    reconvergence_time: list[float]
    coverage_fraction: float
    initial_assignment: dict[int, list[int]]
    event_log: list[dict]
    msg_dropped: int = 0
    gossip_messages: int = 0
    false_failure_detections: int = 0
    detection_delays: list[float] = field(default_factory=list)
    planner_calls: int = 0
    planning_delay: float = 0.0


BASE_ID = -1000


class Simulator:
    def __init__(
        self,
        tasks: list[Task],
        harvesters: list[Harvester],
        comm_range: float,
        hops: int = 8,
        dt: float = 0.5,
        comm_period: float = 2.0,
        alloc_max_rounds: int = 60,
        t_max: float = 20000.0,
        loss_time: Optional[float] = None,
        loss_id: Optional[int] = None,
        loss_detect_delay: float = 4.0,
        help_margin: float = 8.0,
        record_traj: bool = False,
        traj_stride: int = 6,
        spatial: bool = False,
        query_k: int = 24,
        mode: str = "gscbba",
        balancing: bool = True,
        turn_distance: float = 0.0,
        turn_time: float = 0.0,
        loss_events: Optional[list[tuple[float, int]]] = None,
        join_events: Optional[list[tuple[float, Harvester]]] = None,
        speed_events: Optional[list[tuple[float, int, float]]] = None,
        speed_perturbation: Optional[tuple[float, int, float]] = None,
        allocation_policy: str = "decentralized",
        initial_assignment: Optional[dict[int, list[int]]] = None,
        information: str = "shared",
        component_balancing: bool = False,
        idle_help: bool = False,
        packet_loss: float = 0.0,
        round_latency: float = 0.0,
        latency_jitter: float = 0.0,
        heartbeat_timeout: float = 6.0,
        detect_fraction: float = 0.8,
        claim_ttl: float = 240.0,
        comm_seed: int = 0,
        base_position: Optional[np.ndarray] = None,
    ) -> None:
        if mode not in {"gscbba", "marginal"}:
            raise ValueError("mode must be 'gscbba' or 'marginal'")
        if allocation_policy not in {
            "decentralized",
            "frozen",
            "centralized",
            "centralized_base",
        }:
            raise ValueError(
                "allocation_policy must be 'decentralized', 'frozen', "
                "'centralized', or 'centralized_base'"
            )
        if information not in {"shared", "local"}:
            raise ValueError("information must be 'shared' or 'local'")
        if turn_distance < 0 or turn_time < 0:
            raise ValueError("turn costs cannot be negative")
        if not 0.0 <= packet_loss < 1.0:
            raise ValueError("packet_loss must lie in [0, 1)")
        if round_latency < 0 or latency_jitter < 0:
            raise ValueError("latencies cannot be negative")
        if allocation_policy == "centralized_base" and base_position is None:
            raise ValueError("centralized_base requires a base_position")
        self.tasks = {t.id: t for t in tasks}
        self.help_margin = help_margin
        _t0 = tasks[0]
        dv = _t0.b - _t0.a
        nrm = float(np.hypot(dv[0], dv[1])) or 1.0
        self._perp = (-dv[1] / nrm, dv[0] / nrm)
        all_offs = sorted(
            c[0] * self._perp[0] + c[1] * self._perp[1]
            for c in (t.centroid for t in tasks)
        )
        gaps = [b - a for a, b in zip(all_offs, all_offs[1:]) if b - a > 1e-6]
        self._spacing = float(np.median(gaps)) if gaps else 1.0
        self.mode = mode
        self.spatial = spatial if mode != "marginal" else False
        self.query_k = query_k
        self._task_ids = list(self.tasks.keys())
        self._task_centroids = np.array(
            [self.tasks[tid].centroid for tid in self._task_ids]
        )
        self._task_tree = cKDTree(self._task_centroids)
        self.greedy_time = 0.0
        self.balance_time = 0.0
        self.harvesters = harvesters
        self.balancing = balancing if mode != "marginal" else False
        self.turn_distance = float(turn_distance)
        self.turn_time = float(turn_time)
        self.allocation_policy = allocation_policy
        self.information = information
        self.component_balancing = component_balancing
        self.idle_help = idle_help
        self.packet_loss = float(packet_loss)
        self.round_latency = float(round_latency)
        self.latency_jitter = float(latency_jitter)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.detect_fraction = float(detect_fraction)
        self.claim_ttl = float(claim_ttl)
        self._rng = np.random.default_rng(comm_seed)
        self.initial_plan = (
            {hid: list(tids) for hid, tids in initial_assignment.items()}
            if initial_assignment is not None
            else None
        )
        for harvester in self.harvesters:
            harvester.turn_distance = self.turn_distance
            harvester.turn_time = self.turn_time
            if mode == "marginal":
                harvester.contiguous_admission = False
                harvester.unrestricted = True
                harvester.bid_mode = "marginal"
        self.comm_range = comm_range
        self.hops = hops
        self.dt = dt
        self.comm_period = comm_period
        self.alloc_max_rounds = (
            max(alloc_max_rounds, len(harvesters) * len(tasks) + 1)
            if mode == "marginal"
            else alloc_max_rounds
        )
        self.t_max = t_max
        self.loss_detect_delay = loss_detect_delay
        self.loss_events = sorted(list(loss_events or []))
        if loss_time is not None:
            if loss_id is None:
                raise ValueError("loss_id is required when loss_time is set")
            self.loss_events.append((float(loss_time), int(loss_id)))
            self.loss_events.sort()
        self.join_events = sorted(list(join_events or []), key=lambda item: item[0])
        self.speed_events = sorted(list(speed_events or []))
        if speed_perturbation is not None:
            self.speed_events.append(speed_perturbation)
            self.speed_events.sort()

        self.done: set[int] = set()
        self._shared_done: set[int] | frozenset[int] = (
            self.done if information == "shared" else frozenset()
        )
        self.completed_by: dict[int, int] = {}
        self.round = 0
        self.consensus_changes = 0
        self.duplicate_services = 0
        self.alloc_cycles = 0
        self.alloc_rounds = 0
        self.alloc_trace: list[int] = []
        self.allocation_stable: list[bool] = []
        all_harvesters = self.harvesters + [h for _, h in self.join_events]
        ids = [h.hid for h in all_harvesters]
        if len(ids) != len(set(ids)):
            raise ValueError("harvester ids must be unique, including join events")
        self.tasks_done_by = {h.hid: 0 for h in all_harvesters}
        self.finish_time = {h.hid: 0.0 for h in all_harvesters}
        self._failed: set[int] = set()
        self._failure_time: dict[int, float] = {}
        self._detected_failures: set[int] = set()
        self._joined: set[int] = set()
        self._speed_events_done: set[int] = set()
        self.partition_changed_by_balancing: list[bool] = []
        self.msg_records_total = 0
        self.msg_records_by_harvester = {h.hid: 0 for h in all_harvesters}
        self.msg_records_peak_round = 0
        self.msg_balancing_total = 0
        self.msg_balancing_by_harvester = {h.hid: 0 for h in all_harvesters}
        self.msg_balancing_peak_round = 0
        self.msg_dropped = 0
        self.gossip_messages = 0
        self.false_failure_detections = 0
        self.detection_delays: list[float] = []
        self._detected_once: set[int] = set()
        self.planner_calls = 0
        self.planning_delay = 0.0
        self.rows_rebid = 0
        self.reconvergence_rounds: list[int] = []
        self.reconvergence_time: list[float] = []
        self.event_log: list[dict] = []
        self.initial_assignment: dict[int, list[int]] = {}
        self.record_traj = record_traj
        self.traj_stride = traj_stride
        self.traj: dict[int, list[np.ndarray]] = {
            h.hid: [h.pos.copy()] for h in all_harvesters
        }
        self._clock = 0.0
        self._cycle_hops = 0
        self._last_round_drops = 0
        self._idle_signature: dict[int, tuple] = {}
        self.base: Optional[Harvester] = None
        self._base_reach: set[int] = set()
        self._base_plan: dict[int, list[int]] = {}
        self._base_signature: Optional[tuple] = None
        self._base_failed_seen: set[int] = set()
        if allocation_policy == "centralized_base":
            assert base_position is not None
            self.base = Harvester(
                hid=BASE_ID, pos=np.asarray(base_position, dtype=float), speed=1.0
            )

    def _off(self, tid: int) -> float:
        c = self.tasks[tid].centroid
        return c[0] * self._perp[0] + c[1] * self._perp[1]

    def _pos_off(self, h: Harvester) -> float:
        return float(h.pos[0] * self._perp[0] + h.pos[1] * self._perp[1])

    def _live(self) -> list[Harvester]:
        if self.information == "local":
            return [h for h in self.harvesters if h.active and not h.failed]
        return [h for h in self.harvesters if h.active]

    def _set_home(self) -> None:
        margin = self.help_margin * self._spacing
        for h in self.harvesters:
            if h.path:
                offs = [self._off(j) for j in h.path]
                h.home_lo, h.home_hi = min(offs) - margin, max(offs) + margin
            else:
                po = self._pos_off(h)
                h.home_lo, h.home_hi = po - margin, po + margin

    def _avail_all(self, h: Harvester) -> list[int]:
        return [
            tid
            for tid in self.tasks
            if tid not in self._shared_done and tid not in h.known_done
        ]

    def _has_avail(self, h: Harvester) -> bool:
        return any(
            tid not in self._shared_done and tid not in h.known_done
            for tid in self.tasks
        )

    def _candidates(self, h: Harvester) -> list[int]:
        if not self.spatial:
            return self._avail_all(h)
        n = len(self._task_ids)
        k = min(self.query_k, n)
        if not h.path:
            anchors = [h.pos]
        else:
            start_off = self._pos_off(h)
            frontier = max(h.path, key=lambda j: abs(self._off(j) - start_off))
            anchors = [h.pos, self.tasks[frontier].centroid]
        cand: list[int] = []
        seen: set[int] = set()
        for anchor in anchors:
            search_k = k
            while True:
                idx = np.atleast_1d(self._task_tree.query(anchor, k=search_k)[1])
                for ii in idx:
                    tid = self._task_ids[int(ii)]
                    if tid in seen:
                        continue
                    seen.add(tid)
                    if (
                        tid not in self._shared_done
                        and tid not in h.known_done
                        and h.z.get(tid, -1) == -1
                    ):
                        cand.append(tid)
                if len(cand) >= self.query_k or search_k == n:
                    break
                search_k = min(n, 2 * search_k)
        if not cand:
            unclaimed = [tid for tid in self._avail_all(h) if h.z.get(tid, -1) == -1]
            return unclaimed
        return cand

    def _adjacency(self) -> frozenset:
        active = self._live()
        if len(active) < 2:
            return frozenset()
        positions = np.array([h.pos for h in active])
        tree = cKDTree(positions)
        edges = {
            (active[i].hid, active[j].hid) for i, j in tree.query_pairs(self.comm_range)
        }
        return frozenset(edges)

    def _dropped(self) -> bool:
        return self.packet_loss > 0.0 and self._rng.random() < self.packet_loss

    def _comm_round(self, members: Optional[list[Harvester]] = None) -> int:
        active = members if members is not None else self._live()
        self._last_round_drops = 0
        if len(active) < 1:
            return 0
        positions = np.array([h.pos for h in active])
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.comm_range)
        adj: dict[int, list[int]] = {i: [] for i in range(len(active))}
        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)

        total = 0
        for _ in range(self.hops):
            self._cycle_hops += 1
            snap_y = [dict(h.y) for h in active]
            snap_z = [dict(h.z) for h in active]
            snap_t = [dict(h.t) for h in active]
            snap_done = [set(h.known_done) for h in active]
            round_changes = 0
            done_grew = False
            records_this_round = 0
            drops = 0
            for i, h in enumerate(active):
                for nb in adj[i]:
                    sent = len(snap_t[nb])
                    records_this_round += sent
                    sender = active[nb].hid
                    self.msg_records_by_harvester[sender] += sent
                    if self._dropped():
                        drops += 1
                        continue
                    round_changes += h.receive(snap_y[nb], snap_z[nb], snap_t[nb])
                    before = len(h.known_done)
                    h.known_done |= snap_done[nb]
                    done_grew = done_grew or len(h.known_done) != before
            for h in active:
                h.drop_done()
            self.msg_dropped += drops
            self._last_round_drops += drops
            self.consensus_changes += round_changes
            total += round_changes
            self.msg_records_total += records_this_round
            self.msg_records_peak_round = max(
                self.msg_records_peak_round, records_this_round
            )
            if round_changes == 0 and not done_grew and drops == 0:
                break
        return total

    def _apply_events(self, t: float) -> list[dict]:
        replanning_events: list[dict] = []
        local_detection = (
            self.information == "local" and self.allocation_policy != "centralized"
        )

        for event_time, hid in self.loss_events:
            if hid not in self._failed and t >= event_time:
                lost = next((h for h in self.harvesters if h.hid == hid), None)
                if lost is None:
                    raise ValueError(f"loss event refers to unknown harvester {hid}")
                lost.failed = True
                self._failed.add(hid)
                self._failure_time[hid] = t
                self.event_log.append({"time": t, "kind": "loss", "hid": hid})
            if local_detection:
                continue
            if hid in self._failed and hid not in self._detected_failures:
                if t < event_time + self.loss_detect_delay:
                    continue
                lost = next(h for h in self.harvesters if h.hid == hid)
                lost.active = False
                self._detected_failures.add(hid)
                self.detection_delays.append(t - self._failure_time[hid])
                affected = [tid for tid in lost.bundle if tid not in self.done]
                record = {
                    "time": t,
                    "kind": "loss_detected",
                    "hid": hid,
                    "affected_rows": len(affected),
                }
                if self.allocation_policy not in {"frozen", "centralized_base"}:
                    for harvester in self.harvesters:
                        if not harvester.active:
                            continue
                        for tid in affected:
                            harvester.y[tid] = NEG_INF
                            harvester.z[tid] = -1
                            harvester.t[tid] = self.round + 1
                        harvester.home_lo = None
                        harvester.home_hi = None
                        harvester.unrestricted = True
                        harvester.needs_build = True
                    replanning_events.append(record)
                elif self.allocation_policy == "centralized_base":
                    replanning_events.append(record)
                else:
                    self.event_log.append(record)

        for index, (event_time, harvester) in enumerate(self.join_events):
            if index in self._joined or t < event_time:
                continue
            harvester.turn_distance = self.turn_distance
            harvester.turn_time = self.turn_time
            if self.mode == "marginal":
                harvester.contiguous_admission = False
                harvester.unrestricted = True
                harvester.bid_mode = "marginal"
            self.harvesters.append(harvester)
            self.traj.setdefault(harvester.hid, [harvester.pos.copy()])
            self._joined.add(index)
            record = {"time": t, "kind": "join", "hid": harvester.hid}
            if self.allocation_policy != "frozen":
                replanning_events.append(record)
            else:
                self.event_log.append(record)

        for index, (event_time, hid, multiplier) in enumerate(self.speed_events):
            if index in self._speed_events_done or t < event_time:
                continue
            if multiplier <= 0:
                raise ValueError("speed multipliers must be positive")
            harvester = next((h for h in self.harvesters if h.hid == hid), None)
            if harvester is None:
                raise ValueError(f"speed event refers to unknown harvester {hid}")
            harvester.speed *= multiplier
            self._speed_events_done.add(index)
            record = {
                "time": t,
                "kind": "speed",
                "hid": hid,
                "multiplier": multiplier,
            }
            if self.allocation_policy != "frozen":
                replanning_events.append(record)
            else:
                self.event_log.append(record)

        return replanning_events

    def _drive_to(self, h: Harvester, target: np.ndarray) -> None:
        vec = target - h.pos
        d = float(np.hypot(*vec))
        if d < 1e-9:
            h.idle_time += self.dt
            return
        step = min(d, h.speed * self.dt)
        h.pos = h.pos + vec / d * step
        h.distance += step
        h.deadhead_distance += step

    def _move(self, h: Harvester) -> bool:
        if not h.active or h.failed:
            return False
        if h.cur_task is None and self._clock < h.hold_until:
            h.idle_time += self.dt
            return False
        if not h.path:
            if (
                self.base is not None
                and h.hid not in self._base_reach
                and len(self.done) < len(self.tasks)
            ):
                self._drive_to(h, self.base.pos)
                return False
            h.idle_time += self.dt
            return False
        step = h.speed * self.dt

        while h.path and h.path[0] in h.known_done:
            h.drop_done()
        if not h.path:
            return False

        t0 = h.path[0]
        if h.cur_task != t0:
            task = self.tasks[t0]
            da = float(np.hypot(*(h.pos - task.a)))
            db = float(np.hypot(*(h.pos - task.b)))
            if da <= db:
                h.cur_entry, h.cur_exit = task.a, task.b
            else:
                h.cur_entry, h.cur_exit = task.b, task.a
            if h._turn_due:
                h.substate = "turn"
                h.turn_distance_remaining = h.turn_distance
                h.turn_time_remaining = h.turn_time
                h._turn_due = False
            else:
                h.substate = "to_entry"
            h.cur_task = t0

        if h.substate == "turn":
            if h.turn_distance_remaining > 1e-12:
                travelled = min(step, h.turn_distance_remaining)
                h.turn_distance_remaining -= travelled
                h.turn_distance_travelled += travelled
                h.distance += travelled
                return False
            if h.turn_time_remaining > 1e-12:
                h.turn_time_remaining = max(0.0, h.turn_time_remaining - self.dt)
                return False
            h.substate = "to_entry"

        target = h.cur_entry if h.substate == "to_entry" else h.cur_exit
        assert target is not None
        vec = target - h.pos
        d = float(np.hypot(*vec))
        if d <= step:
            h.pos = target.copy()
            h.distance += d
            if h.substate == "to_entry":
                h.deadhead_distance += d
                h.substate = "to_exit"
            else:
                h.row_distance += d
                dup = t0 in self.done
                if dup:
                    self.duplicate_services += 1
                else:
                    self.done.add(t0)
                    self.completed_by[t0] = h.hid
                    self.tasks_done_by[h.hid] += 1
                h.known_done.add(t0)
                h.path = [x for x in h.path if x != t0]
                h.bundle = [x for x in h.bundle if x != t0]
                h.cur_task = None
                h._turn_due = True
                h.needs_build = True
                return not dup
        else:
            h.pos = h.pos + vec / d * step
            h.distance += step
            if h.substate == "to_entry":
                h.deadhead_distance += step
            else:
                h.row_distance += step
        return False

    def _components(self, active: list[Harvester]) -> list[list[Harvester]]:
        n = len(active)
        if n == 0:
            return []
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pos = np.array([h.pos for h in active])
        for i, j in cKDTree(pos).query_pairs(self.comm_range):
            parent[find(i)] = find(j)
        comps: dict[int, list[Harvester]] = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(active[i])
        return list(comps.values())

    def _component_signature(self) -> frozenset:
        return frozenset(
            frozenset(h.hid for h in comp) for comp in self._components(self._live())
        )

    def _done_view(self, machines: list[Harvester]) -> set[int]:
        view = set(self._shared_done)
        for h in machines:
            view |= h.known_done
        return view

    def _serpentine(self, h: Harvester, path: list[int]) -> list[int]:
        if len(path) <= 1:
            return list(path)
        cur = h.cur_task if h.cur_task in path else None
        rest = sorted((j for j in path if j != cur), key=self._off)
        return ([cur] if cur is not None else []) + rest

    def _finish_estimate(self, h: Harvester, path: list[int]) -> float:
        pos = h.pos
        distance = 0.0
        seq = list(path)
        if (
            h.cur_task is not None
            and seq
            and seq[0] == h.cur_task
            and h.cur_entry is not None
            and h.cur_exit is not None
        ):
            if h.substate == "to_exit":
                distance += float(np.hypot(*(h.cur_exit - pos)))
            else:
                distance += (
                    float(np.hypot(*(h.cur_entry - pos)))
                    + self.tasks[h.cur_task].length
                )
            pos = h.cur_exit
            seq = seq[1:]
            distance += len(seq) * (h.turn_distance + h.speed * h.turn_time)
        elif seq:
            distance += (len(seq) - 1) * (h.turn_distance + h.speed * h.turn_time)
        for j in seq:
            task = self.tasks[j]
            da = float(np.hypot(*(pos - task.a)))
            db = float(np.hypot(*(pos - task.b)))
            if da <= db:
                distance += da + task.length
                pos = task.b
            else:
                distance += db + task.length
                pos = task.a
        return distance / h.speed

    def _balanced_partition(
        self,
        machines: list[Harvester],
        rows: Optional[list[int]] = None,
        accept_if_better: bool = False,
    ) -> bool:
        view = self._done_view(machines)
        pool = self.tasks if rows is None else rows
        avail = [tid for tid in pool if tid not in view]
        if not avail or not machines:
            return False
        locked = {
            h.cur_task
            for h in machines
            if h.cur_task is not None and h.cur_task not in view
        }
        free = sorted((tid for tid in avail if tid not in locked), key=self._off)
        ms = sorted(machines, key=self._pos_off)
        total_w = sum(self.tasks[tid].length for tid in free)
        total_v = sum(h.speed for h in ms)
        if total_w <= 0 or total_v <= 0:
            return False
        bounds, cum_v = [], 0.0
        for h in ms:
            cum_v += h.speed
            bounds.append(total_w * cum_v / total_v)
        chunks: dict[int, list[int]] = {h.hid: [] for h in ms}
        acc, k = 0.0, 0
        for tid in free:
            chunks[ms[k].hid].append(tid)
            acc += self.tasks[tid].length
            while k < len(ms) - 1 and acc >= bounds[k] - 1e-9:
                k += 1
        owner = {tid: hid for hid, tids in chunks.items() for tid in tids}
        for h in ms:
            if h.cur_task is not None and h.cur_task not in view:
                owner[h.cur_task] = h.hid
        if accept_if_better:
            proposed = {}
            for h in ms:
                path = []
                if h.cur_task is not None and h.cur_task not in view:
                    path.append(h.cur_task)
                proposed[h.hid] = self._serpentine(h, path + chunks[h.hid])
            before = max(
                self._finish_estimate(h, [j for j in h.path if j not in view])
                for h in ms
            )
            after = max(self._finish_estimate(h, proposed[h.hid]) for h in ms)
            if after >= before * (1.0 - 1e-3):
                return False
        for h in ms:
            path = []
            if h.cur_task is not None and h.cur_task not in view:
                path.append(h.cur_task)
            path += chunks[h.hid]
            h.path = path
            h.bundle = list(path)
            h._now = self.round
            for tid, hid in owner.items():
                h.y[tid] = LOCK_BID
                h.z[tid] = hid
                h.t[tid] = self.round
            h._order_serpentine(self.tasks)
        return True

    def _owner_map(self) -> dict[int, int]:
        return {
            tid: harvester.hid
            for harvester in self.harvesters
            if harvester.active
            for tid in harvester.path
            if tid not in self.done
        }

    def _apply_assignment(
        self,
        assignment: dict[int, list[int]],
        machines: Optional[list[Harvester]] = None,
        done: Optional[set[int]] = None,
    ) -> None:
        active = (
            machines
            if machines is not None
            else [h for h in self.harvesters if h.active]
        )
        done_view = self.done if done is None else done
        owner = {
            tid: hid
            for hid, tids in assignment.items()
            for tid in tids
            if tid not in done_view
        }
        self.round += 1
        for harvester in active:
            tids = [
                tid for tid in assignment.get(harvester.hid, []) if tid not in done_view
            ]
            harvester.path = tids
            harvester.bundle = list(tids)
            if harvester.cur_task is not None and harvester.cur_task not in done_view:
                if not tids or tids[0] != harvester.cur_task:
                    raise ValueError(
                        "a new plan must preserve the current committed row"
                    )
            else:
                harvester.cur_task = None
            harvester.needs_build = False
            harvester._now = self.round
            for tid in self.tasks:
                if tid in done_view:
                    continue
                hid = owner.get(tid, -1)
                harvester.z[tid] = hid
                harvester.y[tid] = LOCK_BID if hid >= 0 else NEG_INF
                harvester.t[tid] = self.round

    def _centralized_allocation(
        self,
        machines: Optional[list[Harvester]] = None,
        excluded: Optional[set[int]] = None,
        done: Optional[set[int]] = None,
    ) -> None:
        from .baselines import optimal_contiguous_partition

        active = (
            machines
            if machines is not None
            else [h for h in self.harvesters if h.active]
        )
        done_view = self.done if done is None else done
        skip = excluded or set()
        locked = {
            h.cur_task
            for h in active
            if h.cur_task is not None and h.cur_task not in done_view
        }
        remaining = [
            task
            for tid, task in self.tasks.items()
            if tid not in done_view and tid not in locked and tid not in skip
        ]
        if not active:
            return
        self.planner_calls += 1
        planning = []
        ready_times = {}
        initial_turns = []
        for h in active:
            pos = h.pos.copy()
            ready = 0.0
            if h.cur_task is not None and h.cur_task not in done_view:
                assert h.cur_entry is not None and h.cur_exit is not None
                if h.substate == "to_exit":
                    ready = float(np.linalg.norm(h.cur_exit - h.pos)) / h.speed
                else:
                    ready = (
                        float(np.linalg.norm(h.cur_entry - h.pos))
                        + self.tasks[h.cur_task].length
                    ) / h.speed
                    if h.substate == "turn":
                        ready += (
                            h.turn_distance_remaining / h.speed + h.turn_time_remaining
                        )
                pos = h.cur_exit.copy()
                initial_turns.append(h.hid)
            elif h._turn_due:
                initial_turns.append(h.hid)
            planning.append(replace(h, pos=pos))
            ready_times[h.hid] = ready
        if remaining:
            result = optimal_contiguous_partition(
                remaining,
                planning,
                turn_distance=self.turn_distance,
                turn_time=self.turn_time,
                ready_times=ready_times,
                initial_turns=initial_turns,
            )
            planned = result.assignment
        else:
            planned = {h.hid: [] for h in active}
        assignment = {
            h.hid: (
                [h.cur_task]
                if h.cur_task is not None and h.cur_task not in done_view
                else []
            )
            + planned[h.hid]
            for h in active
        }
        self._apply_assignment(assignment, active, done_view)

    def _expire_stale(self, h: Harvester) -> int:
        released = 0
        for tid, owner in list(h.z.items()):
            if owner in (-1, h.hid) or tid in h.known_done:
                continue
            if self._clock - h.last_heard.get(owner, 0.0) > self.claim_ttl:
                h.y[tid] = NEG_INF
                h.z[tid] = -1
                h.t[tid] = self.round
                released += 1
        return released

    def _stale_count(self, h: Harvester) -> int:
        return sum(
            1
            for tid, owner in h.z.items()
            if owner not in (-1, h.hid)
            and tid not in h.known_done
            and self._clock - h.last_heard.get(owner, 0.0) > self.claim_ttl
        )

    def _greedy_allocation(self, participants: list[Harvester]) -> bool:
        self.round += 1
        for h in participants:
            h._now = self.round
            h.drop_done()
            if h.cur_task is not None and h.cur_task not in h.known_done:
                h.y[h.cur_task] = LOCK_BID
                h.z[h.cur_task] = h.hid
                h.t[h.cur_task] = self.round
            if self.information == "local" and not h.path:
                self._expire_stale(h)
            if not h.path and self._has_avail(h):
                h.needs_build = True
        order = sorted(participants, key=lambda x: -x.speed)
        quiet = 0
        needed = 1 if self.packet_loss <= 0.0 else 2
        for _ in range(self.alloc_max_rounds):
            self.round += 1
            self.alloc_rounds += 1
            built = False
            for h in order:
                h._now = self.round
                if h.clear_stale_claims():
                    h.needs_build = True
                if not h.path and self._has_avail(h):
                    h.needs_build = True
                if h.needs_build:
                    added = h.build_bundle(self._candidates(h), self.tasks)
                    if added:
                        built = True
                    h.needs_build = added and len(h.bundle) < h.max_bundle
            changed = self._comm_round(participants)
            if self.alloc_cycles == 1:
                holders: dict[int, list[int]] = {}
                for h in participants:
                    for tid in h.path:
                        holders.setdefault(tid, []).append(h.hid)
                self.alloc_trace.append(
                    sum(
                        len(owners) == 1
                        and all(h.z.get(tid, -1) == owners[0] for h in participants)
                        for tid, owners in holders.items()
                        if tid not in self.done
                    )
                )
            if not built and changed == 0:
                quiet += 1
                if quiet >= needed:
                    return True
            else:
                quiet = 0
        return False

    def _whole_team(self, comp: list[Harvester], live: list[Harvester]) -> bool:
        hids = {h.hid for h in comp}
        if self.information == "shared":
            return len(hids) == len(live)
        roster: set[int] = set()
        failed: set[int] = set()
        for h in comp:
            roster.add(h.hid)
            roster |= set(h.peer_state)
            failed |= h.failed_view
        roster.discard(BASE_ID)
        return (roster - failed) <= hids

    def _allocation_cycle(self, participants: Optional[list[Harvester]] = None) -> None:
        live = self._live()
        if participants is None:
            participants = live
        else:
            keep = {id(h) for h in participants}
            participants = [h for h in live if id(h) in keep]
        if not participants:
            return
        self.alloc_cycles += 1
        self._cycle_hops = 0
        _t0 = time.perf_counter()
        self.allocation_stable.append(self._greedy_allocation(participants))
        self.greedy_time += time.perf_counter() - _t0
        comps = self._components(participants)
        changed_by_balancing = False
        balancing_retries = 0
        if self.balancing:
            for comp in comps:
                whole = self._whole_team(comp, live)
                if not whole and not (self.component_balancing and len(comp) > 1):
                    continue
                auction_owner = self._owner_map()
                self.round += 1
                _t1 = time.perf_counter()
                if whole:
                    self._balanced_partition(comp)
                else:
                    held: set[int] = set()
                    for h in comp:
                        held |= set(h.path)
                    self._balanced_partition(comp, sorted(held), accept_if_better=True)
                self.balance_time += time.perf_counter() - _t1
                if whole:
                    changed_by_balancing = auction_owner != self._owner_map()
                sent = len(comp) * max(0, len(comp) - 1)
                self.msg_balancing_total += sent
                self.msg_balancing_peak_round = max(self.msg_balancing_peak_round, sent)
                for harvester in comp:
                    self.msg_balancing_by_harvester[harvester.hid] += max(
                        0, len(comp) - 1
                    )
                if self.packet_loss > 0 and sent:
                    retries = self._rng.geometric(1.0 - self.packet_loss, size=sent)
                    balancing_retries = max(balancing_retries, int(retries.max()))
        self.partition_changed_by_balancing.append(changed_by_balancing)
        if self.round_latency > 0 or self.latency_jitter > 0:
            passes = self._cycle_hops + balancing_retries
            delay = passes * self.round_latency
            if self.latency_jitter > 0 and passes:
                delay += float(
                    self._rng.uniform(0.0, self.latency_jitter, passes).sum()
                )
            self.planning_delay += delay
            for h in participants:
                h.hold_until = max(h.hold_until, self._clock + delay)

    def _waypoints(self, h: Harvester) -> list[np.ndarray]:
        points: list[np.ndarray] = []
        pos = h.pos
        rest = list(h.path)
        if (
            h.cur_task is not None
            and h.cur_entry is not None
            and h.cur_exit is not None
        ):
            if h.substate == "to_exit":
                points.append(h.cur_exit)
            else:
                points += [h.cur_entry, h.cur_exit]
            pos = h.cur_exit
            rest = [j for j in rest if j != h.cur_task]
        for j in rest:
            task = self.tasks[j]
            if float(np.hypot(*(pos - task.a))) <= float(np.hypot(*(pos - task.b))):
                points += [task.a, task.b]
                pos = task.b
            else:
                points += [task.b, task.a]
                pos = task.a
        return points

    @staticmethod
    def _predict(state: tuple, t: float) -> tuple[np.ndarray, bool]:
        pos, speed, points, stamp = state
        budget = speed * max(0.0, t - stamp)
        cur = pos
        for point in points:
            d = float(np.hypot(*(point - cur)))
            if d >= budget:
                if d <= 1e-12:
                    return point, False
                return cur + (point - cur) * (budget / d), False
            budget -= d
            cur = point
        return cur, True

    def _learn_failures(
        self, h: Harvester, hids: set[int], t: float, direct: bool = False
    ) -> None:
        for hid in hids:
            if direct and hid not in self._failed:
                self.false_failure_detections += 1
            elif hid in self._failed and hid not in self._detected_once:
                self._detected_once.add(hid)
                self.detection_delays.append(t - self._failure_time[hid])
        if h.hid == BASE_ID:
            return
        for tid, owner in list(h.z.items()):
            if owner in hids and tid not in h.known_done:
                h.y[tid] = NEG_INF
                h.z[tid] = -1
                h.t[tid] = self.round + 1
        h.home_lo = None
        h.home_hi = None
        h.unrestricted = True
        h.needs_build = True

    def _init_knowledge(self) -> None:
        nodes = list(self.harvesters) + ([self.base] if self.base is not None else [])
        for h in nodes:
            for x in self.harvesters:
                if x.hid == h.hid:
                    continue
                h.last_heard[x.hid] = 0.0
                h.peer_state[x.hid] = (x.pos.copy(), x.speed, [], 0.0)

    def _gossip(self, t: float) -> list[dict]:
        nodes = self._live() + ([self.base] if self.base is not None else [])
        events: list[dict] = []
        for comp in self._components(nodes):
            payload = {}
            for x in comp:
                own = (
                    (x.pos.copy(), x.speed, self._waypoints(x), t)
                    if x.hid != BASE_ID
                    else None
                )
                payload[x.hid] = (
                    set(x.known_done),
                    set(x.failed_view),
                    dict(x.last_heard),
                    dict(x.peer_state),
                    own,
                )
            for y in comp:
                heard: set[int] = set()
                done_union: set[int] = set()
                failed_union: set[int] = set()
                freshness: dict[int, float] = {}
                states: dict[int, tuple] = {}
                for x in comp:
                    if x is y:
                        continue
                    self.gossip_messages += 1
                    if self._dropped():
                        self.msg_dropped += 1
                        continue
                    done, failed, last_heard, peer_state, own = payload[x.hid]
                    heard.add(x.hid)
                    done_union |= done
                    failed_union |= failed
                    for k, stamp in last_heard.items():
                        if stamp > freshness.get(k, -1.0):
                            freshness[k] = stamp
                    for k, state in peer_state.items():
                        if state[3] > states.get(k, (None, None, None, -1.0))[3]:
                            states[k] = state
                    if own is not None:
                        freshness[x.hid] = t
                        states[x.hid] = own
                y.known_done |= done_union
                for k, stamp in freshness.items():
                    if k != y.hid and stamp > y.last_heard.get(k, -1.0):
                        y.last_heard[k] = stamp
                for k, state in states.items():
                    if (
                        k != y.hid
                        and state[3] > y.peer_state.get(k, (None, None, None, -1.0))[3]
                    ):
                        y.peer_state[k] = state
                recent = {
                    k for k, stamp in freshness.items() if t - stamp <= self.comm_period
                }
                y.failed_view -= heard | recent
                learned = failed_union - heard - recent - y.failed_view - {y.hid}
                if learned:
                    y.failed_view |= learned
                    self._learn_failures(y, learned, t)
                    events.append({"time": t, "kind": "failure_learned", "hid": y.hid})
                if y.hid != BASE_ID:
                    y.drop_done()
        for y in nodes:
            for k, state in list(y.peer_state.items()):
                if k == y.hid or k in y.failed_view:
                    continue
                if t - y.last_heard.get(k, 0.0) <= self.heartbeat_timeout:
                    continue
                predicted, finished = self._predict(state, t)
                if finished:
                    continue
                if float(np.hypot(*(predicted - y.pos))) <= (
                    self.detect_fraction * self.comm_range
                ):
                    y.failed_view.add(k)
                    self._learn_failures(y, {k}, t, direct=True)
                    events.append(
                        {"time": t, "kind": "loss_detected", "hid": y.hid, "lost": k}
                    )
        return events

    def _update_base_reach(self) -> None:
        if self.base is None:
            return
        nodes = self._live() + [self.base]
        for comp in self._components(nodes):
            if any(h.hid == BASE_ID for h in comp):
                self._base_reach = {h.hid for h in comp if h.hid != BASE_ID}
                return
        self._base_reach = set()

    def _base_knowledge(self) -> tuple[set[int], set[int]]:
        if self.information == "local":
            assert self.base is not None
            return set(self.base.known_done), set(self.base.failed_view)
        return set(self.done), set(self._detected_failures)

    def _base_replan(self, force: bool = False) -> bool:
        if self.base is None:
            return False
        done_view, failed = self._base_knowledge()
        machines = [
            h
            for h in self.harvesters
            if h.active and not h.failed and h.hid in self._base_reach
        ]
        if not machines:
            return False
        signature = (
            frozenset(self._base_reach),
            len(done_view),
            frozenset(failed),
            tuple(sorted((h.hid, round(h.speed, 6)) for h in machines)),
        )
        idle = any(not h.path for h in machines)
        if not force and not (idle and signature != self._base_signature):
            return False
        self._base_signature = signature
        excluded: set[int] = set()
        for hid, rows in self._base_plan.items():
            if hid in self._base_reach or hid in failed:
                continue
            excluded |= {tid for tid in rows if tid not in done_view}
        self._centralized_allocation(machines, excluded, done_view)
        if self.round_latency > 0 or self.latency_jitter > 0:
            passes = 2
            if self.packet_loss > 0:
                passes += (
                    int(
                        self._rng.geometric(
                            1.0 - self.packet_loss, size=len(machines)
                        ).max()
                    )
                    - 1
                )
            delay = passes * self.round_latency
            if self.latency_jitter > 0:
                delay += float(
                    self._rng.uniform(0.0, self.latency_jitter, passes).sum()
                )
            self.planning_delay += delay
            for h in machines:
                h.hold_until = max(h.hold_until, self._clock + delay)
        for h in machines:
            self._base_plan[h.hid] = list(h.path)
        for hid in failed:
            self._base_plan[hid] = []
        return True

    def _idle_events(self, t: float) -> list[dict]:
        events: list[dict] = []
        live = self._live()
        comp_of: dict[int, frozenset] = {}
        whole: set[int] = set()
        for comp in self._components(live):
            key = frozenset(h.hid for h in comp)
            complete = self._whole_team(comp, live)
            for h in comp:
                comp_of[h.hid] = key
                if complete:
                    whole.add(h.hid)
        for h in live:
            if h.hid in whole or h.path or not self._has_avail(h):
                continue
            unclaimed = sum(
                1
                for tid in self.tasks
                if tid not in self._shared_done
                and tid not in h.known_done
                and h.z.get(tid, -1) == -1
            )
            signature = (
                comp_of.get(h.hid),
                len(h.known_done),
                unclaimed,
                self._stale_count(h) if self.information == "local" else 0,
            )
            if self._idle_signature.get(h.hid) == signature:
                continue
            self._idle_signature[h.hid] = signature
            events.append({"time": t, "kind": "idle", "hid": h.hid})
        return events

    def _event_participants(self, events: list[dict]) -> Optional[list[Harvester]]:
        if self.information == "shared":
            return None
        live = self._live()
        wanted: set[int] = set()
        for event in events:
            if "hids" in event:
                wanted |= set(event["hids"])
            elif "hid" in event:
                wanted.add(event["hid"])
        chosen: list[Harvester] = []
        for comp in self._components(live):
            if any(h.hid in wanted for h in comp):
                chosen.extend(comp)
        return chosen

    def _record_replan(
        self, events: list[dict], owner_before: dict, rounds_before: int, started: float
    ) -> None:
        elapsed = time.perf_counter() - started
        owner_after = self._owner_map()
        changed_rows = sum(
            owner_before.get(tid) != owner_after.get(tid)
            for tid in self.tasks
            if tid not in self.done
        )
        self.rows_rebid += changed_rows
        self.reconvergence_rounds.append(self.alloc_rounds - rounds_before)
        self.reconvergence_time.append(elapsed)
        for event in events:
            event["rows_rebid"] = changed_rows
            event["rounds"] = self.alloc_rounds - rounds_before
            event["wall_time"] = elapsed
            self.event_log.append(event)

    def run(self) -> SimResult:
        t = 0.0
        step = 0
        n_tasks = len(self.tasks)
        if self.information == "local":
            self._init_knowledge()
        if self.allocation_policy == "decentralized":
            self._allocation_cycle()
            self._set_home()
        else:
            self.alloc_cycles += 1
            if self.initial_plan is None:
                self._centralized_allocation()
            else:
                self._apply_assignment(self.initial_plan)
            if self.base is not None:
                self._base_plan = {h.hid: list(h.path) for h in self.harvesters}
                self._update_base_reach()
        if self.information == "local":
            self._gossip(0.0)
        self.initial_assignment = {
            harvester.hid: list(harvester.path) for harvester in self.harvesters
        }
        prev_comps = self._component_signature()
        next_topo_check = self.comm_period
        while len(self.done) < n_tasks and t <= self.t_max:
            self._clock = t
            replanning_events = self._apply_events(t)

            for h in self.harvesters:
                completed = self._move(h)
                if completed:
                    self.finish_time[h.hid] = t + self.dt
            t += self.dt
            self._clock = t
            step += 1
            if self.record_traj and step % self.traj_stride == 0:
                for h in self.harvesters:
                    if h.active:
                        self.traj[h.hid].append(h.pos.copy())

            periodic = t >= next_topo_check
            if periodic:
                if self.information == "local":
                    replanning_events += self._gossip(t)
                if self.base is not None:
                    self._update_base_reach()
                comps = self._component_signature()
                if comps != prev_comps and self.allocation_policy == "decentralized":
                    changed = (
                        set().union(*(comps ^ prev_comps))
                        if comps ^ prev_comps
                        else set()
                    )
                    replanning_events.append(
                        {"time": t, "kind": "topology", "hids": sorted(changed)}
                    )
                    prev_comps = comps
                if self.allocation_policy == "decentralized" and self.idle_help:
                    replanning_events += self._idle_events(t)
                next_topo_check += self.comm_period

            if self.allocation_policy == "centralized_base":
                if replanning_events or periodic:
                    owner_before = self._owner_map()
                    rounds_before = self.alloc_rounds
                    started = time.perf_counter()
                    if self._base_replan(force=bool(replanning_events)):
                        self.alloc_cycles += 1
                        self._record_replan(
                            replanning_events or [{"time": t, "kind": "base_update"}],
                            owner_before,
                            rounds_before,
                            started,
                        )
                    else:
                        self.event_log.extend(replanning_events)
                    prev_comps = self._component_signature()
            elif replanning_events:
                owner_before = self._owner_map()
                rounds_before = self.alloc_rounds
                started = time.perf_counter()
                if self.allocation_policy == "decentralized":
                    self._allocation_cycle(self._event_participants(replanning_events))
                elif self.allocation_policy == "centralized":
                    self.alloc_cycles += 1
                    self._centralized_allocation()
                self._record_replan(
                    replanning_events, owner_before, rounds_before, started
                )
                prev_comps = self._component_signature()

            active_paths = any(
                h.active and not h.failed and h.path for h in self.harvesters
            )
            future_join = any(
                index not in self._joined and event_time > t
                for index, (event_time, _) in enumerate(self.join_events)
            )
            if not active_paths and not future_join and len(self.done) < n_tasks:
                if self.allocation_policy == "centralized_base":
                    if any(h.active and not h.failed for h in self.harvesters):
                        continue
                    break
                if self.allocation_policy == "decentralized":
                    if self.information == "local" and not periodic:
                        continue
                    owner_before = self._owner_map()
                    rounds_before = self.alloc_rounds
                    for harvester in self.harvesters:
                        if harvester.active:
                            harvester.home_lo = None
                            harvester.home_hi = None
                            harvester.unrestricted = True
                            harvester.needs_build = True
                    started = time.perf_counter()
                    self._allocation_cycle()
                    owner_after = self._owner_map()
                    if owner_after:
                        self._record_replan(
                            [{"time": t, "kind": "allocation_exhausted"}],
                            owner_before,
                            rounds_before,
                            started,
                        )
                        prev_comps = self._component_signature()
                        continue
                    if self.information == "local" and any(
                        h.active and not h.failed for h in self.harvesters
                    ):
                        continue
                break

        converged = len(self.done) >= n_tasks
        return SimResult(
            completion_time=t,
            total_distance=sum(h.distance for h in self.harvesters),
            row_distance=sum(h.row_distance for h in self.harvesters),
            deadhead_distance=sum(h.deadhead_distance for h in self.harvesters),
            turn_distance=sum(h.turn_distance_travelled for h in self.harvesters),
            per_harvester_tasks=[self.tasks_done_by[h.hid] for h in self.harvesters],
            per_harvester_distance=[h.distance for h in self.harvesters],
            per_harvester_finish=[self.finish_time[h.hid] for h in self.harvesters],
            per_harvester_speed=[h.speed for h in self.harvesters],
            per_harvester_idle=[h.idle_time for h in self.harvesters],
            consensus_changes=self.consensus_changes,
            total_releases=sum(h.n_releases for h in self.harvesters),
            alloc_cycles=self.alloc_cycles,
            alloc_rounds=self.alloc_rounds,
            n_tasks=n_tasks,
            converged=converged,
            allocation_stable=list(self.allocation_stable),
            duplicate_services=self.duplicate_services,
            greedy_rows_scored=sum(h.rows_scored for h in self.harvesters),
            greedy_time=self.greedy_time,
            balance_time=self.balance_time,
            partition_changed_by_balancing=list(self.partition_changed_by_balancing),
            msg_records_total=self.msg_records_total,
            msg_records_by_harvester=dict(self.msg_records_by_harvester),
            msg_records_peak_round=self.msg_records_peak_round,
            msg_balancing_total=self.msg_balancing_total,
            msg_balancing_by_harvester=dict(self.msg_balancing_by_harvester),
            msg_balancing_peak_round=self.msg_balancing_peak_round,
            rows_rebid=self.rows_rebid,
            reconvergence_rounds=list(self.reconvergence_rounds),
            reconvergence_time=list(self.reconvergence_time),
            coverage_fraction=len(self.done) / n_tasks if n_tasks else 1.0,
            initial_assignment={
                hid: list(tids) for hid, tids in self.initial_assignment.items()
            },
            event_log=list(self.event_log),
            msg_dropped=self.msg_dropped,
            gossip_messages=self.gossip_messages,
            false_failure_detections=self.false_failure_detections,
            detection_delays=list(self.detection_delays),
            planner_calls=self.planner_calls,
            planning_delay=self.planning_delay,
        )


def point_in_polygon(p, poly) -> bool:
    x, y = p[0], p[1]
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi
            if x < xc:
                inside = not inside
        j = i
    return inside


def _clip_line_to_polygon(p0, d, poly) -> list[tuple[np.ndarray, np.ndarray]]:
    ts: list[float] = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        det = ex * d[1] - d[0] * ey
        if abs(det) < 1e-12:
            continue
        rx, ry = a[0] - p0[0], a[1] - p0[1]
        s = (d[0] * ry - d[1] * rx) / det
        if -1e-9 <= s <= 1 + 1e-9:
            ts.append((-rx * ey + ex * ry) / det)
    ts.sort()
    segs = []
    for i in range(len(ts) - 1):
        t1, t2 = ts[i], ts[i + 1]
        if t2 - t1 < 1e-6:
            continue
        mx = p0[0] + 0.5 * (t1 + t2) * d[0]
        my = p0[1] + 0.5 * (t1 + t2) * d[1]
        if point_in_polygon((mx, my), poly):
            a = np.array([p0[0] + t1 * d[0], p0[1] + t1 * d[1]])
            b = np.array([p0[0] + t2 * d[0], p0[1] + t2 * d[1]])
            segs.append((a, b))
    return segs


def make_rows(
    polygon,
    n_rows: int,
    heading_deg: float = 90.0,
    min_len: float = 1.0,
) -> list[Task]:
    th = np.radians(heading_deg)
    d = np.array([np.cos(th), np.sin(th)])
    perp = np.array([-np.sin(th), np.cos(th)])
    poly = [(float(x), float(y)) for x, y in polygon]
    projs = np.array(poly) @ perp
    pmin, pmax = float(projs.min()), float(projs.max())
    step = (pmax - pmin) / n_rows
    tasks: list[Task] = []
    tid = 0
    for k in range(n_rows):
        off = pmin + (k + 0.5) * step
        p0 = perp * off
        for a, b in _clip_line_to_polygon(p0, d, poly):
            if float(np.linalg.norm(b - a)) >= min_len:
                tasks.append(Task(tid, a, b))
                tid += 1
    return tasks


def square_field(size: float = 200.0) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]


def rect_field(w: float, h: float) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def l_field(size: float = 240.0) -> list[tuple[float, float]]:
    s = size
    return [(0, 0), (s, 0), (s, 0.5 * s), (0.5 * s, 0.5 * s), (0.5 * s, s), (0, s)]


def hex_field(radius: float = 130.0, cx: float = 130.0, cy: float = 120.0):
    return [
        (cx + radius * np.cos(a), cy + radius * np.sin(a))
        for a in np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    ]


def trapezoid_field(bottom: float = 240.0, top: float = 130.0, h: float = 200.0):
    off = 0.5 * (bottom - top)
    return [(0.0, 0.0), (bottom, 0.0), (bottom - off, h), (off, h)]


def _entry_point(x: float, polygon, ymin: float, ymax: float) -> np.ndarray:
    step = (ymax - ymin) / 300.0
    y = ymin
    while y < ymax and not point_in_polygon((x, y + step), polygon):
        y += step
    return np.array([x, y + step])


def _speeds(n, rng, speeds, speed_range):
    if speeds is not None:
        return [float(s) for s in speeds]
    if speed_range is not None:
        return [float(s) for s in rng.uniform(speed_range[0], speed_range[1], size=n)]
    return [1.5] * n


def make_harvesters(
    n: int,
    polygon,
    rng: np.random.Generator,
    layout: str = "headland",
    speeds=None,
    speed_range=None,
    **kwargs,
) -> list[Harvester]:
    P = np.array(polygon)
    xmin, ymin = float(P[:, 0].min()), float(P[:, 1].min())
    xmax, ymax = float(P[:, 0].max()), float(P[:, 1].max())
    sp = _speeds(n, rng, speeds, speed_range)
    hs = []
    if layout == "headland":
        lo, hi = xmin + 0.05 * (xmax - xmin), xmax - 0.05 * (xmax - xmin)
        base = np.linspace(lo, hi, n)
        gap = (hi - lo) / max(n, 1)
        xs = np.sort(np.clip(base + rng.uniform(-0.28, 0.28, size=n) * gap, lo, hi))
        for i, x in enumerate(xs):
            hs.append(
                Harvester(
                    hid=i,
                    pos=_entry_point(float(x), polygon, ymin, ymax),
                    speed=sp[i],
                    **kwargs,
                )
            )
    else:
        for i in range(n):
            for _ in range(2000):
                p = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
                if point_in_polygon(p, polygon):
                    hs.append(Harvester(hid=i, pos=np.array(p), speed=sp[i], **kwargs))
                    break
            else:
                hs.append(
                    Harvester(
                        hid=i,
                        pos=np.array([0.5 * (xmin + xmax), ymin]),
                        speed=sp[i],
                        **kwargs,
                    )
                )
    return hs


def make_gate_harvesters(
    n: int,
    polygon,
    rng: np.random.Generator,
    spread: float = 0.04,
    speeds=None,
    speed_range=None,
    **kwargs,
) -> list[Harvester]:
    P = np.array(polygon)
    xmin, ymin = float(P[:, 0].min()), float(P[:, 1].min())
    xmax, ymax = float(P[:, 0].max()), float(P[:, 1].max())
    gate_x = xmin + 0.5 * (xmax - xmin)
    sp = _speeds(n, rng, speeds, speed_range)
    hs = []
    for i in range(n):
        x = gate_x + rng.uniform(-spread, spread) * (xmax - xmin)
        x = float(np.clip(x, xmin + 1.0, xmax - 1.0))
        hs.append(
            Harvester(
                hid=i, pos=_entry_point(x, polygon, ymin, ymax), speed=sp[i], **kwargs
            )
        )
    return hs
