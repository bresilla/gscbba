from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.spatial import cKDTree

from .cbba import (
    Simulator,
    hex_field,
    l_field,
    make_harvesters,
    make_rows,
    square_field,
    trapezoid_field,
)

RESULTS = os.path.abspath("results")
FIGDIR = os.path.abspath("figures")

sns.set_theme(
    context="notebook",
    style="whitegrid",
    font_scale=1.05,
    rc={
        "axes.titlesize": "medium",
        "axes.titleweight": "semibold",
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.9,
        "grid.color": "0.9",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    },
)

PALETTE = sns.color_palette("colorblind", 6).as_hex()
ACCENT = PALETTE[0]
ACCENT2 = PALETTE[3]


def _load(name: str) -> dict:
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg")


def fig_field() -> None:
    n_rows = 20
    xs = np.linspace(0.5, 19.5, n_rows)
    y0, y1 = 0.5, 9.5

    init_owner = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 3 + [4] * 3 + [5] * 2
    redist_owner = list(init_owner)
    redist_owner[0] = 1
    redist_owner[1] = 1
    redist_owner[2] = 2
    redist_owner[3] = 2

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    titles = [
        "Initial allocation  $(|B_i|) = (4,4,4,3,3,2)$",
        "After $h_1$ failure  $(|B_i|) = (0,6,6,3,3,2)$",
    ]
    for ax, owners, title in zip(axes, (init_owner, redist_owner), titles):
        for x, o in zip(xs, owners):
            ax.plot([x, x], [y0, y1], color=PALETTE[o], lw=3.2, solid_capstyle="round")
        ax.set_title(title)
        ax.set_xlim(-0.5, 20.0)
        ax.set_ylim(-0.6, 10.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("20 AB-line rows in one shared field")
    axes[1].scatter([], [])

    handles = [
        Line2D([0], [0], color=PALETTE[i], lw=3.2, label=f"$h_{i + 1}$")
        for i in range(6)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Single-field row allocation and consensus-driven redistribution", y=1.02
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig_field")


def fig_scalability() -> None:
    d = _load("scalability")
    Ns = d["Ns"]
    ct = [d["data"][str(n)]["completion_time"] for n in Ns]
    td = [d["data"][str(n)]["total_distance"] for n in Ns]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    a1.errorbar(
        Ns,
        [s["mean"] for s in ct],
        yerr=[s["std"] for s in ct],
        marker="o",
        color=ACCENT,
        capsize=3,
        lw=2,
    )
    a1.set_xlabel("Number of harvesters $N$")
    a1.set_ylabel("Completion time (s)")
    a1.set_title("Makespan vs team size")
    a1.set_xticks(Ns)
    a1.grid(alpha=0.25)

    a2.errorbar(
        Ns,
        [s["mean"] for s in td],
        yerr=[s["std"] for s in td],
        marker="s",
        color=ACCENT2,
        capsize=3,
        lw=2,
    )
    a2.set_xlabel("Number of harvesters $N$")
    a2.set_ylabel("Total distance travelled (m)")
    a2.set_title("Total travel vs team size")
    a2.set_xticks(Ns)
    a2.grid(alpha=0.25)

    fig.suptitle(
        f"Scalability: {d['M']} rows in a {int(d['extent'])}×{int(d['extent'])} m "
        f"field ({d['trials']} trials)",
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "fig_scalability")


def fig_consensus() -> None:
    d = _load("scalability")
    diagnostics = _load("diagnostics")
    Ns = d["Ns"]
    rounds = [d["data"][str(n)]["consensus_rounds"]["mean"] for n in Ns]
    rounds_sd = [d["data"][str(n)]["consensus_rounds"]["std"] for n in Ns]

    finish = diagnostics["consensus"]["per_harvester_finish"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    a1.bar(Ns, rounds, yerr=rounds_sd, color=ACCENT, width=1.1, capsize=3, alpha=0.85)
    a1.set_xlabel("Number of harvesters $N$")
    a1.set_ylabel("Consensus rounds per allocation")
    a1.set_title("Observed auction stopping rounds")
    a1.set_xticks(Ns)
    a1.set_ylim(0, max(rounds) * 1.25)

    mean_finish = float(np.mean(finish))
    a2.bar(
        [f"$h_{i + 1}$" for i in range(len(finish))],
        finish,
        color=PALETTE[: len(finish)],
        alpha=0.9,
    )
    a2.axhline(
        mean_finish, color="0.35", ls="--", lw=1.2, label=f"mean = {mean_finish:.0f} s"
    )
    a2.set_ylabel("Finish time (s)")
    a2.set_title(f"Finish times ($N=6$, {d['M']} rows)")
    a2.set_ylim(0, max(finish) * 1.18)
    a2.legend(frameon=False, loc="upper right")

    fig.suptitle("Consensus effort and workload balance", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_consensus")


def fig_commrange() -> None:
    d = _load("commrange")
    Rcs = d["Rcs"]

    def series(hops, field):
        return (
            [d["data"][str(hops)][str(r)][field]["mean"] for r in Rcs],
            [d["data"][str(hops)][str(r)][field]["std"] for r in Rcs],
        )

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    m, s = series(1, "completion_time")
    a1.errorbar(Rcs, m, yerr=s, marker="o", color=ACCENT, capsize=3, lw=2)
    a1.set_xlabel("Communication range $R_c$ (m)")
    a1.set_ylabel("Completion time (s)")
    a1.set_title("Connectivity unlocks the balanced partition")
    a1.set_ylim(0, max(m) * 1.2)
    a1.grid(alpha=0.25)

    m, s = series(1, "total_distance")
    a2.errorbar(Rcs, m, yerr=s, marker="s", color=ACCENT2, capsize=3, lw=2)
    a2.set_xlabel("Communication range $R_c$ (m)")
    a2.set_ylabel("Total distance travelled (m)")
    a2.set_title("Field always fully covered")
    a2.set_ylim(0, max(m) * 1.3)
    a2.grid(alpha=0.25)

    fig.suptitle(
        f"Communication constraints: $N={d['N']}$, {d['M']} rows "
        f"({d['trials']} trials)",
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "fig_commrange")


def fig_loss() -> None:
    d = _load("loss")
    fractions = d["failure_fractions"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    for range_key, label, color in (
        ("50", "$R_c=50$ m", PALETTE[1]),
        ("100", "$R_c=100$ m", PALETTE[2]),
        ("inf", "full connectivity", PALETTE[0]),
    ):
        values = [
            100 * d["data"][range_key][str(fraction)]["1"]["overhead_fraction"]["mean"]
            for fraction in fractions
        ]
        errors = [
            100 * d["data"][range_key][str(fraction)]["1"]["overhead_fraction"]["std"]
            for fraction in fractions
        ]
        a1.errorbar(
            np.asarray(fractions) * 100,
            values,
            yerr=errors,
            marker="o",
            capsize=3,
            lw=2,
            color=color,
            label=label,
        )
    a1.set_xlabel("Failure time (% of baseline makespan)")
    a1.set_ylabel("Makespan overhead (%)")
    a1.set_title("Cost of one failure")
    a1.legend(frameon=False, fontsize=8.5)
    a1.grid(alpha=0.25)

    rows_rebid = [
        d["data"]["inf"][str(fraction)]["1"]["rows_rebid"]["mean"]
        for fraction in fractions
    ]
    rounds = [
        d["data"]["inf"][str(fraction)]["1"]["reconvergence_rounds"]["mean"]
        for fraction in fractions
    ]
    x = np.arange(len(fractions))
    a2.bar(x, rows_rebid, color=ACCENT, alpha=0.85, label="rows re-bid")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{fraction:.0%}" for fraction in fractions])
    a2.set_xlabel("Failure time")
    a2.set_ylabel("Rows re-bid")
    a2.set_title("Reallocation under full connectivity")
    twin = a2.twinx()
    twin.plot(x, rounds, "o-", color=ACCENT2, lw=2, label="consensus rounds")
    twin.set_ylabel("Re-convergence rounds")
    twin.set_ylim(0, max(rounds) * 1.25)
    handles1, labels1 = a2.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    a2.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8.5)
    a2.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"Harvester-loss response ($N={d['N']}$, $M={d['M']}$, {d['trials']} trials)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_loss")


def _box(ax, x, y, w, h, text, fc, ec="#444", fs: float = 9, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.10",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight
    )


def _arrow(ax, p, q, color="#444", style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=14,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    _box(ax, 0.3, 2.7, 2.4, 1.6, "Simulator\n(ground-truth\nfield state)", "#eef3fb")
    _box(ax, 4.2, 1.9, 3.4, 3.4, "", "#f6f9f3", ec="#7aa86b")
    ax.text(5.9, 5.0, "Harvester $i$ runtime", ha="center", fontsize=10, weight="bold")
    _box(
        ax,
        4.5,
        3.35,
        2.8,
        1.15,
        "Task allocation\n(GS-CBBA + KD-tree)",
        "#edf6ea",
        fs=8.5,
    )
    _box(ax, 4.5, 2.15, 2.8, 1.0, "Navigation /\ncontrol", "#edf6ea", fs=8.5)
    _box(
        ax,
        9.2,
        2.7,
        2.5,
        1.6,
        "Inter-harvester\nnetwork (Zenoh)",
        "#fbf0e6",
    )
    _box(ax, 8.7, 0.3, 1.4, 0.9, "Harvester $j$", "#f6f9f3", ec="#7aa86b", fs=8)
    _box(ax, 10.4, 0.3, 1.4, 0.9, "Harvester $N$", "#f6f9f3", ec="#7aa86b", fs=8)

    _arrow(ax, (2.7, 3.5), (4.2, 3.5))
    _arrow(ax, (4.2, 3.2), (2.7, 3.2))
    ax.text(3.45, 3.75, "State\nupdates", ha="center", fontsize=7.5, color="#444")
    _arrow(ax, (7.6, 3.7), (9.2, 3.7))
    _arrow(ax, (9.2, 3.4), (7.6, 3.4))
    ax.text(
        8.4, 3.95, "Zenoh\nbids /\nreleases", ha="center", fontsize=7.5, color="#444"
    )
    _arrow(ax, (9.4, 2.7), (9.4, 1.2), rad=0.0)
    _arrow(ax, (11.1, 2.7), (11.1, 1.2), rad=0.0)

    ax.set_title("Multi-harvester coordination architecture", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_architecture")


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 4)
    ax.axis("off")

    _box(
        ax, 0.2, 1.4, 2.1, 1.2, "Local row list\n+ neighbor\nupdates", "#f2f2f2", fs=8
    )
    _box(
        ax,
        2.9,
        1.3,
        2.4,
        1.4,
        "1. Spatial filter\n($k$-NN index,\nnearby rows)",
        "#eaf2fb",
        fs=8.5,
    )
    _box(
        ax,
        5.7,
        1.3,
        2.4,
        1.4,
        "2. Greedy bundle\n(score by\nproximity)",
        "#edf6ea",
        fs=8.5,
    )
    _box(
        ax,
        8.5,
        1.3,
        2.4,
        1.4,
        "3. Consensus\n(timestamped\nbids, resolve)",
        "#fbf0e6",
        fs=8.5,
    )
    _box(ax, 11.3, 1.4, 1.5, 1.2, "Next row\nto harvest", "#f2f2f2", fs=8)

    for x0, x1 in ((2.3, 2.9), (5.3, 5.7), (8.1, 8.5), (10.9, 11.3)):
        _arrow(ax, (x0, 2.0), (x1, 2.0))
    _arrow(ax, (9.7, 1.3), (4.1, 1.3), rad=-0.28, color="#999", lw=1.4)
    ax.text(
        6.9, -0.32, "repeat every planning cycle", ha="center", fontsize=8, color="#777"
    )

    ax.set_title("Three-stage decentralized planner on each harvester", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_pipeline")


def _plot_rows(ax, tasks, colors, lw=2.4):
    for tid, t in tasks.items():
        ax.plot(
            [t.a[0], t.b[0]],
            [t.a[1], t.b[1]],
            color=colors[tid],
            lw=lw,
            solid_capstyle="round",
        )


def _draw_boundary(ax, polygon):
    P = list(polygon) + [polygon[0]]
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    ax.plot(xs, ys, color="#333", lw=1.3, zorder=1)


def _run(polygon, n_rows, N, seed, heading=90.0, layout="headland", speed_range=None):
    rng = np.random.default_rng(seed)
    tasks = make_rows(polygon, n_rows=n_rows, heading_deg=heading)
    harv = make_harvesters(
        N, polygon, rng, layout=layout, speed_range=speed_range, max_bundle=n_rows
    )
    starts = [h.start_pos.copy() for h in harv]
    sim = Simulator(tasks, harv, comm_range=1e9, hops=2)
    sim.run()
    return {t.id: t for t in tasks}, starts, sim


def fig_spatial() -> None:
    extent, M, N = 200.0, 100, 6
    poly = square_field(extent)
    tasks, starts, sim = _run(poly, M, N, seed=7, heading=90.0)
    owner = sim.completed_by

    ids = list(tasks.keys())
    cents = np.array([tasks[t].centroid for t in ids])
    tree = cKDTree(cents)
    k = 16
    p0 = starts[2]
    d, idx = tree.query(p0, k=k)
    cand = set(int(i) for i in np.atleast_1d(idx))
    r_query = float(np.max(d))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.9))

    colA = {tid: (ACCENT if i in cand else "#cfcfcf") for i, tid in enumerate(ids)}
    _plot_rows(a1, tasks, colA, lw=2.0)
    _draw_boundary(a1, poly)
    a1.add_patch(Circle(tuple(p0), r_query, fill=False, ls="--", lw=1.6, color=ACCENT2))
    a1.plot(*p0, marker="*", ms=20, color=ACCENT2, markeredgecolor="k", zorder=5)
    a1.set_title(f"KD-tree query: a harvester's {k} nearest rows")

    colB = {tid: PALETTE[owner.get(tid, 0) % len(PALETTE)] for tid in ids}
    _plot_rows(a2, tasks, colB, lw=2.4)
    _draw_boundary(a2, poly)
    for hid, s in enumerate(starts):
        a2.plot(
            *s,
            marker="*",
            ms=14,
            color=PALETTE[hid % len(PALETTE)],
            markeredgecolor="k",
            zorder=5,
        )
    a2.set_title("After connected-team balancing")

    for ax in (a1, a2):
        ax.set_xlim(-10, extent + 10)
        ax.set_ylim(-12, extent + 10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("field x (m)")
    a1.set_ylabel("field y (m)")

    fig.suptitle(
        f"Parallel AB-line rows: spatial query and resulting partition "
        f"($N=6$, {M} rows)",
        y=1.0,
    )
    fig.tight_layout()
    _save(fig, "fig_spatial")


def fig_shapes() -> None:
    cases = [
        ("Square, diagonal rows", square_field(200.0), 70, 35.0, 5),
        ("L-shaped field", l_field(220.0), 64, 90.0, 5),
        ("Hexagonal field", hex_field(120.0), 60, 90.0, 5),
        ("Trapezoidal field", trapezoid_field(230.0, 120.0, 190.0), 60, 90.0, 4),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5))
    for ax, (title, poly, nr, head, N) in zip(axes, cases):
        tasks, starts, sim = _run(poly, nr, N, seed=11, heading=head)
        owner = sim.completed_by
        col = {tid: PALETTE[owner.get(tid, 0) % len(PALETTE)] for tid in tasks}
        _plot_rows(ax, tasks, col, lw=2.2)
        _draw_boundary(ax, poly)
        for hid, s in enumerate(starts):
            ax.plot(
                *s,
                marker="*",
                ms=12,
                color=PALETTE[hid % len(PALETTE)],
                markeredgecolor="k",
                zorder=5,
            )
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Parallel-row allocation generalises across field shapes and row headings",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_shapes")


def fig_routes() -> None:
    extent, M, N = 200.0, 60, 4
    poly = square_field(extent)
    rng = np.random.default_rng(0)
    speeds = [1.0, 1.3, 1.6, 2.0]
    tasks = make_rows(poly, n_rows=M, heading_deg=90.0)
    harv = make_harvesters(N, poly, rng, layout="headland", speeds=speeds, max_bundle=M)
    starts = [h.start_pos.copy() for h in harv]
    sim = Simulator(tasks, harv, comm_range=1e9, hops=2, record_traj=True)
    r = sim.run()

    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    for t in {t.id: t for t in tasks}.values():
        ax.plot(
            [t.a[0], t.b[0]],
            [t.a[1], t.b[1]],
            color="#d9d9d9",
            lw=1.6,
            solid_capstyle="round",
            zorder=1,
        )
    _draw_boundary(ax, poly)
    for hid in range(N):
        tr = np.array(sim.traj[hid])
        ax.plot(
            tr[:, 0],
            tr[:, 1],
            color=PALETTE[hid % len(PALETTE)],
            lw=1.5,
            alpha=0.95,
            zorder=3,
        )
        ax.plot(
            *starts[hid],
            marker="*",
            ms=16,
            color=PALETTE[hid % len(PALETTE)],
            markeredgecolor="k",
            zorder=5,
            label=f"$h_{hid + 1}$ ({speeds[hid]:.1f} m/s, {r.per_harvester_tasks[hid]} rows)",
        )
    ax.set_aspect("equal")
    ax.set_xlim(-10, extent + 10)
    ax.set_ylim(-30, extent + 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.16),
        fontsize=8.5,
    )
    ax.set_title(
        f"Serpentine routes with mixed speeds: faster machines\ntake wider blocks "
        f"($N={N}$, {M} rows)"
    )
    fig.tight_layout()
    _save(fig, "fig_routes")


def fig_speed() -> None:
    cached = _load("diagnostics")["speed"]
    extent = _load("diagnostics")["extent"]
    M, N = cached["M"], cached["N"]
    speeds = cached["speeds"]
    starts = np.asarray(cached["starts"])
    owner = {int(tid): hid for tid, hid in cached["owner"].items()}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.3))

    for task in cached["tasks"]:
        color = PALETTE[owner.get(task["id"], 0) % len(PALETTE)]
        a1.plot(
            [task["a"][0], task["b"][0]],
            [task["a"][1], task["b"][1]],
            color=color,
            lw=2.2,
            solid_capstyle="round",
        )
    _draw_boundary(a1, square_field(extent))
    for hid, s in enumerate(starts):
        a1.plot(
            *s,
            marker="*",
            ms=16,
            color=PALETTE[hid % len(PALETTE)],
            markeredgecolor="k",
            zorder=5,
        )
        a1.text(
            s[0],
            -16,
            f"{speeds[hid]:.1f}",
            ha="center",
            va="top",
            fontsize=9,
            color=PALETTE[hid % len(PALETTE)],
            weight="bold",
        )
    a1.text(extent / 2, -30, "harvester speed (m/s)", ha="center", va="top", fontsize=9)
    a1.set_xlim(-10, extent + 10)
    a1.set_ylim(-34, extent + 10)
    a1.set_aspect("equal")
    a1.set_xticks([])
    a1.set_yticks([])
    a1.set_title("Faster machines get wider blocks (one field)")

    sp_sorted = sorted(cached["rows_by_speed"], key=float)
    means = [cached["rows_by_speed"][speed]["mean"] for speed in sp_sorted]
    stds = [cached["rows_by_speed"][speed]["std"] for speed in sp_sorted]
    x = np.arange(len(sp_sorted))
    bars = a2.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(sp_sorted))],
        alpha=0.9,
        width=0.62,
    )
    a2.set_xticks(x)
    a2.set_xticklabels(sp_sorted)
    a2.set_xlabel("harvester speed (m/s)")
    a2.set_ylabel("rows harvested")
    a2.set_ylim(0, max(means) * 1.34)
    for b, v, e in zip(bars, means, stds):
        a2.text(
            b.get_x() + b.get_width() / 2,
            v + e + 0.8,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    a2.set_title(f"Mean rows vs speed ({cached['trials']} trials)")

    fig.suptitle(f"Heterogeneous harvester speeds ($N={N}$, {M} rows)", y=1.0)
    fig.tight_layout()
    _save(fig, "fig_speed")


def _max_block_gap(sim, tasks):
    perp = sim._perp
    offs = sorted(c[0] * perp[0] + c[1] * perp[1] for c in (t.centroid for t in tasks))
    sp = np.median(np.diff(offs)) if len(offs) > 1 else 1.0
    worst = 0.0
    for h in sim.harvesters:
        os = sorted(sim._off(tid) for tid, o in sim.completed_by.items() if o == h.hid)
        if len(os) > 1:
            worst = max(worst, max(np.diff(os)))
    return worst / sp


def fig_efficiency() -> None:
    benchmark = _load("benchmark")["scalability"]
    Ns = [int(value) for value in benchmark]
    full = [benchmark[str(n)]["methods"]["full"] for n in Ns]
    dp = [benchmark[str(n)]["methods"]["dp"] for n in Ns]
    lower = [benchmark[str(n)]["methods"]["lower_bound"] for n in Ns]
    mk = [method["makespan"]["mean"] for method in full]
    opt = [method["makespan"]["mean"] for method in dp]
    lb = [method["makespan"]["mean"] for method in lower]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    a1.plot(Ns, mk, "o-", color=ACCENT, lw=2, label="full method")
    a1.plot(Ns, opt, "D--", color=ACCENT2, lw=2, label="centralized DP")
    a1.plot(Ns, lb, "s:", color="gray", lw=2, label="deadhead-free bound")
    a1.set_xlabel("Number of harvesters $N$")
    a1.set_ylabel("Completion time (s)")
    a1.set_title("Makespan vs lower bound")
    a1.set_xticks(Ns)
    a1.legend(frameon=False)
    a1.grid(alpha=0.25)

    feasible_efficiency = [value / achieved * 100 for value, achieved in zip(opt, mk)]
    bound_efficiency = [value / achieved * 100 for value, achieved in zip(lb, mk)]
    a2.plot(
        Ns,
        feasible_efficiency,
        "D-",
        color=ACCENT2,
        lw=2,
        label="centralized DP / full",
    )
    a2.plot(Ns, bound_efficiency, "s--", color="gray", lw=2, label="lower bound / full")
    a2.axhline(100, color="gray", ls=":", lw=1)
    a2.set_xlabel("Number of harvesters $N$")
    a2.set_ylabel("Efficiency (%)")
    a2.set_title("Efficiency against feasible and ideal references")
    a2.set_xticks(Ns)
    a2.set_ylim(0, 110)
    a2.legend(frameon=False, fontsize=8.5)
    a2.grid(alpha=0.25)

    fig.suptitle("Allocation efficiency (100 rows, homogeneous team)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_efficiency")


def fig_deadhead() -> None:
    benchmark = _load("benchmark")["scalability"]
    Ns = [int(value) for value in benchmark]
    trav, dead, total_sd = [], [], []
    for n in Ns:
        method = benchmark[str(n)]["methods"]["full"]
        totals = method["total_distance"]["values"]
        shares = method["deadhead_share"]["values"]
        total_sd.append(method["total_distance"]["std"])
        trav.append(
            np.mean([total * (1.0 - share) for total, share in zip(totals, shares)])
        )
        dead.append(np.mean([total * share for total, share in zip(totals, shares)]))

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = np.arange(len(Ns))
    ax.bar(x, trav, color=ACCENT, label="row traversal (productive)")
    ax.bar(
        x,
        dead,
        bottom=trav,
        color=ACCENT2,
        label="straight deadhead",
        yerr=total_sd,
        capsize=3,
    )
    for i, (a, b) in enumerate(zip(trav, dead)):
        ax.text(i, a + b + 300, f"{100 * b / (a + b):.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of harvesters $N$")
    ax.set_ylabel("Total distance travelled (m)")
    ax.set_title("Deadhead is a small share of total travel")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_ylim(0, max(t + d for t, d in zip(trav, dead)) * 1.12)
    fig.tight_layout()
    _save(fig, "fig_deadhead")


def fig_convergence() -> None:
    diagnostics = _load("diagnostics")
    M = diagnostics["M"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for N, color in zip(
        (2, 4, 6, 10), (PALETTE[0], PALETTE[2], PALETTE[3], PALETTE[1])
    ):
        traces = diagnostics["convergence"][str(N)]
        L = max(len(t) for t in traces)
        padded = [t + [t[-1]] * (L - len(t)) for t in traces]
        mean = np.mean(padded, axis=0)
        sd = np.std(padded, axis=0, ddof=1) if len(traces) > 1 else np.zeros(L)
        ax.errorbar(
            range(1, L + 1),
            mean,
            yerr=sd,
            fmt="o-",
            color=color,
            lw=2,
            capsize=3,
            label=f"$N={N}$",
        )
    ax.axhline(M, color="gray", ls="--", lw=1)
    ax.text(0.7, M + 2, f"all {M} rows", fontsize=8.5, color="gray")
    ax.set_xlabel("Consensus round")
    ax.set_ylabel("Uniquely held rows with agreed ownership")
    ax.set_title("Initial ownership agreement")
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(0, M * 1.08)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_convergence")


def fig_benchmark() -> None:
    data = _load("benchmark")["scalability"]
    Ns = [int(value) for value in data]
    styles = (
        ("lower_bound", "deadhead-free bound", "gray", ":", "s"),
        ("dp", "executed fixed-sweep DP", ACCENT2, "--", "D"),
        ("static", "static speed split", PALETTE[2], "-.", "^"),
        ("standard_cbba", "standard CBBA", PALETTE[5], "-", "v"),
        ("auction_only", "GS-CBBA: auction only", PALETTE[4], "-", "P"),
        ("full", "GS-CBBA: auction + balance", ACCENT, "-", "o"),
    )
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for method, label, color, line_style, marker in styles:
        means = [data[str(n)]["methods"][method]["makespan"]["mean"] for n in Ns]
        errors = [data[str(n)]["methods"][method]["makespan"]["std"] for n in Ns]
        a1.errorbar(
            Ns,
            means,
            yerr=errors,
            label=label,
            color=color,
            ls=line_style,
            marker=marker,
            capsize=2.5,
            lw=1.8,
        )
    a1.set_xlabel("Number of harvesters $N$")
    a1.set_ylabel("Makespan (s)")
    a1.set_xticks(Ns)
    a1.set_title("Executed makespan")
    a1.legend(frameon=False, ncol=1, fontsize=8.0)
    a1.grid(alpha=0.25)
    for method, label, color, line_style, marker in styles[1:]:
        means = [
            1000.0 * data[str(n)]["methods"][method]["compute_time"]["mean"] for n in Ns
        ]
        a2.plot(
            Ns, means, color=color, ls=line_style, marker=marker, lw=1.8, label=label
        )
    a2.set_yscale("log")
    a2.set_xlabel("Number of harvesters $N$")
    a2.set_ylabel("Allocation compute time (ms, log)")
    a2.set_xticks(Ns)
    a2.set_title("Computation")
    a2.grid(alpha=0.25, which="both")
    fig.suptitle("Centralized and decentralized allocation methods (100 rows)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_benchmark")


EVENT_LABELS = [
    "one\nloss",
    "two\nlosses",
    "one\njoin",
    "loss then\njoin",
    "speed\ndrop",
]
EVENT_METHODS = (
    ("frozen", "frozen plan", "#9a9a9a"),
    ("centralized_resolve", "oracle re-solve", "#5b5b5b"),
    ("centralized_base", "base-station re-solve", ACCENT2),
    ("decentralized", "GS-CBBA replanning", ACCENT),
)


def fig_events() -> None:
    result = _load("events")
    scenarios = result["scenarios"]
    ranges = (
        ("full", "full connectivity"),
        ("100", "$R_c=100$ m"),
        ("50", "$R_c=50$ m"),
    )
    infos = (("shared", "shared state"), ("local", "local information"))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), sharey=True)
    width = 0.2
    x = np.arange(len(scenarios))
    for row, (info, info_label) in enumerate(infos):
        for col, (range_name, range_label) in enumerate(ranges):
            ax = axes[row, col]
            for index, (method, label, color) in enumerate(EVENT_METHODS):
                cell = result["data"][info][range_name]
                values = [cell[s][method]["completion_time"]["mean"] for s in scenarios]
                coverage = [
                    cell[s][method]["coverage_fraction"]["mean"] for s in scenarios
                ]
                bars = ax.bar(
                    x + (index - 1.5) * width,
                    values,
                    width,
                    color=color,
                    alpha=0.9,
                    label=label,
                    hatch="//" if method == "frozen" else None,
                )
                for bar, fraction in zip(bars, coverage):
                    if fraction < 0.999:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 30,
                            f"{fraction:.0%}",
                            ha="center",
                            va="bottom",
                            fontsize=6.5,
                            rotation=90,
                        )
            ax.set_xticks(x)
            ax.set_xticklabels(EVENT_LABELS, fontsize=8)
            ax.set_title(f"{range_label}, {info_label}", fontsize=10)
            ax.grid(alpha=0.2, axis="y")
        axes[row, 0].set_ylabel("Completion or stop time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Dynamic events ($N={result['N']}$, $M={result['M']}$, "
        f"{result['trials']} paired trials; labels mark incomplete coverage)",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig_events")


def fig_network() -> None:
    result = _load("network")
    names = list(result["conditions"])
    labels = [
        "ideal",
        "10% loss",
        "20% loss",
        "30% loss",
        "50 ms",
        "250 ms",
        "1 s",
        "20% +\n250 ms",
    ]
    x = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.1))
    colors = {"one_loss": ACCENT, "two_losses": PALETTE[2], "speed_drop": PALETTE[4]}
    for scenario in result["scenarios"]:
        for method, style, tag in (
            ("decentralized", "o-", "GS-CBBA"),
            ("centralized_base", "s--", "base station"),
        ):
            means = [
                result["data"][name][scenario][method]["completion_time"]["mean"]
                for name in names
            ]
            a1.plot(
                x,
                means,
                style,
                color=colors[scenario],
                lw=1.8,
                label=f"{scenario.replace('_', ' ')}: {tag}",
            )
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("Completion time (s)")
    a1.set_title("Local information, $R_c=100$ m")
    a1.legend(frameon=False, fontsize=7.4, ncol=2)
    a1.grid(alpha=0.25)
    for key, marker, color, tag in (
        ("duplicate_services", "o-", ACCENT, "duplicate services"),
        ("false_failure_detections", "^-", ACCENT2, "false failure alarms"),
    ):
        means = [
            np.mean(
                [
                    result["data"][name][scenario]["decentralized"][key]["mean"]
                    for scenario in result["scenarios"]
                ]
            )
            for name in names
        ]
        a2.plot(x, means, marker, color=color, lw=1.8, label=tag)
    a2.set_xticks(x)
    a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("Events per run (mean over scenarios)")
    a2.set_title("GS-CBBA side effects")
    a2.legend(frameon=False, fontsize=8.5)
    a2.grid(alpha=0.25)
    fig.suptitle(
        f"Packet loss and latency ($N={result['N']}$, $M={result['M']}$, "
        f"{result['trials']} paired trials)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_network")


def fig_turncost() -> None:
    result = _load("turncost")
    turns = result["turn_distances"]
    shares = [
        100 * result["data"][str(turn)]["deadhead_share"]["mean"] for turn in turns
    ]
    gap_dp = [100 * result["data"][str(turn)]["gap_to_dp"]["mean"] for turn in turns]
    gap_lb = [
        100 * result["data"][str(turn)]["gap_to_lower_bound"]["mean"] for turn in turns
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    a1.plot(turns, shares, "o-", color=ACCENT, lw=2)
    a1.set_xlabel("Turn distance per row transition (m)")
    a1.set_ylabel("Deadhead and turns (% of travel)")
    a1.set_title("Nonproductive travel")
    a1.grid(alpha=0.25)
    a2.plot(turns, gap_dp, "D-", color=ACCENT2, lw=2, label="gap to centralized DP")
    a2.plot(turns, gap_lb, "s--", color="gray", lw=2, label="gap to ideal bound")
    a2.set_xlabel("Turn distance per row transition (m)")
    a2.set_ylabel("Makespan gap (%)")
    a2.set_title("Makespan sensitivity")
    a2.legend(frameon=False, fontsize=8.5)
    a2.grid(alpha=0.25)
    fig.suptitle(
        f"Turn-cost sensitivity ($N={result['N']}$, $M={result['M']}$, "
        f"{result['trials']} trials)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_turncost")


def fig_ksweep() -> None:
    result = _load("ksweep")
    labels = [str(value) for value in result["candidate_counts"]]
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for range_name, title, color in (
        ("full", "full connectivity", ACCENT),
        ("75", "$R_c=75$ m", ACCENT2),
    ):
        makespan = [
            result["data"][range_name][label]["makespan"]["mean"] for label in labels
        ]
        scored = [
            result["data"][range_name][label]["rows_scored"]["mean"] for label in labels
        ]
        a1.plot(x, makespan, "o-", lw=2, color=color, label=title)
        a2.plot(x, scored, "o-", lw=2, color=color, label=title)
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Candidate rows $k$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8.5)
    a1.set_ylabel("Makespan (s)")
    a1.set_title("Allocation result")
    a2.set_ylabel("Candidate rows scored")
    a2.set_yscale("log")
    a2.set_title("Auction work")
    fig.suptitle(
        f"Candidate-count sweep ($N={result['N']}$, $M={result['M']}$, "
        f"{result['trials']} trials)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_ksweep")


def fig_ablation() -> None:
    d = _load("ablation")
    Ns = d["Ns"]
    rows_off = [d["data"][str(n)]["rows_scored_off"] for n in Ns]
    rows_on = [d["data"][str(n)]["rows_scored_on"] for n in Ns]
    ms_off = [d["data"][str(n)]["greedy_ms_off"] for n in Ns]
    ms_on = [d["data"][str(n)]["greedy_ms_on"] for n in Ns]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    def _pair(ax, off, on, ylabel, title):
        ax.errorbar(
            Ns,
            [s["mean"] for s in off],
            yerr=[s["std"] for s in off],
            marker="s",
            color=ACCENT2,
            capsize=3,
            lw=2,
            label="exhaustive (no index)",
        )
        ax.errorbar(
            Ns,
            [s["mean"] for s in on],
            yerr=[s["std"] for s in on],
            marker="o",
            color=ACCENT,
            capsize=3,
            lw=2,
            label="KD-tree query",
        )
        ax.set_xlabel("Number of harvesters $N$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(Ns)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

    _pair(a1, rows_off, rows_on, "Candidate rows scored per allocation", "Auction work")
    _pair(a2, ms_off, ms_on, "Greedy auction wall-clock (ms)", "Auction time")

    fig.suptitle(
        f"Cost of spatial filtering: {d['M']} rows, full connectivity, "
        f"same scenarios and balancing pass ({d['trials']} trials)",
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "fig_ablation")


def fig_timing() -> None:
    d = _load("timing")
    Ms = d["Ms"]
    rows_off = [d["data"][str(m)]["rows_scored_off"]["mean"] for m in Ms]
    rows_on = [d["data"][str(m)]["rows_scored_on"]["mean"] for m in Ms]
    ms_off = [d["data"][str(m)]["greedy_ms_off"]["mean"] for m in Ms]
    ms_on = [d["data"][str(m)]["greedy_ms_on"]["mean"] for m in Ms]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    def _pair(ax, off, on, ylabel, title):
        ax.plot(Ms, off, marker="s", color=ACCENT2, lw=2, label="exhaustive (no index)")
        ax.plot(Ms, on, marker="o", color=ACCENT, lw=2, label="KD-tree query")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of rows $M$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, which="both")

    _pair(
        a1,
        rows_off,
        rows_on,
        "Candidate rows scored per allocation",
        "Auction work vs problem size",
    )
    _pair(
        a2,
        ms_off,
        ms_on,
        "Greedy auction wall-clock (ms)",
        "Auction time vs problem size",
    )

    fig.suptitle(
        f"Scaling of the auction with $M$ ($N={d['N']}$, {d['trials']} trials)", y=1.03
    )
    fig.tight_layout()
    _save(fig, "fig_timing")


def main() -> None:
    print("Generating figures:")
    fig_field()
    fig_architecture()
    fig_pipeline()
    fig_spatial()
    fig_shapes()
    fig_routes()
    fig_speed()
    fig_scalability()
    fig_consensus()
    fig_commrange()
    fig_loss()
    fig_benchmark()
    fig_events()
    fig_network()
    fig_turncost()
    fig_ksweep()
    fig_efficiency()
    fig_deadhead()
    fig_convergence()
    fig_ablation()
    fig_timing()
    print(f"Done -> {FIGDIR}")


if __name__ == "__main__":
    main()
