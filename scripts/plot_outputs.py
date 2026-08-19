#!/usr/bin/env python
"""Turn `outputs/` artifacts into figures.

Every plot answers a question the port raised:

  approach_spread   Did going 6-DoF actually buy approach diversity, or does the
                    model just reproduce the top-down grasp the planar pipeline
                    was locked to? This is the headline figure.
  score_distribution Was the model confident and focused or scattered, and how
                    much did the language (mask-containment) constraint discard?
  stage_timing      Where does wall-clock actually go on CPU?
  cpu_latency       What do the planner / num_grasps knobs cost?
  ocid_iou          How does the back-projected 6-DoF grasp score against the
                    planar ground truth?

    conda run -n graspmas python scripts/plot_outputs.py            # newest run
    conda run -n graspmas python scripts/plot_outputs.py --run <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

# Validated categorical slots (light mode). Only the first three are used
# together in any one figure — that subset clears the all-pairs CVD floors.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e5e4df"


def style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: the data should be the only assertive thing here."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12, fontweight="medium")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, path: Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    try:
        shown = path.relative_to(REPO)
    except ValueError:  # a --run outside the repo
        shown = path
    print(f"  wrote {shown}")
    return path


# ---------------------------------------------------------------------------


def load_run_grasps(run_dir: Path):
    """Concatenate every candidate set the run produced."""
    grasps, scores, kept_flags = [], [], []
    for npz_path in sorted((run_dir / "grasps").glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        g, s = data["grasps"], data["scores"]
        keep = np.zeros(len(g), bool)
        if "kept_indices" in data:
            keep[data["kept_indices"].astype(int)] = True
        grasps.append(g)
        scores.append(s)
        kept_flags.append(keep)
    if not grasps:
        return None, None, None
    return np.concatenate(grasps), np.concatenate(scores), np.concatenate(kept_flags)


def plot_approach_spread(run_dir: Path, out_dir: Path):
    """Approach elevation: everything the sampler proposed vs. what survived.

    The planar predecessor could only ever express one approach direction (a
    fixed top-down vector bolted on after the fact), so spread here is capability
    that did not exist before.

    Raw and filtered are plotted separately on purpose. GraspMoE's OBB branch
    sweeps candidate poses over every face of the box it fits, including the far
    side a single depth view never observed, so the raw distribution piles up
    near 180 degrees (a hand travelling back toward the camera). Showing only the
    raw set would badly misrepresent what the system actually does.
    """
    grasps, scores, kept = load_run_grasps(run_dir)
    if grasps is None or len(grasps) == 0:
        return None

    approach = grasps[:, :3, 2]
    norms = np.linalg.norm(approach, axis=1)
    ok = norms > 1e-9
    elev = np.full(len(grasps), np.nan)
    elev[ok] = np.degrees(np.arccos(np.clip(approach[ok, 2] / norms[ok], -1, 1)))

    sel = kept & np.isfinite(elev)
    bins = np.arange(0, 185, 7.5)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.hist(elev[np.isfinite(elev)], bins=bins, color=C_BLUE, alpha=0.28,
            edgecolor=SURFACE, linewidth=0.8, label="all candidates generated")
    ax.hist(elev[sel], bins=bins, color=C_BLUE, edgecolor=SURFACE,
            linewidth=0.8, label="kept: visible side, on the referred region")

    ax.axvline(0, color=C_ORANGE, linewidth=2, linestyle="--")
    top = ax.get_ylim()[1]
    ax.annotate("planar 2D pipeline:\nfixed top-down only",
                xy=(0, top * 0.95), xytext=(14, top * 0.95),
                color=C_ORANGE, fontsize=9, va="top",
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2))

    if sel.any():
        median = float(np.median(elev[sel]))
        ax.axvline(median, color=INK_2, linewidth=1.5)
        ax.annotate(f"kept median {median:.0f}°", xy=(median, top * 0.5),
                    xytext=(median + 8, top * 0.5), color=INK_2, fontsize=9)

    style_axes(ax, f"6-DoF approach directions  ({int(sel.sum())} kept "
                   f"of {int(np.isfinite(elev).sum())} generated)",
               "angle between approach axis and camera optical axis (degrees)",
               "grasp candidates")
    ax.set_xlim(0, 180)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper center")
    for text in leg.get_texts():
        text.set_color(INK_2)
    return save(fig, out_dir / "approach_spread.png")


def plot_score_distribution(run_dir: Path, out_dir: Path):
    """Score histogram split by whether the language constraint kept the grasp."""
    grasps, scores, kept = load_run_grasps(run_dir)
    if grasps is None or len(scores) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bins = np.linspace(0, 1, 26)
    ax.hist(scores[kept], bins=bins, color=C_BLUE, edgecolor=SURFACE,
            linewidth=0.8, label="on the referred region")
    ax.hist(scores[~kept], bins=bins, color=C_ORANGE, edgecolor=SURFACE,
            linewidth=0.8, alpha=0.75, label="discarded (outside mask)")

    n_keep, n_drop = int(kept.sum()), int((~kept).sum())
    style_axes(
        ax,
        f"Discriminator scores  ({n_keep} kept, {n_drop} discarded by the language constraint)",
        "grasp confidence", "grasp candidates",
    )
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK_2)
    return save(fig, out_dir / "score_distribution.png")


def plot_stage_timing(run_dir: Path, out_dir: Path):
    """Where the wall clock goes. Horizontal bars: labels are long, values ranked."""
    timings_path = run_dir / "timings.json"
    if not timings_path.exists():
        return None
    stages = json.loads(timings_path.read_text()).get("stages_s", {})
    if not stages:
        return None

    items = sorted(stages.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(items) + 1.8))
    bars = ax.barh(labels, values, color=C_BLUE, height=0.62)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}s", va="center", color=INK_2, fontsize=9)

    style_axes(ax, "Wall-clock by stage", "seconds")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, max(values) * 1.18)
    return save(fig, out_dir / "stage_timing.png")


def plot_cpu_latency(out_dir: Path):
    """Latency of the two planners across sample sizes. Grouped bars, one axis."""
    path = OUTPUTS / "bench" / "cpu_latency.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    records = blob["records"]
    if not records:
        return None

    planners = sorted({r["planner"] for r in records})
    sizes = sorted({r["num_grasps"] for r in records})
    colors = {planners[0]: C_BLUE}
    if len(planners) > 1:
        colors[planners[1]] = C_ORANGE

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.36
    x = np.arange(len(sizes))
    for i, planner in enumerate(planners):
        means = []
        for n in sizes:
            vals = [r["latency_s_mean"] for r in records
                    if r["planner"] == planner and r["num_grasps"] == n]
            means.append(float(np.mean(vals)) if vals else np.nan)
        offset = (i - (len(planners) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width * 0.92, label=planner,
                      color=colors.get(planner, C_AQUA))
        for bar, v in zip(bars, means):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.1f}",
                        ha="center", color=INK_2, fontsize=9)

    meta = blob.get("meta", {})
    style_axes(ax, f"CPU inference latency  ({meta.get('cpu_count', '?')} cores, "
                   f"{meta.get('num_diffusion_iters_eval', '?')} DDPM steps)",
               "diffusion samples requested", "seconds per call")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    leg = ax.legend(frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_2)
    return save(fig, out_dir / "cpu_latency.png")


def plot_ocid_iou(out_dir: Path, eval_path: Path | None = None):
    """Success rate as the IoU threshold sweeps — the standard grasp-metric curve."""
    if eval_path is None:
        candidates = sorted((OUTPUTS / "eval").glob("*.jsonl"))
        if not candidates:
            return None
        eval_path = candidates[-1]
    if not eval_path.exists():
        return None

    ious, angles = [], []
    for line in eval_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("best_iou") is not None:
            ious.append(rec["best_iou"])
            angles.append(rec.get("best_angle_err") or np.nan)
    if not ious:
        return None

    ious = np.asarray(ious)
    thresholds = np.linspace(0, 1, 51)
    rate = [(ious >= t).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(thresholds, rate, color=C_BLUE, linewidth=2)
    ax.axvline(0.25, color=INK_MUTED, linewidth=1, linestyle=":")
    at25 = float((ious >= 0.25).mean())
    ax.plot([0.25], [at25], "o", color=C_ORANGE, markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"IoU@0.25 = {at25:.1%}", xy=(0.25, at25), xytext=(0.32, at25 + 0.06),
                color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1))

    style_axes(ax, f"OCID-VLG: 6-DoF grasp back-projected to a rectangle  (n={len(ious)})",
               "IoU threshold", "success rate")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    return save(fig, out_dir / "ocid_iou.png")


# ---------------------------------------------------------------------------


def newest_run() -> Path | None:
    runs = sorted((OUTPUTS / "runs").glob("*"))
    runs = [r for r in runs if r.is_dir()]
    return runs[-1] if runs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="Run directory (default: the newest under outputs/runs)")
    ap.add_argument("--eval", type=str, default=None, help="An eval JSONL to curve")
    ap.add_argument("--all-runs", action="store_true", help="Plot every run")
    args = ap.parse_args()

    targets = []
    if args.all_runs:
        targets = [r for r in sorted((OUTPUTS / "runs").glob("*")) if r.is_dir()]
    elif args.run:
        targets = [Path(args.run)]
    else:
        newest = newest_run()
        if newest:
            targets = [newest]

    if not targets:
        print("No runs found under outputs/runs/", file=sys.stderr)
    for run_dir in targets:
        print(f"\n{run_dir.name}")
        out_dir = run_dir / "plots"
        made = [
            plot_approach_spread(run_dir, out_dir),
            plot_score_distribution(run_dir, out_dir),
            plot_stage_timing(run_dir, out_dir),
        ]
        if not any(m for m in made):
            print("  (no plottable artifacts)")

    print("\nbenchmarks + evaluation")
    bench_dir = OUTPUTS / "bench"
    if plot_cpu_latency(bench_dir) is None:
        print("  (no outputs/bench/cpu_latency.json)")
    if plot_ocid_iou(OUTPUTS / "eval",
                     Path(args.eval) if args.eval else None) is None:
        print("  (no OCID evaluation data yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
