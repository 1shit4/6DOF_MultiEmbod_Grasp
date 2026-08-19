#!/usr/bin/env python
"""Standalone CPU inference check + latency benchmark for GraspGen-X.

Runs entirely inside the `graspgenx` env — no ZMQ server, no GraspMAS. This is
the gate that proves the CPU patch works and establishes the latency budget the
rest of the system has to live within.

    conda run -n graspgenx python scripts/bench_cpu.py --quick

Writes outputs/bench/cpu_latency.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def peak_rss_gb() -> float:
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def load_sample_clouds(limit: int) -> list[tuple[str, np.ndarray]]:
    """Object point clouds from the shipped sample data (metres, camera frame)."""
    clouds: list[tuple[str, np.ndarray]] = []
    pc_dir = REPO_ROOT / "GraspGenX" / "assets" / "sample_data" / "object_pc"
    for path in sorted(pc_dir.glob("*.json"))[:limit]:
        with open(path) as f:
            data = json.load(f)
        pc = np.asarray(data["pc"] if "pc" in data else data, dtype=np.float32)
        pc = pc.reshape(-1, pc.shape[-1])[:, :3]
        clouds.append((path.stem, pc))
    return clouds


def synthetic_box(n: int = 3000, seed: int = 0) -> np.ndarray:
    """A 6x6x12 cm box surface at 60 cm in front of the camera.

    Deterministic, dependency-free fallback so the benchmark still runs if the
    sample data is missing, and a sanity target with a known answer: a parallel
    jaw gripper should close across one of the short axes.
    """
    rng = np.random.default_rng(seed)
    half = np.array([0.03, 0.03, 0.06], dtype=np.float32)
    pts = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32) * half
    # Push each point onto the nearest face so it is a surface, not a volume.
    axis = rng.integers(0, 3, size=n)
    sign = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    pts[np.arange(n), axis] = sign * half[axis]
    pts += np.array([0.0, 0.0, 0.60], dtype=np.float32)
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", default="franka_panda")
    ap.add_argument("--num-grasps", type=int, nargs="+", default=[50, 200])
    ap.add_argument("--planner", nargs="+", default=["diffusion", "graspmoe"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-clouds", type=int, default=2)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="One cloud, one config, one repeat — just proves CPU inference works.",
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "bench" / "cpu_latency.json"))
    args = ap.parse_args()

    if args.quick:
        args.num_grasps, args.planner, args.repeats, args.max_clouds = [50], ["diffusion"], 1, 1

    torch.set_num_threads(os.cpu_count() or 8)
    os.environ.setdefault("GRASPGENX_DEVICE", "cpu")

    from graspgenx.grasp_server import GraspGenXSampler, load_grasp_gen_model
    from graspgenx.samplers.planner import run_planner_on_object
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx import get_checkpoints_version_dir

    ckpt_root = Path(os.environ.get("GRASPGENX_CHECKPOINT_DIR", "")) / "release"
    if not ckpt_root.exists():
        ckpt_root = Path(get_checkpoints_version_dir())

    t0 = time.time()
    cfg = load_model_cfg(f"{ckpt_root}/gen", f"{ckpt_root}/dis", None, None)
    model = load_grasp_gen_model(cfg)
    load_s = time.time() - t0

    device = next(model.parameters()).device
    flash = {
        "diffusion": bool(cfg.diffusion.ptv3vanilla.enable_flash),
        "discriminator": bool(cfg.discriminator.ptv3vanilla.enable_flash),
    }
    print(f"[bench] device={device}  load={load_s:.1f}s  enable_flash={flash}")
    print(f"[bench] backbones: gen={cfg.diffusion.object_backbone} "
          f"dis={cfg.discriminator.object_backbone}  num_points={cfg.data.num_points}")

    # Hard gates: if either of these is wrong, every downstream number is junk.
    assert device.type == "cpu", f"expected CPU, got {device}"
    assert not any(flash.values()), "enable_flash must be False on CPU"

    sampler = GraspGenXSampler(
        cfg,
        gripper_name=args.gripper,
        assets_dir=os.environ.get(
            "GRASPGENX_ASSETS_DIR", str(REPO_ROOT / "GraspGenX" / "assets")
        ),
        model=model,
    )

    clouds = load_sample_clouds(args.max_clouds)
    clouds.append(("synthetic_box", synthetic_box()))

    records = []
    for name, pc in clouds:
        centered = pc - pc.mean(axis=0)
        for planner in args.planner:
            for n in args.num_grasps:
                times, counts, best = [], [], []
                for _ in range(args.repeats):
                    t = time.time()
                    grasps, conf, _tags, _obb = run_planner_on_object(
                        centered,
                        sampler,
                        planner=planner,
                        grasp_threshold=-1.0,
                        num_grasps=n,
                        topk_num_grasps=-1,
                    )
                    times.append(time.time() - t)
                    g = np.asarray(grasps.cpu() if hasattr(grasps, "cpu") else grasps)
                    c = np.asarray(conf.cpu() if hasattr(conf, "cpu") else conf)
                    counts.append(int(len(g)))
                    best.append(float(c.max()) if len(c) else float("nan"))

                    if len(g):
                        assert g.shape[1:] == (4, 4), f"bad grasp shape {g.shape}"
                        assert np.isfinite(g).all(), "non-finite grasp pose"
                        assert ((c >= 0) & (c <= 1)).all(), "score outside [0,1]"

                rec = {
                    "cloud": name,
                    "n_points": int(len(pc)),
                    "planner": planner,
                    "num_grasps": n,
                    "latency_s_mean": float(np.mean(times)),
                    "latency_s_min": float(np.min(times)),
                    "latency_s_max": float(np.max(times)),
                    "n_returned_mean": float(np.mean(counts)),
                    "best_score_mean": float(np.nanmean(best)),
                }
                records.append(rec)
                print(f"[bench] {name:16s} {planner:9s} n={n:4d}  "
                      f"{rec['latency_s_mean']:6.1f}s  "
                      f"{rec['n_returned_mean']:5.0f} grasps  "
                      f"best={rec['best_score_mean']:.3f}")

    out = {
        "meta": {
            "device": str(device),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "gripper": args.gripper,
            "model_load_s": load_s,
            "enable_flash": flash,
            "backbone_gen": str(cfg.diffusion.object_backbone),
            "backbone_dis": str(cfg.discriminator.object_backbone),
            "num_points": int(cfg.data.num_points),
            "num_diffusion_iters_eval": int(cfg.diffusion.num_diffusion_iters_eval),
            "peak_rss_gb": peak_rss_gb(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[bench] peak RSS {out['meta']['peak_rss_gb']:.2f} GB -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
