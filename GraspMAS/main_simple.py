"""Single-image 6-DoF language-driven grasp detection.

    python main_simple.py --query "grasp the mug by its handle" \
        --image-path scene/rgb.png --depth-path scene/depth.npy \
        --intrinsics scene/meta_data.json --gripper-name franka_panda

Depth is optional: without --depth-path the pipeline estimates metric depth
monocularly, at the cost of absolute-scale accuracy.

Requires the GraspGen-X server (scripts/run_server.sh).
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np

from agents.graspmas import GraspMAS
from perception3d import approach_elevation_deg, load_intrinsics
from run_artifacts import RunRecorder


def parse():
    p = argparse.ArgumentParser(
        description="Language-driven 6-DoF grasp detection (GraspMAS + GraspGen-X)"
    )
    p.add_argument("--query", type=str, required=True, help="Natural language grasp instruction")
    p.add_argument("--image-path", type=str, required=True, help="RGB image")

    g = p.add_argument_group("3D input (optional; monocular depth is used if omitted)")
    g.add_argument("--depth-path", type=str, default=None,
                   help="Metric depth map (.npy) or 16-bit depth image (.png)")
    g.add_argument("--depth-scale", type=float, default=1.0,
                   help="Multiplier to convert depth into METRES (e.g. 0.001 for millimetres)")
    g.add_argument("--intrinsics", type=str, default=None,
                   help="3x3 camera matrix as .npy or .json (accepts GraspGen-X meta_data.json)")
    g.add_argument("--fov-deg", type=float, default=60.0,
                   help="Assumed horizontal FOV when no intrinsics are given")

    r = p.add_argument_group("grasp generation")
    r.add_argument("--gripper-name", type=str, default="franka_panda",
                   help="Any gripper the server has assets for, e.g. robotiq_2f_85, unitree_g1")
    r.add_argument("--num-grasps", type=int, default=200, help="Diffusion samples per call")
    r.add_argument("--planner", type=str, default="graspmoe",
                   choices=["graspmoe", "diffusion"],
                   help="graspmoe = diffusion + OBB candidates (better); diffusion = faster")

    a = p.add_argument_group("agent loop")
    a.add_argument("--api-file", type=str, default="api.key", help="LLM API key file")
    a.add_argument("--llm-config", type=str, default=None, help="Path to llm_config.yaml")
    a.add_argument("--max-round", type=int, default=4, help="Planner/Coder/Observer rounds")

    o = p.add_argument_group("output")
    o.add_argument("--save-folder", type=str, default=None,
                   help="Where to write images (defaults to the run directory)")
    o.add_argument("--run-name", type=str, default=None, help="Label for the outputs/runs entry")
    o.add_argument("--no-artifacts", action="store_true", help="Skip writing outputs/")
    o.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_depth(path: str) -> np.ndarray:
    """Load depth from .npy or a 16-bit PNG/TIFF."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    import cv2

    depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not read depth from {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def main() -> int:
    args = parse()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    depth = load_depth(args.depth_path) if args.depth_path else None
    K = load_intrinsics(args.intrinsics) if args.intrinsics else None

    if depth is not None and args.depth_scale == 1.0 and np.nanmax(depth) > 100:
        # Almost certainly millimetres; metres are ~0.2-3.0 for tabletop scenes.
        print(f"WARNING: depth max is {np.nanmax(depth):.0f}, which looks like millimetres.\n"
              f"         Pass --depth-scale 0.001 if so, or the grasp scale will be wrong.",
              file=sys.stderr)

    name = args.run_name or args.query[:40]
    recorder = RunRecorder(name=name, enabled=not args.no_artifacts)
    recorder.write_config(vars(args))
    save_folder = args.save_folder or (
        str(recorder.images_dir) if recorder.enabled else "imgs/"
    )

    try:
        graspmas = GraspMAS(
            api_file=args.api_file,
            max_round=args.max_round,
            gripper_name=args.gripper_name,
            num_grasps=args.num_grasps,
            planner=args.planner,
            recorder=recorder,
            llm_config=args.llm_config,
        )
    except RuntimeError as exc:
        # A missing API key is the expected first failure for a new user; a
        # traceback buries the one line that tells them what to do.
        print(f"\n{exc}\n", file=sys.stderr)
        print("The grasp pipeline itself needs no key — try:\n"
              "  conda run -n graspmas python scripts/verify_pipeline.py",
              file=sys.stderr)
        return 2

    print(f"Query   : {args.query}")
    print(f"Image   : {args.image_path}")
    print(f"Depth   : {args.depth_path or 'monocular estimate'}")
    print(f"Gripper : {args.gripper_name}")
    print()

    _out, grasp = asyncio.run(
        graspmas.query(
            args.query,
            args.image_path,
            save_folder=save_folder,
            depth=depth,
            intrinsics=K,
            depth_scale=args.depth_scale,
        )
    )

    print("\n" + "=" * 60)
    if grasp is None:
        print("NO GRASP FOUND")
        if recorder.enabled:
            print(f"Artifacts: {recorder.dir}")
        return 1

    pose = np.asarray(grasp["pose"])
    print("6-DoF GRASP")
    print(f"  gripper   : {grasp['gripper']}")
    print(f"  score     : {grasp['score']:.3f}")
    print(f"  position  : {np.round(grasp['position'], 4).tolist()} m (camera frame)")
    print(f"  approach  : {np.round(grasp['approach'], 3).tolist()}"
          f"  ({approach_elevation_deg(grasp['approach']):.1f} deg off optical axis)")
    print(f"  closing   : {np.round(grasp['closing'], 3).tolist()}")
    print(f"  jaw width : {grasp['width'] * 100:.1f} cm")
    print("  pose (4x4):")
    for row in pose:
        print("    [" + "  ".join(f"{v:+.4f}" for v in row) + "]")
    print(f"  LLM calls : {recorder.llm_call_count}")
    if recorder.enabled:
        print(f"\nArtifacts : {recorder.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
