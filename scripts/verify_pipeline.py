#!/usr/bin/env python
"""End-to-end verification of the 2D -> 6-DoF upgrade, without the LLM.

The agent loop (Planner/Coder/Observer) needs an LLM key, but the part that
actually changed — language grounding -> metric point cloud -> GraspGen-X ->
6-DoF pose -> visualization — does not: `find()` and `find_part()` run on local
GroundingDINO / SAM / VLPart weights. So this script exercises the whole upgrade
offline and for free, and it is what backs the numbers in SUMMARY.md.

Covers:
  1. RGB-D grounding + 6-DoF grasping on the shipped sample scene
  2. Multi-embodiment: the same object across several grippers
  3. Part-level grounding (the language constraint reaching the sampler)
  4. RGB-only monocular depth fallback, and its scale error vs. real depth

Requires the GraspGen-X server (scripts/run_server.sh --daemon).

    conda run -n graspmas python scripts/verify_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "GraspMAS"))

SCENE = REPO / "GraspGenX" / "assets" / "sample_data" / "real_world" / "00"


def load_scene():
    import cv2

    rgb = cv2.imread(str(SCENE / "rgb.png"))[:, :, ::-1].copy()
    depth = np.load(SCENE / "depth.npy").astype(np.float32)
    K = np.asarray(json.loads((SCENE / "meta_data.json").read_text())["intrinsics"])
    return rgb, depth, K


def summarize(grasp: dict) -> dict:
    from perception3d import approach_elevation_deg

    return {
        "score": round(float(grasp["score"]), 4),
        "position_m": [round(float(v), 4) for v in grasp["position"]],
        "approach": [round(float(v), 3) for v in grasp["approach"]],
        "approach_deg_off_axis": round(approach_elevation_deg(grasp["approach"]), 1),
        "jaw_width_cm": round(float(grasp["width"]) * 100, 1),
        "gripper": grasp["gripper"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+",
                    default=["yellow cup", "mustard bottle", "cracker box"])
    ap.add_argument("--grippers", nargs="+",
                    default=["franka_panda", "robotiq_2f_85", "unitree_g1"])
    ap.add_argument("--part-object", default="mustard bottle")
    ap.add_argument("--part-name", default="cap")
    ap.add_argument("--num-grasps", type=int, default=100)
    ap.add_argument("--planner", default="graspmoe")
    ap.add_argument("--skip-mono", action="store_true",
                    help="Skip the RGB-only monocular-depth section")
    args = ap.parse_args()

    import image_patch as ip
    from perception3d import SceneContext, approach_elevation_deg
    from run_artifacts import RunRecorder
    from vis6d import visualize_grasp_6dof, visualize_grasp_candidates

    rgb, depth, K = load_scene()
    print(f"scene: rgb{rgb.shape} depth[{depth[depth>0].min():.2f}-{depth.max():.2f}]m")

    recorder = RunRecorder(name="verify_pipeline")
    recorder.write_config(vars(args))
    recorder.save_inputs(image=rgb, depth=depth, intrinsics=K,
                         meta={"scene": str(SCENE)})

    client = ip.get_graspgen_client()
    if not client.is_alive():
        print("ERROR: GraspGen-X server is not reachable. "
              "Start it with scripts/run_server.sh --daemon", file=sys.stderr)
        return 2
    print(f"server: {client.metadata['model']}\n")

    report: dict = {"rgbd": {}, "multi_embodiment": {}, "part_level": {}, "monocular": {}}

    ip.configure_scene(
        scene=SceneContext(depth=depth, intrinsics=K, image_shape=rgb.shape[:2]),
        recorder=recorder, gripper_name="franka_panda",
        num_grasps=args.num_grasps, planner=args.planner,
    )
    root = ip.ImagePatch(rgb)

    # -- 1. RGB-D grounding + 6-DoF grasping -------------------------------
    print("=" * 70)
    print("1. RGB-D: language grounding -> metric cloud -> 6-DoF grasp")
    print("=" * 70)
    for obj in args.objects:
        t0 = time.time()
        patches = root.find(obj)
        if not patches:
            print(f"  {obj:20s} NOT FOUND by the grounding stack")
            report["rgbd"][obj] = {"status": "not_grounded"}
            continue
        grasp = root.grasp_detection(patches[0], round_idx=1)
        dt = time.time() - t0
        if grasp is None:
            print(f"  {obj:20s} grounded, but no grasp produced ({dt:.1f}s)")
            report["rgbd"][obj] = {"status": "no_grasp", "elapsed_s": round(dt, 1)}
            continue

        s = summarize(grasp)
        print(f"  {obj:20s} score={s['score']:.3f} pos={s['position_m']} "
              f"approach={s['approach_deg_off_axis']:5.1f}deg  ({dt:.1f}s)")
        visualize_grasp_6dof(rgb, grasp, K, save_folder=recorder.images_dir,
                             filename=f"rgbd_{obj.replace(' ', '_')}.png",
                             mask=patches[0].mask)
        report["rgbd"][obj] = {"status": "ok", "elapsed_s": round(dt, 1), **s}

    # -- 2. Multi-embodiment ------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Multi-embodiment: one object, several robot hands")
    print("=" * 70)
    target = args.objects[0]
    patches = root.find(target)
    if patches:
        for gripper in args.grippers:
            ip._GRASP_CACHE.clear()
            t0 = time.time()
            grasp = root.grasp_detection(patches[0], gripper_name=gripper, round_idx=2)
            dt = time.time() - t0
            if grasp is None:
                print(f"  {gripper:16s} no grasp")
                report["multi_embodiment"][gripper] = {"status": "no_grasp"}
                continue
            s = summarize(grasp)
            print(f"  {gripper:16s} score={s['score']:.3f} jaw={s['jaw_width_cm']:5.1f}cm "
                  f"approach={s['approach_deg_off_axis']:5.1f}deg  ({dt:.1f}s)")
            visualize_grasp_6dof(rgb, grasp, K, save_folder=recorder.images_dir,
                                 filename=f"gripper_{gripper}.png", mask=patches[0].mask)
            report["multi_embodiment"][gripper] = {"status": "ok", **s}

    # -- 3. Part-level grounding -------------------------------------------
    print("\n" + "=" * 70)
    print(f"3. Part-level: '{args.part_object}' -> part '{args.part_name}'")
    print("=" * 70)
    ip._GRASP_CACHE.clear()
    obj_patches = root.find(args.part_object)
    if not obj_patches:
        print(f"  {args.part_object} not found; skipping part test")
        report["part_level"] = {"status": "object_not_grounded"}
    else:
        whole = root.grasp_detection(obj_patches[0], round_idx=3)
        try:
            part_patch = obj_patches[0].find_part(args.part_object, args.part_name)
        except Exception as exc:
            part_patch = None
            print(f"  find_part failed: {exc}")

        if part_patch is None or getattr(part_patch, "mask", None) is None:
            print("  part not segmented; skipping")
            report["part_level"] = {"status": "part_not_grounded"}
        elif getattr(part_patch, "part_found", None) is False:
            # VLPart has no such part; it handed back the whole object, so
            # any "part-level" result here would be object-level in disguise.
            print(f"  VLPart has no '{args.part_name}' for a {args.part_object}; "
                  f"it fell back to the whole object -- test is inconclusive")
            report["part_level"] = {"status": "part_vocabulary_miss"}
        else:
            ip._GRASP_CACHE.clear()
            part_grasp = root.grasp_detection(part_patch, round_idx=4)
            if part_grasp is None:
                n_px = int(np.asarray(part_patch.mask).astype(bool).sum())
                print(f"  part segmented ({n_px} px) but no grasp produced")
                report["part_level"] = {"status": "no_grasp", "part_mask_px": n_px}
            else:
                # The whole point: does constraining the mask move the grasp?
                pw = np.asarray(whole["position"]) if whole else None
                pp = np.asarray(part_grasp["position"])
                shift = float(np.linalg.norm(pp - pw)) if pw is not None else None
                inside = _projects_inside(part_grasp, part_patch.mask, K)
                print(f"  whole-object grasp : {summarize(whole)['position_m'] if whole else 'n/a'}")
                print(f"  part-level   grasp : {summarize(part_grasp)['position_m']}")
                print(f"  moved by           : {shift:.3f} m" if shift else "")
                print(f"  centre inside part mask: {inside}")
                visualize_grasp_6dof(rgb, part_grasp, K,
                                     save_folder=recorder.images_dir,
                                     filename="part_level.png", mask=part_patch.mask)
                report["part_level"] = {
                    "status": "ok",
                    "part_grasp": summarize(part_grasp),
                    "whole_grasp": summarize(whole) if whole else None,
                    "position_shift_m": round(shift, 4) if shift else None,
                    "centre_inside_part_mask": inside,
                }

    # -- 4. Monocular fallback ----------------------------------------------
    if not args.skip_mono:
        print("\n" + "=" * 70)
        print("4. RGB-only: monocular metric depth vs. real sensor depth")
        print("=" * 70)
        ip._GRASP_CACHE.clear()
        mono_scene = SceneContext(intrinsics=K, image_shape=rgb.shape[:2])
        ip.configure_scene(scene=mono_scene, recorder=recorder)
        mono_root = ip.ImagePatch(rgb)
        t0 = time.time()
        mono_patches = mono_root.find(target)
        mono_grasp = (mono_root.grasp_detection(mono_patches[0], round_idx=5)
                      if mono_patches else None)
        dt = time.time() - t0

        if mono_grasp is None:
            print("  no grasp from monocular depth")
            report["monocular"] = {"status": "no_grasp"}
        else:
            est = mono_scene.depth
            m = mono_patches[0].mask.astype(bool)
            real_med = float(np.median(depth[m & (depth > 0)]))
            est_med = float(np.median(est[m]))
            s = summarize(mono_grasp)
            print(f"  depth on target: sensor {real_med:.3f} m  vs  estimated {est_med:.3f} m"
                  f"   (error {abs(est_med-real_med)/real_med*100:.0f}%)")
            print(f"  grasp: score={s['score']:.3f} pos={s['position_m']} ({dt:.1f}s)")
            visualize_grasp_6dof(rgb, mono_grasp, K, save_folder=recorder.images_dir,
                                 filename="monocular.png", mask=m)
            from vis6d import depth_colormap
            depth_colormap(est, recorder.images_dir / "monocular_depth.png")
            report["monocular"] = {
                "status": "ok",
                "sensor_median_depth_m": round(real_med, 4),
                "estimated_median_depth_m": round(est_med, 4),
                "relative_error_pct": round(abs(est_med - real_med) / real_med * 100, 1),
                **s,
            }

    recorder.finish(result=report, status="success")
    out = recorder.dir / "verification_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport   : {out}")
    print(f"Artifacts: {recorder.dir}")
    return 0


def _projects_inside(grasp: dict, mask: np.ndarray, K: np.ndarray) -> bool:
    """Does the point where the hand actually closes land on the mask?

    Measured at the FINGERTIPS. The pose is anchored at the gripper base, ~10 cm
    behind the contact, so testing there answers a different question.
    """
    import image_patch as ip
    from perception3d import gripper_finger_points, project_points

    _w, fingertip_depth = ip.ImagePatch._gripper_geometry(grasp["gripper"])
    tips = gripper_finger_points(
        np.asarray(grasp["pose"]), grasp["width"], fingertip_depth
    )
    uv = project_points(tips.mean(axis=0, keepdims=True), K)[0]
    if not np.isfinite(uv).all():
        return False
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = mask.shape[:2]
    return bool(0 <= u < w and 0 <= v < h and mask[v, u])


if __name__ == "__main__":
    sys.exit(main())
