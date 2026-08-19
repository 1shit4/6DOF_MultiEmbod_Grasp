"""OCID-VLG evaluation for the 6-DoF pipeline.

OCID-VLG gives us three things for free that the single-image path has to guess:
real metric depth (`dataset.py` divides millimetres by 1000), a segmentation
mask of the *referred* instance, and rectangle ground truth. Upstream computed
all three and threw them away — `main_batch.py` only read `sentence` and `img`.

Ground truth is planar, so 6-DoF predictions are scored through
`Grasp6D.rect_2d`, the back-projection of the grasp's closing segment. That
keeps the number directly comparable to the 2D literature.

Two practical constraints shape the design:

* **The free LLM tier allows ~1500 requests/day** and one query costs 15-25.
  So results are appended as JSONL and already-evaluated `sent_id`s are skipped,
  letting an evaluation legitimately span several days.
* **OCID ships no camera intrinsics.** `--intrinsics` supplies them; without it
  we fall back to a synthesized K, which is recorded in the results so the
  distinction never gets lost.

    python main_batch.py --dataset-path /data/OCID-VLG --limit 20 \
        --intrinsics ocid_K.json --output outputs/eval/ocid.jsonl
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from agents.graspmas import GraspMAS
from OCID_VLG.dataset import OCIDVLGDataset
from perception3d import approach_elevation_deg, load_intrinsics
from run_artifacts import RunRecorder, append_eval_record, read_eval_records
from utils import eval_grasp

logger = logging.getLogger(__name__)


def parse():
    p = argparse.ArgumentParser(description="Evaluate 6-DoF GraspMAS on OCID-VLG")
    p.add_argument("--dataset-path", type=str, required=True, help="OCID-VLG root")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--api-file", type=str, default="api.key")
    p.add_argument("--llm-config", type=str, default=None)
    p.add_argument("--output", type=str, default="../outputs/eval/ocid_vlg.jsonl",
                   help="Append-only JSONL; re-running skips completed sent_ids")

    p.add_argument("--limit", type=int, default=20,
                   help="Samples to evaluate this run. The free LLM tier allows "
                        "roughly 60-100 queries/day, so keep this modest and resume later.")
    p.add_argument("--offset", type=int, default=0, help="Skip the first N samples")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Concurrent samples. Above 1 the shared rate limiter still "
                        "paces requests, but latency per sample gets noisy.")

    p.add_argument("--max-round", type=int, default=3)
    p.add_argument("--gripper-name", type=str, default="franka_panda")
    p.add_argument("--num-grasps", type=int, default=200)
    p.add_argument("--planner", type=str, default="graspmoe", choices=["graspmoe", "diffusion"])

    p.add_argument("--intrinsics", type=str, default=None,
                   help="OCID camera matrix (.json/.npy). Without it a synthesized K is used "
                        "and absolute scale is approximate.")
    p.add_argument("--iou-threshold", type=float, default=0.25)
    p.add_argument("--angle-threshold", type=float, default=30.0)
    p.add_argument("--no-artifacts", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def run_sample(graspmas, sample, idx, args, K, out_path) -> dict:
    query = sample["sentence"]
    image = np.asarray(sample["img"])
    depth = np.asarray(sample["depth"]) if "depth" in sample else None
    gt_mask = np.asarray(sample["mask"]) if "mask" in sample else None
    gt_rects = sample.get("grasp_rects")

    recorder = RunRecorder(
        name=f"ocid_{sample['sent_id']}", enabled=not args.no_artifacts
    )
    recorder.write_config(vars(args), extra={"sent_id": sample["sent_id"],
                                             "scene_id": sample.get("scene_id")})
    graspmas.recorder = recorder
    graspmas.llm.recorder = recorder

    t0 = time.time()
    status, grasp, error = "success", None, None
    try:
        _out, grasp = await graspmas.query(
            query, image,
            save_folder=str(recorder.images_dir) if recorder.enabled else "imgs/",
            depth=depth, intrinsics=K,
        )
        if grasp is None:
            status = "no_grasp"
    except Exception as exc:  # a single bad sample must not kill the sweep
        logger.exception("Sample %s failed", sample["sent_id"])
        status, error = "error", f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0

    record = {
        "index": idx,
        "sent_id": int(sample["sent_id"]),
        "scene_id": sample.get("scene_id"),
        "query": query,
        "target": sample.get("target"),
        "status": status,
        "error": error,
        "elapsed_s": round(elapsed, 2),
        "llm_calls": recorder.llm_call_count,
        "gripper": args.gripper_name,
        "planner": args.planner,
        "intrinsics_source": "supplied" if K is not None else "synthesized",
        "artifacts": str(recorder.dir) if recorder.enabled else None,
    }

    if grasp is not None:
        record["score"] = grasp["score"]
        record["position_m"] = [round(float(v), 4) for v in grasp["position"]]
        record["approach"] = [round(float(v), 4) for v in grasp["approach"]]
        record["approach_deg_off_axis"] = round(
            approach_elevation_deg(grasp["approach"]), 2
        )
        record["jaw_width_m"] = grasp["width"]
        record["rect_2d"] = grasp.get("rect_2d")

        # Score against the planar ground truth via the back-projected rectangle.
        if grasp.get("rect_2d") is not None and gt_rects is not None and len(gt_rects):
            success, iou, angle = eval_grasp(
                grasp["rect_2d"], gt_rects,
                iou_threshold=args.iou_threshold,
                angle_threshold=args.angle_threshold,
                dataset_type="OCID",
                return_details=True,
            )
            record["grasp_success"] = bool(success)
            record["best_iou"] = round(iou, 4)
            record["best_angle_err"] = None if not np.isfinite(angle) else round(angle, 2)
        else:
            record["grasp_success"] = None

        # Did the grasp land on the referred object at all? This separates
        # "grounded the wrong thing" from "grounded right, grasped awkwardly",
        # which the IoU number alone cannot distinguish.
        if gt_mask is not None and grasp.get("rect_2d") is not None:
            _, cx, cy = grasp["rect_2d"][0], grasp["rect_2d"][1], grasp["rect_2d"][2]
            h, w = gt_mask.shape[:2]
            u, v = int(round(cx)), int(round(cy))
            record["centre_on_target"] = bool(
                0 <= u < w and 0 <= v < h and gt_mask[v, u]
            )

    append_eval_record(out_path, record)
    return record


def summarize(records: list, args) -> dict:
    total = len(records)
    if total == 0:
        return {"total": 0}

    ok = [r for r in records if r["status"] == "success"]
    scored = [r for r in ok if r.get("grasp_success") is not None]
    hits = [r for r in scored if r["grasp_success"]]
    on_target = [r for r in ok if r.get("centre_on_target")]
    elev = [r["approach_deg_off_axis"] for r in ok if "approach_deg_off_axis" in r]

    summary = {
        "total": total,
        "produced_grasp": len(ok),
        "no_grasp": sum(1 for r in records if r["status"] == "no_grasp"),
        "errors": sum(1 for r in records if r["status"] == "error"),
        "scored_against_gt": len(scored),
        f"success_rate_iou{args.iou_threshold}_ang{args.angle_threshold}":
            round(len(hits) / len(scored), 4) if scored else None,
        "centre_on_target_rate":
            round(len(on_target) / len(ok), 4) if ok else None,
        "mean_confidence":
            round(float(np.mean([r["score"] for r in ok])), 4) if ok else None,
        "mean_best_iou":
            round(float(np.mean([r["best_iou"] for r in scored])), 4) if scored else None,
        "approach_deg_off_axis": {
            "mean": round(float(np.mean(elev)), 1) if elev else None,
            "min": round(float(np.min(elev)), 1) if elev else None,
            "max": round(float(np.max(elev)), 1) if elev else None,
            # The old 2D pipeline was structurally incapable of anything but a
            # fixed top-down approach; this is the headline 6-DoF statistic.
            "frac_beyond_20deg":
                round(float(np.mean([e > 20 for e in elev])), 3) if elev else None,
        },
        "mean_elapsed_s": round(float(np.mean([r["elapsed_s"] for r in records])), 1),
        "total_llm_calls": int(sum(r.get("llm_calls", 0) for r in records)),
    }
    return summary


async def evaluate(args) -> int:
    dataset = OCIDVLGDataset(args.dataset_path, split=args.split)
    print(f"OCID-VLG {args.split}: {len(dataset)} samples")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()

    done = read_eval_records(out_path)
    done_ids = {r["sent_id"] for r in done if r.get("status") != "error"}
    if done_ids:
        print(f"Resuming: {len(done_ids)} samples already recorded in {out_path}")

    K = load_intrinsics(args.intrinsics) if args.intrinsics else None
    if K is None:
        print("WARNING: no --intrinsics; using a synthesized camera matrix.\n"
              "         Grasp positions will be metrically approximate.", file=sys.stderr)

    graspmas = GraspMAS(
        api_file=args.api_file,
        max_round=args.max_round,
        gripper_name=args.gripper_name,
        num_grasps=args.num_grasps,
        planner=args.planner,
        llm_config=args.llm_config,
    )

    todo = []
    for idx in range(args.offset, len(dataset)):
        if len(todo) >= args.limit:
            break
        sample = dataset[idx]
        if int(sample["sent_id"]) in done_ids:
            continue
        todo.append((idx, sample))

    print(f"Evaluating {len(todo)} samples (limit={args.limit})\n")

    results = []
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        batch = await asyncio.gather(*[
            run_sample(graspmas, sample, idx, args, K, out_path)
            for idx, sample in chunk
        ])
        results.extend(batch)
        for r in batch:
            flag = ("HIT " if r.get("grasp_success") else
                    "miss" if r.get("grasp_success") is False else "----")
            print(f"[{r['index']:5d}] {flag} iou={r.get('best_iou', float('nan')):.3f} "
                  f"{r['status']:9s} {r['elapsed_s']:6.1f}s  {r['query'][:50]}")

    all_records = read_eval_records(out_path)
    summary = summarize(all_records, args)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (all records in this file, including prior runs)")
    print(json.dumps(summary, indent=2))

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "args": vars(args)}, f, indent=2)
    print(f"\nResults : {out_path}")
    print(f"Summary : {summary_path}")
    return 0


def main() -> int:
    args = parse()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    sys.exit(main())
