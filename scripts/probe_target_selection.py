#!/usr/bin/env python
"""Test the task planner's stage-1 target selection on real photographs.

Why this exists separately from the declutter runs: every synthetic scenario
hands the planner **ground-truth labels**. `MutationExecutor` returns a
segmentation mask with known identities, so `SceneRegistry` builds its prompt
table from the literal strings written when the scene was authored — `"knife"`,
`"apple"`. The planner is therefore doing pure semantic reasoning over text, and
the rendered blocks it also receives carry no information it needs. That is a
legitimate test of *"can it read an abstract goal"* and no test at all of
*"can it tell what these objects are"*.

Here the labels come from GroundingDINO run on a real photograph, and the photo
the planner sees is a photograph. Nothing else in the loop is involved: no
depth, no placement, no execution — just

    goal + scene table + image  ->  ranked candidates with priorities

so a wrong answer is the planner's, not the geometry's.

    python scripts/probe_target_selection.py --photos <dir> --report out.md

Costs one LLM request per (photo, goal) pair. Detection runs on CPU and is the
slow part, ~10-20 s per photo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "GraspMAS"))

# The open vocabulary offered to the detector. Deliberately wider than any one
# goal needs: a planner that only ever sees the right answer is not being tested.
VOCABULARY = [
    "knife", "fork", "spoon", "plate", "bowl", "cup", "mug", "bottle", "glass",
    "apple", "banana", "orange", "bread", "fruit",
    "scissors", "hammer", "screwdriver", "wrench", "pliers", "saw", "tape",
    "pen", "pencil", "book", "box", "can", "cloth", "brush",
]

#: Goals and the answer a person would accept, for scoring by hand afterwards.
GOALS = [
    ("i need something to cut", "a knife, scissors, or a saw"),
    ("i am hungry", "any food item"),
    ("i need something to drink from", "a cup, mug, glass, or bottle"),
    ("i need to drive in a nail", "a hammer"),
]


#: Longest edge the detector sees. Full-resolution photographs plus a
#: 28-word vocabulary OOM-killed this on a 15 GB box; boxes are all the scene
#: table needs and they survive downscaling perfectly well.
MAX_EDGE_PX = 720

#: At most this many instances of any one label, and this many objects overall.
#: Per-label first: the overall cap alone lets one over-detected class evict
#: every other object in the scene.
MAX_PER_LABEL = 3
MAX_DETECTIONS = 28

#: Measured on these photographs: real objects score 0.39-0.93 while pure
#: hallucinations sit at 0.15-0.30 — an "apple" and a "banana" on a photo of a
#: tool set, a "glass" among the wrenches. At 0.15 that garbage out-competed a
#: correctly-detected hammer (0.443) for a place in the table.


def detect(image_path: Path, vocabulary, threshold: float = 0.35):
    """Real open-vocabulary detection. Returns a list of detection dicts.

    Boxes only — **no segmentation**. `ImagePatch.find` runs SAM per label, and
    28 labels x SAM on a 1 MP photograph exhausts memory here for masks this
    never uses. GroundingDINO takes the whole vocabulary in one call anyway,
    which is both lighter and how it is meant to be used.
    """
    import cv2
    import image_patch as ip
    from PIL import Image

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"could not read {image_path}")
    scale = min(1.0, MAX_EDGE_PX / max(bgr.shape[:2]))
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    labels = [x if x.endswith(".") else x + "." for x in vocabulary]
    results = ip.object_detector(
        Image.fromarray(rgb), candidate_labels=labels, threshold=threshold
    )

    found = []
    for r in results:
        box = r["box"]
        bw, bh = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
        if bw < 8 or bh < 8:
            continue
        found.append({
            "label": r["label"].rstrip("."),
            "score": float(r["score"]),
            "cx": int((box["xmin"] + box["xmax"]) / 2),
            "cy": int((box["ymin"] + box["ymax"]) / 2),
            "w": int(bw), "h": int(bh),
            "area_frac": float(bw * bh) / (h * w),
        })
    # De-duplicate by IoU, best score first: several vocabulary words hit the
    # same object ("cup" and "mug"), and a table listing both invites the
    # planner to treat one object as two.
    #
    # IoU rather than centre distance, which was the first attempt and was
    # wrong: a large detection *containing* small ones sits at roughly their
    # centre, so the plate suppressed every piece of cutlery on it — the exact
    # objects the goals are about.
    def iou(a, b):
        ax0, ay0 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
        ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
        bx0, by0 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
        bx1, by1 = bx0 + b["w"], by0 + b["h"]
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union > 0 else 0.0

    # Cap PER LABEL before capping overall. A flat "best 12 by score" looks
    # reasonable and silently deletes whole object classes: on a photo of a tool
    # set, GroundingDINO returned 330 boxes — 35 scissors, 41 saws, and 36
    # hammers — and score-ranking alone kept five scissors and no hammer at all,
    # so the planner was asked to drive a nail with no hammer on the table. It
    # improvised a wrench, which then read as a planner defect and was mine.
    per_label: dict = {}
    found.sort(key=lambda d: -d["score"])
    for cand in found:
        bucket = per_label.setdefault(cand["label"], [])
        if len(bucket) >= MAX_PER_LABEL:
            continue
        if any(iou(cand, k) > 0.5 for k in bucket):
            continue
        bucket.append(cand)

    kept: list = []
    for cand in sorted(
        (c for bucket in per_label.values() for c in bucket),
        key=lambda d: -d["score"],
    ):
        if any(iou(cand, k) > 0.5 for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= MAX_DETECTIONS:
            break
    return kept, (h, w)


def scene_table(detections, shape) -> str:
    """The same shape of table the registry produces, in honest units.

    Columns match `SceneRegistry.as_prompt_table` in spirit but say *pixels*,
    because there is no depth here and pretending to centimetres would be a lie
    the planner cannot check.
    """
    if not detections:
        return "(no objects detected)"
    rows = ["| id | object | position (image x,y px) | size px | image fraction |",
            "|---|---|---|---|---|"]
    for i, d in enumerate(detections, start=1):
        rows.append(
            f"| obj_{i:03d} | the {d['label']} | {d['cx']}, {d['cy']} | "
            f"{d['w']}x{d['h']} | {d['area_frac']:.1%} |"
        )
    del shape
    return "\n".join(rows)


async def probe(photos, goals, out_path=None, json_path=None):
    import cv2
    from agents.llm import get_shared_llm
    from agents.observer import encode_image
    from agents.task_planner import TaskPlanner

    planner = TaskPlanner(get_shared_llm())
    results = []

    for photo in photos:
        print(f"\n=== {photo.name}", flush=True)
        detections, shape = detect(photo, VOCABULARY)
        table = scene_table(detections, shape)
        labels = [d["label"] for d in detections]
        print(f"  detected: {labels}", flush=True)

        b64 = encode_image(str(photo))
        for goal, expected in goals:
            # One provider timeout must not discard every result before it.
            # It did: a single 120 s timeout on the 7th of 20 calls took the
            # whole probe down and left nothing on disk, so the six answers
            # already paid for were lost along with the failure.
            try:
                choice = await planner.select_target(goal, table, b64)
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
                print(f"  {goal!r}\n      !! {type(exc).__name__}: {exc}", flush=True)
                results.append({
                    "photo": photo.name, "goal": goal, "expected": expected,
                    "detected": labels, "interpretation": "", "confidence": "",
                    "ranked": [], "error": str(exc),
                })
                _dump(results, out_path, json_path)
                continue
            ranked = [
                {
                    "id": c.object_id,
                    "label": next(
                        (d["label"] for i, d in enumerate(detections, 1)
                         if f"obj_{i:03d}" == c.object_id),
                        "?",
                    ),
                    "priority": c.priority,
                    "why": c.why,
                }
                for c in choice.candidates
            ]
            results.append({
                "photo": photo.name, "goal": goal, "expected": expected,
                "detected": labels, "interpretation": choice.interpretation,
                "confidence": choice.confidence, "ranked": ranked,
            })
            top = ", ".join(
                f"{r['label']}(p{r['priority']})" for r in ranked
            ) or "(nothing)"
            print(f"  {goal!r}\n      -> {top}\n      expected: {expected}", flush=True)
            # Written after every answer, so an interrupted probe still leaves
            # everything it managed to measure.
            _dump(results, out_path, json_path)

    _dump(results, out_path, json_path)
    if out_path:
        print(f"\nwrote {out_path}")
    return results


def _dump(results, out_path, json_path) -> None:
    if out_path:
        Path(out_path).write_text(render(results))
    if json_path:
        Path(json_path).write_text(json.dumps(results, indent=2))


def render(results) -> str:
    out = [
        "# Task planner — target selection on real photographs",
        "",
        "Stage 1 only: goal + scene table + photo → ranked candidates. Labels come",
        "from GroundingDINO run on the photo, not from ground truth, so this tests",
        "the planner's reasoning against real detections rather than against strings",
        "written when a synthetic scene was authored.",
        "",
        "**Verdict column is filled in by hand.** The point is to read the reasoning,",
        "not to score it automatically — an answer can name the right object for the",
        "wrong reason, and that is exactly what is worth catching.",
        "",
    ]
    by_photo = {}
    for r in results:
        by_photo.setdefault(r["photo"], []).append(r)
    for photo, rows in by_photo.items():
        out += [f"## {photo}", "", f"**Detected:** {', '.join(rows[0]['detected']) or '(none)'}", ""]
        out += ["| goal | chose | priority | reasoning | a person would accept | ok? |",
                "|---|---|---|---|---|---|"]
        for r in rows:
            if not r["ranked"]:
                why = (f"**call failed:** {r['error']}" if r.get("error")
                       else "_declined — nothing here serves the goal_")
                out.append(f"| `{r['goal']}` | — | — | {why} | {r['expected']} | |")
                continue
            for j, c in enumerate(r["ranked"]):
                goal_cell = f"`{r['goal']}`" if j == 0 else ""
                exp_cell = r["expected"] if j == 0 else ""
                out.append(
                    f"| {goal_cell} | {c['label']} (`{c['id']}`) | {c['priority']} "
                    f"| {c['why']} | {exp_cell} | |"
                )
        out.append("")
        for r in rows:
            if r["interpretation"]:
                out.append(f"- `{r['goal']}` read as: *{r['interpretation']}* "
                           f"(confidence {r['confidence']})")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True, help="directory of photographs")
    ap.add_argument("--report", default=None, help="write a markdown report here")
    ap.add_argument("--json", default=None, help="also dump raw results here")
    args = ap.parse_args(argv)

    photos = sorted(
        p for p in Path(args.photos).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not photos:
        raise SystemExit(f"no images in {args.photos}")

    asyncio.run(probe(
        photos, GOALS,
        Path(args.report) if args.report else None,
        Path(args.json) if args.json else None,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
