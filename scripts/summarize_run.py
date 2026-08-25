#!/usr/bin/env python
"""Digest one declutter run directory into a report-ready summary.

A run writes a lot: progress.json, grand_plan.json, an LLM trace, per-iteration
images. This pulls out the parts a write-up needs — what was decided, what
happened, what it cost — without opening five files by hand.

    conda run -n graspmas python scripts/summarize_run.py outputs/runs/<dir> [...]
    conda run -n graspmas python scripts/summarize_run.py --all --json digest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def llm_stats(run: Path) -> dict:
    """Per-agent call counts and token spend, plus any starved replies.

    `truncated` is the number worth watching: a thinking model draws its
    reasoning from the same budget as its answer, so a reply that stops for
    `length` rather than `stop` is usually a cut-off JSON object, not a verbose
    one. That is what aborted the first LLM run of this loop.
    """
    trace = run / "llm_trace.jsonl"
    if not trace.is_file():
        return {}
    calls, agents, truncated = [], {}, 0
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        calls.append(rec)
        agents[rec["agent"]] = agents.get(rec["agent"], 0) + 1
        if (rec.get("usage") or {}).get("finish_reason") == "length":
            truncated += 1
    usage = [c.get("usage") or {} for c in calls]

    def total(key):
        return sum(u.get(key) or 0 for u in usage)

    return {
        "calls": len(calls),
        "by_agent": agents,
        "errors": sum(1 for c in calls if c.get("error")),
        "retries": sum(c.get("retries") or 0 for c in calls),
        "truncated_replies": truncated,
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "latency_s": round(sum(c.get("latency_s") or 0 for c in calls), 1),
        "model": calls[0]["model"] if calls else None,
    }


def summarize(run: Path) -> dict:
    progress = load(run / "progress.json") or {}
    plan = load(run / "grand_plan.json") or {}
    config = load(run / "config.json") or {}

    iterations = []
    for rec in progress.get("iterations", []):
        ev = rec.get("evaluation") or {}
        ex = rec.get("execution") or {}
        iterations.append({
            "index": rec.get("index"),
            "action": rec.get("action"),
            "object": rec.get("object_id"),
            "label": rec.get("object_label"),
            "rationale": rec.get("rationale"),
            "execution": ex.get("status"),
            "grasped": ex.get("grasped_object"),
            "verdict": ev.get("action_succeeded"),
            "still_blocking": ev.get("still_blocking_target"),
            "displacement_cm": ev.get("displacement_cm"),
            "place_error_cm": ev.get("place_error_cm"),
            "collateral": ev.get("collateral"),
            "evidence": ev.get("evidence"),
            "notes": rec.get("notes") or [],
        })

    images = sorted(
        str(p.relative_to(run)) for p in (run / "images").rglob("*.png")
    ) if (run / "images").is_dir() else []

    return {
        "run": run.name,
        "inject": (config.get("args") or {}).get("inject") or "none",
        "scenario": (config.get("args") or {}).get("scenario"),
        "no_llm": (config.get("args") or {}).get("no_llm"),
        "goal": progress.get("goal"),
        "target": progress.get("target"),
        "status": progress.get("status"),
        "outcome": progress.get("outcome"),
        "grand_plan": {
            "removal_order": [
                s.get("object_id") for s in plan.get("removal_order", [])
            ],
            "success_criterion": plan.get("success_criterion"),
            "revisions": len(plan.get("revisions", [])),
        },
        "iterations": iterations,
        "llm": llm_stats(run),
        "images": images,
    }


def render(d: dict) -> str:
    out = [
        f"=== {d['run']} ===",
        f"  scenario {d['scenario']}   inject: {d['inject'] or 'none'}",
        f"  goal     {d['goal']!r} -> {d['status'].upper() if d['status'] else '?'}",
    ]
    if d["outcome"] and d["outcome"].get("reason"):
        out.append(f"  reason   {d['outcome']['reason']}")
    order = d["grand_plan"]["removal_order"]
    out.append(f"  plan     {' -> '.join(order) if order else '(empty)'}"
               f"   revisions: {d['grand_plan']['revisions']}")
    for it in d["iterations"]:
        blocking = it["still_blocking"]
        blk = "still blocking" if blocking else "cleared" if blocking is False else "?"
        head = f"  [{it['index']}] {it['action']} {it['object'] or '-'}"
        if it["label"]:
            head += f" ({it['label']})"
        out.append(f"{head}: exec={it['execution']} verdict={it['verdict']} {blk}")
        if it["displacement_cm"] is not None:
            out.append(f"        moved {it['displacement_cm']:.1f} cm"
                       + (f", {it['place_error_cm']:.1f} cm off plan"
                          if it["place_error_cm"] is not None else ""))
        if it["collateral"]:
            out.append(f"        collateral: {it['collateral']}")
        for n in it["notes"]:
            out.append(f"        note: {n}")
    llm = d["llm"]
    if llm:
        out.append(
            f"  llm      {llm['calls']} calls ({llm['by_agent']}), "
            f"{llm['truncated_replies']} truncated, {llm['errors']} errors, "
            f"{llm['retries']} retries"
        )
        out.append(
            f"           tokens: {llm['prompt_tokens']} in / "
            f"{llm['completion_tokens']} out / {llm['reasoning_tokens']} reasoning"
            f"   {llm['latency_s']}s"
        )
    out.append(f"  images   {len(d['images'])}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="*", type=Path)
    p.add_argument("--all", action="store_true", help="every run under outputs/runs")
    p.add_argument("--json", type=Path, help="also write the digests here")
    args = p.parse_args(argv)

    runs = list(args.runs)
    if args.all:
        runs += sorted(
            d for d in (REPO / "outputs" / "runs").iterdir()
            if (d / "progress.json").is_file()
        )
    if not runs:
        p.error("name a run directory, or pass --all")

    digests = []
    for run in runs:
        if not (run / "progress.json").is_file():
            print(f"skip {run} (no progress.json)")
            continue
        d = summarize(run)
        digests.append(d)
        print(render(d))
        print()

    if args.json:
        args.json.write_text(json.dumps(digests, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
