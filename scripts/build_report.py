#!/usr/bin/env python
"""Turn a finished declutter run into a readable per-iteration narrative.

`summarize_run.py` answers "what happened"; this answers "how did it decide
that". For each outer iteration it lays out, in the order they occurred:

    task planner  ->  what it decided and why, plus any grand-plan amendment
    planner       ->  its <thought> and the plan it handed the coder
    coder         ->  the generated code, the grasp, the overlay image
    observer      ->  its verdict and critique of that grasp
    executor      ->  the scene photographed afterwards
    evaluator     ->  the geometric verdict, and the state file as it then stood

Writes Markdown with relative image links, so the report renders in place
inside the run directory.

    conda run -n graspmas python scripts/build_report.py outputs/runs/<dir>
    conda run -n graspmas python scripts/build_report.py --all --index
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "outputs" / "reports"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fence(text, lang: str = "") -> str:
    """Fenced block, with a longer fence when the body contains one."""
    body = (text if isinstance(text, str) else json.dumps(text, indent=2)).strip()
    if not body:
        return "_(nothing recorded)_"
    bars = "```"
    while bars in body:
        bars += "`"
    return f"{bars}{lang}\n{body}\n{bars}"


def img(path: Path | str | None, root: Path, alt: str) -> str:
    """Markdown image link, relative to the run directory.

    Callers pass a mix: paths already rooted at the run (built here) and
    absolute paths recorded by the pipeline. Trying `root / p` on the first
    kind doubles the prefix, so resolve by what actually exists.
    """
    if not path:
        return f"_(no {alt} image)_"
    for candidate in (Path(path), root / path):
        if candidate.is_file():
            return f"![{alt}]({os.path.relpath(candidate, root)})"
    return f"_(no {alt} image)_"


def grasp_line(g: dict | None) -> str:
    if not g:
        return "_(no grasp)_"
    pos = g.get("position") or []
    parts = [f"score **{g.get('score', 0):.3f}**", f"gripper `{g.get('gripper', '?')}`"]
    if len(pos) == 3:
        parts.append("position " + ", ".join(f"{v:+.3f}" for v in pos) + " m")
    if g.get("width") is not None:
        parts.append(f"jaw {g['width'] * 100:.1f} cm")
    if g.get("source"):
        parts.append(f"source `{g['source']}`")
    return " · ".join(parts)


def place_line(p: dict | None) -> str:
    if not p:
        return "_(no place pose — this iteration grasped the target rather than moving something)_"
    return (
        f"travel **{p.get('travel_m', 0) * 100:.1f} cm** · "
        f"clearance {p.get('clearance_m', 0) * 100:.1f} cm"
    )


def render_iteration(run: Path, rec: dict, trace_by_iter: dict) -> list[str]:
    idx = rec.get("index")
    out = [f"\n---\n\n## Iteration {idx}\n"]

    # -- task planner ------------------------------------------------------
    decision = rec.get("decision") or {}
    out.append("### 1 · Task planner — what to do next\n")
    action = rec.get("action") or decision.get("action") or "?"
    who = rec.get("object_id") or "—"
    label = f" ({rec['object_label']})" if rec.get("object_label") else ""
    out.append(f"**Decision:** `{action}` → **{who}**{label}\n")
    if rec.get("subgoal"):
        out.append(f"**Subgoal handed to the grasping agents:** {rec['subgoal']}\n")
    if rec.get("rationale"):
        out.append(f"**Rationale:** {rec['rationale']}\n")

    blockers = rec.get("blockers") or []
    if blockers:
        out.append("\n**Blocking analysis it was given:**\n")
        out.append("| object | reasons | occlusion | sweep points | gap |")
        out.append("|---|---|---|---|---|")
        for b in blockers:
            occ = b.get("occlusion_frac")
            gap = b.get("gap_m")
            out.append(
                f"| `{b.get('object_id')}` ({b.get('label')}) "
                f"| {', '.join(b.get('reasons', []))} "
                f"| {f'{occ:.0%}' if occ is not None else '—'} "
                f"| {b.get('sweep_points', '—')} "
                f"| {f'{gap*100:.1f} cm' if gap is not None else '—'} |"
            )
        out.append("")
    else:
        out.append("\n_Blocking analysis: nothing was listed as blocking the target._\n")

    if decision.get("grand_plan_update"):
        out.append("\n**Grand-plan amendment proposed:**\n")
        out.append(fence(decision["grand_plan_update"], "json"))
    if decision.get("corrections"):
        out.append("\n**Corrections Python applied to that decision:**\n")
        for c in decision["corrections"]:
            out.append(f"- {c}")
        out.append("")

    out.append(f"\n**Scene as the planner saw it:**\n")
    out.append(img(run / "images" / f"iter{idx}" / f"scene_iter{idx}_before.png",
                   run, f"scene before iteration {idx}"))

    # -- inner loop, round by round ---------------------------------------
    rounds = rec.get("agent_rounds") or []
    if not rounds:
        out.append("\n### 2 · Grasping agents\n")
        out.append("_The inner loop was not run for this iteration._\n")
    for r in rounds:
        n = r.get("round")
        out.append(f"\n### 2.{n} · Grasping agents — inner round {n}\n")
        out.append(f"**Query given to them:** `{r.get('query', '')}`\n")

        out.append("\n**Planner thought:**\n")
        out.append(fence(r.get("thought")))
        out.append("\n**Plan handed to the coder:**\n")
        out.append(fence(r.get("plan")))

        if r.get("concluded"):
            out.append("\n_The planner judged the grasp sound and returned to the caller; "
                       "no code was generated this round._\n")
            continue

        out.append("\n**Coder — generated code:**\n")
        out.append(fence(r.get("code"), "python"))

        out.append("\n**Coder — result:**\n")
        out.append(f"- Grasp: {grasp_line(r.get('grasp'))}")
        if r.get("grasp_summary"):
            gs = r["grasp_summary"]
            out.append(
                f"- Approach {gs.get('approach_deg_off_camera_axis', '?')}° off the "
                f"optical axis · depth from `{gs.get('depth_source', '?')}`"
            )
        if r.get("place"):
            out.append(f"- Place: {place_line(r.get('place'))}")
        if r.get("error_logs"):
            out.append(f"- **Errors:** {r['error_logs']}")
        out.append("")
        out.append(img(r.get("overlay_image"), run, f"grasp overlay, round {n}"))

        obs = r.get("observer") or {}
        out.append(f"\n**Observer — verdict `{obs.get('verdict', '?')}`**\n")
        if obs.get("checklist"):
            checks = " · ".join(f"{k}: **{v}**" for k, v in obs["checklist"].items())
            out.append(f"{checks}\n")
        if obs.get("summary"):
            out.append(f"> {obs['summary']}\n")
        if obs.get("error_logs") and obs["error_logs"] != "none":
            out.append(f"**Observer noted errors:** {obs['error_logs']}\n")

    # -- what was actually executed ---------------------------------------
    planned = rec.get("planned") or {}
    out.append("\n### 3 · Executor\n")
    out.append(f"**Grasp sent to the executor:** {grasp_line(planned.get('grasp'))}\n")
    out.append(f"**Place sent to the executor:** {place_line(planned.get('place'))}\n")

    ex = rec.get("execution") or {}
    if ex:
        out.append(
            f"**Reported:** status `{ex.get('status', '?')}` · "
            f"reached stage `{ex.get('stage_reached', '?')}`"
        )
        if ex.get("grasped_object"):
            out.append(f" · actually grasped **{ex['grasped_object']}**")
        out.append("")
        if ex.get("error"):
            out.append(f"**Executor error:** {ex['error']}\n")
        if ex.get("disturbed"):
            out.append(f"**Disturbed by the move:** {ex['disturbed']}\n")
        if ex.get("notes"):
            out.append(f"_{ex['notes']}_\n")
    else:
        out.append("_Nothing was executed this iteration._\n")

    out.append("\n**Scene after the executor:**\n")
    out.append(img(run / "images" / f"iter{idx}" / f"scene_iter{idx}_after.png",
                   run, f"scene after iteration {idx}"))

    # -- evaluator ---------------------------------------------------------
    ev = rec.get("evaluation") or {}
    out.append("\n### 4 · Evaluator (geometric — no LLM)\n")
    if ev:
        blocking = ev.get("still_blocking_target")
        out.append(
            f"| verdict | still blocking target | displacement | error vs plan | source |\n"
            f"|---|---|---|---|---|\n"
            f"| **{ev.get('action_succeeded', '?')}** "
            f"| **{blocking}** "
            f"| {ev.get('displacement_cm', '—')} cm "
            f"| {ev.get('place_error_cm', '—')} cm "
            f"| `{ev.get('source', '?')}` |\n"
        )
        if ev.get("evidence"):
            out.append(f"> {ev['evidence']}\n")
        if ev.get("collateral"):
            out.append(f"**Collateral movement:** `{ev['collateral']}`\n")
    else:
        out.append("_No move to evaluate — this iteration grasped the target._\n")

    for n in rec.get("notes") or []:
        out.append(f"- _note:_ {n}")
    if rec.get("notes"):
        out.append("")

    # -- state as it then stood -------------------------------------------
    snap = run / "states" / f"iter{idx}"
    if (snap / "progress.json").is_file():
        out.append(
            f"\n**State after this iteration:** "
            f"[`progress.json`]({os.path.relpath(snap / 'progress.json', run)})"
        )
        if (snap / "grand_plan.json").is_file():
            out.append(
                f" · [`grand_plan.json`]({os.path.relpath(snap / 'grand_plan.json', run)})"
            )
        out.append("")

    calls = trace_by_iter.get(f"iter{idx}", [])
    if calls:
        agents = {}
        for c in calls:
            agents[c["agent"]] = agents.get(c["agent"], 0) + 1
        spend = sum((c.get("usage") or {}).get("total_tokens") or 0 for c in calls)
        out.append(
            f"\n_LLM cost this iteration: {len(calls)} calls "
            f"({', '.join(f'{k} ×{v}' for k, v in sorted(agents.items()))}), "
            f"{spend:,} tokens._\n"
        )
    return out


def build(run: Path) -> str:
    progress = load(run / "progress.json") or {}
    plan = load(run / "grand_plan.json") or {}
    config = load(run / "config.json") or {}
    args = config.get("args") or {}
    trace = jsonl(run / "llm_trace.jsonl")

    by_iter: dict = {}
    for c in trace:
        by_iter.setdefault(c.get("iteration") or "setup", []).append(c)

    inject = args.get("inject") or "none"
    target = progress.get("target") or {}

    out = [
        f"# {run.name}",
        "",
        f"**Scenario:** `{args.get('scenario', '?')}`   ",
        f"**Injected failure:** `{inject}`   ",
        f"**Planner:** {'scripted (no LLM)' if args.get('no_llm') else 'LLM'}   ",
        "",
        "## Inputs",
        "",
        f"- **User prompt / goal:** `{progress.get('goal', args.get('goal', '?'))}`",
        f"- **Target object:** `{target.get('id', '?')}` ({target.get('label', '?')})",
        f"- **Gripper:** `{args.get('gripper_name', '?')}`",
        f"- **Inner rounds per grasp (`--max-round`):** {args.get('max_round', '?')}",
        f"- **Iteration cap:** {args.get('max_iterations', '?')}",
        f"- **Model:** `{trace[0]['model'] if trace else 'n/a'}`",
        f"- **Seed:** {args.get('seed', '?')}",
        "",
        "**Initial scene:**",
        "",
        img(run / "images" / "iter0" / "scene_iter0_before.png", run, "initial scene"),
        "",
        img(run / "images" / "depth_colormap.png", run, "initial depth"),
        "",
    ]

    # Stage 1: how the goal was read, and what it could have meant. Omitted
    # when --target was given, because then nothing was inferred. This was
    # missing from every earlier report, which showed only the target that came
    # out and none of the reasoning that produced it — so a run could not be
    # audited for *why* it went after what it did.
    choice = progress.get("target_choice") or {}
    if choice:
        out += ["## Reading the goal — what the task planner decided it meant", ""]
        if choice.get("interpretation"):
            out += [f"> {choice['interpretation']}", ""]
        conf = choice.get("confidence")
        if conf:
            out += [f"**Confidence:** {conf}", ""]
        cands = choice.get("candidates") or []
        if cands:
            out.append("| priority | object | why it serves the goal | blockers |")
            out.append("|---|---|---|---|")
            for c in cands:
                n = c.get("n_blockers")
                cost = "not measured" if n is None else ("none" if n == 0 else str(n))
                out.append(
                    f"| {c.get('priority', '?')} | `{c.get('object_id', '?')}` "
                    f"({c.get('label', '?')}) | {c.get('why', '')} | {cost} |"
                )
            out += [
                "",
                "_Priority is suitability alone — stage 1 is not shown what is in "
                "the way. Blocker counts are measured afterwards and weighed "
                "against priority to produce the final order._",
                "",
            ]
        for correction in choice.get("corrections") or []:
            out += [f"- **Correction:** {correction}"]
        out.append("")

    notes = progress.get("run_notes") or []
    if notes:
        out += ["**Run-level notes:**", ""]
        out += [f"- {n}" for n in notes]
        out.append("")

    retargets = progress.get("retargets") or []
    if retargets:
        out += ["## Target changes", ""]
        for r in retargets:
            out += [
                f"- **Iteration {r.get('iteration', '?')}:** "
                f"`{(r.get('from') or {}).get('id', '?')}` "
                f"({(r.get('from') or {}).get('label', '?')}) → "
                f"`{(r.get('to') or {}).get('id', '?')}` "
                f"({(r.get('to') or {}).get('label', '?')})",
                f"  - {r.get('reason', '')}",
            ]
        out.append("")

    if plan:
        out += ["## Grand plan, as first drafted", ""]
        order = plan.get("removal_order") or []
        if order:
            out.append("| # | object | why it is in the way |")
            out.append("|---|---|---|")
            for i, step in enumerate(order):
                # Runs recorded before `_normalise_removal_order` existed can
                # hold bare ids here. A report builder that dies on the archive
                # it exists to read is worse than one that renders it plainly.
                if not isinstance(step, dict):
                    step = {"object_id": str(step)}
                out.append(
                    f"| {i + 1} | `{step.get('object_id')}` ({step.get('label', '?')}) "
                    f"| {step.get('reason', '')} |"
                )
        else:
            out.append("_Empty removal order._")
        out += [
            "",
            f"**Success criterion:** {plan.get('success_criterion', '—')}",
            "",
            f"**Reasoning:** {plan.get('reasoning', '—')}",
            "",
        ]
        if plan.get("revisions"):
            out += ["**Revisions made during the run:**", ""]
            for r in plan["revisions"]:
                out.append(
                    f"- _iteration {r.get('iteration')}_ — changed "
                    f"`{', '.join(r.get('changed', []))}`: {r.get('reason', '')}"
                )
            out.append("")

    for rec in progress.get("iterations", []):
        out += render_iteration(run, rec, by_iter)

    # -- final report ------------------------------------------------------
    outcome = progress.get("outcome") or {}
    status = (progress.get("status") or "?").upper()
    out += ["\n---\n", "## Final result", "", f"### {status}", ""]
    if outcome.get("reason"):
        out.append(f"**Reason:** {outcome['reason']}")
        out.append("")
    moved = outcome.get("moved") or []
    out.append(f"- **Iterations run:** {len(progress.get('iterations', []))}")
    out.append(f"- **Objects moved:** {', '.join(f'`{m}`' for m in moved) or 'none'}")
    if outcome.get("grasp"):
        out.append(f"- **Final grasp on the target:** {grasp_line(outcome['grasp'])}")
    out.append("")

    if progress.get("iterations"):
        out += ["**Iteration summary:**", "",
                "| # | action | object | execution | verdict | still blocking |",
                "|---|---|---|---|---|---|"]
        for r in progress["iterations"]:
            ev, ex = r.get("evaluation") or {}, r.get("execution") or {}
            out.append(
                f"| {r.get('index')} | `{r.get('action')}` | `{r.get('object_id') or '—'}` "
                f"| {ex.get('status', '—')} | {ev.get('action_succeeded', '—')} "
                f"| {ev.get('still_blocking_target', '—')} |"
            )
        out.append("")

    if trace:
        usage = [c.get("usage") or {} for c in trace]
        tot = lambda k: sum(u.get(k) or 0 for u in usage)
        truncated = sum(1 for u in usage if u.get("finish_reason") == "length")
        agents: dict = {}
        for c in trace:
            agents[c["agent"]] = agents.get(c["agent"], 0) + 1
        out += [
            "**LLM cost:**", "",
            f"- {len(trace)} calls — {', '.join(f'{k} ×{v}' for k, v in sorted(agents.items()))}",
            f"- {tot('prompt_tokens'):,} prompt / {tot('completion_tokens'):,} completion "
            f"/ {tot('reasoning_tokens'):,} reasoning tokens",
            f"- {sum(c.get('latency_s') or 0 for c in trace):.1f} s of model latency, "
            f"{sum(c.get('retries') or 0 for c in trace)} retries, "
            f"{sum(1 for c in trace if c.get('error'))} errors",
            f"- **{truncated} truncated replies**"
            + ("" if truncated == 0 else "  ⚠️ raise `min_max_tokens`"),
            "",
            "Full prompts and replies, one JSON object per call, are in "
            "[`llm_trace.jsonl`](llm_trace.jsonl).",
            "",
        ]

    out += [
        "## Everything this run kept", "",
        "| path | what it holds |",
        "|---|---|",
        "| `progress.json` | the whole run: decisions, agent rounds, execution, evaluation |",
        "| `grand_plan.json` | the plan and every amendment to it |",
        "| `states/iter<N>/` | both files frozen as they stood after iteration N |",
        "| `images/iter<N>/` | scene before, grasp overlay, scene after |",
        "| `llm_trace.jsonl` | every LLM call: prompt, reply, tokens, latency, iteration |",
        "| `masks/`, `clouds/`, `grasps/` | the perception behind each grasp; `grasps/` keeps ALL candidates, not just the winner |",
        "| `inputs/` | the exact rgb, depth and intrinsics the run started from |",
        "| `config.json`, `timings.json`, `log.txt` | how it was invoked, where the time went |",
        "",
    ]
    return "\n".join(out)


def build_index(runs: list[Path]) -> str:
    rows = []
    for run in runs:
        progress = load(run / "progress.json") or {}
        args = (load(run / "config.json") or {}).get("args") or {}
        trace = jsonl(run / "llm_trace.jsonl")
        outcome = progress.get("outcome") or {}
        rows.append({
            "run": run.name,
            "planner": "scripted" if args.get("no_llm") else "LLM",
            "inject": args.get("inject") or "none",
            "scenario": args.get("scenario", "?"),
            "status": progress.get("status", "?"),
            "iters": len(progress.get("iterations", [])),
            "moved": outcome.get("moved") or [],
            "calls": len(trace),
            "reason": outcome.get("reason", ""),
        })

    out = [
        "# Decluttering runs — index",
        "",
        "Which run is which scenario, and where its narrative lives. Each report",
        "walks the run iteration by iteration: what every agent thought, what it",
        "produced, what the executor did, and what the evaluator measured.",
        "",
        "The **evaluator is geometric in every run** — it measures the scene and",
        "spends no LLM calls. The `scripted` row is the control: identical",
        "machinery with the planning decisions made by a rule instead of a model,",
        "which is what isolates a planning failure from a geometry failure.",
        "",
        "| scenario (injected failure) | planner | outcome | iters | objects moved | LLM calls | report |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["planner"] == "LLM", r["inject"] != "none")):
        moved = ", ".join(f"`{m}`" for m in r["moved"]) or "none"
        mark = {"success": "✅", "failed": "❌", "aborted": "⚠️"}.get(r["status"], "•")
        out.append(
            f"| **{r['inject']}** | {r['planner']} | {mark} {r['status']} | {r['iters']} "
            f"| {moved} | {r['calls']} | [{r['run']}](../runs/{r['run']}/report.md) |"
        )
    out += ["", "### Why each failing run failed", ""]
    for r in rows:
        if r["status"] != "success" and r["reason"]:
            out.append(f"- **{r['inject']}** ({r['planner']}) — {r['reason']}")
    if all(r["status"] == "success" for r in rows):
        out.append("_Every run reached its target._")
    out += [
        "",
        "### What the injected failures do",
        "",
        "| mode | what the executor does |",
        "|---|---|",
        "| `none` | executes the plan faithfully |",
        "| `drop` | the object slips from the gripper and stays where it was |",
        "| `offset` | the release lands several cm from the planned spot |",
        "| `tip` | the object topples on release |",
        "| `collateral` | a neighbouring object is knocked as the hand passes |",
        "| `wrong_object` | the hand closes on a different object than the one planned |",
        "",
        "**Each fault fires once, on the first pick-and-place** (`--inject-at`, "
        "default `0`). That is what makes these recovery tests: a fault on every "
        "attempt makes the task impossible by construction — no object the "
        "planner intends to move can ever move — so such a run can only ever "
        "show that the loop gives up. Firing once asks whether the system "
        "notices what went wrong and reaches the goal anyway.",
        "",
        "`--inject-at every` is its own scenario, a gripper that is simply "
        "broken. There the loop is expected to fail, and does so in 2 "
        "iterations with a named diagnosis rather than running to the cap.",
        "",
        "Failures are **injected, not simulated** — execution is scene mutation, "
        "not physics. See [`../../docs/declutter.md`](../../docs/declutter.md).",
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="*", type=Path)
    p.add_argument("--all", action="store_true",
                   help="every run under outputs/runs with a progress.json")
    p.add_argument("--index", action="store_true",
                   help="also write outputs/reports/index.md")
    args = p.parse_args(argv)

    runs = list(args.runs)
    if args.all:
        runs += sorted(
            d for d in (REPO / "outputs" / "runs").iterdir()
            if (d / "progress.json").is_file()
        )
    if not runs:
        p.error("name a run directory, or pass --all")

    written = []
    for run in runs:
        if not (run / "progress.json").is_file():
            print(f"skip {run} (no progress.json)")
            continue
        out = run / "report.md"
        out.write_text(build(run))
        written.append(run)
        print(f"wrote {out}")

    if args.index and written:
        REPORTS.mkdir(parents=True, exist_ok=True)
        idx = REPORTS / "index.md"
        idx.write_text(build_index(written))
        print(f"wrote {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
