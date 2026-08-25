#!/usr/bin/env python
"""End-to-end verification of the decluttering loop, with no LLM and no GPU.

The counterpart to `verify_pipeline.py`, which checks that a grasp can be found
on a visible object. This checks the harder claim: that a target which is *not*
reachable can be made reachable, by repeatedly picking things up and putting
them somewhere sensible.

It runs the whole machinery — registry, blocking analysis, placement, collision
checking, executor, evaluator, state files — against synthetic scenes with exact
ground truth, so every claim is checked against what actually happened rather
than against what the pipeline believes happened.

    conda run -n graspmas python scripts/verify_declutter.py

Needs no API key and no GraspGen-X server: grasps come from the geometric
`nominal_grasp` path. Exit code 0 if every check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "GraspMAS"))

import synth_scene as ss  # noqa: E402
from declutter import DeclutterLoop  # noqa: E402
from execution import MutationExecutor  # noqa: E402
from run_artifacts import RunRecorder  # noqa: E402
from scene_registry import SceneRegistry  # noqa: E402
from session_state import SessionState  # noqa: E402


class Checks:
    """Collects pass/fail results so one failure does not hide the rest."""

    def __init__(self):
        self.results = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    @property
    def failed(self):
        return [r for r in self.results if not r["ok"]]


async def scenario_reaches_target(check, recorder, scenario="occluded_target",
                                  target="banana", inject=()):
    """The core claim: an unreachable target becomes reachable."""
    label = f"{scenario}" + (f" +{','.join(inject)}" if inject else "")
    print(f"\n--- {label} ---")

    executor = MutationExecutor(ss.SCENARIOS[scenario](), inject=list(inject), seed=0)
    run_dir = Path(recorder.dir) / f"declutter_{scenario}_{'_'.join(inject) or 'clean'}"
    state = SessionState(run_dir)

    before = executor.true_positions()

    # The target must genuinely start out unreachable, or the run proves nothing.
    reg = SceneRegistry()
    obs = executor.capture(0)
    reg.update_from_segmentation(obs.depth, obs.K, obs.seg, 0, obs.label_map)
    start_target = reg.resolve_target(target)
    start_blockers = reg.blocking_objects(start_target.id)
    check(
        f"{label}: the target starts blocked",
        len(start_blockers) > 0,
        f"{len(start_blockers)} blocker(s), target {start_target.visibility:.0%} visible",
    )

    loop = DeclutterLoop(executor=executor, state=state, max_iterations=6)
    result = await loop.run(f"pick up the {target}", target)

    after = executor.true_positions()
    moved_truth = {
        name for name in before
        if np.linalg.norm(after[name][:2] - before[name][:2]) > 0.02
    }

    if not inject:
        check(f"{label}: reached the target", result.status == "success", result.reason)
        check(
            f"{label}: every blocker was cleared",
            moved_truth == {b.label for b in start_blockers},
            f"moved {sorted(moved_truth)}, blockers were "
            f"{sorted(b.label for b in start_blockers)}",
        )
    else:
        # A ONE-SHOT fault, so the question is recovery, not termination.
        # Injecting on every attempt makes the task impossible by construction
        # — no object the planner intends to move can ever move — and then the
        # only thing the run can demonstrate is that it gives up. A single
        # fault asks the question worth asking: did the loop notice, and did it
        # carry on to the goal anyway?
        check(
            f"{label}: recovered and still reached the target",
            result.status == "success",
            f"{result.status}: {result.reason}",
        )
        progress_now = json.loads((run_dir / "progress.json").read_text())
        # A trace anywhere in the record, not only in the verdict. `collateral`
        # and `tip` move the intended object exactly as planned — the fault is a
        # side effect — so `action_succeeded` is legitimately "success" and the
        # evidence lives in what the executor reported instead.
        noticed = [
            r for r in progress_now["iterations"]
            if (r.get("evaluation") or {}).get("action_succeeded")
            not in (None, "success")
            or (r.get("evaluation") or {}).get("collateral")
            or (r.get("execution") or {}).get("disturbed")
            or (r.get("execution") or {}).get("notes")
            or (r.get("execution") or {}).get("status") not in (None, "ok")
        ]
        check(
            f"{label}: the injected failure left a trace in the record",
            bool(noticed),
            f"{len(noticed)} iteration(s) recorded something other than a clean move",
        )
        check(
            f"{label}: recovery cost at most one extra iteration",
            result.iterations <= 4,
            f"{result.iterations} iterations (a clean run takes 3)",
        )

    check(f"{label}: the target was never moved", target not in moved_truth)
    check(
        f"{label}: stayed within the iteration cap",
        result.iterations <= 6, f"{result.iterations} iterations",
    )

    unplanned = moved_truth - {b.label for b in start_blockers}
    if not inject:
        check(
            f"{label}: nothing was moved that was not a blocker",
            not unplanned, f"moved {sorted(moved_truth)}",
        )
    else:
        # The injections deliberately disturb things the loop did not choose to
        # move. The interesting property is not that it never happens, it is
        # that the evaluator *noticed* — silent collateral damage is the failure.
        progress_now = json.loads((run_dir / "progress.json").read_text())
        reported = any(
            r.get("evaluation", {}).get("collateral")
            or r.get("execution", {}).get("disturbed")
            for r in progress_now["iterations"]
        )
        check(
            f"{label}: unplanned movement was noticed, not silent",
            not unplanned or reported,
            f"unplanned {sorted(unplanned)}, reported={reported}",
        )

    # Ground truth: everything that was moved is still on the table, upright.
    on_table = all(
        abs(executor.spec.by_name(n).primitive.position[2]
            - executor.spec.by_name(n).primitive.height / 2.0) < 1e-6
        for n in moved_truth
    )
    check(f"{label}: moved objects are resting on the surface", on_table)

    # The record has to be a faithful account, not a summary of intent.
    progress = json.loads((run_dir / "progress.json").read_text())
    check(
        f"{label}: progress.json matches the run",
        progress["status"] == result.status
        and len(progress["iterations"]) == result.iterations,
        f"{len(progress['iterations'])} iterations recorded",
    )
    check(
        f"{label}: every attempted move was evaluated",
        all(
            r.get("evaluation")
            for r in progress["iterations"]
            if r.get("action") == "remove"
        ),
        "the final grasp_target iteration has no move to evaluate",
    )
    check(f"{label}: a grand plan was written", (run_dir / "grand_plan.json").is_file())

    return result


async def placement_is_sound(check, recorder):
    """Every placement the loop chose must have been collision-free and useful."""
    print("\n--- placement quality ---")
    import collision as col
    import placement as pl

    executor = MutationExecutor(ss.occluded_target_scene(), seed=0)
    state = SessionState(Path(recorder.dir) / "declutter_placement")
    loop = DeclutterLoop(executor=executor, state=state, max_iterations=6)
    await loop.run("pick up the banana", "banana")

    clearances, travels = [], []
    for record in state.iterations:
        place = (record.planned or {}).get("place")
        if place:
            clearances.append(place["clearance_m"])
            travels.append(place["travel_m"])

    check("every removal produced a place pose", len(clearances) == 2, f"{len(clearances)}")
    if clearances:
        check(
            "placements cleared their own footprint",
            min(clearances) > 0.02,
            f"min clearance {min(clearances)*100:.1f} cm",
        )
        check(
            "objects were actually relocated",
            min(travels) > 0.05,
            f"min travel {min(travels)*100:.0f} cm",
        )

    # The final grasp must be reachable in the cleared scene.
    obs = executor.capture(9)
    reg = SceneRegistry()
    reg.update_from_segmentation(obs.depth, obs.K, obs.seg, 9, obs.label_map)
    banana = reg.resolve_target("banana")
    pose = reg.nominal_grasp(banana.id)
    clear = col.sweep_is_clear(
        pose, reg.scene_cloud_excluding(banana.id),
        col.load_gripper_points("franka_panda"), approach_len=0.10,
    )
    check("the final grasp is collision-free", bool(clear))
    check(
        "the target is fully visible at the end",
        banana.visibility > 0.95, f"{banana.visibility:.0%}",
    )


async def keep_out_is_load_bearing(check, recorder):
    """Show the failure the placement keep-out exists to prevent."""
    print("\n--- keep-out region ---")
    import placement as pl
    from execution import PickPlacePlan

    def clear_with(use_keep_out, steps=4):
        executor = MutationExecutor(ss.occluded_target_scene(), seed=0)
        moved = []
        for step in range(steps):
            obs = executor.capture(step)
            reg = SceneRegistry()
            reg.update_from_segmentation(obs.depth, obs.K, obs.seg, step, obs.label_map)
            target = reg.resolve_target("banana")
            blockers = reg.blocking_objects(target.id)
            if not blockers:
                return moved, target.visibility
            inst = reg.get(blockers[0].object_id)
            grasp = reg.nominal_grasp(inst.id)
            keep = reg.keep_out_for(target.id, moving_id=inst.id) if use_keep_out else None
            place = pl.plan_place(
                grasp, inst.cloud, reg.scene_cloud_excluding(), reg.plane,
                keep_out=keep, hmap=reg.hmap,
            )
            if place is None:
                break
            executor.execute_pick_place(
                PickPlacePlan(inst.id, {"pose": grasp.tolist()}, place.as_dict())
            )
            moved.append(inst.label)

        obs = executor.capture(steps)
        reg = SceneRegistry()
        reg.update_from_segmentation(obs.depth, obs.K, obs.seg, steps, obs.label_map)
        return moved, reg.resolve_target("banana").visibility

    without, vis_without = clear_with(False)
    with_, vis_with = clear_with(True)

    check(
        "without a keep-out the loop re-blocks the target",
        len(without) > len(set(without)),
        f"moved {without}, target {vis_without:.0%} visible",
    )
    check(
        "with a keep-out one move per blocker suffices",
        len(with_) == len(set(with_)) == 2 and vis_with > 0.95,
        f"moved {with_}, target {vis_with:.0%} visible",
    )


async def persistent_fault_terminates(check, recorder):
    """The other half: a fault that never stops must end the run, not hang it.

    `--inject-at every` is a real scenario — a gripper that is simply broken —
    and the correct outcome there is failure with a diagnosis, reached quickly
    rather than at the iteration cap.
    """
    print("\n--- persistent fault ---")
    executor = MutationExecutor(
        ss.occluded_target_scene(), inject=["drop"], inject_at=None, seed=0
    )
    state = SessionState(Path(recorder.dir) / "declutter_persistent_drop")
    loop = DeclutterLoop(executor=executor, state=state, max_iterations=6)
    result = await loop.run("pick up the banana", "banana")

    check(
        "a permanently broken gripper fails rather than looping",
        result.status == "failed",
        result.reason[:80],
    )
    check(
        "it gives up well before the iteration cap",
        result.iterations <= 3,
        f"{result.iterations} iterations",
    )
    check(
        "and says which fault it was",
        "same stage" in result.reason,
        result.reason.split("Diagnosis:")[-1].strip()[:90],
    )


async def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="skip the injected-failure runs")
    args = p.parse_args(argv)

    recorder = RunRecorder(name="verify_declutter")
    recorder.write_config(vars(args))
    check = Checks()

    print("=" * 70)
    print("  verifying the decluttering loop — no LLM, no GPU, no server")
    print("=" * 70)

    await scenario_reaches_target(check, recorder)
    await placement_is_sound(check, recorder)
    await keep_out_is_load_bearing(check, recorder)

    if not args.quick:
        for inject in (["drop"], ["offset"], ["collateral"], ["wrong_object"],
                       ["tip"]):
            await scenario_reaches_target(check, recorder, inject=inject)
        await persistent_fault_terminates(check, recorder)

    print("\n" + "=" * 70)
    passed = len(check.results) - len(check.failed)
    print(f"  {passed}/{len(check.results)} checks passed")
    for bad in check.failed:
        print(f"    FAILED: {bad['check']} — {bad['detail']}")
    print("=" * 70)

    report = {"checks": check.results, "passed": passed, "total": len(check.results)}
    (Path(recorder.dir) / "verification_report.json").write_text(
        json.dumps(report, indent=2)
    )
    recorder.finish(
        result=report, status="success" if not check.failed else "failed"
    )
    print(f"  report: {recorder.dir}/verification_report.json")

    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
