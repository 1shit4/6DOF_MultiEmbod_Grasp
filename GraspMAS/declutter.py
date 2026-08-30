"""The outer loop: clear the table until the target can be grasped.

It **wraps** `GraspMAS.query()` rather than modifying it. The inner
Planner/Coder/Observer loop keeps doing what it does — turn one instruction into
one grasp — and everything long-horizon lives out here. That way the existing
suite keeps passing and the two concerns stay separable.

One iteration:

    capture ──▶ registry ──▶ evaluate the last move ──▶ decide what to do next
        ▲                                                        │
        │                                                        ▼
    execute ◀── plan the pick and the place ◀────── grasp it (inner loop)

Termination, in the order checked:

1. the planner says `grasp_target` and a grasp survives collision filtering — **success**
2. the planner says `abort`, or a validation check forces it — **aborted**
3. no progress for two iterations running — **failed**, stated as such
4. the iteration cap — **failed**

The third is what stops a plausible-looking loop running forever. Progress is
measured by whether the target is still blocked, not by whether actions
succeeded, because those come apart in both directions — see `evaluator`.

The loop can run with `planner=None`, in which case blockers are cleared
worst-first by severity. That path spends no LLM requests at all and is what
`scripts/verify_declutter.py` uses to check the machinery end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import placement as pl
from agents.task_planner import (
    MAX_CANDIDATES,
    Decision,
    TargetCandidate,
    TargetChoice,
    TaskPlanner,
    format_blocking,
    format_candidates,
)
from evaluator import Evaluation, evaluate
from execution import ExecutionReport, Observation, PickPlacePlan
from scene_registry import Blocker, SceneRegistry
from session_state import SessionState

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 6
STALL_WINDOW = 2


@dataclass
class DeclutterResult:
    status: str  # success | aborted | failed
    reason: str = ""
    grasp: Optional[dict] = None
    iterations: int = 0
    moved: List[str] = field(default_factory=list)
    run_dir: Optional[str] = None

    def describe(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "iterations": self.iterations,
            "moved": list(self.moved),
            "grasp": self.grasp,
            "run_dir": self.run_dir,
        }


class DeclutterLoop:
    """Runs pick-and-place iterations until the target is reachable."""

    def __init__(
        self,
        executor,
        state: SessionState,
        planner: Optional[TaskPlanner] = None,
        graspmas=None,
        gripper_name: str = "franka_panda",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        recorder=None,
        vlm_review: bool = True,
    ):
        self.executor = executor
        self.state = state
        self.planner = planner
        self.graspmas = graspmas
        self.gripper_name = gripper_name
        self.max_iterations = int(max_iterations)
        self.recorder = recorder
        self.vlm_review = bool(vlm_review)

        self.registry = SceneRegistry()
        # The person's own words, kept for the whole run. The inner loop needs
        # them when fetching the target: how a knife should be held depends on
        # what was asked for, and only this string carries that.
        self.goal: str = ""
        # Why the last grasp attempt produced nothing, when the reason is more
        # informative than "no grasp found" — currently, that the critic
        # rejected everything it was shown. Read once and cleared, so a stale
        # reason cannot be attributed to a later iteration.
        self._last_grasp_refusal: Optional[str] = None
        # Objects a real grasp attempt on the target ran into: named by the
        # collision filter over the ~416 poses the sampler actually proposed,
        # not by a synthesised one. Populated only once a grasp has been tried,
        # which is why the loop clears occluders first and asks this afterwards.
        self._grasp_blockers: List[dict] = []
        self._previous: Optional[SceneRegistry] = None
        self._last_plan: Optional[PickPlacePlan] = None
        self._last_place_xy: Optional[np.ndarray] = None
        self._last_object: Optional[str] = None
        self._last_report: Optional[ExecutionReport] = None

    # -- perception --------------------------------------------------------

    def _perceive(self, iteration: int, phase: str = "before") -> Observation:
        """Capture and rebuild the registry.

        A **fresh** `SceneRegistry` is not used here on purpose — ids must
        survive across iterations, which is what `update_from_*` is for. But a
        fresh `SceneContext` *is* required per capture downstream, because
        `image_patch._GRASP_CACHE` is keyed on the mask and only cleared when
        the scene object changes; reusing one would serve a grasp computed
        against the previous arrangement of the table.

        The last move's intended destination is passed as a matching hint, so
        an object the robot deliberately relocated is recognised where it was
        put rather than retired as missing — see `SceneRegistry._ingest`.
        """
        obs = self.executor.capture(iteration)
        if self.recorder is not None:
            # Namespace this iteration's artifacts. The inner loop names its
            # files by inner round, so without this every outer iteration
            # overwrites the last and the run keeps only the final pick.
            self.recorder.set_scope(f"iter{iteration}")
        hint = (
            {self._last_object: self._last_place_xy}
            if self._last_object and self._last_place_xy is not None
            else None
        )
        # `phase` matters: this runs twice per iteration — once to decide, and
        # once after executing to evaluate. Both are worth keeping and they are
        # different pictures, so naming them the same would mean the second
        # silently overwrote the first.
        self._save_scene(obs, f"scene_iter{iteration}_{phase}.png")
        if obs.has_ground_truth:
            self.registry.update_from_segmentation(
                obs.depth, obs.K, obs.seg, iteration, obs.label_map,
                expected_moves=hint,
            )
        else:
            self.registry.update_from_patches(
                obs.depth, obs.K, self._detect(obs), iteration, expected_moves=hint
            )
        return obs

    def _detect(self, obs: Observation):
        """Open-vocabulary detection, for observations with no ground truth."""
        import image_patch as ip
        from perception3d import SceneContext

        ip.configure_scene(
            scene=SceneContext(depth=obs.depth, intrinsics=obs.K,
                               image_shape=obs.depth.shape),
            gripper_name=self.gripper_name,
        )
        root = ip.ImagePatch(obs.rgb)
        found = []
        for name in self.vocabulary:
            for patch in root.find(name) or []:
                if patch is not None and patch.mask is not None:
                    found.append((name, patch.mask))
        return found

    vocabulary: List[str] = []

    # -- the loop ----------------------------------------------------------

    async def run(self, goal: str, target: Optional[str] = None) -> DeclutterResult:
        self._stall_deferred = False
        self._grasp_blockers = []
        self.goal = goal or ""
        obs = self._perceive(0)

        choice, instance = await self._choose_target(goal, target, obs)
        if instance is None:
            named = f"the target {target!r} is not in the scene" if target else (
                f"nothing in the scene serves the goal {goal!r}"
            )
            reason = (
                f"{named}; detected "
                f"{sorted(i.label for i in self.registry.instances.values())}"
            )
            if choice is not None and choice.corrections:
                reason += f". {'; '.join(choice.corrections)}"
            self.state.start(goal, None)
            self.state.finish("failed", {"reason": reason})
            return DeclutterResult("failed", reason, run_dir=str(self.state.run_dir))

        target_id = instance.id
        self.state.start(
            goal,
            {"id": target_id, "label": instance.label},
            target_choice=choice.describe() if choice is not None else None,
        )
        if choice is not None and choice.interpretation:
            self._log(f"read the goal as: {choice.interpretation}")
            for note in choice.corrections:
                self.state.note(f"target selection corrected: {note}")
        self._log(f"target {target_id} ({instance.label}) for goal {goal!r}")

        target_id = await self._draft_grand_plan(goal, target_id, obs, choice)

        moved: List[str] = []
        for index in range(self.max_iterations):
            if index > 0:
                obs = self._perceive(index)

            if target_id not in self.registry.instances:
                # The id can lapse for two very different reasons: the target
                # really is gone (knocked off the table, or picked up by
                # mistake), or it merely shifted far enough to be re-registered.
                # Re-resolving by label separates them, and only the first is
                # fatal.
                recovered = (
                    self.registry.resolve_target(target) if target else None
                )
                if recovered is None:
                    return self._fail(
                        f"the target {target_id} is no longer in the scene", index, moved
                    )
                self._log(
                    f"target {target_id} was re-registered as {recovered.id}; "
                    "it moved further than the tracker expected"
                )
                target_id = recovered.id
                self.state.progress["target"] = {
                    "id": target_id, "label": recovered.label
                }
                self.state.snapshot(f"iter{index}_reidentified")

            # Occluders and objects touching the target, plus anything a real
            # grasp attempt has already shown to stand in the hand's way.
            blockers = self._blockers_for(target_id)
            decision = await self._decide(goal, target_id, blockers, obs)

            if decision.action == "defer":
                # A refusal about timing, not possibility. Spend the iteration
                # on the geometrically obvious move rather than on nothing: it
                # makes real progress, and an iteration that acts is what turns
                # "no evidence yet" into evidence.
                fallback = self._scripted_decision(target_id, blockers)
                fallback.corrections = list(decision.corrections) + [
                    f"falling back to the geometric choice for this iteration: "
                    f"{fallback.action}"
                    + (f" {fallback.object_id}" if fallback.object_id else "")
                ]
                decision = fallback

            self.state.begin_iteration(
                index,
                subgoal=decision.subgoal,
                action=decision.action,
                object_id=decision.object_id,
                object_label=(
                    self.registry.instances[decision.object_id].label
                    if decision.object_id in self.registry.instances else None
                ),
                rationale=decision.rationale,
                blockers=[b.describe() for b in blockers],
            )
            self.state.record_decision(decision.describe())
            for note in decision.corrections:
                self.state.note(f"planner corrected: {note}")

            self._apply_plan_update(decision, index)

            if decision.action == "abort":
                self.state.end_iteration()
                return self._abort(decision.rationale or "planner aborted", index, moved)

            if decision.action == "retarget":
                target_id = await self._retarget(decision, target_id, index)
                self.state.end_iteration()
                # The scene has not changed, only what we are aiming at, so the
                # next iteration re-derives blockers for the new target from the
                # same observation. It does not count as progress and does not
                # count as a stall.
                continue

            if decision.action == "grasp_target":
                result = await self._grasp_target(target_id, obs, index, moved)
                if result is not None:
                    return result
                continue

            outcome = await self._remove(decision, target_id, obs, index)
            self.state.end_iteration()

            if outcome is not None:
                moved.append(decision.object_id)

            if self.state.is_stalled(window=STALL_WINDOW):
                # A stall is the evidence a retarget needs, so ending the run on
                # it would make retargeting unreachable: the gate wants
                # STALL_WINDOW iterations of getting nowhere, and the stall check
                # fires on exactly that iteration. Give the planner one — and only
                # one — further iteration to propose the switch, and only when an
                # unused ranked alternative actually exists.
                alternatives = [
                    i for i in self.state.target_candidates if i != target_id
                ]
                if (
                    not self._stall_deferred
                    and alternatives
                    and self.state.can_retarget()
                ):
                    self._stall_deferred = True
                    self.state.note_run(
                        f"no progress in {STALL_WINDOW} iterations, but "
                        f"{len(alternatives)} ranked alternative target(s) remain — "
                        "one more iteration to switch before giving up"
                    )
                    continue

                reason = (
                    f"no progress in {STALL_WINDOW} iterations: the last attempts "
                    "neither moved anything usefully nor cleared the target"
                )
                # Say *which* failure this is when the reports agree on one.
                # The generic sentence above is true of every stalled run, and
                # a gripper that drops everything and an arm that reaches for
                # the wrong object need opposite responses.
                diagnosis = self.state.stall_diagnosis()
                if diagnosis:
                    reason += f". Diagnosis: {diagnosis}"
                return self._fail(reason, index + 1, moved)

        return self._fail(
            f"reached the {self.max_iterations}-iteration limit with the target "
            "still blocked",
            self.max_iterations, moved,
        )

    # -- what the goal refers to -------------------------------------------

    async def _choose_target(self, goal: str, target: Optional[str], obs):
        """Resolve the goal to one instance, and to the ranking behind it.

        An explicit `--target` short-circuits the whole thing and takes the
        original label-matching path, so every previously recorded run,
        `verify_declutter.py`, and `--no-llm` behave exactly as before. The LLM
        is asked only when nobody has already said what the object is — which is
        the case this exists for ("I need something to cut").
        """
        if target:
            return None, self.registry.resolve_target(target)

        if self.planner is None:
            # Nothing here can read an abstract goal, and guessing at one would
            # be worse than saying so.
            raise ValueError(
                "no target was given and there is no planner to infer one; "
                "pass --target, or run without --no-llm"
            )

        choice = await self.planner.select_target(
            goal, self.registry.as_prompt_table(None), self._image_b64(obs)
        )
        choice = TaskPlanner.validate_target(choice, self.registry)
        best = choice.best
        return choice, (self.registry.instances.get(best.object_id) if best else None)

    async def _retarget(self, decision: Decision, target_id: str, index: int) -> str:
        """Switch to a ranked alternative, or keep the current target.

        Every guard lives in `TaskPlanner._validate_retarget` and in
        `SessionState.retarget`; by the time this runs the decision has already
        been checked. A refusal here is therefore a bug rather than a model
        error, and is noted as one instead of being swallowed.
        """
        new_id = decision.object_id
        inst = self.registry.instances.get(new_id)
        if inst is None:
            self.state.note(f"retarget to {new_id} failed: it is not in the scene")
            return target_id

        ok = self.state.retarget(
            {"id": new_id, "label": inst.label},
            decision.rationale or "the previous target could not be reached",
            iteration=index,
        )
        if not ok:
            self.state.note(f"retarget refused: {self.state.last_refusal}")
            return target_id

        self._log(f"retargeted from {target_id} to {new_id} ({inst.label})")
        return new_id

    # -- steps -------------------------------------------------------------

    def _candidate_blocking(self, choice, target_id: str) -> str:
        """Blocking analysis for each alternative target, as prompt text.

        Geometry, not a request — so the model gets to compare what clearing
        each candidate would cost, which is the one thing a mid-turn tool call
        would have bought, at no extra cost in calls.
        """
        if choice is None:
            return ""
        entries = []
        for cand in choice.candidates[:MAX_CANDIDATES]:
            if cand.object_id not in self.registry.instances:
                continue
            blockers = self.registry.blocking_objects(
                cand.object_id, gripper_name=self.gripper_name,
            )
            # Recorded on the candidate as well as rendered, so the count the
            # model was shown is the count `progress.json` reports afterwards.
            cand.n_blockers = len(blockers)
            entries.append(
                (cand, format_blocking(blockers, self.registry.get(cand.object_id)))
            )
        return format_candidates(entries)

    def _apply_target_order(self, order, choice, target_id: str) -> str:
        """Validate stage 2's ordering and adopt its head as the target.

        The model may only *order* the candidates stage 1 produced. Unknown ids
        are dropped, omitted ones keep their existing relative position, and the
        result becomes the retarget ranking as well as the target — so a later
        `retarget` advances to whatever this decided was second best.

        A head that is not from the best priority tier is allowed, because
        effort is a real cost the user asked to weigh, but it is *recorded*:
        fetching the wrong thing quickly is a failure, not an efficiency, and a
        run that did it should say so in its own state file.
        """
        if choice is None or not choice.candidates:
            return target_id

        known = set(choice.ids)
        clean = [i for i in order if i in known and i in self.registry.instances]
        dropped = [i for i in order if i not in known]
        if dropped:
            self.state.note_run(
                f"grand plan named {dropped} in target_order, which were not "
                "candidates; ignored"
            )
        if clean:
            choice.reorder(clean)

        best = choice.best
        if best is None:
            return target_id

        top_priority = min(c.priority for c in choice.candidates)
        if best.priority > top_priority:
            preferred = [
                c.object_id for c in choice.candidates if c.priority == top_priority
            ]
            self.state.note_run(
                f"target order put {best.object_id} (priority {best.priority}, "
                f"{best.n_blockers} blocker(s)) ahead of priority-{top_priority} "
                f"{preferred} — effort was traded against suitability"
            )

        self.state.progress["target_choice"] = choice.describe()
        if best.object_id != target_id:
            inst = self.registry.get(best.object_id)
            self._log(
                f"target order chose {best.object_id} ({inst.label}) over "
                f"{target_id}: priority {best.priority}, {best.n_blockers} blocker(s)"
            )
            self.state.progress["target"] = {"id": best.object_id, "label": inst.label}
            return best.object_id
        return target_id

    async def _draft_grand_plan(self, goal: str, target_id: str, obs, choice=None) -> str:
        """Write the plan, and return the target it is actually for."""
        blockers = self.registry.blocking_objects(
            target_id, gripper_name=self.gripper_name
        )
        blocking = format_blocking(blockers, self.registry.get(target_id))

        if self.planner is None:
            self.state.set_grand_plan(
                removal_order=[
                    {"object_id": b.object_id, "label": b.label,
                     "reason": ", ".join(b.reasons)}
                    for b in blockers
                ],
                success_criterion="nothing blocks the target and it is fully visible",
                reasoning="scripted planner: clear blockers worst-first by severity",
            )
            return target_id

        try:
            plan = await self.planner.make_grand_plan(
                goal, target_id,
                self.registry.as_prompt_table(target_id),
                blocking,
                self._image_b64(obs),
                candidates=self._candidate_blocking(choice, target_id),
            )
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
            # The grand plan is guidance, not a precondition. Losing it costs
            # continuity, not correctness: the blocking analysis still names
            # what is in the way on every iteration.
            logger.warning("could not draft a grand plan: %s", exc)
            plan = {
                "removal_order": [
                    {"object_id": b.object_id, "label": b.label,
                     "reason": ", ".join(b.reasons)}
                    for b in blockers
                ],
                "success_criterion": "nothing blocks the target and it is fully visible",
                "reasoning": f"drafted geometrically: the planner was unreachable ({exc})",
            }

        # Stage 2 orders the candidates, weighing the suitability priority stage
        # 1 set against the effort measured since. It is an *ordering* job over a
        # fixed candidate list, not a free choice of target: an earlier version
        # let the plan name any target it liked, which was a second retargeting
        # path with none of the guards — no evidence, no budget, no record — and
        # it promptly swapped a priority-1 knife for the scissors because they
        # were "fully visible and unobstructed". Ordering is validated against
        # the candidates, and a swap across priorities is recorded.
        target_id = self._apply_target_order(plan.pop("target_order", []), choice,
                                             target_id)
        # Only the three fields `set_grand_plan` accepts. A model that emits an
        # extra key — `target_id` was the one that did it — would otherwise take
        # the run down with a TypeError raised from inside the state writer,
        # which names neither the model nor the key.
        self.state.set_grand_plan(**{
            k: plan[k] for k in
            ("removal_order", "success_criterion", "reasoning") if k in plan
        })
        self._log(f"grand plan: {[s.get('object_id') for s in plan['removal_order']]}")
        return target_id

    async def _decide(self, goal, target_id, blockers, obs) -> Decision:
        if self.planner is None:
            return self._scripted_decision(target_id, blockers)

        try:
            decision = await self.planner(
                goal=goal,
                target_id=target_id,
                scene_table=self.registry.as_prompt_table(target_id),
                blocking=format_blocking(blockers, self.registry.get(target_id)),
                progress=self.state.planner_context(),
                image_b64=self._image_b64(obs),
            )
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
            # The provider being unreachable is not a reason to lose the run.
            # `_grasp_via_agents` has always caught this and fallen back to a
            # geometric grasp; the outer planner had no such guard, so a daily
            # quota exhausted mid-run raised straight through `run()` and the
            # process died with `progress.json` still saying "in_progress" —
            # no verdict, no reason, and `--resume` reading a half-open
            # iteration. Aborting deliberately keeps the record honest and the
            # run resumable once quota returns.
            # Reported via `corrections`, not `state.note`: `_decide` runs
            # before `begin_iteration`, so there is no open iteration to note
            # against yet. `run()` replays corrections once the record is open.
            #
            # The wording distinguishes a provider outage from a defect on our
            # side, because the first message this guard ever produced said "the
            # task planner could not be reached" for an `AttributeError` raised
            # while rendering our own state file. That reads as a network
            # problem and sent the diagnosis in the wrong direction entirely.
            outage = isinstance(exc, (RuntimeError, TimeoutError, ConnectionError))
            what = (
                "could not be reached" if outage
                else f"failed with {type(exc).__name__}"
            )
            logger.warning("task planner %s: %s", what, exc)
            return Decision(
                action="abort",
                rationale=f"the task planner {what}: {exc}",
                corrections=[f"task planner {what}: {exc}"],
            )
        return TaskPlanner.validate(decision, self.registry, target_id, self.state)

    def _scripted_decision(self, target_id, blockers) -> Decision:
        """The no-LLM path: clear the worst blocker, or finish."""
        target = self.registry.get(target_id)
        if not blockers:
            if target.footprint_is_reliable:
                return Decision(
                    action="grasp_target",
                    subgoal=f"compute the best 6-DoF grasp for {target_id}",
                    rationale="nothing blocks the target and it is fully visible",
                )
            return Decision(
                action="abort",
                rationale=(
                    f"nothing is listed as blocking, but the target is only "
                    f"{target.visibility:.0%} visible, so the analysis cannot be trusted"
                ),
            )
        worst = blockers[0]
        return Decision(
            action="remove",
            object_id=worst.object_id,
            subgoal=f"pick up {worst.object_id} and place it clear of {target_id}",
            rationale=f"worst blocker: {', '.join(worst.reasons)}",
        )

    async def _remove(self, decision, target_id, obs, index) -> Optional[str]:
        """Plan and execute one pick-and-place. Returns the object id on success."""
        object_id = decision.object_id
        inst = self.registry.get(object_id)

        grasp_dict = await self._grasp_for(object_id, obs, index, phase="declutter")
        if grasp_dict is None:
            # Nothing was attempted, so nothing is reported as attempted. The
            # *reason* is what lets the planner tell "this object cannot be
            # picked up" from "something has to be cleared before it can be".
            why = self._take_grasp_refusal() or f"no grasp could be computed for {object_id}"
            self.state.record_execution(
                {"status": "failed", "stage_reached": "plan", "error": why}
            )
            self.state.record_evaluation(
                {"action_succeeded": "not_moved", "still_blocking_target": True,
                 "evidence": why}
            )
            return None

        place = pl.plan_place(
            np.asarray(grasp_dict["pose"], dtype=np.float64),
            inst.cloud,
            self.registry.scene_cloud_excluding(),
            self.registry.plane,
            gripper=self.gripper_name,
            width=float(grasp_dict.get("width", 0.08)),
            keep_out=self.registry.keep_out_for(target_id, moving_id=object_id),
            hmap=self.registry.hmap,
        )
        if place is None:
            self.state.note(
                f"no placement available for {object_id}: nowhere on the table "
                "clears its footprint without obstructing the target again"
            )
            self.state.record_execution(
                {"status": "failed", "error": "no placement available"}
            )
            self.state.record_evaluation(
                {"action_succeeded": "not_moved", "still_blocking_target": True,
                 "evidence": "the object could not be placed anywhere useful"}
            )
            return None

        self.state.record_plan(grasp_dict, place.as_dict())

        plan = PickPlacePlan(
            object_id=object_id,
            grasp=grasp_dict,
            place=place.as_dict(),
            gripper=self.gripper_name,
            object_label=inst.label,
        )
        self._previous = _snapshot(self.registry)
        self._last_place_xy = place.place_xy
        self._last_object = object_id

        report = self.executor.execute_pick_place(plan)
        self._last_report = report
        self.state.record_execution(report)
        self._log(
            f"[{index}] moved {object_id} ({inst.label}): {report.status}"
            + (f" — {report.error}" if report.error else "")
        )

        evaluation = self._evaluate_now(index, target_id)
        # "Moved" means the object is somewhere else now, not that it landed
        # precisely. A release 9 cm off plan still cleared the target, and
        # reporting that as having moved nothing reads as a bug when it is not.
        moved = evaluation.action_succeeded in ("success", "moved_off_target")
        return object_id if moved else None

    def _evaluate_now(self, index: int, target_id: str) -> Evaluation:
        """Re-perceive and judge the move that just happened."""
        # The "after" capture is the only image of what the action actually did;
        # for the last iteration there is no following capture to stand in.
        self._perceive(index, phase="after")
        if self.recorder is not None and getattr(self.recorder, "enabled", False):
            self.state.record_artifacts(
                scene_after=str(
                    Path(self.recorder.images_dir) / f"scene_iter{index}_after.png"
                )
            )
        result = evaluate(
            self._previous,
            self.registry,
            self._last_object,
            target_id,
            intended_place_xy=self._last_place_xy,
            execution_report=self._last_report,
        )
        self.state.record_evaluation(result)
        self._log(f"[{index}] evaluated: {result.action_succeeded}, {result.evidence}")
        return result

    async def _grasp_target(self, target_id, obs, index, moved):
        """Final step: grasp the target, or report why we cannot."""
        grasp = await self._grasp_for(target_id, obs, index, phase="deliver")
        self.state.record_plan(grasp, None)

        if grasp is None:
            self.state.note(
                self._take_grasp_refusal() or "no grasp survived filtering on the target"
            )
            # Nothing occluding the target, nothing touching it, and still no
            # grasp. The attempt itself says why: the collision filter rejected
            # candidates and recorded which objects they ran into. Those become
            # blockers for the next iteration — named by a real search over the
            # poses the sampler proposed, not guessed from one top-down sweep.
            if self._grasp_blockers:
                named = ", ".join(
                    f"{e['object_id']} ({e['label']})"
                    for e in self._grasp_blockers[:3]
                )
                self.state.note(
                    f"nothing occludes or touches the target, but the best grasps "
                    f"on it collide with {named}; clearing those next"
                )
            self.state.end_iteration()
            return None  # let the loop continue; something is still in the way

        self.state.end_iteration()
        self.state.finish("success", {"grasp": grasp, "moved": moved})
        self._log(f"target grasped after {index} iteration(s)")
        return DeclutterResult(
            "success", "the target is reachable", grasp, index + 1, moved,
            str(self.state.run_dir),
        )

    async def _grasp_for(
        self, object_id: str, obs, index: int, phase: str = "declutter"
    ) -> Optional[dict]:
        """A grasp on one instance, from the inner loop or from geometry alone.

        `phase` is "declutter" (this object is in the way and is being moved
        aside) or "deliver" (this is the object the person actually asked for).
        The two want different grasps on the same object — a knife being
        shifted off the table should be taken by the handle, and a knife being
        handed to a person is taken by the blade so the handle is what they
        receive. The inner loop could not tell them apart, because both paths
        sent it the same bare instance id.
        """
        inst = self.registry.get(object_id)

        if self.graspmas is not None:
            grasp = await self._grasp_via_agents(inst, obs, index, phase)
            if grasp is not None:
                return grasp

            # Why the loop came back empty decides whether a geometric grasp is
            # a reasonable substitute. Two very different cases used to share
            # this branch:
            #
            #   * the loop never reached a verdict — a provider outage, a code
            #     error, no candidates at all. A nominal grasp is a sensible
            #     fallback and is what the outage guard is for.
            #   * the Observer looked at a grasp and rejected it. Substituting a
            #     geometric pose here silently overrides the critic, and it is
            #     how a run came to report `success` carrying `score: 0.0,
            #     source: "nominal"` after every round had been rejected.
            review = getattr(self.graspmas, "observation_json", None) or {}
            why = str(review.get("failure") or "unspecified")
            # Which rejections should stop us acting depends on what the action
            # is FOR, and the two phases differ:
            #
            #   * `geometry` and `wrong_object` stop us either way — the hand
            #     would collide, or would move something nobody asked to move.
            #   * `wrong_part` is a soft failure while decluttering. The job is
            #     to get the object out of the way; a physically sound grasp on
            #     the whole object does that, whatever sub-region was requested.
            #     Blocking it here cost every clean run: on synthetic scenes an
            #     object has no parts at all, so `find_part` correctly finds
            #     none, the Observer correctly says `wrong_part` — and the loop
            #     refused a grasp it had itself judged "physically feasible and
            #     collision-free", then aborted for lack of options.
            #   * when delivering, `wrong_part` is decisive. How the object is
            #     held IS the request; handing someone a knife by the wrong end
            #     is the failure the phase exists to prevent.
            blocking_failures = ("geometry", "wrong_object")
            refuses = str(review.get("verdict", "")).upper() == "INVALID" and (
                phase == "deliver" or why in blocking_failures
            )
            if refuses:
                summary = str(review.get("summary") or "").strip()
                self.state.note(
                    f"no grasp on {object_id} passed review ({why})"
                    + (f": {summary[:240]}" if summary else "")
                )
                self._last_grasp_refusal = (
                    f"every grasp found on {object_id} was rejected on review "
                    f"({why})"
                )
                return None

            if str(review.get("verdict", "")).upper() == "INVALID":
                self.state.note(
                    f"the grasp on {object_id} was judged {why}, which does not "
                    "prevent moving it aside; using a nominal top-down grasp"
                )
            else:
                self.state.note(
                    f"the agent loop returned no grasp for {object_id}; "
                    "falling back to a nominal top-down grasp"
                )

        pose = self.registry.nominal_grasp(object_id, gripper_name=self.gripper_name)
        if pose is None:
            return None
        if not self._reachable(pose, object_id):
            self.state.note(
                f"the nominal top-down grasp on {object_id} is not collision-free"
            )
            self._diagnose_obstruction(object_id, pose)
            return None

        from perception3d import Grasp6D

        width, _ = _gripper_geometry(self.gripper_name)
        # Score 0: this is geometry, not a discriminator judgement, and saying so
        # keeps it distinguishable from a genuinely low-confidence GraspGen-X
        # result in `progress.json`.
        grasp = Grasp6D(pose=pose, score=0.0, gripper=self.gripper_name, width=width)
        return {**grasp.as_dict(), "source": "nominal"}

    def _take_grasp_refusal(self) -> Optional[str]:
        """The last refusal reason, consumed so it cannot be reused."""
        why, self._last_grasp_refusal = self._last_grasp_refusal, None
        return why

    @staticmethod
    def _grasp_blockers_from_rounds(rounds) -> List[dict]:
        """Objects the rejected grasp candidates ran into, gathered per round.

        `grasp_detection` attributes each rejection to a named instance; this
        merges what every round of the inner loop found, so an object that
        fouled grasps twice counts twice.
        """
        tally: Dict[str, dict] = {}
        for rd in rounds or []:
            grasp = rd.get("grasp") or {}
            for entry in (grasp.get("filters") or {}).get("fouled_by") or []:
                oid = entry.get("object_id")
                if not oid:
                    continue
                got = tally.setdefault(
                    oid, {"object_id": oid, "label": entry.get("label", ""),
                          "grasps_fouled": 0}
                )
                got["grasps_fouled"] += int(entry.get("grasps_fouled", 0))
        return sorted(tally.values(), key=lambda e: -e["grasps_fouled"])

    def _blockers_for(self, target_id: str):
        """What must move before the target can be picked up.

        Two measured reasons from the scene — it hides the target, or it is
        touching it — plus anything a real grasp attempt has already shown to be
        in the hand's way. That third source only exists after a grasp has been
        tried, which is the whole shape of the loop: clear what obviously has to
        go, try the grasp, and let the attempt itself name what else is wrong.
        """
        blockers = self.registry.blocking_objects(
            target_id, gripper_name=self.gripper_name
        )
        if not self._grasp_blockers:
            return blockers

        by_id = {b.object_id: b for b in blockers}
        for entry in self._grasp_blockers:
            oid = entry.get("object_id")
            if oid is None or oid == target_id or oid not in self.registry.instances:
                continue
            b = by_id.get(oid)
            if b is None:
                b = Blocker(
                    object_id=oid,
                    label=entry.get("label", "")
                    or self.registry.instances[oid].label,
                    reasons=[],
                )
                by_id[oid] = b
            if "fouls_grasp" not in b.reasons:
                b.reasons.append("fouls_grasp")
            b.grasps_fouled = int(entry.get("grasps_fouled", 0))
        return sorted(by_id.values(), key=lambda b: -b.severity)

    def _diagnose_obstruction(self, object_id: str, pose) -> None:
        """Name what is in the way of `object_id`, and what is in the way of THAT.

        Gated on a geometric fact rather than on a model's opinion: it runs only
        when a specific object is measurably inside the hand's swept volume. A
        grasp that failed because the object is wider than the jaw, or because
        the hand kept slipping, produces no attribution and no cascade — those
        failures are not fixed by moving anything.

        One level deep, deliberately. `blocking_objects` was only ever called on
        the *target*, so a chain — B blocks A, A blocks the target — left B
        invisible: nothing ever asked what was in A's way, and the run simply
        failed to grasp A over and over. One level finds B. Recursing further
        would turn a stuck run into a tour of the table.
        """
        fouled_by = self.registry.colliding_instances(
            pose, gripper_name=self.gripper_name, exclude=(object_id,)
        )
        if not fouled_by:
            # Unreachable, but not because of any one object — nothing to clear.
            self.state.note(
                f"nothing identifiable obstructs {object_id}; the hand does not "
                "fit here for another reason"
            )
            return

        named = ", ".join(f"{oid} ({label})" for oid, label, _ in fouled_by[:3])
        self.state.note(f"{object_id} is obstructed by {named}")

        # And what stands in the way of the worst offender, so the planner can
        # see a two-step path rather than rediscovering the same wall.
        worst = fouled_by[0][0]
        try:
            second = self.registry.blocking_objects(
                worst, gripper_name=self.gripper_name
            )
        except Exception as exc:  # a diagnostic must never take the run down
            logger.debug("second-level blocking analysis failed: %s", exc)
            return
        others = [b for b in second if b.object_id != object_id]
        if others:
            chain = ", ".join(f"{b.object_id} ({b.label})" for b in others[:3])
            self.state.note(
                f"and {worst} is itself blocked by {chain} — clearing {worst} "
                f"may need {chain} moved first"
            )

    async def _grasp_via_agents(self, inst, obs, index, phase: str = "declutter"):
        """Run the inner Planner/Coder/Observer loop for one object."""
        import image_patch as ip
        from perception3d import SceneContext

        self.graspmas.reset()
        ip.set_registry(self.registry)
        try:
            _out, grasp = await self.graspmas.query(
                self._instruction(inst, phase, self.goal),
                obs.rgb,
                depth=obs.depth,
                intrinsics=obs.K,
                gripper_name=self.gripper_name,
            )
        except Exception as exc:  # a generated-code failure must not kill the run
            logger.warning("inner loop raised for %s: %s", inst.id, exc)
            self.state.note(f"inner loop raised: {type(exc).__name__}: {exc}")
            return None
        finally:
            ip.set_registry(None)

        # What the collision filter said stood in the hand's way, from the
        # candidates the sampler actually produced. Only meaningful for the
        # target — a blocker's own blockers are a different question, handled
        # by `_diagnose_obstruction`.
        if phase == "deliver":
            self._grasp_blockers = self._grasp_blockers_from_rounds(
                getattr(self.graspmas, "rounds", None)
            )

        self.state.record_agent_rounds(getattr(self.graspmas, "rounds", None))
        # Recorded whether or not a grasp came back. The failure kind is most
        # informative precisely when there is no grasp — and `wrong_object` is
        # the one the inner loop cannot fix by construction, since the id was
        # chosen out here. Gating this on success threw it away exactly when it
        # mattered.
        self.state.record_observer(getattr(self.graspmas, "observation_json", {}) or {})
        return grasp

    @staticmethod
    def _instruction(inst, phase: str = "declutter", goal: str = "") -> str:
        """What to ask the inner loop for. The instance id has to be in the words.

        `find_by_id` exists, the registry is installed before the call, and the
        Coder prompt documents both — but asking for "the mug" gives the Coder
        no id to pass it, so it falls back to `find("mug")`. That re-runs
        detection and indexes into a list whose order is not stable between
        iterations, which is the exact ambiguity instance ids were introduced to
        remove: with two bottles on the table, `find("bottle")[0]` can mean a
        different physical object on each iteration.

        Naming the id costs nothing when labels happen to be unique, and is the
        difference between clearing the right bottle and the wrong one when they
        are not.

        The id alone was all the inner loop ever got, which left it unable to
        tell the two jobs apart. Clearing an obstruction wants the most secure
        hold available; fetching what the person asked for is governed by what
        they asked. On a knife those are opposite grips, so the phase and the
        person's own words are stated here rather than inferred.
        """
        label = inst.descriptor or inst.label
        who = f"{inst.id} ({label})" if label else inst.id

        if phase == "deliver":
            asked = f' The person asked: "{goal}".' if goal else ""
            return (
                f"Grasp {who} so it can be handed to the person.{asked}"
                " Their request governs how it should be held."
            )
        return (
            f"Grasp {who} in order to move it out of the way."
            " It is an obstruction, not what the person asked for, so take"
            " whichever hold is most secure — grasp a tool by its handle."
        )

    def _reachable(self, pose, object_id: str) -> bool:
        """Is the hand clear of everything except the object it is grasping?"""
        import collision as col

        scene = self.registry.scene_cloud_excluding(object_id)
        pts = col.load_gripper_points(self.gripper_name)
        return col.sweep_is_clear(pose, scene, pts, approach_len=0.10)

    # -- plumbing ----------------------------------------------------------

    def _apply_plan_update(self, decision: Decision, index: int) -> None:
        update = decision.grand_plan_update
        if not update:
            return
        reason = str(update.pop("reason", ""))
        if self.state.amend_grand_plan(update, reason, iteration=index):
            self.state.note(f"grand plan amended: {reason}")
        else:
            self.state.note(f"grand plan amendment rejected: {self.state.last_refusal}")

    def _save_scene(self, obs, filename: str) -> Optional[Path]:
        """Write one scene photograph into the current iteration's directory."""
        if self.recorder is None or not getattr(self.recorder, "enabled", False):
            return None
        try:
            import cv2

            path = Path(self.recorder.images_dir) / filename
            cv2.imwrite(str(path), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
            return path
        except Exception as exc:  # pragma: no cover - artifacts are optional
            logger.debug("could not write %s: %s", filename, exc)
            return None

    def _image_b64(self, obs) -> Optional[str]:
        """Encode the scene for the planner's prompt. Written by `_perceive`."""
        if self.recorder is None or not getattr(self.recorder, "enabled", False):
            return None
        path = Path(self.recorder.images_dir) / f"scene_iter{obs.iteration}_before.png"
        if not path.is_file():
            path = self._save_scene(obs, f"scene_iter{obs.iteration}_before.png")
        if path is None:
            return None
        try:
            from agents.observer import encode_image

            return encode_image(str(path))
        except Exception as exc:  # pragma: no cover - vision is optional
            logger.debug("could not encode the scene image: %s", exc)
            return None

    def _log(self, message: str) -> None:
        logger.info(message)
        if self.recorder is not None:
            self.recorder.log(message)

    def _abort(self, reason, index, moved) -> DeclutterResult:
        self.state.finish("aborted", {"reason": reason, "moved": moved})
        self._log(f"aborted: {reason}")
        return DeclutterResult("aborted", reason, None, index + 1, moved,
                               str(self.state.run_dir))

    def _fail(self, reason, index, moved) -> DeclutterResult:
        self.state.finish("failed", {"reason": reason, "moved": moved})
        self._log(f"failed: {reason}")
        return DeclutterResult("failed", reason, None, index, moved,
                               str(self.state.run_dir))


def _snapshot(registry: SceneRegistry) -> SceneRegistry:
    """A shallow copy of the registry, frozen for before/after comparison.

    The evaluator needs the state *before* the move, and `update_from_*` mutates
    in place so ids survive. Copying the instance dict is enough: instances are
    replaced wholesale on each update rather than edited.
    """
    import copy

    frozen = SceneRegistry(match_radius_m=registry.match_radius_m)
    frozen.instances = dict(registry.instances)
    frozen.plane = registry.plane
    frozen.hmap = registry.hmap
    frozen.iteration = registry.iteration
    frozen._depth = registry._depth
    frozen._K = registry._K
    return frozen


def _gripper_geometry(name: str):
    from scene_registry import _gripper_geometry as g

    return g(name)
