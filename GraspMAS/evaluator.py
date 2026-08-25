"""Did the last move actually work, and does it matter?

Every question worth asking after a pick-and-place is measurable, so this is
arithmetic rather than an LLM call:

* **Did the object move?**            distance between its centroid before and after
* **Did it land where intended?**     distance from that centroid to the planned place position
* **Is it still in the way?**         re-run the blocking test on the new scene

Doing it geometrically is not only cheaper. An LLM asked to grade its own
previous decision *and* choose the next one has an incentive to find that the
last one worked, which is exactly the premature-completion failure a long-horizon
loop has to be defended against. Arithmetic has no such incentive.

**Two verdicts, not one.** `action_succeeded` and `still_blocking_target` are
computed independently and both reported, because they come apart in both
directions and the planner needs the second one, not the first:

* the gripper slipped and the object only got nudged — the action *failed*, but
  it is no longer in the way, so **nothing needs redoing**;
* the object went exactly where it was told — the action *succeeded*, and it is
  still blocking the target, so **the plan was wrong**.

**Where geometry cannot see.** Position is observable; state is not. An object
that toppled barely moves its centroid. Re-identification can quietly match the
wrong one of two identical bottles. And an occluded object's centroid is not
even a stable quantity — uncovering one shifts its apparent centre by centimetres
without it moving (measured at 2.3 cm), which is why every comparison here is
gated on visibility. When the evidence does not support a verdict this returns
`unknown` rather than guessing, and `unknown` is the only case that spends an
LLM call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from scene_registry import SceneRegistry

logger = logging.getLogger(__name__)

# Below this the object has not moved: within depth noise and centroid jitter
# for an object seen from a fixed camera.
MOVED_THRESH_M = 0.03

# Landing this close to the planned spot counts as success. Generous next to the
# 3 cm placement margin, because the point is "it ended up where we meant", not
# "it ended up to the millimetre".
PLACE_TOLERANCE_M = 0.06

# Centroids of objects less visible than this are not compared at all — see the
# module docstring. Matches `scene_registry.RELIABLE_VISIBILITY`.
MIN_COMPARABLE_VISIBILITY = 0.95

VERDICTS = (
    "success",  # moved, and landed where the plan said
    "not_moved",  # still where it started: the grasp failed
    "moved_off_target",  # moved, but not to the planned spot
    "object_missing",  # no longer detected at all
    "unknown",  # the evidence does not support any of the above
)


@dataclass
class Evaluation:
    """What the measurements support, and nothing more."""

    action_succeeded: str = "unknown"
    still_blocking_target: Optional[bool] = None
    displacement_m: Optional[float] = None
    place_error_m: Optional[float] = None
    collateral: List[Tuple[str, float]] = field(default_factory=list)
    evidence: str = ""
    source: str = "geometric"
    target_visibility: Optional[float] = None
    # Every object still blocking the target, not just the one that was acted
    # on. Progress toward the goal can come from anywhere — a hand that closed
    # on the wrong object can still carry a real blocker away — and a run that
    # only asks about the object it chose cannot see that it is winning.
    target_blockers: Optional[List[str]] = None
    needs_review: bool = False

    def __post_init__(self):
        if self.action_succeeded not in VERDICTS:
            raise ValueError(
                f"verdict must be one of {VERDICTS}; got {self.action_succeeded!r}"
            )

    @property
    def helped(self) -> bool:
        """Whether the goal advanced, regardless of whether the action worked."""
        return self.still_blocking_target is False

    def describe(self) -> dict:
        return {
            "action_succeeded": self.action_succeeded,
            "still_blocking_target": self.still_blocking_target,
            "displacement_cm": (
                round(self.displacement_m * 100, 1)
                if self.displacement_m is not None else None
            ),
            "place_error_cm": (
                round(self.place_error_m * 100, 1)
                if self.place_error_m is not None else None
            ),
            "collateral": [
                [oid, round(d * 100, 1) if np.isfinite(d) else None]
                for oid, d in self.collateral
            ],
            "target_visibility": (
                round(self.target_visibility, 2)
                if self.target_visibility is not None else None
            ),
            "target_blockers": self.target_blockers,
            "evidence": self.evidence,
            "source": self.source,
            "needs_review": self.needs_review,
        }


def evaluate(
    before: SceneRegistry,
    after: SceneRegistry,
    object_id: str,
    target_id: str,
    intended_place_xy: Optional[np.ndarray] = None,
    execution_report=None,
    moved_thresh_m: float = MOVED_THRESH_M,
    place_tolerance_m: float = PLACE_TOLERANCE_M,
) -> Evaluation:
    """Compare two registries and report what the last action achieved.

    `before` and `after` must be *separate* registry snapshots; passing the same
    object updated in place gives a comparison of a thing with itself.
    """
    ev = Evaluation()
    notes: List[str] = []

    # --- did it move? -----------------------------------------------------
    prior = before.instances.get(object_id)
    now = after.instances.get(object_id)

    if prior is None:
        ev.action_succeeded = "unknown"
        notes.append(f"{object_id} was not in the scene before the action")
        ev.needs_review = True
    elif now is None:
        ev.action_succeeded = "object_missing"
        notes.append(
            f"{object_id} is no longer detected — it may be off the table, "
            "out of frame, or hidden by something else"
        )
        ev.needs_review = True
    else:
        ev.displacement_m = float(
            np.linalg.norm(now.centroid_table[:2] - prior.centroid_table[:2])
        )
        comparable = (
            prior.visibility >= MIN_COMPARABLE_VISIBILITY
            and now.visibility >= MIN_COMPARABLE_VISIBILITY
        )

        if intended_place_xy is not None:
            ev.place_error_m = float(
                np.linalg.norm(
                    now.centroid_table[:2] - np.asarray(intended_place_xy)[:2]
                )
            )

        if not comparable:
            ev.action_succeeded = "unknown"
            ev.needs_review = True
            notes.append(
                f"{object_id} was {prior.visibility:.0%} visible before and "
                f"{now.visibility:.0%} after; its centroid is the mean of whatever "
                "was in view, so the measured displacement is not trustworthy"
            )
        elif ev.displacement_m < moved_thresh_m:
            ev.action_succeeded = "not_moved"
            notes.append(
                f"{object_id} moved only {ev.displacement_m*100:.1f} cm — the grasp "
                "did not take hold"
            )
        elif ev.place_error_m is None:
            ev.action_succeeded = "moved_off_target"
            notes.append(
                f"{object_id} moved {ev.displacement_m*100:.0f} cm, but no intended "
                "position was recorded to compare against"
            )
        elif ev.place_error_m <= place_tolerance_m:
            ev.action_succeeded = "success"
            notes.append(
                f"{object_id} moved {ev.displacement_m*100:.0f} cm and landed "
                f"{ev.place_error_m*100:.1f} cm from the planned spot"
            )
        else:
            ev.action_succeeded = "moved_off_target"
            notes.append(
                f"{object_id} moved {ev.displacement_m*100:.0f} cm but landed "
                f"{ev.place_error_m*100:.0f} cm from where it was meant to go"
            )

    # An executor that reported a failure outranks a geometric guess: it knows
    # things the camera cannot see, such as whether the jaw closed on anything.
    if execution_report is not None:
        status = getattr(execution_report, "status", None)
        error = getattr(execution_report, "error", None)
        if status == "failed" and ev.action_succeeded == "success":
            ev.action_succeeded = "unknown"
            ev.needs_review = True
            notes.append(
                f"the executor reported failure ({error}) but the object appears "
                "to have moved as planned; these disagree"
            )
        elif error:
            notes.append(f"executor: {error}")

    # --- does it still matter? -------------------------------------------
    # Recomputed from scratch, deliberately independent of the verdict above.
    target = after.instances.get(target_id)
    if target is None:
        notes.append(f"target {target_id} is not visible in the new scene")
        ev.needs_review = True
    else:
        ev.target_visibility = target.visibility
        blockers = {b.object_id for b in after.blocking_objects(target_id)}
        ev.target_blockers = sorted(blockers)
        ev.still_blocking_target = object_id in blockers
        if ev.still_blocking_target:
            notes.append(f"{object_id} is still in the way of {target_id}")
        else:
            notes.append(f"{object_id} no longer obstructs {target_id}")
        if not blockers:
            notes.append(f"nothing now blocks {target_id}")

    # --- did anything else move? -----------------------------------------
    ev.collateral = [
        (oid, dist)
        for oid, dist in after.moved_since(
            before.positions(),
            thresh_m=moved_thresh_m,
            min_visibility=MIN_COMPARABLE_VISIBILITY,
            previous_visibility=before.visibilities(),
        )
        if oid != object_id
    ]

    # An object shifted far enough is re-registered under a new id, so it drops
    # out of `moved_since` entirely — it is not "an object that moved", it is
    # one id gone and another appeared. Measured: with the hand closing on the
    # wrong object, the bystander it carried away went completely unreported.
    # A bystander that vanished is at least as alarming as one that shifted, so
    # it is reported with an unknown distance rather than not at all.
    vanished = set(before.instances) - set(after.instances) - {object_id}
    for oid in sorted(vanished):
        ev.collateral.append((oid, float("nan")))
        notes.append(
            f"{oid} was in the scene before this action and is not now — it was "
            "moved far enough to lose its identity, or it left the table"
        )

    if ev.collateral:
        listed = ", ".join(
            f"{oid} by {d*100:.0f} cm" if np.isfinite(d) else f"{oid} (gone)"
            for oid, d in ev.collateral
        )
        notes.append(f"other objects moved: {listed}")
        ev.needs_review = True

    ev.evidence = "; ".join(notes)
    return ev


# ---------------------------------------------------------------------------
# Optional VLM fallback
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """
**Role**: You are checking whether a robot's pick-and-place actually worked.

You are shown two photographs of the same tabletop: BEFORE the robot acted, and
AFTER. You are told what it was trying to do. Geometry could not settle the
question, so the specific doubt is stated below.

Answer only from what you can see. If the two images do not let you tell, say so
— "unclear" is a useful answer here and a wrong guess is not.

--- WHAT THE ROBOT TRIED TO DO ---
{intent}

--- WHY THIS NEEDS A HUMAN-LIKE LOOK ---
{doubt}

--- WHAT WAS MEASURED ---
{measurements}

--- OUTPUT FORMAT ---
Wrap the answer in <review> ... </review> as JSON, and output nothing else.

<review>
{{
  "object_moved": "yes|no|unclear",
  "landed_upright": "yes|no|unclear",
  "target_now_reachable": "yes|no|unclear",
  "anything_else_disturbed": "yes|no|unclear",
  "summary": "one or two sentences describing what changed between the images"
}}
</review>
"""


async def review_with_vlm(
    evaluation: Evaluation,
    before_image: str,
    after_image: str,
    intent: str,
    llm,
) -> Evaluation:
    """Ask a VLM about a case geometry could not settle. Returns a new Evaluation.

    Fires only when `evaluation.needs_review` is set, which in a healthy run is
    never — so the loop's LLM cost does not depend on it. It answers the
    questions geometry structurally cannot: whether an object toppled, and
    whether the scene changed in a way no position comparison would show.

    The result never *overrides* a measurement. It can only move a verdict to or
    from `unknown`, and it is recorded with `source="vlm_review"` so the
    provenance of every verdict stays visible in `progress.json`.
    """
    from agents.llm import extract_json, extract_tag
    from agents.observer import encode_image

    prompt = REVIEW_PROMPT.format(
        intent=intent,
        doubt=evaluation.evidence or "the measurements were inconclusive",
        measurements=str(evaluation.describe()),
    )

    try:
        raw = await llm.chat_with_image(
            llm.system_prompt,
            prompt,
            encode_image(after_image),
            agent="evaluator",
            max_tokens=600,
        )
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("VLM review failed: %s", exc)
        evaluation.evidence += f"; VLM review unavailable ({exc})"
        return evaluation

    parsed = extract_json(extract_tag(raw, "review") or raw) or {}
    if not parsed:
        evaluation.evidence += "; VLM review returned nothing parseable"
        return evaluation

    evaluation.source = "vlm_review"
    moved = str(parsed.get("object_moved", "unclear")).lower()
    if evaluation.action_succeeded == "unknown":
        if moved == "yes":
            evaluation.action_succeeded = "moved_off_target"
        elif moved == "no":
            evaluation.action_succeeded = "not_moved"

    reachable = str(parsed.get("target_now_reachable", "unclear")).lower()
    if evaluation.still_blocking_target is None and reachable in ("yes", "no"):
        evaluation.still_blocking_target = reachable == "no"

    if str(parsed.get("landed_upright", "unclear")).lower() == "no":
        evaluation.evidence += "; VLM: the object is not upright"
    if summary := parsed.get("summary"):
        evaluation.evidence += f"; VLM: {summary}"
    return evaluation
