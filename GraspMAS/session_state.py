"""The two files that stop a long-horizon run from forgetting or bluffing.

A multi-round agent loop has no memory beyond what you hand it. GraspMAS's inner
loop already shows the failure: it carries exactly two strings between rounds
(`self.plan` and the Observer's `summary`), and it never clears them, so
`main_batch.py` leaks one sample's plan into the next. Over a dozen iterations of
picking things up, that is not survivable — an LLM with no durable record of what
it has already done will redo it, or decide it must be finished by now.

So two files, both owned by Python:

* **`progress.json`** — what was attempted each iteration and what actually
  happened. Written by the loop, read by the planner. This is the history.
* **`grand_plan.json`** — the goal, the target instance, and a hypothesised
  removal order. Written once at the start, amended only with a stated reason.
  This is the anchor.

**The LLM never writes either file.** Agents return structured JSON; this module
validates it and applies it. That is the whole point: a model that can rewrite
its own success criterion has no success criterion, and one that can silently
edit its history can talk itself into being done. `amend_grand_plan` refuses to
touch `goal` or `target` at all, and records every accepted change with the
reason that justified it.

Writes are atomic (temp file + `os.replace`) because the alternative is a run
that dies mid-write and resumes from a truncated file it cannot parse.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

PROGRESS_FILE = "progress.json"
PLANNER_STATE_FILE = "planner_state.json"
GRAND_PLAN_FILE = "grand_plan.json"

# Fields of the grand plan no agent may change. The goal is the user's, not the
# model's, and the target is what the whole run is for.
IMMUTABLE_PLAN_FIELDS = ("goal", "target", "created_at", "schema_version")

# A plan revised more often than this is not being guided by evidence, it is
# oscillating. The loop treats hitting the cap as a reason to stop and say so.
MAX_REVISIONS = 8

#: How often the run may change *what it is fetching*. One, deliberately.
#: The goal is the person's words and never moves; the target is the system's
#: own inference about what those words meant, so it may be revised once on
#: evidence. More than that and the run is shopping for an easy object rather
#: than converging on the right one.
MAX_RETARGETS = 1

#: Iterations that decide something but move nothing. They are neither progress
#: nor evidence of its absence, so anything counting "iterations that got
#: nowhere" has to skip them.
NON_ACTING_ACTIONS = ("defer", "retarget")

TERMINAL_STATUSES = ("success", "failed", "aborted")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj):
    """Numpy-safe encoding, matching `run_artifacts.JsonNumpyEncoder`."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write via a temp file in the same directory, then rename.

    `os.replace` is atomic within a filesystem, so a reader either sees the old
    file or the new one, never a half-written one. A run killed mid-write must
    still be resumable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """One pass of the outer loop: what was planned, done, and observed."""

    index: int
    subgoal: str = ""
    action: str = ""  # remove | grasp_target | abort
    object_id: Optional[str] = None
    object_label: Optional[str] = None
    rationale: str = ""
    blockers: List[dict] = field(default_factory=list)
    planned: Dict[str, Any] = field(default_factory=dict)  # grasp / place
    observer: Dict[str, Any] = field(default_factory=dict)  # verdict / checklist
    # The inner Planner/Coder/Observer loop, round by round: what each agent
    # thought, what it produced, and what the Observer made of it. Kept so a
    # run can be explained after the fact rather than only summarised.
    agent_rounds: List[dict] = field(default_factory=list)
    # The outer task planner's own decision payload, including any grand-plan
    # amendment it proposed and any correction Python applied to it.
    decision: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "subgoal": self.subgoal,
            "action": self.action,
            "object_id": self.object_id,
            "object_label": self.object_label,
            "rationale": self.rationale,
            "blockers": self.blockers,
            "planned": self.planned,
            "observer": self.observer,
            "agent_rounds": self.agent_rounds,
            "decision": self.decision,
            "execution": self.execution,
            "evaluation": self.evaluation,
            "artifacts": self.artifacts,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IterationRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def succeeded(self) -> bool:
        return self.evaluation.get("action_succeeded") == "success"

    @property
    def still_blocking(self) -> Optional[bool]:
        return self.evaluation.get("still_blocking_target")


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))[:40]


def _normalise_removal_order(order) -> Optional[list]:
    """Coerce a planner-supplied removal order into the shape the rest expects.

    Guarding *which* fields an agent may change is not enough — the value has to
    be well formed too. A live planner amended `removal_order` to
    `["obj_002"]`, a list of plain strings rather than of objects. Nothing
    rejected it, so the state file was written that way, and the next
    `planner_context()` called `.get("object_id")` on a string and raised
    `AttributeError` on every subsequent iteration. The run then aborted with
    "the task planner could not be reached", which is not what happened: an
    agent had poisoned its own state file and Python let it.

    A bare id is an obvious intent, so it is upgraded rather than refused.
    Anything that is neither a mapping nor a string is refused, which is what
    `None` signals.
    """
    if not isinstance(order, list):
        return None
    out = []
    for item in order:
        if isinstance(item, dict):
            if not item.get("object_id"):
                return None
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({"object_id": item.strip()})
        else:
            return None
    return out


def _require_removal_order(order) -> list:
    """`_normalise_removal_order`, but raising — for the one-shot initial plan."""
    normalised = _normalise_removal_order(list(order))
    if normalised is None:
        raise ValueError(
            "removal_order must be a list of steps like "
            '{"object_id": "obj_003", "reason": "..."}'
        )
    return normalised


class SessionState:
    """Owns `progress.json` and `grand_plan.json` for one run."""

    def __init__(self, run_dir, autosave: bool = True):
        self.run_dir = Path(run_dir)
        self.autosave = bool(autosave)
        self.progress: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "goal": "",
            "target": None,
            "status": "not_started",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "iterations": [],
            "outcome": None,
        }
        self.grand_plan: Optional[Dict[str, Any]] = None
        self._current: Optional[IterationRecord] = None

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        goal: str,
        target: Optional[dict] = None,
        target_choice: Optional[dict] = None,
    ) -> None:
        self.progress["goal"] = str(goal)
        self.progress["target"] = target
        self.progress["status"] = "in_progress"
        self.progress["created_at"] = _utc_now()
        # How the target was arrived at, and what else would have served. Kept
        # so a retarget can advance the ranking without another LLM call, and so
        # a questionable choice can be traced to the reading that produced it.
        self.progress["target_choice"] = target_choice
        self.progress["retargets"] = []
        self._touch()

    # -- the target, and changing it ---------------------------------------

    @property
    def target_id(self) -> Optional[str]:
        target = self.progress.get("target") or {}
        return target.get("id")

    @property
    def target_candidates(self) -> List[str]:
        """Ranked instance ids the goal could refer to, best first."""
        choice = self.progress.get("target_choice") or {}
        return [
            c["object_id"] for c in choice.get("candidates", [])
            if isinstance(c, dict) and c.get("object_id")
        ]

    @property
    def retargets_used(self) -> int:
        return len(self.progress.get("retargets") or [])

    def can_retarget(self) -> bool:
        return self.retargets_used < MAX_RETARGETS

    def iterations_without_progress(self) -> int:
        """How many trailing iterations achieved nothing toward the target.

        The same question `is_stalled` asks, counted rather than thresholded, so
        the retarget gate and the stall check cannot disagree about whether a
        target is working.

        Counting *failed attempts on the target* instead — the obvious reading —
        measures the wrong thing: a target is usually defeated without ever
        being touched, by iteration after iteration failing to clear the things
        in front of it. Measured on `affordance_table` with a permanently broken
        gripper, the knife accumulated **zero** attempts of its own while the run
        went nowhere, so a gate counting attempts could never fire.
        """
        n = 0
        for rec in reversed(self.iterations):
            if rec.action in NON_ACTING_ACTIONS:
                # A retarget or a declined decision moved no object, so it is
                # not evidence that the target is unreachable — it is only
                # evidence that a decision was made. Counting it let a refusal
                # supply the very evidence that would justify the next request,
                # so a single real failure plus one declined ask was enough to
                # clear a threshold meant to need two failures.
                continue
            if not self._made_progress(rec):
                n += 1
            else:
                break
        return n

    def _made_progress(self, rec) -> bool:
        """Did this one iteration move the run toward the target?"""
        if rec.still_blocking is False:
            return True
        blockers = (rec.evaluation or {}).get("target_blockers")
        if blockers is None:
            return False
        history = self.iterations
        i = history.index(rec)
        previous = next(
            (
                (history[j].evaluation or {}).get("target_blockers")
                for j in range(i - 1, -1, -1)
                if (history[j].evaluation or {}).get("target_blockers") is not None
            ),
            None,
        )
        return previous is not None and len(blockers) < len(previous)

    def retarget(self, new_target: dict, reason: str, iteration: int = -1) -> bool:
        """Switch what the run is fetching, once, with the reason recorded.

        Deliberately *not* an amendment. `amend_grand_plan` refuses `target`
        outright and continues to; a retarget supersedes the whole plan instead,
        archiving the old one, so the record shows two plans with a stated reason
        between them rather than a criterion quietly edited underneath a run.
        """
        self.last_refusal = None

        reason = (reason or "").strip()
        if not reason:
            self.last_refusal = "a retarget must state why; none given"
            return False
        if not isinstance(new_target, dict) or not new_target.get("id"):
            self.last_refusal = "a retarget needs the new target's instance id"
            return False
        if not self.can_retarget():
            self.last_refusal = (
                f"refused: the target has already been changed {self.retargets_used} "
                f"time(s); the limit is {MAX_RETARGETS}"
            )
            return False
        if new_target["id"] == self.target_id:
            self.last_refusal = "refused: that is already the target"
            return False

        before = dict(self.progress.get("target") or {})
        self.progress.setdefault("retargets", []).append(
            {
                "iteration": iteration,
                "at": _utc_now(),
                "from": before,
                "to": dict(new_target),
                "reason": reason,
            }
        )
        self.progress["target"] = dict(new_target)

        if self.grand_plan is not None:
            superseded = dict(self.grand_plan)
            superseded["superseded_at"] = _utc_now()
            superseded["superseded_because"] = reason
            archive = list(self.grand_plan.get("superseded") or [])
            archive.append(superseded)
            # A fresh plan for the new target, carrying the archive forward.
            self.grand_plan = None
            self.set_grand_plan([], success_criterion="", reasoning=reason)
            self.grand_plan["superseded"] = archive

        self._touch()
        return True

    def set_grand_plan(
        self,
        removal_order: Sequence[dict],
        success_criterion: str = "",
        reasoning: str = "",
    ) -> dict:
        """Fix the run's anchor. Callable once; later changes go through amend."""
        if self.grand_plan is not None:
            raise RuntimeError(
                "grand plan already set; use amend_grand_plan so the change is "
                "recorded, or retarget() if the target itself has changed"
            )
        self.grand_plan = {
            "schema_version": SCHEMA_VERSION,
            "goal": self.progress["goal"],
            "target": self.progress["target"],
            # Same normalisation as `amend_grand_plan`: a bare id is upgraded to
            # a step, anything else raises here rather than being written out to
            # break a later render.
            "removal_order": _require_removal_order(removal_order),
            "success_criterion": success_criterion,
            "reasoning": reasoning,
            "created_at": _utc_now(),
            "revisions": [],
        }
        self._touch()
        return self.grand_plan

    def amend_grand_plan(self, edit: dict, reason: str, iteration: int = -1) -> bool:
        """Apply a planner-proposed change, or refuse it and say why.

        Refuses when: there is no plan yet, no reason is given, the edit touches
        an immutable field, or the revision budget is spent. Returns whether it
        was applied; the caller surfaces `last_refusal` to the planner so the
        next decision is made knowing the edit did not happen.
        """
        self.last_refusal: Optional[str] = None

        if self.grand_plan is None:
            self.last_refusal = "no grand plan exists yet"
            return False

        reason = (reason or "").strip()
        if not reason:
            self.last_refusal = "an amendment must state why; none given"
            return False

        if not isinstance(edit, dict) or not edit:
            self.last_refusal = "amendment was empty"
            return False

        touched = [k for k in edit if k in IMMUTABLE_PLAN_FIELDS]
        if touched:
            self.last_refusal = (
                f"refused: {', '.join(sorted(touched))} cannot be changed. "
                "The goal and the target are the run's definition of success."
            )
            logger.warning("grand plan amendment refused: %s", self.last_refusal)
            return False

        unknown = [k for k in edit if k not in self.grand_plan]
        if unknown:
            self.last_refusal = f"refused: unknown field(s) {', '.join(sorted(unknown))}"
            return False

        if "removal_order" in edit:
            normalised = _normalise_removal_order(edit["removal_order"])
            if normalised is None:
                self.last_refusal = (
                    "refused: removal_order must be a list of objects like "
                    '{"object_id": "obj_003", "reason": "..."}'
                )
                logger.warning("grand plan amendment refused: %s", self.last_refusal)
                return False
            edit = {**edit, "removal_order": normalised}

        if len(self.grand_plan["revisions"]) >= MAX_REVISIONS:
            self.last_refusal = (
                f"refused: already revised {MAX_REVISIONS} times; the plan is "
                "oscillating rather than converging"
            )
            logger.warning("grand plan amendment refused: %s", self.last_refusal)
            return False

        before = {k: self.grand_plan.get(k) for k in edit}
        self.grand_plan.update(edit)
        self.grand_plan["revisions"].append(
            {
                "iteration": int(iteration),
                "at": _utc_now(),
                "changed": sorted(edit),
                "before": before,
                "after": {k: self.grand_plan.get(k) for k in edit},
                "reason": reason,
            }
        )
        self._touch()
        return True

    @property
    def revisions_left(self) -> int:
        if self.grand_plan is None:
            return MAX_REVISIONS
        return max(MAX_REVISIONS - len(self.grand_plan["revisions"]), 0)

    # -- iterations --------------------------------------------------------

    def begin_iteration(
        self,
        index: int,
        subgoal: str = "",
        action: str = "",
        object_id: Optional[str] = None,
        object_label: Optional[str] = None,
        rationale: str = "",
        blockers: Optional[Sequence[dict]] = None,
    ) -> IterationRecord:
        if self._current is not None:
            raise RuntimeError(
                f"iteration {self._current.index} is still open; call end_iteration first"
            )
        self._current = IterationRecord(
            index=int(index),
            subgoal=subgoal,
            action=action,
            object_id=object_id,
            object_label=object_label,
            rationale=rationale,
            blockers=[dict(b) for b in (blockers or [])],
            started_at=_utc_now(),
        )
        return self._current

    @property
    def current(self) -> IterationRecord:
        if self._current is None:
            raise RuntimeError("no iteration is open; call begin_iteration first")
        return self._current

    @staticmethod
    def _slim_grasp(grasp: Optional[dict]) -> Optional[dict]:
        """A grasp as it should be *stored*, without its alternatives.

        `grasp_detection` returns up to 20 candidates so that a rejection can be
        answered with a different pose. They are consumed from an in-memory
        registry, never read back from here, and the recorder already writes all
        ~416 of them to a `.npz` — in a better form than JSON. Keeping a copy in
        the state file costs ~3 KB per grasp, duplicated per round, for nothing.
        """
        if not isinstance(grasp, dict) or "candidates" not in grasp:
            return grasp
        slim = {k: v for k, v in grasp.items() if k != "candidates"}
        slim["n_candidates_available"] = len(grasp.get("candidates") or [])
        return slim

    def record_plan(self, grasp: Optional[dict], place: Optional[dict]) -> None:
        self.current.planned = {"grasp": self._slim_grasp(grasp), "place": place}

    def record_observer(self, observation: Optional[dict]) -> None:
        self.current.observer = dict(observation or {})

    def record_agent_rounds(self, rounds: Optional[Sequence[dict]]) -> None:
        """The inner loop's reasoning for this iteration, round by round."""
        self.current.agent_rounds = [
            {**r, "grasp": self._slim_grasp(r.get("grasp"))} if "grasp" in r else dict(r)
            for r in (rounds or [])
        ]

    def record_decision(self, decision: Optional[dict]) -> None:
        """What the outer task planner decided, and why."""
        self.current.decision = dict(decision or {})

    def record_execution(self, report: Any) -> None:
        self.current.execution = report if isinstance(report, dict) else _as_dict(report)

    def record_evaluation(self, evaluation: Any) -> None:
        self.current.evaluation = (
            evaluation if isinstance(evaluation, dict) else _as_dict(evaluation)
        )

    def record_artifacts(self, **paths: str) -> None:
        self.current.artifacts.update({k: str(v) for k, v in paths.items() if v})

    def note(self, message: str) -> None:
        self.current.notes.append(str(message))

    def note_run(self, message: str) -> None:
        """A note about the run rather than about one iteration.

        Needed because the interesting decisions happen *between* iterations —
        before `begin_iteration` or after `end_iteration` — where `note()` raises.
        That has now caught out two guards (see CLAUDE.md §7.11 on the outage
        guard, and the stall deferral in `declutter.run`), so there is somewhere
        legitimate to put them.
        """
        self.progress.setdefault("run_notes", []).append(str(message))
        self._touch()

    def end_iteration(self) -> IterationRecord:
        rec = self.current
        rec.ended_at = _utc_now()
        self.progress["iterations"].append(rec.to_dict())
        self._current = None
        self._touch()
        self.snapshot(f"iter{rec.index}")
        return rec

    def snapshot(self, label: str) -> Optional[Path]:
        """Freeze both state files as they stand, under `states/<label>/`.

        `progress.json` is rewritten in place every iteration, so the finished
        file shows the end state and nothing else. A snapshot per iteration is
        what makes "the plan as it stood when this decision was taken" a thing
        you can read rather than reconstruct — including a grand plan that was
        later amended away.
        """
        if not self.autosave:
            return None
        out = self.run_dir / "states" / _safe_label(label)
        out.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out / PROGRESS_FILE, self.progress)
        if self.grand_plan is not None:
            _atomic_write_json(out / GRAND_PLAN_FILE, self.grand_plan)
        return out

    def finish(self, status: str, outcome: Optional[dict] = None) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"status must be one of {TERMINAL_STATUSES}; got {status!r}")
        if self._current is not None:
            self.note(f"run finished ({status}) with this iteration still open")
            self.end_iteration()
        self.progress["status"] = status
        self.progress["outcome"] = outcome
        self._touch()

    # -- queries -----------------------------------------------------------

    @property
    def iterations(self) -> List[IterationRecord]:
        return [IterationRecord.from_dict(d) for d in self.progress["iterations"]]

    @property
    def next_index(self) -> int:
        return len(self.progress["iterations"])

    def has_acted(self) -> bool:
        """Has this run actually tried to move anything yet?

        The bar a retarget has to clear. It replaces a count of iterations
        without progress, which conflated two things: a target that has
        genuinely resisted effort, and a target nobody has lifted a finger for.
        Only the first is evidence.

        Deliberately low. It rules out the failure that was *measured* — the
        model switching at iteration 0 because the alternative was "fully
        visible and unobstructed, making it a more efficient and immediate
        solution", i.e. because it was easier — without demanding that a run
        grind through two futile iterations before it may say so.
        """
        return any(
            rec.action not in NON_ACTING_ACTIONS and rec.object_id
            for rec in self.iterations
        )

    def attempts_on(self, object_id: str) -> List[IterationRecord]:
        return [r for r in self.iterations if r.object_id == object_id]

    def moved_objects(self) -> List[str]:
        """Ids the loop believes it successfully moved."""
        return [r.object_id for r in self.iterations if r.succeeded and r.object_id]

    def is_stalled(self, window: int = 2) -> bool:
        """True when the last `window` iterations achieved nothing.

        Not "the action failed" — an action can fail and still help, and can
        succeed and still leave the target blocked. The question is whether the
        run is making progress **toward the goal**.

        That question is asked of the *target*, not of the object the loop
        happened to choose. Asking only "did the object I picked stop blocking?"
        misses progress the loop did not plan, and that is not hypothetical:
        under the `wrong_object` fault the hand closed on a real blocker by
        accident and carried it away, taking the target from 32% visible with
        two blockers to 78% with one — and the run declared "no progress" and
        stopped, because neither *chosen* object had moved. Measured on
        `occluded_target`; see `test_session_state.py::TestStallDetection`.

        So an iteration counts as progress if the object acted on stopped
        blocking, **or** if the target ended up with fewer blockers than it had
        the iteration before, whoever moved them.
        """
        if len(self.iterations) < window:
            return False
        return self.iterations_without_progress() >= window

    def stall_diagnosis(self, window: int = 3) -> str:
        """Name the pattern behind a stall, when the reports show one.

        "No progress in 2 iterations" is true of every stalled run and useful
        for none of them. The executor already reports which stage it reached,
        what went wrong, and which object it *actually* grasped — and those
        distinguish faults needing completely different responses: a gripper
        that drops everything is not an arm that reaches for the wrong object,
        and neither is a table with nowhere left to put things.

        Every claim here carries the count it is based on. An earlier version
        wrote "in the last 3 attempts" from a single matching error, because
        filtering empties before `all()` makes it vacuously true — an overclaim
        in exactly the place a person is trying to diagnose a fault.

        Returns "" when no pattern covers at least half the attempts; inventing
        a story from inconsistent evidence is worse than admitting there is none.
        """
        recent = [r for r in self.iterations[-window:] if r.action == "remove"]
        n = len(recent)
        if n < 2:
            return ""

        reports = [r.execution or {} for r in recent]

        def phrase(count: str) -> str:
            return f"{count} of the last {n} attempts"

        # Every attempt dying at the same stage: the grasp, not the choice.
        failed = [r for r in reports if r.get("status") == "failed"]
        stages = {r.get("stage_reached") for r in failed if r.get("stage_reached")}
        if len(failed) == n and len(stages) == 1:
            return (
                f"all {n} of the last attempts failed at the same stage "
                f"({stages.pop()}) — this looks like the grasp itself rather than "
                f"the choice of object, so moving something else will not help"
            )

        # The hand reaching the wrong object: targeting, not planning.
        mismatched = [
            r for r, rec in zip(reports, recent)
            if r.get("grasped_object") and rec.object_label
            and r["grasped_object"] != rec.object_label
        ]
        if len(mismatched) * 2 >= n:
            got = ", ".join(str(r["grasped_object"]) for r in mismatched)
            return (
                f"the hand closed on a different object than planned in "
                f"{phrase(str(len(mismatched)))} (got: {got}) — the grasps were "
                f"computed for the right objects, so this points at targeting or "
                f"calibration rather than at the plan"
            )

        errors = [(r.get("error") or "").lower() for r in reports]
        no_place = sum("placement" in e for e in errors)
        if no_place * 2 >= n:
            return (
                f"no placement could be found in {phrase(str(no_place))} — the "
                f"table has nowhere left to put things that clears the target"
            )
        no_grasp = sum("no grasp" in e for e in errors)
        if no_grasp * 2 >= n:
            return (
                f"no grasp could be computed in {phrase(str(no_grasp))} — the "
                f"objects may be unreachable, or too poorly seen to grasp"
            )
        return ""

    def plan_steps(self) -> List[dict]:
        """The grand plan's steps, each with what became of it.

        The plan and the history were two disconnected lists: `removal_order`
        carried no status, so "attempted twice and failed" was neither done nor
        skipped nor pending, and the planner had to re-derive its own position
        by joining the two in-context on every call. That join is arithmetic; it
        belongs here.
        """
        order = (self.grand_plan or {}).get("removal_order") or []
        outcome: Dict[str, tuple] = {}
        for rec in self.iterations:
            if rec.action != "remove" or not rec.object_id:
                continue
            got = (rec.evaluation or {}).get("action_succeeded")
            # Later iterations supersede earlier ones for the same object.
            outcome[rec.object_id] = (got, rec.index)

        steps, pending_seen = [], False
        for step in order:
            if not isinstance(step, dict):
                continue
            oid = step.get("object_id")
            got, at = outcome.get(oid, (None, None))
            if got == "success":
                status = "done"
            elif got is not None:
                status = "attempted"
            elif not pending_seen:
                status, pending_seen = "next", True
            else:
                status = "pending"
            steps.append({
                "object_id": oid,
                "label": step.get("label", ""),
                "status": status,
                "iteration": at,
            })
        return steps

    def off_plan_actions(self, within=None) -> List[dict]:
        """Iterations that acted on something the grand plan never named.

        The geometric fallback does this routinely, and nothing marked it — so
        the planner saw an object move that its plan does not mention and had no
        way to tell that from a step it had forgotten.

        Scoped to the same window as the history it accompanies: an unbounded
        list of off-plan notes for iterations too old to be shown is noise, and
        the digest earns its keep by being short.
        """
        named = {
            s.get("object_id")
            for s in ((self.grand_plan or {}).get("removal_order") or [])
            if isinstance(s, dict)
        }
        records = self.iterations if within is None else within
        return [
            {"iteration": rec.index, "object_id": rec.object_id}
            for rec in records
            if rec.action == "remove" and rec.object_id and rec.object_id not in named
        ]

    def planner_state(self, max_iterations: int = 6) -> dict:
        """Exactly what the task planner is told, as data.

        Written to `planner_state.json` beside the archive, so the boundary
        between "what the system runs on" and "what we keep to explain a run"
        is a file you can open rather than a function you have to call. What is
        *not* here — poses, waypoints, generated code, agent transcripts,
        timestamps — is in `progress.json` and is for us, not for the planner.
        """
        history = []
        shown = self.iterations[-max_iterations:]
        for rec in shown:
            ev = rec.evaluation or {}
            outcome = ev.get("action_succeeded", "not evaluated")
            entry = {
                "iteration": rec.index,
                "action": rec.action or "n/a",
                "object_id": rec.object_id,
                "object_label": rec.object_label,
                "outcome": outcome,
                "still_blocking_target": ev.get("still_blocking_target"),
                "evidence": ev.get("evidence"),
                "notes": list(rec.notes),
            }
            if outcome not in ("success", "not evaluated"):
                detail = self._grasp_difficulty(rec)
                if detail:
                    entry["grasp_difficulty"] = detail
            history.append(entry)

        steps = self.plan_steps()
        return {
            "goal": self.progress.get("goal"),
            "target": self.progress.get("target"),
            "grand_plan": {
                "success_criterion": (self.grand_plan or {}).get("success_criterion", ""),
                "steps": steps,
                "done": sum(1 for s in steps if s["status"] == "done"),
                "total": len(steps),
                "revisions": [
                    {"iteration": r.get("iteration"), "changed": r.get("changed"),
                     "reason": r.get("reason")}
                    for r in ((self.grand_plan or {}).get("revisions") or [])
                ],
                "revisions_left": self.revisions_left,
            },
            "off_plan_actions": self.off_plan_actions(within=shown),
            "iterations_omitted": max(len(self.iterations) - len(shown), 0),
            "history": history,
            "retargets_used": self.retargets_used,
            "target_candidates": list(self.target_candidates),
        }

    def planner_context(self, max_iterations: int = 6) -> str:
        """A compact digest of the run so far, for the planner prompt.

        Rendered from `planner_state()` so the prompt and `planner_state.json`
        cannot drift: whatever the file says is what the planner was told.

        Deliberately terse — the planner needs what was tried and what came of
        it, not a transcript. But the *selection* used to be wrong: it kept the
        evaluator's verdict and dropped both the grasp score and where the run
        had got to in its own plan, so a planner could not tell "the grasp was
        sound and the hand slipped" from "we never found a sound grasp", nor
        which step it was on.
        """
        st = self.planner_state(max_iterations=max_iterations)
        plan = st["grand_plan"]
        lines = []

        if plan["total"]:
            lines.append(
                f"GRAND PLAN — {plan['done']} of {plan['total']} steps done"
            )
            mark = {"done": "[x]", "attempted": "[!]", "next": "->", "pending": "[ ]"}
            for step in plan["steps"]:
                who = f"{step['object_id']}"
                if step["label"]:
                    who += f" ({step['label']})"
                tail = {
                    "done": f" — done, iteration {step['iteration']}",
                    "attempted": f" — attempted at iteration {step['iteration']}, not cleared",
                    "next": " — next",
                    "pending": "",
                }[step["status"]]
                lines.append(f"  {mark[step['status']]} {who}{tail}")
        elif self.grand_plan is None:
            lines.append("GRAND PLAN — not set yet")
        else:
            # Set, then revised down to nothing. Saying "not set yet" here reads
            # as "you have not planned", when what happened is "your plan is
            # complete or was amended away" — opposite advice.
            lines.append(
                "GRAND PLAN — no removal steps remain (the plan was revised "
                "to empty, or nothing needed moving)"
            )

        if plan["success_criterion"]:
            lines.append(f"  done when: {plan['success_criterion']}")
        for rev in plan["revisions"]:
            lines.append(
                f"  revised at iteration {rev['iteration']}: "
                f"{', '.join(rev['changed'] or [])} — {rev['reason']}"
            )
        if plan["total"] or plan["revisions"]:
            lines.append(f"  revisions left: {plan['revisions_left']}")

        for off in st["off_plan_actions"]:
            lines.append(
                f"  ! iteration {off['iteration']} acted on {off['object_id']}, "
                "which is not in the plan"
            )

        lines.append("")
        if not st["history"]:
            lines.append("HISTORY — nothing attempted yet; this is the first iteration.")
            return "\n".join(lines)

        if st["iterations_omitted"]:
            lines.append(
                f"HISTORY — {st['iterations_omitted']} earlier iteration(s) omitted"
            )
        else:
            lines.append("HISTORY")

        for rec in st["history"]:
            blocking = rec["still_blocking_target"]
            blocking_txt = (
                "still blocking" if blocking is True
                else "no longer blocking" if blocking is False
                else "blocking status unknown"
            )
            who = (
                f"{rec['object_id']} ({rec['object_label']})"
                if rec["object_id"] else "-"
            )
            lines.append(
                f"  [{rec['iteration']}] {rec['action']} {who}: "
                f"{rec['outcome']}, {blocking_txt}"
            )
            if rec.get("evidence"):
                lines.append(f"        {rec['evidence']}")
            if rec.get("grasp_difficulty"):
                lines.append(f"        {rec['grasp_difficulty']}")
            for n in rec["notes"]:
                lines.append(f"        note: {n}")
        return "\n".join(lines)

    @staticmethod
    def _grasp_difficulty(rec) -> str:
        """One line on what the inner loop went through to produce this grasp."""
        bits = []
        rounds = rec.agent_rounds or []
        attempts = [r for r in rounds if r.get("grasp")]
        if attempts:
            best = max(
                float((r.get("grasp") or {}).get("score") or 0.0) for r in attempts
            )
            bits.append(f"grasp {best:.2f}")
        elif rounds:
            bits.append("no grasp found")

        obs = rec.observer or {}
        verdict = str(obs.get("verdict", "") or "").upper()
        failure = str(obs.get("failure", "") or "")
        if verdict == "INVALID":
            bits.append(
                f"critic rejected it ({failure})" if failure and failure != "none"
                else "critic rejected it"
            )
        elif verdict:
            bits.append(f"critic: {verdict.lower()}")

        if len(rounds) > 1:
            # Say WHOSE attempts these are. "2 attempts" was read by the planner
            # as two failed *iterations* on the object — which is the trigger
            # for its own two-strikes rule — and it aborted a recoverable run
            # after a single failure. These are rounds inside one iteration.
            bits.append(f"{len(rounds)} grasp attempts within this one iteration")
        return "; ".join(bits)

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        self.progress["updated_at"] = _utc_now()
        _atomic_write_json(self.run_dir / PROGRESS_FILE, self.progress)
        # The system-consumed view, written beside the archive. Two files with
        # two jobs: this one is everything the planner is told and nothing else,
        # so a bad decision can be read against exactly what informed it.
        _atomic_write_json(self.run_dir / PLANNER_STATE_FILE, self.planner_state())
        if self.grand_plan is not None:
            _atomic_write_json(self.run_dir / GRAND_PLAN_FILE, self.grand_plan)

    def _touch(self) -> None:
        if self.autosave:
            self.save()

    @classmethod
    def resume(cls, run_dir, autosave: bool = True) -> "SessionState":
        """Reload a run. Raises if `progress.json` is missing or unreadable.

        An iteration left open by a crash is closed and marked, rather than
        silently dropped — "we were part way through moving obj_3" is exactly
        the thing the next planner call needs to know.
        """
        run_dir = Path(run_dir)
        path = run_dir / PROGRESS_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no {PROGRESS_FILE} in {run_dir}")

        state = cls(run_dir, autosave=False)
        with open(path) as f:
            state.progress = json.load(f)

        version = state.progress.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} is schema version {version}, this code speaks {SCHEMA_VERSION}"
            )
        state.progress.setdefault("iterations", [])

        plan_path = run_dir / GRAND_PLAN_FILE
        if plan_path.is_file():
            with open(plan_path) as f:
                state.grand_plan = json.load(f)
            state.grand_plan.setdefault("revisions", [])

        state.autosave = autosave
        return state

    def describe(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "goal": self.progress["goal"],
            "status": self.progress["status"],
            "iterations": len(self.progress["iterations"]),
            "has_grand_plan": self.grand_plan is not None,
            "revisions_left": self.revisions_left,
        }


def _as_dict(obj: Any) -> dict:
    """Best-effort dict for a dataclass, a `.describe()`-able, or anything else."""
    if obj is None:
        return {}
    if hasattr(obj, "describe") and callable(obj.describe):
        return obj.describe()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    return {"value": str(obj)}
