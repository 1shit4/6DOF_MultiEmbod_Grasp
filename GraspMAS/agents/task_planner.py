"""The outer planner: which object to move next, or whether to stop.

Distinct from `Planner`, which turns one subgoal into steps for the Coder. This
one operates a level up — it never sees code, only the scene and the history,
and it emits a structured decision rather than prose.

It is the only LLM call in the outer loop. The evaluator is geometric, so a
healthy iteration costs one request here plus whatever the inner
Planner/Coder/Observer loop spends.

Every decision it returns is validated against the actual scene before use. A
planner that names an object which does not exist, or tries to remove the target
itself, is corrected in Python rather than trusted — see `TaskPlanner.validate`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.prompt import task_planner_prompt

from .llm import ChatLLM, extract_json, extract_tag

logger = logging.getLogger(__name__)

ACTIONS = ("remove", "grasp_target", "retarget", "abort")

#: Internal only — never emitted by a model, never in `ACTIONS`. Means "that
#: decision was refused, and the run continues", as opposed to `abort`, which is
#: terminal.
#:
#: This survived a deliberate attempt to delete it. Making every refusal
#: terminal is tidier and re-creates the §7.12 defect exactly: a planner that
#: proposes a switch too early — a defensible thing to propose — kills the run.
#: What changed instead is how rarely it fires. The bar used to be two
#: iterations of getting nowhere; it is now simply "this run has acted at all",
#: so the only refusal left is the one that was actually measured: switching at
#: iteration 0 because the alternative looked easier.
DEFER = "defer"

#: Below this an instance is too sparse to fit a footprint to, so it cannot be
#: placed or reasoned about geometrically. Taken from the registry rather than
#: restated, so the two cannot drift apart. Imported lazily inside the function
#: that needs it: importing `scene_registry` at module scope would pull the
#: perception stack into every process that merely wants the prompt constants.
def _min_target_points() -> int:
    from scene_registry import MIN_INSTANCE_POINTS

    return MIN_INSTANCE_POINTS

#: How many ranked alternatives are worth computing a blocking analysis for.
#: Each costs a geometric test, not a request, but a longer list only adds
#: prompt the model has to read.
MAX_CANDIDATES = 3


@dataclass
class TargetCandidate:
    """One object that would satisfy the goal, and how well.

    `priority` is *suitability only* — 1 is best, and ties are meaningful:
    equal priority means either object serves the person equally well, which is
    what licenses picking whichever is cheaper to reach. Effort is measured
    separately and weighed against this, never folded into it.
    """

    object_id: str
    label: str = ""
    why: str = ""
    priority: int = 1
    #: Blockers measured for this candidate, filled in before the grand plan.
    n_blockers: Optional[int] = None

    def describe(self) -> dict:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "priority": self.priority,
            "why": self.why,
            "n_blockers": self.n_blockers,
        }


@dataclass
class TargetChoice:
    """What the goal turned out to refer to, ranked.

    The ranking is the whole point of returning a list: `retarget` advances it
    without another LLM call, so a run that discovers its first choice is
    unreachable does not have to pay to re-decide.
    """

    candidates: List[TargetCandidate] = field(default_factory=list)
    interpretation: str = ""
    confidence: str = "high"
    corrections: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[TargetCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def ids(self) -> List[str]:
        return [c.object_id for c in self.candidates]

    def get(self, object_id: str) -> Optional[TargetCandidate]:
        return next((c for c in self.candidates if c.object_id == object_id), None)

    def reorder(self, order: List[str]) -> None:
        """Apply a final ordering, keeping any candidate the order left out."""
        by_id = {c.object_id: c for c in self.candidates}
        ranked = [by_id[i] for i in order if i in by_id]
        ranked += [c for c in self.candidates if c.object_id not in set(order)]
        self.candidates = ranked

    def describe(self) -> dict:
        return {
            "interpretation": self.interpretation,
            "confidence": self.confidence,
            "candidates": [c.describe() for c in self.candidates],
            "corrections": list(self.corrections),
        }


@dataclass
class Decision:
    """One outer-loop decision, after validation."""

    action: str = "abort"
    object_id: Optional[str] = None
    subgoal: str = ""
    rationale: str = ""
    grand_plan_update: Optional[dict] = None
    corrections: List[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.action in ("grasp_target", "abort")

    @property
    def is_deferred(self) -> bool:
        """Refused, but the run continues — not a model-emitted action."""
        return self.action == DEFER

    def describe(self) -> dict:
        return {
            "action": self.action,
            "object_id": self.object_id,
            "subgoal": self.subgoal,
            "rationale": self.rationale,
            "grand_plan_update": self.grand_plan_update,
            "corrections": list(self.corrections),
        }


def format_candidates(entries) -> str:
    """Render each alternative target with what clearing it would cost.

    `entries` is a sequence of `(TargetCandidate, blocking_text)`. Showing the
    blocking analysis for *every* candidate is what a mid-turn tool call would
    have bought — the model can see that the knife is buried and the scissors
    are not — obtained here for no extra request, because the analysis is
    geometry and geometry is free.

    Priority and cost are put side by side deliberately: the trade the model is
    being asked to make is between them, so it should not have to hold one of
    them in its head while reading the other.
    """
    if not entries:
        return "No candidates were considered."
    lines = []
    for cand, blocking in entries:
        n = cand.n_blockers
        cost = (
            "nothing in the way" if n == 0
            else f"{n} object(s) to move first" if n
            else "cost not measured"
        )
        head = f"- {cand.object_id}"
        if cand.label:
            head += f" ({cand.label})"
        head += f" — priority {cand.priority}, {cost}"
        lines.append(head)
        if cand.why:
            lines.append(f"    suitability: {cand.why}")
        for line in (blocking or "").strip().splitlines():
            lines.append(f"    {line.strip()}")
    return "\n".join(lines)


def format_blocking(blockers, target=None) -> str:
    """Render the blocking analysis for the prompt, in the agent's vocabulary."""
    if not blockers:
        line = "Nothing blocks the target."
        if target is not None:
            line += f" Target visibility {target.visibility:.0%}."
            if not target.footprint_is_reliable:
                line += (
                    " NOTE: the target is still partly hidden, so the geometric"
                    " tests are not yet meaningful — this list may be incomplete."
                )
        return line

    lines = []
    for b in blockers:
        why = []
        if "occlusion" in b.reasons:
            why.append(f"occlusion ({b.occlusion_frac:.0%} of the target's outline)")
        if "fouls_grasp" in b.reasons:
            why.append(
                f"blocks the hand ({b.grasps_fouled} of the best grasps on the "
                "target collide with it)"
            )
        if "contact" in b.reasons:
            why.append(
                f"touching the target (gap {b.gap_m*100:.1f} cm) — may resist the "
                "pick, or topple when the target is lifted away"
            )
        lines.append(f"  {b.object_id} ({b.label}) — {', '.join(why)}")

    if target is not None and not target.footprint_is_reliable:
        lines.append(
            f"  NOTE: the target is only {target.visibility:.0%} visible, so only"
            " occlusion could be measured. More blockers may appear once it is clear."
        )
    return "\n".join(lines)


class TaskPlanner:
    """Chooses the next object to move. One LLM call per outer iteration."""

    def __init__(
        self,
        llm: ChatLLM,
        prompt=task_planner_prompt,
        max_tokens: int = 900,
    ):
        self.llm = llm
        self.prompt = prompt
        self.max_tokens = max_tokens

    # -- what the goal refers to -------------------------------------------

    async def select_target(
        self, goal: str, scene_table: str, image_b64: Optional[str] = None,
    ) -> TargetChoice:
        """Work out which object the request means, ranked best first.

        This is the only place in the system that can read *"I need something to
        cut"*. Everything downstream — the registry, the blocking analysis, the
        placement search — works on an instance id and has no notion of what an
        object is for.

        Degrades rather than raising, like `make_grand_plan`: an unparseable
        reply yields an empty choice, and the caller decides whether that is
        fatal. A run should fail with "nothing here serves that goal", not with
        a JSONDecodeError.
        """
        prompt = self.prompt.SELECT_TARGET.format(goal=goal, scene_table=scene_table)
        raw = await self._ask(prompt, image_b64, agent="task_planner", temperature=0.3)
        parsed = extract_json(extract_tag(raw, "target") or raw) or {}

        choice = TargetChoice(
            interpretation=str(parsed.get("interpretation", "")),
            confidence=str(parsed.get("confidence", "high")).lower(),
        )
        raw_candidates = parsed.get("candidates")
        if not isinstance(raw_candidates, list):
            logger.warning("target selection returned no usable candidate list")
            return choice
        for item in raw_candidates[:MAX_CANDIDATES]:
            if isinstance(item, str):
                # A bare id is a legible-enough answer to accept; refusing it
                # would throw away a correct choice over its packaging.
                choice.candidates.append(TargetCandidate(object_id=item))
            elif isinstance(item, dict) and item.get("object_id"):
                try:
                    priority = max(1, int(item.get("priority", 1)))
                except (TypeError, ValueError):
                    priority = 1
                choice.candidates.append(
                    TargetCandidate(
                        object_id=str(item["object_id"]),
                        label=str(item.get("label", "")),
                        why=str(item.get("why", "")),
                        priority=priority,
                    )
                )
        # Stable sort by priority, so the list is in suitability order before
        # effort is measured. Ties keep the order the model gave them in.
        choice.candidates.sort(key=lambda c: c.priority)
        return choice

    @staticmethod
    def validate_target(choice: TargetChoice, registry) -> TargetChoice:
        """Drop candidates the scene cannot support, keeping the ranking.

        Same contract as `validate`: the model proposes, Python disposes. An id
        that is not in the registry, or an instance too sparse to fit a
        footprint to, is removed here rather than failing later inside the
        placement search where the cause would be unrecognisable.
        """
        out = TargetChoice(
            interpretation=choice.interpretation,
            confidence=choice.confidence,
            corrections=list(choice.corrections),
        )
        seen: set = set()
        for cand in choice.candidates:
            if cand.object_id in seen:
                continue
            seen.add(cand.object_id)
            inst = registry.instances.get(cand.object_id)
            if inst is None:
                out.corrections.append(
                    f"{cand.object_id} is not in the scene "
                    f"(have {sorted(registry.instances)}); dropped"
                )
                continue
            if inst.n_points < _min_target_points():
                out.corrections.append(
                    f"{cand.object_id} has only {inst.n_points} points, too few to "
                    "grasp reliably; dropped"
                )
                continue
            out.candidates.append(
                TargetCandidate(
                    object_id=cand.object_id,
                    label=cand.label or inst.label,
                    why=cand.why,
                    priority=cand.priority,
                    n_blockers=cand.n_blockers,
                )
            )
        return out

    # -- grand plan --------------------------------------------------------

    async def make_grand_plan(
        self, goal: str, target_id: str, scene_table: str, blocking: str,
        image_b64: Optional[str] = None, candidates: str = "",
    ) -> dict:
        """Draft the run's anchor. Degrades to an empty plan rather than failing."""
        prompt = self.prompt.GRAND_PLAN.format(
            goal=goal, target_id=target_id, scene_table=scene_table, blocking=blocking,
            candidates=candidates or "No alternatives were considered.",
        )
        raw = await self._ask(prompt, image_b64, agent="task_planner", temperature=0.4)
        parsed = extract_json(extract_tag(raw, "plan") or raw) or {}

        order = parsed.get("removal_order")
        if not isinstance(order, list):
            logger.warning("grand plan had no usable removal_order; starting empty")
            order = []
        return {
            "removal_order": [s for s in order if isinstance(s, dict)],
            "success_criterion": str(parsed.get("success_criterion", "")),
            "reasoning": str(parsed.get("reasoning", "")),
            # The order the robot should try the candidates in, weighing the
            # priority stage 1 set against the effort measured since. Validated
            # against the candidate list by the caller; never a free choice.
            "target_order": [
                str(i) for i in (parsed.get("target_order") or [])
                if isinstance(i, (str, int))
            ],
        }

    # -- per-iteration decision -------------------------------------------

    async def __call__(
        self,
        goal: str,
        target_id: str,
        scene_table: str,
        blocking: str,
        progress: str,
        image_b64: Optional[str] = None,
    ) -> Decision:
        prompt = self.prompt.TASK_PLAN.format(
            goal=goal,
            target_id=target_id,
            scene_table=scene_table,
            blocking=blocking,
            progress=progress,
            examples=self.prompt.EXAMPLES_TASK_PLANNER,
        )

        raw = await self._ask(prompt, image_b64, agent="task_planner", temperature=0.5)
        parsed = self._parse(raw)

        if parsed is None:
            # One corrective re-prompt. Losing an iteration to a dropped tag is
            # expensive on a rate-limited tier, and models do drop them.
            logger.warning("task planner reply was unparseable; re-prompting once")
            raw = await self._ask(
                prompt
                + "\n\nIMPORTANT: reply with exactly one JSON object wrapped in "
                  "<decision> ... </decision> and nothing else.",
                image_b64,
                agent="task_planner",
                temperature=0.0,
            )
            parsed = self._parse(raw)

        if parsed is None:
            # Aborting is the safe failure. Guessing an object to move is not.
            return Decision(
                action="abort",
                rationale="the task planner's reply could not be parsed twice running",
                corrections=["unparseable reply"],
            )

        return Decision(
            action=str(parsed.get("action", "abort")),
            object_id=parsed.get("object_id") or None,
            subgoal=str(parsed.get("subgoal", "")),
            rationale=str(parsed.get("rationale", "")),
            grand_plan_update=(
                parsed.get("grand_plan_update")
                if isinstance(parsed.get("grand_plan_update"), dict)
                else None
            ),
        )

    async def _ask(self, prompt, image_b64, agent, temperature):
        if image_b64:
            return await self.llm.chat_with_image(
                self.llm.system_prompt, prompt, image_b64,
                agent=agent, temperature=temperature, max_tokens=self.max_tokens,
            )
        return await self.llm.chat(
            self.llm.system_prompt, prompt,
            agent=agent, temperature=temperature, max_tokens=self.max_tokens,
        )

    @staticmethod
    def _parse(raw: str) -> Optional[dict]:
        parsed = extract_json(extract_tag(raw, "decision") or raw or "")
        return parsed if isinstance(parsed, dict) else None

    # -- validation --------------------------------------------------------

    @staticmethod
    def validate(decision: Decision, registry, target_id: str, state=None) -> Decision:
        """Check a decision against the scene, correcting it rather than trusting it.

        Corrections are recorded on the decision and land in `progress.json`, so
        a planner that keeps needing them is visible rather than silently
        patched over. Every branch here fails *safe*: an unusable decision
        becomes `abort`, never a guess at which object to move.
        """
        d = Decision(**{**decision.__dict__, "corrections": list(decision.corrections)})

        if d.action not in ACTIONS:
            d.corrections.append(f"unknown action {d.action!r}; aborting")
            d.action = "abort"
            d.object_id = None
            return d

        if d.action == "retarget":
            return TaskPlanner._validate_retarget(d, registry, target_id, state)

        if d.action != "remove":
            d.object_id = None
            return d

        if not d.object_id:
            d.corrections.append("action was 'remove' with no object named; aborting")
            d.action = "abort"
            return d

        if d.object_id not in registry.instances:
            d.corrections.append(
                f"{d.object_id} is not in the scene "
                f"(have {sorted(registry.instances)}); aborting"
            )
            d.action = "abort"
            d.object_id = None
            return d

        if d.object_id == target_id:
            d.corrections.append(
                "tried to remove the target itself; aborting rather than "
                "moving the thing we were sent to fetch"
            )
            d.action = "abort"
            d.object_id = None
            return d

        # Two failed attempts on one object means the approach is not working.
        # The prompt says so; this enforces it, because a loop that keeps
        # retrying the same impossible grasp never terminates.
        if state is not None:
            attempts = state.attempts_on(d.object_id)
            failed = [a for a in attempts if not a.succeeded]
            if len(failed) >= 2:
                d.corrections.append(
                    f"{d.object_id} has already failed {len(failed)} times; aborting "
                    "rather than repeating it"
                )
                d.action = "abort"
                d.object_id = None
        return d

    @staticmethod
    def _validate_retarget(d: Decision, registry, target_id: str, state) -> Decision:
        """Allow a target switch only on evidence, and only once.

        The goal is the person's words and never moves. The *target* is the
        system's own inference about what those words referred to, so it may be
        revised — but a model that can re-pick its objective whenever the
        current one gets hard has no objective, so every condition here has to
        hold and failure falls back to `abort`, never to a silent switch.
        """
        if state is None:
            d.corrections.append("retarget needs run state; aborting")
            d.action, d.object_id = "abort", None
            return d

        if not state.can_retarget():
            d.corrections.append(
                f"the retarget budget is spent ({state.retargets_used} used); aborting"
            )
            d.action, d.object_id = "abort", None
            return d

        alternatives = [i for i in state.target_candidates if i != target_id]
        if not alternatives:
            d.corrections.append(
                "no alternative target was ever identified — the goal named one "
                "object, so there is nothing to switch to; aborting"
            )
            d.action, d.object_id = "abort", None
            return d

        # Switching before the run has tried anything is the failure this
        # guards, and it is a measured one: on `affordance_choice` the model
        # moved from the knife to the scissors because they were "fully visible
        # and unobstructed, making them a more efficient and immediate
        # solution" — that is, because they were easier, which the prompt
        # forbids. Fetching the wrong thing quickly is a failure, not an
        # efficiency.
        #
        # But the bar is "has this run acted at all", not a count of fruitless
        # iterations. Counting iterations conflated a target that has resisted
        # effort with one nobody has touched, and it made a *planner reasoning
        # correctly* wait: asked to switch after one honest failure, with sound
        # reasons, it used to be refused. Trusting the planner once it has
        # evidence is the point; the guard only stops it choosing on comfort.
        if not state.has_acted():
            # Refused, not fatal. Proposing a switch early is a reasonable thing
            # for a planner to do; dying for it is not, and §7.12 records a run
            # that did exactly that. The iteration proceeds with the current
            # target instead — which both makes progress and produces the
            # evidence that would justify asking again.
            d.corrections.append(
                f"retarget refused: nothing has been attempted yet, so there is "
                f"no evidence against {target_id} — an easier target is not a "
                f"better one. Continuing with the current target"
            )
            d.action, d.object_id = DEFER, None
            return d

        wanted = d.object_id
        if wanted not in alternatives:
            # Take the next ranked candidate rather than refuse: the ranking is
            # ours, computed at selection time, and it is the answer the model
            # was told would be used.
            fallback = alternatives[0]
            d.corrections.append(
                f"retarget named {wanted!r}, which was not a ranked alternative; "
                f"using {fallback} instead"
            )
            wanted = fallback

        if wanted not in registry.instances:
            d.corrections.append(
                f"{wanted} is no longer in the scene; aborting"
            )
            d.action, d.object_id = "abort", None
            return d

        d.object_id = wanted
        return d
