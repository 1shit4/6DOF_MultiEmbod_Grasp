# Long-horizon decluttering

Reference for the outer loop: its contract, its state files, and what a real
robot backend has to provide. Rationale and measurements live in
[`CLAUDE.md`](../CLAUDE.md) §7; this is the interface.

---

## The loop

```
                    ┌──────────────────────────────────────────┐
                    │              DeclutterLoop               │
                    └──────────────────────────────────────────┘
 capture ──▶ SceneRegistry ──▶ evaluate last move ──▶ decide next
    ▲            (ids,             (geometric,          (1 LLM call,
    │          blocking)            no LLM)             or scripted)
    │                                                        │
    └──── executor ◀──── plan_place ◀──── grasp ◀────────────┘
```

One iteration, in order:

| step | module | LLM |
|---|---|---|
| capture RGB-D | `execution` | — |
| build/refresh instance ids, fit the plane, analyse blocking | `scene_registry` | — |
| work out what the goal refers to (once, if no `--target`) | `agents/task_planner` | 1 call |
| judge the previous move | `evaluator` | only if `unknown` |
| choose the next object | `agents/task_planner` | 1 call |
| find a grasp | `agents/graspmas` (inner loop) or `nominal_grasp` | 3/round |
| find a place | `placement` | — |
| execute | `execution` | — |
| record | `session_state` | — |

**Termination**, checked in this order:

1. planner says `grasp_target` and a grasp survives collision filtering → `success`
2. planner says `abort`, or validation forces it → `aborted`
3. no progress for 2 iterations → `failed`, with a diagnosis when the reports show one
   — **unless** an unused ranked alternative target remains and the retarget budget
   is unspent, in which case the run gets exactly one further iteration to switch
4. iteration cap → `failed`

`retarget` and `defer` are not terminal. A `defer` is a decision declined on
*timing* rather than possibility — a retarget asked for too early. The iteration
falls back to the geometric choice and does **real work**, carrying the refusal
into the next decision.

Deferred and retarget iterations are skipped by `iterations_without_progress()`,
which both stall detection and the retarget gate read. That matters: counting a
declined iteration let **a refusal supply the very evidence justifying the next
request**, so one real failure plus one declined ask cleared a bar meant to need
two failures.

Progress means *the target stopped being blocked*, not *the action succeeded* —
those come apart in both directions, which is the whole point of the evaluator
returning two independent verdicts.

It is also asked of the **target**, not of the object the loop chose to move.
An iteration counts as progress if the object acted on cleared, **or** if the
target's blocker set shrank — whoever shrank it. Progress the loop did not plan
is still progress: under the `wrong_object` fault a hand that closed on the
wrong thing carried away a real blocker and took the target from 32% visible to
78%, and a check that only asked about the chosen object called that "no
progress" and gave up.

When a run does stall, `SessionState.stall_diagnosis` names the pattern behind
it, because "no progress in 2 iterations" is true of every stalled run and
actionable for none. It distinguishes a repeated failure at the same stage (the
grasp, not the choice of object), the hand reaching a different object than
planned (targeting or calibration), no placement (the table is full), and no
grasp (unreachable or poorly seen). Every claim states the count it rests on,
and a pattern covering fewer than half the attempts produces no claim at all.

---

## Running it

```bash
# The whole machinery, no API key, no server, no GPU
conda run -n graspmas python scripts/verify_declutter.py

# One run, scripted planner
cd GraspMAS && python main_declutter.py --goal "pick up the banana" \
    --target banana --no-llm

# Watch it recover from a fault: injected on the FIRST pick only
python main_declutter.py --target banana --no-llm --inject drop

# A fault that never lifts — a gripper that is simply broken
python main_declutter.py --target banana --no-llm --inject drop --inject-at every

# With the agent loop
export LLM_API_KEY=...          # or several, comma-separated; see the key pool
python main_declutter.py --goal "pick up the banana" --target banana --max-round 2

# Abstract goal: omit --target and the planner works out what is meant
python main_declutter.py --goal "i am hungry" --scenario occluded_target
python main_declutter.py --goal "i need something to cut" --scenario affordance_table

# Two cutters, one blocked — exercises the retarget path
python main_declutter.py --goal "i need something to cut" \
    --scenario affordance_choice --inject drop --inject-at every
```

**Choosing the target.** With `--target` the label is matched against detections
exactly as before and nothing else changes. Without it:

1. **Stage 1** returns candidates with a `priority` — 1 is best, and **ties are
   meaningful**: equal priority means either object serves equally well. It sees
   the scene and the photo but *not* what is in the way, so suitability is judged
   on its own.
2. **Python** drops candidates that are missing or too sparse, then measures
   blockers for the top 3.
3. **Stage 2** (the grand plan) sees priority and cost together and returns
   `target_order`. Priority leads; within a tier effort decides; across tiers a
   swap needs a large gap and is recorded in `run_notes`. Python validates the
   order is a permutation of the real candidates — it can reorder, never invent.

Cost: **one extra request per run**. `--no-llm` requires `--target`, because
nothing offline can read a goal like "something to cut".

> **Risk: a label is not an object.** Suitability is decided once, from detector
> labels, and validated only for *existence* — never for sensibleness. Everything
> downstream works on an instance id and treats the choice as settled, so a
> confident wrong label propagates unchecked. Measured on real photographs
> (`outputs/reports/target_selection_real_photos.md`): 20/20 correct, and it did
> reject a "bottle" the image showed to be a screwdriver — a capability observed,
> not a guarantee. Synthetic scenarios cannot test it at all, because they hand
> the planner ground-truth labels.

**Changing the target.** The goal is the person's words and never changes. The
target is the system's *inference* about them, so it may be revised — through the
`retarget` action, never through an amendment. It requires an unused ranked
alternative, two iterations of no progress, and a budget of one; it archives the
old grand plan rather than editing it, with the reason beside it. `progress.json`
carries `target_choice` (the ranking and the reading that produced it) and
`retargets` (every switch, with before, after and why).

**LLM budget.** One outer planner call per iteration, plus the inner loop's
3 per round. At `--max-round 2` a three-object declutter is roughly 25-30
requests. The evaluator is geometric and normally spends nothing.

**Free-tier quota is per model and is the binding constraint.** Measured
2026-08-20: `gemini-3.5-flash` gives 5 requests/minute and **20 per day**, which
one run exhausts. `llm_config.yaml` therefore orders `model_candidates` by
quota rather than capability, with the flash-lite family first. Check what a key
can actually reach with `python -m agents.llm --probe`.

**Two provider settings are load-bearing**, both in `llm_config.yaml`:
`min_max_tokens` (thinking tokens are drawn from the same budget as the reply —
without a floor every structured answer truncates mid-JSON) and
`reasoning_effort`. See `CLAUDE.md` §6.

**The loop survives losing the LLM entirely.** An unreachable outer planner
aborts with a stated reason and a resumable `progress.json`; an unreachable
inner loop falls back to a geometric top-down grasp. Verified by exhausting a
daily quota mid-run: the iteration in flight still completed its pick and place
using the fallback grasp.

---

## State files

Both live in the run directory and are written atomically. **Python owns them;
agents never write them directly** — they return structured JSON which
`session_state` validates and applies.

### `progress.json`

```jsonc
{
  "schema_version": 1,
  "goal": "pick up the banana",
  "target": {"id": "obj_001", "label": "banana"},
  "status": "in_progress | success | failed | aborted",
  "iterations": [
    {
      "index": 0,
      "action": "remove",              // remove | grasp_target | retarget | abort
                                       //   (plus "defer": declined, run continues)
      "object_id": "obj_003",
      "subgoal": "pick up obj_003 and place it clear of obj_001",
      "rationale": "worst blocker: occlusion",
      "blockers": [ {"object_id": "...", "reasons": ["occlusion"], ...} ],
      "planned":    {"grasp": {...Grasp6D...}, "place": {...PlacePose...}},
      "observer":   {"verdict": "VALID", "checklist": {...}},
      "execution":  {"status": "ok", "stage_reached": "retreat",
                     "grasped_object": "mug", "disturbed": []},
      "evaluation": {"action_succeeded": "success",
                     "still_blocking_target": false,
                     "displacement_cm": 29.0, "place_error_cm": 2.0,
                     "collateral": [], "source": "geometric"},
      "notes": ["planner corrected: ..."],
      "started_at": "...", "ended_at": "..."
    }
  ],
  "outcome": {"grasp": {...}, "moved": ["obj_003", "obj_002"]}
}
```

### `grand_plan.json`

```jsonc
{
  "goal": "pick up the banana",          // immutable, always
  "target": {"id": "obj_001", ...},      // immutable here; changed only by retarget,
                                         //   which supersedes the whole plan
  "removal_order": [ {"object_id": "obj_002", "label": "bottle",
                      "reason": "hides most of the target"} ],
  "success_criterion": "nothing blocks the target and it is fully visible",
  "revisions": [ {"iteration": 2, "changed": ["removal_order"],
                  "before": {...}, "after": {...}, "reason": "..."} ]
}
```

`amend_grand_plan` refuses an edit that touches `goal` or `target`, that carries
no reason, that names an unknown field, or that would be the ninth revision.
Refusals are recorded in the iteration's `notes`, so a planner that keeps
needing correction is visible rather than silently patched over.

### Reading a finished run

```bash
conda run -n graspmas python scripts/summarize_run.py <run_dir>   # one-screen digest
conda run -n graspmas python scripts/build_report.py  <run_dir>   # full narrative
conda run -n graspmas python scripts/build_report.py --all --index
```

`build_report.py` writes `<run_dir>/report.md`: per iteration, the task
planner's decision and blocking analysis, the inner loop's reasoning round by
round (thought, plan, generated code, grasp, overlay, observer verdict), the
scene before and after the executor, the evaluator's verdict, and a link to the
state files as they stood at that moment. `--index` additionally writes
`outputs/reports/index.md` mapping each run to its scenario and injected
failure.

---

## Object identity

The planner names **instances** (`obj_003`), never labels. Ids are carried by 3D
position across iterations. Two things make that work:

* Objects that did not move are matched within `DEFAULT_MATCH_RADIUS_M` (8 cm).
* Objects the loop *deliberately moved* are matched against where it put them,
  via `expected_moves` — a pick-and-place travels 20-30 cm, far outside the
  static radius. The hint is **additive**: the old position is still checked,
  because the move may have failed.

Generated code should use `find_by_id("obj_003")` rather than `find("bottle")`
whenever the plan names an id.

---

## Adding an executor backend

Implement three methods (`execution/base.py`):

```python
class MyExecutor:
    def capture(self, iteration: int = 0) -> Observation: ...
    def execute_pick_place(self, plan: PickPlacePlan) -> ExecutionReport: ...
    def reset(self) -> None: ...
```

`RobotExecutor` in `execution/robot.py` documents what a real arm has to supply.
The four things that matter:

1. **Depth in metres, with the intrinsics of the camera that took it.**
   Everything downstream is in that camera's frame.
2. **Move through `plan.waypoints`** — `pre_grasp, grasp, lift, pre_place,
   place, retreat`. Lift and descend along the **table normal**, not the
   gripper's own axis: a side grasp retreating along its −Z drags the object
   sideways across the table.
3. **These are gripper *base* poses.** The fingertips are `fingertip_depth`
   further along +Z (0.1034 m for a Panda). Confusing the two puts the hand
   10 cm into the table.
4. **Report what happened, not what was asked.** An executor that echoes the
   plan back as success makes the evaluator meaningless. If the gripper reports
   its width after closing, use it — a jaw that shut on nothing is the single
   most useful signal a real gripper offers, and no camera can recover it.

Motion planning *between* waypoints is out of scope. This repo decides where to
grasp and where to release, and checks the hand is clear at those poses and
along the approach.

---

## Failure injection

`--inject` names the fault; `--inject-at` says **when** it fires, defaulting to
the first pick-and-place only.

That default is the whole point. A fault on *every* attempt makes the task
impossible by construction — no object the planner intends to move can ever
move — so such a run can only ever demonstrate that the loop gives up. A
one-shot fault asks the question worth asking: does the system notice what went
wrong, and does it reach the goal anyway?

Measured on `occluded_target` with the scripted planner, one fault each:

| injected | outcome | iterations | how it recovered |
|---|---|---|---|
| `drop` | success | 4 | verdict `not_moved`, retried the same object, succeeded |
| `wrong_object` | success | 4 | `not_moved` plus the bystander reported gone; retried |
| `offset` | success | 3 | released 8 cm off plan, still cleared, not retried |
| `collateral` | success | 3 | intended move fine; the knock is in the executor's report |
| `tip` | success | 3 | position correct, so geometry sees success (see Limits) |

A clean run takes 3 iterations, so recovery costs at most one.

`--inject-at every` is still a legitimate scenario — a gripper that is simply
broken. There the correct outcome is failure, and the loop reaches it in 2
iterations with a diagnosis rather than running to the cap.

## Limits

* **Execution is scene mutation, not physics.** Objects are teleported and the
  scene re-rendered. Failures are injected (`offset`, `drop`, `tip`,
  `collateral`, `wrong_object`), so the evaluator is only ever tested against
  failures somebody wrote down. A CPU physics backend (PyBullet) is viable and
  deferred.
* **Collateral detection only sees near-fully-visible objects.** `evaluate`
  gates it at 95% visibility, because an object's perceived centroid moves when
  it is *uncovered* — measured: a banana that nobody touched appeared to shift
  3.8 cm while 81% visible. The gate stops that phantom, and in doing so also
  misses real collateral on anything partly hidden: a box genuinely nudged
  3.3 cm went unreported at 82% visible. The executor's `disturbed` field caught
  it, but a real robot cannot supply that. Loosening the gate trades a missed
  nudge for a false alarm on the target itself, which is the worse error.
* **No grasp here has been executed on a robot.**
* **A heavily occluded object's footprint is wrong**, and its centroid is not
  even stable. Both are gated on `visibility`; see `CLAUDE.md` §7.4.
* **Placement is 2.5D.** Objects are placed on the support surface, never
  stacked, never inside containers.
* The loop clears obstructions **one at a time**. It does not plan a joint
  rearrangement, and will not find a solution that requires moving two objects
  simultaneously or swapping their positions.
