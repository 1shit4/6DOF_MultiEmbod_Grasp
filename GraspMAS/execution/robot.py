"""The real-hardware backend: a contract, not an implementation.

Nothing here runs. It exists so the interface a real arm has to satisfy is
written down while the reasoning is fresh, rather than reconstructed later from
whatever the simulator happened to do.

**No grasp produced by this repository has been executed on a robot.** That
statement is in `SUMMARY.md` and stays true; this file does not change it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ExecutionReport, Observation, PickPlacePlan


class RobotExecutor:
    """What a real arm must provide. Every method raises.

    Implementing this means supplying four things, in rough order of difficulty:

    1. **A calibrated RGB-D capture.** `capture()` must return depth in metres
       and the intrinsics of the camera that took it. Everything downstream
       assumes the camera frame, so if the camera is on the wrist rather than
       fixed, the poses this repo returns are in *that* frame at *that* instant
       — either capture from a repeatable pose or apply the extrinsic yourself.

    2. **Motion through the plan's waypoints.** `plan.waypoints` gives six poses
       in order: `pre_grasp, grasp, lift, pre_place, place, retreat`. Lift and
       descend along the table normal, not the gripper's own axis — a side grasp
       retreating along its -Z drags the object across the table.
       These are gripper *base* poses; the fingertips are `fingertip_depth`
       further along +Z (0.1034 m for a Panda). Confusing the two puts the hand
       10 cm into the table.

    3. **Honest reporting.** `ExecutionReport.status` must reflect what happened.
       `stage_reached` should name the last waypoint actually achieved, so a
       failure localises. If the gripper reports its own width after closing,
       compare it against the object and set `status="failed"` when the jaw shut
       on nothing — that is the single most useful signal a real gripper offers
       and the evaluator cannot recover it from images.

    4. **`reset()`**, or an honest refusal. There is no undo on real hardware; a
       robot backend may raise `NotImplementedError` here rather than pretend.

    Collision-free *motion planning* between waypoints is deliberately out of
    scope. This repo decides where to grasp and where to release, and checks the
    hand is clear at those poses and along the approach. Getting the arm between
    them without hitting anything is a motion planner's job.
    """

    backend = "robot"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RobotExecutor is a documented contract, not an implementation. "
            "See this class's docstring for what a real backend must provide."
        )

    def capture(self, iteration: int = 0) -> Observation:  # pragma: no cover
        raise NotImplementedError

    def execute_pick_place(self, plan: PickPlacePlan) -> ExecutionReport:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError
