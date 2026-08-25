#!/usr/bin/env python
"""Clear a cluttered table until the target object can be grasped.

The long-horizon counterpart to `main_simple.py`: instead of one grasp on a
reachable object, it repeatedly picks up and sets aside whatever is in the way.

    # No LLM at all — the machinery end to end, on a synthetic scene
    python main_declutter.py --goal "pick up the banana" --target banana --no-llm

    # With the agent loop
    export LLM_API_KEY=...
    python main_declutter.py --goal "pick up the banana" --target banana

    # Watch it cope with things going wrong
    python main_declutter.py --target banana --no-llm --inject offset,collateral

Exit codes: 0 success, 1 the target was not reached, 2 no API key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import synth_scene as ss
from declutter import DeclutterLoop
from execution import INJECTABLE, MutationExecutor
from run_artifacts import NullRecorder, RunRecorder
from session_state import SessionState

logger = logging.getLogger("declutter")


def parse(argv=None):
    p = argparse.ArgumentParser(
        description="Clear obstructions until a target object can be grasped.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--goal", default="pick up the target object",
                   help="what the run is for, in the user's words. May be "
                        "abstract (\"I need something to cut\"); the task "
                        "planner works out which object that means")
    p.add_argument("--target", default=None,
                   help="the object to reach, matched against detected labels. "
                        "Omit it to have the task planner infer the target from "
                        "--goal and the scene. Required with --no-llm, which has "
                        "nothing that can read an abstract goal")

    scene = p.add_argument_group("scene")
    scene.add_argument("--scenario", default="occluded_target", choices=sorted(ss.SCENARIOS),
                       help="which synthetic tabletop to clear")
    scene.add_argument("--height", type=int, default=480)
    scene.add_argument("--width", type=int, default=640)

    ex = p.add_argument_group("execution")
    ex.add_argument("--inject", default="",
                    help=f"comma-separated failure modes to inject: {', '.join(INJECTABLE)}")
    ex.add_argument("--inject-at", default="0", metavar="N|every",
                    help="which pick-and-place the fault fires on, counting from 0. "
                         "A one-shot fault asks whether the loop notices and recovers; "
                         "'every' makes the task impossible and only tests that it "
                         "gives up")
    ex.add_argument("--inject-offset-m", type=float, default=0.08, metavar="M",
                    help="how far the 'offset' fault misses by, in metres")
    ex.add_argument("--inject-offset-dir", default="random",
                    choices=("random", "short"),
                    help="'random' slips sideways and usually still clears the "
                         "target; 'short' releases early, along the path, so the "
                         "object lands between the camera and the target and the "
                         "move succeeds while the target stays blocked")
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--gripper-name", default="franka_panda")

    loop = p.add_argument_group("loop")
    loop.add_argument("--max-iterations", type=int, default=6,
                      help="hard cap on pick-and-place attempts")
    loop.add_argument("--max-round", type=int, default=2,
                      help="inner Planner/Coder/Observer rounds per grasp")
    loop.add_argument("--no-llm", action="store_true",
                      help="scripted planner and geometric grasps; spends no requests")
    loop.add_argument("--api-file", default="api.key")
    loop.add_argument("--llm-config", default=None)

    out = p.add_argument_group("output")
    out.add_argument("--run-name", default=None)
    out.add_argument("--resume", default=None, metavar="RUN_DIR",
                    help="continue a run from its progress.json")
    out.add_argument("--no-artifacts", action="store_true")
    out.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    # Inferring the target *is* an LLM job — it is the one step that needs to
    # know what an object is for. Saying so here beats failing several seconds
    # later inside the loop with a less legible message.
    if args.no_llm and not args.target:
        p.error("--no-llm needs an explicit --target: there is nothing without "
                "an LLM that can work out which object a goal refers to")
    return args


async def _run(args) -> int:
    inject = [m.strip() for m in args.inject.split(",") if m.strip()]
    at = str(args.inject_at).strip().lower()
    inject_at = None if at in ("every", "all", "always") else int(at)
    executor = MutationExecutor(
        ss.SCENARIOS[args.scenario](),
        height=args.height,
        width=args.width,
        inject=inject,
        inject_at=inject_at,
        offset_m=args.inject_offset_m,
        offset_dir=args.inject_offset_dir,
        seed=args.seed,
    )

    recorder = (
        NullRecorder()
        if args.no_artifacts
        else RunRecorder(name=args.run_name or f"declutter_{args.target or 'inferred'}")
    )
    if not args.no_artifacts:
        recorder.write_config(vars(args), extra={"scenario": args.scenario})

    run_dir = Path(args.resume) if args.resume else Path(
        recorder.dir if getattr(recorder, "enabled", False) else "outputs/runs/declutter"
    )
    state = (
        SessionState.resume(run_dir) if args.resume else SessionState(run_dir)
    )

    planner = None
    graspmas = None
    if not args.no_llm:
        from agents.graspmas import GraspMAS
        from agents.llm import get_shared_llm
        from agents.task_planner import TaskPlanner

        llm = get_shared_llm(
            config_path=args.llm_config, api_file=args.api_file, recorder=recorder
        )
        planner = TaskPlanner(llm)
        graspmas = GraspMAS(
            api_file=args.api_file,
            max_round=args.max_round,
            gripper_name=args.gripper_name,
            recorder=recorder,
            llm_config=args.llm_config,
        )

    loop = DeclutterLoop(
        executor=executor,
        state=state,
        planner=planner,
        graspmas=graspmas,
        gripper_name=args.gripper_name,
        max_iterations=args.max_iterations,
        recorder=recorder if getattr(recorder, "enabled", False) else None,
    )

    result = await loop.run(args.goal, args.target)

    print("=" * 66)
    print(f"  {result.status.upper()}: {result.reason}")
    print(f"  iterations: {result.iterations}   moved: {result.moved or 'nothing'}")
    if result.grasp:
        pos = result.grasp.get("position", [])
        print(f"  grasp at ({', '.join(f'{v:+.3f}' for v in pos)}) m, "
              f"score {result.grasp.get('score', 0):.3f}")
    print(f"  state: {state.run_dir}")
    print("=" * 66)

    if getattr(recorder, "enabled", False):
        recorder.finish(result=result.describe(), status=result.status)

    return 0 if result.status == "success" else 1


def main(argv=None) -> int:
    args = parse(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("declutter").setLevel(logging.INFO)

    try:
        return asyncio.run(_run(args))
    except RuntimeError as exc:
        if "api" in str(exc).lower() or "key" in str(exc).lower():
            print(f"error: {exc}", file=sys.stderr)
            print(
                "\nGet a free key at https://aistudio.google.com/apikey, then\n"
                "  export LLM_API_KEY=...\n"
                "or run with --no-llm to exercise the machinery without one.",
                file=sys.stderr,
            )
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
