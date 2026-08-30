import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

import image_patch as image_patch_module
from agents.prompt import coder_prompt, observer_prompt, planner_prompt
from collision import gripper_morphology
from perception3d import Grasp6D, SceneContext, approach_elevation_deg
from run_artifacts import NullRecorder
from vis6d import visualize_grasp_6dof, visualize_grasp_candidates

from .coder import Coder
from .llm import get_shared_llm
from .observer import Observer
from .planner import Planner

logger = logging.getLogger(__name__)


class GraspMAS:
    """Planner / Coder / Observer loop producing a 6-DoF grasp.

    The structure is unchanged from the original: the Planner writes a step, the
    Coder turns it into Python against the ImagePatch API, we exec it, and the
    Observer looks at the rendered result and critiques it. What changed is the
    payload — `grasp_detection` now returns a 6-DoF pose dict instead of a
    planar rectangle, so the result gate, the visualization and the Observer's
    input all had to move with it.
    """

    def __init__(
        self,
        api_file: str = "api.key",
        max_round: int = 5,
        gripper_name: str = "franka_panda",
        num_grasps: int = 200,
        planner: str = "graspmoe",
        recorder=None,
        llm_config: Optional[str] = None,
    ):
        self.recorder = recorder or NullRecorder()
        llm = get_shared_llm(
            config_path=llm_config, api_file=api_file, recorder=self.recorder
        )
        self.llm = llm
        self.planner = Planner(planner_prompt, llm)
        self.coder = Coder(coder_prompt, llm)
        self.observer = Observer(observer_prompt, llm)

        self.plan = None
        self.observation = None
        self.observation_json = None
        self.code = None
        self.thought = None
        # Every round, not just the last. The fields above are overwritten each
        # round, so after a query only the final round survived — which is
        # enough to decide what to do next and useless for explaining how the
        # loop got there.
        self.rounds: list[dict] = []
        self.max_round = max_round

        self.gripper_name = gripper_name
        self.num_grasps = num_grasps
        self.grasp_planner = planner

    def reset(self) -> None:
        """Clear the state carried between rounds.

        `query()` does not do this, so a second call inherits the first's plan
        and observation. Harmless for a one-shot run, wrong for anything that
        calls `query` more than once — the decluttering loop calls it per
        object, and `main_batch.py` per sample, where it silently leaks one
        sample's reasoning into the next.
        """
        self.plan = None
        self.observation = None
        self.observation_json = None
        self.code = None
        self.thought = None
        # Every round, not just the last. The fields above are overwritten each
        # round, so after a query only the final round survived — which is
        # enough to decide what to do next and useless for explaining how the
        # loop got there.
        self.rounds: list[dict] = []

    @staticmethod
    def _prepare_image(
        image: Union[str, np.ndarray], save_folder: Union[str, Path] = "imgs/"
    ) -> Tuple[np.ndarray, Path]:
        """Normalize input to an RGB array and save a copy for reference."""
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        image_path = save_folder / "query_image.png"

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise ValueError("image must be str (path) or numpy.ndarray")

        cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img, image_path

    async def query(
        self,
        query: str,
        image: Union[str, np.ndarray],
        save_folder: Union[str, Path] = "imgs/",
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
        gripper_name: Optional[str] = None,
    ) -> Tuple[Any, Optional[dict]]:
        """Run the closed loop; return (final output, best grasp dict or None)."""
        img_np, image_path = self._prepare_image(image, save_folder)

        scene = SceneContext(
            depth=depth,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            image_shape=img_np.shape[:2],
        )
        image_patch_module.configure_scene(
            scene=scene,
            recorder=self.recorder,
            save_folder=str(save_folder),
            gripper_name=gripper_name or self.gripper_name,
            num_grasps=self.num_grasps,
            planner=self.grasp_planner,
        )

        self.recorder.save_inputs(
            image=img_np,
            depth=scene.depth,
            intrinsics=scene.intrinsics,
            meta={"query": query, "gripper": gripper_name or self.gripper_name},
        )
        self.recorder.log(f"query={query!r} depth_source={scene.depth_source}")

        grasp_pose: Optional[dict] = None
        out: Any = None

        for idx in range(self.max_round):
            print("=" * 10, f"Round {idx}", "=" * 10)
            self.recorder.log(f"--- round {idx} ---")

            with self.recorder.stage("planner"):
                self.thought, self.plan = await self.planner(
                    query=query, previous_plan=self.plan, observation=self.observation
                )
            print("----- Thought -----\n", self.thought)
            print("----- Plan -----\n", self.plan)

            if "return to user" in (self.plan or "").lower():
                # Recorded rather than dropped: "the grasp is good, stop" is a
                # decision, and a report that omits it looks like the loop ran
                # out of rounds instead of finishing.
                self.rounds.append({
                    "round": idx,
                    "query": query,
                    "thought": self.thought,
                    "plan": self.plan,
                    "concluded": True,
                })
                break

            with self.recorder.stage("coder"):
                self.code = await self.coder(self.plan)
            print("----- Code -----\n", self.code)

            out_raw = self._execute(self.code, img_np, idx)
            result = self._build_result(out_raw, img_np, scene, image_path, save_folder, idx)

            print("----- Execution Result -----\n", {
                k: (v if k != "grasp" else self._short_grasp(v)) for k, v in result.items()
            })

            with self.recorder.stage("observer"):
                observation = await self.observer(result, query)
            print("----- Observation -----\n", observation)

            # Keep the LATEST grasp the Observer did not reject, not the
            # best-scoring one. A rejected grasp was rejected for a reason and
            # the next round exists to fix that reason, so a later grasp
            # supersedes an earlier one even when it scores lower — the score is
            # the discriminator's opinion of the grasp in isolation and knows
            # nothing about the scene or the query.
            #
            # An unparseable verdict keeps the grasp: a provider hiccup should
            # not discard a sound pose. Only an explicit rejection does.
            if result["grasp"] is not None:
                if self._keeps_grasp(observation):
                    grasp_pose = result["grasp"]
                    out = grasp_pose
                else:
                    self.recorder.log(
                        f"round {idx}: grasp rejected by the Observer, not kept"
                    )
            self.observation = self._observation_for_planner(observation)
            # The Observer's verdict and checklist used to be discarded here.
            # The prompt tells the Planner to reason from the summary alone, and
            # that stays true — but the *outer* loop needs a programmatic signal
            # of whether a grasp was judged sound, and it was already being
            # computed and thrown away.
            self.observation_json = observation

            self.rounds.append({
                "round": idx,
                "query": query,
                "thought": self.thought,
                "plan": self.plan,
                "code": self.code,
                "grasp": result.get("grasp"),
                "grasp_summary": result.get("grasp_summary"),
                "place": result.get("place"),
                "place_summary": result.get("place_summary"),
                "error_logs": result.get("error_logs"),
                "overlay_image": result.get("image"),
                "observer": observation,
            })

        status = "success" if grasp_pose is not None else "no_grasp"
        self.recorder.finish(result=grasp_pose, status=status)
        return out, grasp_pose

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _observation_for_planner(observation: dict) -> str:
        """Render the Observer's judgement for the Planner, in full.

        This used to be `observation.get("summary", "")` — one prose sentence,
        with the verdict, the failure kind and all five checklist fields thrown
        away, because the Planner's prompt forbade reading them. The Planner
        then reconstructed pass/fail from the prose and behaved as a switch,
        which is roughly all a sentence can carry.
        """
        obs = observation or {}
        lines = []
        verdict = str(obs.get("verdict", "") or "").upper()
        if verdict:
            lines.append(f"VERDICT: {verdict}")
        failure = str(obs.get("failure", "") or "")
        if failure and failure != "none":
            lines.append(f"FAILURE KIND: {failure}")

        checklist = obs.get("checklist")
        if isinstance(checklist, dict) and checklist:
            lines.append("CHECKS:")
            for key, value in checklist.items():
                lines.append(f"  {key}: {value}")

        errors = obs.get("error_logs")
        if errors and str(errors).strip().lower() not in ("none", "null"):
            lines.append(f"ERRORS: {errors}")

        summary = str(obs.get("summary", "") or "").strip()
        if summary:
            lines.append(f"SUMMARY: {summary}")
        return "\n".join(lines)

    @staticmethod
    def _selection_summary(grasp: Optional[dict]) -> Optional[dict]:
        """The funnel that produced this grasp, in the Observer's terms."""
        f = (grasp or {}).get("filters") or {}
        if not f:
            return None
        return {
            "candidates_generated": f.get("n_candidates"),
            "approaching_from_the_observed_side": f.get("n_visible"),
            "landing_on_the_requested_region": f.get("n_on_target"),
            "clear_of_the_rest_of_the_scene": f.get("n_clear_of_scene"),
            "scene_collision_was_checked": f.get("collision_checked"),
        }

    @staticmethod
    def _selection_warnings(grasp: Optional[dict]) -> Optional[str]:
        """Anything that makes this grasp the least-bad rather than a good one.

        These are the three ways `grasp_detection` returns a pose it is not
        happy with. Each was a `logger.warning` and nothing more, so the
        Observer was left to infer from a 2D projection that the approach was
        fouled — a question Python had already answered exactly.
        """
        f = (grasp or {}).get("filters") or {}
        if not f:
            return None

        notes = []
        if f.get("every_candidate_collides"):
            notes.append(
                f"COLLISION: all {f.get('n_on_target')} grasps on this object "
                "put the hand through something else in the scene. The one shown "
                "is the least bad of them, not a clear one — the constraint was "
                "dropped to return anything at all."
            )
        if f.get("no_candidate_inside_mask"):
            notes.append(
                f"OFF-TARGET: none of {f.get('n_visible')} candidates had their "
                "centre inside the requested region, so they were ranked by score "
                "alone. The grasp shown may be on the wrong part of the object."
            )
        fouled = f.get("fouled_by") or []
        if fouled:
            named = ", ".join(
                f"{e['object_id']} ({e['label']}) fouled {e['grasps_fouled']}"
                for e in fouled[:3]
            )
            notes.append(
                f"BLOCKED BY: of the best grasps found on this object, these "
                f"were rejected for colliding with — {named}. These are the "
                f"objects standing between the hand and a good grasp; say so, "
                f"because moving them is what would help."
            )
        if f.get("part_found") is False:
            notes.append(
                "WRONG REGION: the requested part could not be located, so the "
                "whole object was used instead. This grasp is NOT on the part "
                "that was asked for."
            )
        elif f.get("part_substituted"):
            notes.append(
                f"SUBSTITUTED PART: the requested part was not found under that "
                f"name, but {f.get('part_term')!r} was, and the grasp is on "
                f"whatever that matched. Check in the image that it is the same "
                f"region that was asked for — a plausible synonym is not always "
                f"the same piece of the object."
            )
        return "\n".join(notes) if notes else None

    @staticmethod
    def _keeps_grasp(observation: dict) -> bool:
        """May this round's grasp stand as the loop's answer?

        Two decisions in one line, and they pull in different directions.

        A grasp the Observer rejected must not be kept, or the loop can finish
        holding a pose its own critic called unsound and hand it to the
        executor. That was the old behaviour: every round overwrote the last
        with no reference to the verdict at all.

        But an *unparseable* verdict must not cost us a sound grasp. The
        Observer already retries once and then degrades to raw text, and a
        provider hiccup on the critique is no evidence against the pose. So the
        rule is "unless explicitly rejected", not "only when explicitly
        approved".
        """
        return str((observation or {}).get("verdict", "")).upper() != "INVALID"

    def _execute(self, code: str, img_np: np.ndarray, round_idx: int):
        """Exec the generated `execute_command` and return its result or the error."""
        namespace: dict = {}
        try:
            with self.recorder.stage("execute"):
                exec(code, image_patch_module.__dict__ | globals() | namespace, namespace)
                fn = namespace.get("execute_command")
                if fn is None:
                    return "Generated code did not define execute_command(image)"
                t0 = time.time()
                out = fn(img_np)
                self.recorder.log(
                    f"execute_command returned {type(out).__name__} in {time.time()-t0:.1f}s"
                )
                return out
        except Exception as exc:
            logger.warning("Generated code raised: %s", exc)
            self.recorder.log(f"execute_command raised: {exc}")
            return f"{type(exc).__name__}: {exc}"

    def _build_result(self, out_raw, img_np, scene, image_path, save_folder, round_idx):
        """Package the round's outcome for the Observer.

        The Observer *always* needs an image path — `agents/observer.py` calls
        `encode_image` on it — so a visualization is written even on failure.
        """
        result = {
            "grasp": None,
            "grasp_summary": None,
            "place": None,
            "place_summary": None,
            "image": str(image_path),
            "selection": None,
            # Only ever filled in when there was NO grasp, which is how every
            # warning raised while successfully returning a poor one stayed
            # invisible. The success path below now populates it too.
            "error_logs": None,
        }

        # The upstream gate was `isinstance(out_raw, list)`, matching the old
        # [quality, x, y, w, h, angle] rectangle. 6-DoF grasps are dicts.
        #
        # A pick-and-place returns `{"grasp": ..., "place": ...}` instead, so
        # both shapes are accepted and the grasp is unwrapped. A plain grasp
        # dict still passes unchanged, which is what keeps `main_simple.py` and
        # the existing tests working.
        place = None
        if isinstance(out_raw, dict) and "grasp" in out_raw:
            place = out_raw.get("place")
            out_raw = out_raw.get("grasp")

        is_grasp = isinstance(out_raw, dict) and "pose" in out_raw
        if not is_grasp:
            result["error_logs"] = out_raw if isinstance(out_raw, str) else (
                f"execute_command returned {type(out_raw).__name__}, "
                f"expected a grasp dict from grasp_detection()"
            )
            vis = visualize_grasp_6dof(
                img_np, None, scene.get_intrinsics(img_np),
                save_folder=self.recorder.images_dir
                if self.recorder.enabled else save_folder,
                filename=f"round{round_idx}_overlay.png",
            )
            result["image"] = str(vis)
            return result

        grasp = Grasp6D.from_dict(out_raw)
        K = scene.get_intrinsics(img_np)
        # The mask is what the Observer's prompt calls "the magenta region …
        # the exact object or part the grasp was computed for". It was never
        # passed, so that region was never drawn and the Observer had to guess
        # which object was aimed at from where the hand landed.
        vis = visualize_grasp_6dof(
            img_np, grasp, K,
            save_folder=self.recorder.images_dir
            if self.recorder.enabled else save_folder,
            filename=f"round{round_idx}_overlay.png",
            mask=image_patch_module.grasp_mask(out_raw),
        )
        result["grasp"] = out_raw
        result["image"] = str(vis)
        # A compact numeric channel alongside the picture: the Observer judges
        # far better with the approach angle stated than inferred from pixels.
        result["grasp_summary"] = {
            "gripper": grasp.gripper,
            "score": round(grasp.score, 3),
            "position_m": [round(float(v), 3) for v in grasp.position],
            "approach_unit_vector": [round(float(v), 3) for v in grasp.approach],
            "approach_deg_off_camera_axis": round(approach_elevation_deg(grasp.approach), 1),
            "jaw_width_cm": round(grasp.width * 100, 1),
            "depth_source": scene.depth_source,
            # What kind of hand this is. A name string and a jaw width is not
            # enough to judge "can this close on that" for anything but the one
            # gripper the Observer has already seen.
            "gripper_morphology": gripper_morphology(grasp.gripper),
        }
        # How the winner was chosen, and whether any constraint had to be
        # dropped to produce one. Both were computed inside `grasp_detection`
        # and logged to a file nobody reads.
        result["selection"] = self._selection_summary(out_raw)
        result["error_logs"] = self._selection_warnings(out_raw)

        if place is not None:
            result["place"] = place
            result["place_summary"] = {
                "position_m": [round(float(v), 3) for v in place.get("position", [])],
                "travel_cm": round(float(place.get("travel_m", 0.0)) * 100, 1),
                "clearance_cm": round(float(place.get("clearance_m", 0.0)) * 100, 1),
            }
        return result

    @staticmethod
    def _short_grasp(grasp):
        if not isinstance(grasp, dict):
            return grasp
        return {
            "score": grasp.get("score"),
            "position": [round(float(v), 3) for v in grasp.get("position", [])],
            "approach": [round(float(v), 3) for v in grasp.get("approach", [])],
            "gripper": grasp.get("gripper"),
        }
