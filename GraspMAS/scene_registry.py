"""Who is who, across iterations, and who is in the way.

Two problems this solves, both of which a name-based pipeline gets wrong.

**Identity.** "Move the bottle" is ambiguous the moment there are two bottles,
and it is *silently* ambiguous — `ImagePatch.find("bottle")` returns a list and
something downstream picks one. Worse, across iterations the ordering of that
list is not stable, so "the bottle" can mean a different object on iteration 2
than it did on iteration 1. So the planner never names objects; it names
**instances** (`obj_003`), and identity is carried by 3D position: an object
that did not move keeps its id.

That is not a heuristic hedge. Between the two real sample scenes the Cheez-It
box is labelled `obj_2` in one and `obj_1` in the other, while sitting 4 mm from
where it was. Label-based identity fails on real data; position-based identity
gets it right.

**Obstruction.** "In the way" has three distinct meanings and a loop that only
implements one will strand itself:

1. *occlusion* — it hides the target, so the target cannot be seen or segmented;
2. *approach* — its points lie inside the swept volume of a candidate grasp;
3. *proximity* — the hand simply does not fit in the gap beside the target.

Measured on the `occluded_target` scene, the bottle blocks by (1) and (2) and
the mug by (2) alone while occluding almost nothing. A purely visual notion of
"in the way" would clear the bottle, declare victory, and fail.

Frames and units follow `perception3d` and `placement`: metres, camera frame,
plus the table frame for anything on the surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import collision as col
import placement as pl
from perception3d import unproject

logger = logging.getLogger(__name__)

# An instance seen this far from a previous one is the same object that moved
# slightly, not a new object. Larger than the depth noise on a static object by
# a wide margin, smaller than any move worth making.
DEFAULT_MATCH_RADIUS_M = 0.08

# Below this many points an "object" is a segmentation fragment. Matches
# perception3d.MIN_OBJECT_POINTS, which is also GraspGen-X's own floor.
MIN_INSTANCE_POINTS = 100

# Below this `visibility`, an instance's footprint is not to be trusted.
#
# The threshold is high because the measurement says it has to be. Occluding the
# banana in `occluded_target` by varying amounts and comparing the fitted extent
# against ground truth:
#
#     occluders      visibility   long-axis error
#     bottle + mug       0.32        -11.8 cm
#     mug                0.78         -7.9 cm
#     bottle             0.81         -5.7 cm
#     none               1.00         -0.6 cm
#
# There is no useful middle ground: at 0.78 the fit is still 8 cm short, which
# would put a nominal grasp in the wrong place with the wrong orientation. Only
# an essentially unoccluded object has a footprint worth doing geometry with.
#
# That is not restrictive in practice, because it matches how the loop moves: a
# target stays occluded until its blockers are gone, and the blockers themselves
# stand in front and are unoccluded, so their footprints — the ones needed to
# *place* them — are reliable from the first iteration.
RELIABLE_VISIBILITY = 0.95


@dataclass
class Blocker:
    """One object standing between the robot and the target, and why."""

    object_id: str
    label: str
    reasons: List[str]  # any of "occlusion", "approach", "proximity"
    occlusion_frac: float = 0.0
    sweep_points: int = 0
    gap_m: float = float("inf")

    @property
    def severity(self) -> float:
        """A rough ordering for the planner. Not a probability, just a rank."""
        return (
            2.0 * self.occlusion_frac
            + min(self.sweep_points / 200.0, 2.0)
            + (1.0 if "proximity" in self.reasons else 0.0)
        )

    def describe(self) -> dict:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "reasons": list(self.reasons),
            "occlusion_frac": round(self.occlusion_frac, 3),
            "sweep_points": int(self.sweep_points),
            "gap_cm": (
                round(self.gap_m * 100, 1) if np.isfinite(self.gap_m) else None
            ),
        }


@dataclass
class ObjectInstance:
    """One object, tracked across iterations."""

    id: str
    label: str
    mask: np.ndarray
    cloud: np.ndarray  # camera frame, metres
    centroid_cam: np.ndarray
    centroid_table: np.ndarray
    footprint: Optional[pl.Footprint] = None
    visibility: float = 1.0
    first_seen: int = 0
    last_seen: int = 0
    descriptor: str = ""

    @property
    def n_points(self) -> int:
        return len(self.cloud)

    @property
    def footprint_is_reliable(self) -> bool:
        """False when occlusion makes the fitted extent untrustworthy."""
        return (
            self.footprint is not None
            and self.visibility >= RELIABLE_VISIBILITY
            and self.n_points >= MIN_INSTANCE_POINTS
        )

    def describe(self) -> dict:
        d = {
            "id": self.id,
            "label": self.label,
            "descriptor": self.descriptor,
            "position_m": [round(float(v), 3) for v in self.centroid_cam],
            "table_xy_m": [round(float(v), 3) for v in self.centroid_table[:2]],
            "n_points": self.n_points,
            "visibility": round(self.visibility, 2),
        }
        if self.footprint is not None:
            d["size_cm"] = [
                round(float(v) * 200, 1) for v in self.footprint.half_extent
            ]
            d["height_cm"] = round(self.footprint.height_m * 100, 1)
            d["footprint_reliable"] = self.footprint_is_reliable
        return d


class SceneRegistry:
    """The set of objects on the table, with ids that survive re-perception.

    Detection is pluggable so the loop is testable without a detector: when the
    observation carries a ground-truth `seg` map — every synthetic scene and both
    real sample scenes do — instances come straight from it. Otherwise it falls
    back to `ImagePatch.find` over a vocabulary.
    """

    def __init__(
        self,
        match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
        min_points: int = MIN_INSTANCE_POINTS,
    ):
        self.instances: Dict[str, ObjectInstance] = {}
        self.plane: Optional[pl.SupportPlane] = None
        self.hmap: Optional[pl.HeightMap] = None
        self.match_radius_m = float(match_radius_m)
        self.min_points = int(min_points)
        self.iteration = -1
        self._next_id = 1
        self._depth: Optional[np.ndarray] = None
        self._K: Optional[np.ndarray] = None

    # -- building ----------------------------------------------------------

    def update_from_segmentation(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        seg: np.ndarray,
        iteration: int,
        label_map: Optional[Dict[str, int]] = None,
        ignore_ids: Sequence[int] = (0, 1),
        expected_moves: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Rebuild the registry from a labelled segmentation map.

        `label_map` names the ids; anything unnamed gets `label_<id>`. Ids in
        `ignore_ids` are background and support surface. `expected_moves` is
        passed through to `_ingest` — see there for why it matters.
        """
        seg = np.asarray(seg)
        names = {v: k for k, v in (label_map or {}).items()}

        detections = []
        for sid in np.unique(seg):
            if int(sid) in ignore_ids:
                continue
            mask = seg == sid
            cloud = unproject(depth, K, mask=mask, max_depth=5.0)
            if len(cloud) < self.min_points:
                logger.debug("dropping %s: only %d points", sid, len(cloud))
                continue
            detections.append((names.get(int(sid), f"label_{int(sid)}"), mask, cloud))

        self._ingest(detections, depth, K, iteration, expected_moves)

    def update_from_patches(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        patches: Sequence[Tuple[str, np.ndarray]],
        iteration: int,
        expected_moves: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Rebuild from `(label, mask)` pairs, e.g. from `ImagePatch.find`."""
        detections = []
        for label, mask in patches:
            mask = np.asarray(mask).astype(bool)
            cloud = unproject(depth, K, mask=mask, max_depth=5.0)
            if len(cloud) < self.min_points:
                continue
            detections.append((label, mask, cloud))
        self._ingest(detections, depth, K, iteration, expected_moves)

    def _ingest(self, detections, depth, K, iteration: int, expected_moves=None) -> None:
        self.iteration = int(iteration)
        self._depth, self._K = np.asarray(depth), np.asarray(K)

        object_mask = np.zeros(np.shape(depth), dtype=bool)
        for _, mask, _ in detections:
            object_mask |= mask

        cloud = pl.support_cloud(depth, K, object_mask=object_mask)
        self.plane = pl.fit_support_plane(cloud)
        self.hmap = pl.build_height_map(cloud, self.plane)

        previous = dict(self.instances)
        self.instances = {}
        unmatched = dict(previous)

        fresh = []
        for label, mask, obj_cloud in detections:
            centroid_cam = obj_cloud.mean(axis=0).astype(np.float64)
            centroid_table = self.plane.to_table(centroid_cam[None, :])[0]
            fresh.append((label, mask, obj_cloud, centroid_cam, centroid_table))

        # Where each previously-known object is expected to be found now.
        #
        # By default that is where it was, and the match radius is sized for
        # depth noise on a static object. But this loop *deliberately* moves
        # things, usually 10-30 cm — far outside that radius — so an object the
        # robot just picked up and set down would be retired and re-registered
        # under a new id, and the evaluator would report it as having vanished.
        # That is exactly what happened on the first end-to-end run.
        #
        # The loop knows where it put things, so it says so. A hinted object is
        # matched against its intended destination *as well as* its old position,
        # never instead of it — the move may have failed, and an object that
        # never left the ground must still be recognised where it always was.
        # Getting that wrong made a dropped object look like a vanished one, and
        # the loop then reported it as no longer blocking, which defeated stall
        # detection and ran the run to its iteration cap.
        hints = {
            k: np.asarray(v, dtype=np.float64)[:2]
            for k, v in (expected_moves or {}).items()
        }
        moved_radius = max(self.match_radius_m * 2.0, 0.15)

        # Match nearest-first across all pairs, so a close pair is never stolen
        # by a worse but earlier-considered candidate.
        pairs = []
        for i, (label, _, _, _, ct) in enumerate(fresh):
            for pid, prev in unmatched.items():
                anchors = [(prev.centroid_table[:2], self.match_radius_m)]
                if pid in hints:
                    anchors.append((hints[pid], moved_radius))
                for anchor, radius in anchors:
                    d = float(np.linalg.norm(ct[:2] - anchor))
                    if d <= radius:
                        pairs.append((d, i, pid))
                        break
        pairs.sort()

        assigned: Dict[int, str] = {}
        taken = set()
        for d, i, pid in pairs:
            if i in assigned or pid in taken:
                continue
            assigned[i] = pid
            taken.add(pid)

        for i, (label, mask, obj_cloud, centroid_cam, centroid_table) in enumerate(fresh):
            pid = assigned.get(i)
            if pid is None:
                pid = f"obj_{self._next_id:03d}"
                self._next_id += 1
                first_seen = self.iteration
            else:
                first_seen = previous[pid].first_seen
                # Keep the original label: re-detection can flip a name between
                # frames, and the id is the thing that must stay meaningful.
                label = previous[pid].label

            try:
                footprint = pl.object_footprint(obj_cloud, self.plane)
            except ValueError:
                footprint = None

            self.instances[pid] = ObjectInstance(
                id=pid,
                label=label,
                mask=mask,
                cloud=obj_cloud,
                centroid_cam=centroid_cam,
                centroid_table=centroid_table,
                footprint=footprint,
                first_seen=first_seen,
                last_seen=self.iteration,
            )

        self._compute_visibility()
        self._assign_descriptors()

    def _compute_visibility(self, halo_px: int = 3) -> None:
        """How much of each object is *not* cut off by something in front of it.

        There is no way to measure the fraction of an unoccluded silhouette that
        survives, because the unoccluded silhouette was never observed. Filling
        the mask's convex hull and comparing looks like it measures that and does
        not: the visible fragment of an occluded object is usually convex too, so
        it scores the 19%-visible banana at 98%.

        What *is* observable is the object's outline, and where that outline runs
        against a nearer object rather than against free space. An unoccluded
        object is bounded by table and background all the way round; a truncated
        one is bounded by its occluder along the cut. So visibility here is
        ``1 - (outline against nearer objects) / (whole outline)``.

        It is a proxy, not a percentage of the object — an object exactly half
        hidden does not necessarily score 0.5. It is monotonic in the right
        direction and cheap, which is what the reliability gate needs.
        """
        import cv2

        k = np.ones((2 * halo_px + 1, 2 * halo_px + 1), np.uint8)
        halos = {}
        for inst in self.instances.values():
            m = inst.mask.astype(np.uint8)
            halos[inst.id] = cv2.dilate(m, k).astype(bool) & ~inst.mask

        for inst in self.instances.values():
            halo = halos[inst.id]
            total = int(halo.sum())
            if total == 0:
                inst.visibility = 0.0
                continue
            occluded = np.zeros_like(halo)
            for other in self.instances.values():
                if other.id == inst.id:
                    continue
                if float(other.centroid_cam[2]) < float(inst.centroid_cam[2]):
                    occluded |= halo & other.mask
            inst.visibility = 1.0 - float(occluded.sum()) / total

    def _assign_descriptors(self) -> None:
        """Human-readable disambiguators, for prompts and logs only.

        Ids are the contract; these exist so a plan reads like a sentence rather
        than a list of opaque handles. Only objects that share a label need one.
        """
        by_label: Dict[str, List[ObjectInstance]] = {}
        for inst in self.instances.values():
            by_label.setdefault(inst.label, []).append(inst)

        for label, group in by_label.items():
            if len(group) == 1:
                group[0].descriptor = f"the {label}"
                continue
            # Left-to-right in the table frame is the least surprising ordering.
            group.sort(key=lambda i: float(i.centroid_table[0]))
            words = ["leftmost", "second from the left", "third from the left"]
            for n, inst in enumerate(group):
                if n == len(group) - 1:
                    inst.descriptor = f"the rightmost {label}"
                elif n < len(words):
                    inst.descriptor = f"the {words[n]} {label}"
                else:
                    inst.descriptor = f"{label} #{n + 1}"

    # -- lookup ------------------------------------------------------------

    def get(self, object_id: str) -> ObjectInstance:
        if object_id not in self.instances:
            raise KeyError(
                f"no instance {object_id!r}; have {sorted(self.instances)}"
            )
        return self.instances[object_id]

    def find_by_label(self, label: str) -> List[ObjectInstance]:
        needle = label.lower().strip()
        return [
            i
            for i in self.instances.values()
            if needle in i.label.lower() or i.label.lower() in needle
        ]

    def resolve_target(self, description: str) -> Optional[ObjectInstance]:
        """Best-effort id for a free-text target, used once to seed the run.

        Deliberately not used per iteration — after the first resolution the
        planner works with the id, which is what makes duplicates tractable.
        """
        matches = self.find_by_label(description)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return None
        # Ambiguous: prefer the one the description's modifiers point at, else
        # the largest, which is the most likely thing a person meant.
        low = description.lower()
        for word, key in (
            ("left", lambda i: -i.centroid_table[0]),
            ("right", lambda i: i.centroid_table[0]),
            ("near", lambda i: -i.centroid_cam[2]),
            ("far", lambda i: i.centroid_cam[2]),
        ):
            if word in low:
                return max(matches, key=key)
        return max(matches, key=lambda i: i.n_points)

    # -- reasoning ---------------------------------------------------------

    def moved_since(
        self,
        previous: Dict[str, np.ndarray],
        thresh_m: float = 0.03,
        min_visibility: float = 0.0,
        previous_visibility: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Which tracked objects have shifted since a recorded set of positions.

        `previous` maps id -> table-frame centroid. Used by `evaluator` both to
        confirm the intended object moved and to catch collateral damage.

        **An occluded object's centroid is not a stable quantity.** It is the
        mean of whatever happens to be visible, so uncovering part of an object
        moves its apparent centre without the object moving at all — measured at
        2.3 cm on the banana when the mug beside it was picked up, which is
        comfortably past any sane movement threshold. Pass `min_visibility` to
        restrict the answer to instances whose centroid actually means something;
        the evaluator does exactly that before reporting collateral damage.

        **Both ends of the comparison have to qualify.** Gating on the instance's
        visibility *now* is not enough: a stored centroid computed from a
        partly-hidden view is just as untrustworthy as a current one. Measured —
        with the mug carried away, the banana went from 78% visible to 100% and
        its apparent centre shifted **3.8 cm** without being touched, which the
        evaluator reported as collateral damage *to the target* and the planner
        reasonably aborted on. Pass `previous_visibility` so an object that was
        badly seen before is skipped too.
        """
        out = []
        previous_visibility = previous_visibility or {}
        for pid, before in previous.items():
            inst = self.instances.get(pid)
            if inst is None or inst.visibility < min_visibility:
                continue
            if previous_visibility.get(pid, 1.0) < min_visibility:
                continue
            d = float(np.linalg.norm(inst.centroid_table[:2] - np.asarray(before)[:2]))
            if d >= thresh_m:
                out.append((pid, d))
        return sorted(out, key=lambda x: -x[1])

    def positions(self) -> Dict[str, np.ndarray]:
        """Table-frame centroids, the snapshot `moved_since` compares against."""
        return {i.id: i.centroid_table.copy() for i in self.instances.values()}

    def visibilities(self) -> Dict[str, float]:
        """The companion to `positions`: how well each centroid was seen.

        `moved_since` needs both ends of its comparison to be trustworthy, and a
        bare position snapshot cannot say whether it was.
        """
        return {i.id: float(i.visibility) for i in self.instances.values()}

    def occlusion_of(self, target_id: str, dilate_px: int = 5) -> Dict[str, float]:
        """How much of the target each other object hides.

        An object occludes the target when it is adjacent to it in the image and
        nearer the camera. Adjacency rather than overlap because segmentation
        assigns each pixel to exactly one object — the occluder has *taken* the
        pixels the target would have had, so they never overlap.
        """
        import cv2

        target = self.get(target_id)
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        halo = cv2.dilate(target.mask.astype(np.uint8), k).astype(bool) & ~target.mask
        halo_area = max(int(halo.sum()), 1)

        out = {}
        for inst in self.instances.values():
            if inst.id == target_id:
                continue
            shared = int((halo & inst.mask).sum())
            if shared == 0:
                continue
            nearer = float(inst.centroid_cam[2]) < float(target.centroid_cam[2])
            out[inst.id] = (shared / halo_area) if nearer else 0.0
        return out

    def blocking_objects(
        self,
        target_id: str,
        grasp_pose: Optional[np.ndarray] = None,
        gripper_name: str = "franka_panda",
        approach_len: float = 0.12,
        occlusion_thresh: float = 0.05,
        proximity_margin_m: float = 0.01,
    ) -> List[Blocker]:
        """Everything standing between the robot and the target.

        `grasp_pose` enables the approach test. Without one, a nominal top-down
        grasp closing across the target's short axis is synthesised — that is the
        grasp a parallel jaw would actually choose, and it means the loop can
        reason about reachability before it has spent an inference on it.

        **The geometric tests are skipped while the target is badly occluded**,
        and this is the important subtlety. Approach and proximity both need the
        target's footprint, and a footprint fitted to a 19%-visible fragment is
        the wrong size in the wrong place with the wrong orientation, so both
        tests would return confident nonsense. While the target is hidden,
        occlusion alone decides. That is not a gap: clearing the occluders makes
        the target visible, at which point its footprint becomes trustworthy and
        the geometric tests take over and find the second-order blockers. The
        loop discovers them in the order it can actually establish them.

        An empty result plus a grasp that survives scene-collision filtering is
        the loop's termination condition.
        """
        target = self.get(target_id)
        gripper_pts = col.load_gripper_points(gripper_name)

        occl = self.occlusion_of(target_id)
        jaw_half = float(np.abs(gripper_pts[:, 0]).max())

        blockers: Dict[str, Blocker] = {}

        def entry(inst: ObjectInstance) -> Blocker:
            if inst.id not in blockers:
                blockers[inst.id] = Blocker(inst.id, inst.label, [])
            return blockers[inst.id]

        for oid, frac in occl.items():
            if frac >= occlusion_thresh:
                b = entry(self.instances[oid])
                b.reasons.append("occlusion")
                b.occlusion_frac = frac

        geometry_is_meaningful = target.footprint_is_reliable or grasp_pose is not None
        if not geometry_is_meaningful:
            logger.info(
                "%s is %.0f%% visible; reporting occlusion only until it is cleared",
                target_id, target.visibility * 100,
            )
            return sorted(blockers.values(), key=lambda b: -b.severity)

        if grasp_pose is None:
            grasp_pose = self.nominal_grasp(target_id, gripper_name=gripper_name)

        for inst in self.instances.values():
            if inst.id == target_id:
                continue

            if grasp_pose is not None:
                hits = col.points_inside_sweep(
                    grasp_pose, inst.cloud, gripper_pts, approach_len=approach_len
                )
                if len(hits):
                    b = entry(inst)
                    if "approach" not in b.reasons:
                        b.reasons.append("approach")
                    b.sweep_points = int(len(hits))

            if inst.footprint is not None and target.footprint is not None:
                gap = float(
                    np.linalg.norm(
                        inst.footprint.centroid_xy - target.footprint.centroid_xy
                    )
                    - inst.footprint.radius_m
                    - target.footprint.radius_m
                )
                if gap < jaw_half + proximity_margin_m:
                    b = entry(inst)
                    if "proximity" not in b.reasons:
                        b.reasons.append("proximity")
                    b.gap_m = gap

        return sorted(blockers.values(), key=lambda b: -b.severity)

    def nominal_grasp(
        self, object_id: str, gripper_name: str = "franka_panda", standoff_m: float = 0.005
    ) -> Optional[np.ndarray]:
        """A plausible top-down grasp, for reachability reasoning before inference.

        Closes across the object's short axis, which is what a parallel jaw does
        with an elongated object, and is anchored at the gripper base so it lines
        up with what GraspGen-X returns.
        """
        inst = self.get(object_id)
        if inst.footprint is None or self.plane is None:
            return None

        fp = inst.footprint
        short = np.array([-np.sin(fp.yaw), np.cos(fp.yaw)])
        if fp.half_extent[0] < fp.half_extent[1]:
            short = np.array([np.cos(fp.yaw), np.sin(fp.yaw)])

        closing = self.plane.rotation[:, :2] @ short
        n = np.linalg.norm(closing)
        if n < 1e-9:
            return None

        _, fingertip = _gripper_geometry(gripper_name)
        T = np.eye(4)
        T[:3, 2] = -self.plane.normal
        T[:3, 0] = closing / n
        T[:3, 1] = np.cross(T[:3, 2], T[:3, 0])
        T[:3, 3] = self.plane.to_camera(
            np.array([[fp.centroid_xy[0], fp.centroid_xy[1], fp.top_m + standoff_m]])
        )[0] + fingertip * self.plane.normal
        return T

    def scene_cloud_excluding(self, *object_ids: str) -> np.ndarray:
        """The scene cloud with the named instances removed."""
        if self._depth is None or self._K is None:
            raise RuntimeError("registry has no observation yet; call update_* first")
        exclude = np.zeros(self._depth.shape, dtype=bool)
        for oid in object_ids:
            exclude |= self.get(oid).mask
        return col.scene_cloud_excluding(self._depth, self._K, exclude_mask=exclude)

    def keep_out_for(
        self, target_id: str, moving_id: Optional[str] = None, dilate_m: float = 0.05
    ) -> np.ndarray:
        """Where `moving_id` must *not* be put down if the target is to stay clear.

        Three regions: the target's own footprint, anywhere the moved object
        would project in front of the target, and the moving object's current
        spot — so "clearing" it cannot mean putting it back.

        The occlusion region is computed in the image rather than on the table
        (see `placement.projected_occlusion_keep_out`), because the target is by
        definition occluded and its footprint is therefore a fragment.

        The image region protected is the target's mask **unioned with the
        moving object's**, and that union is the point rather than a safety
        margin. The target's visible mask is itself truncated by whatever is in
        front of it, so protecting only those pixels leaves the hidden part
        unprotected — measured, the bottle was moved 20 cm and released straight
        back over the part of the banana it had been hiding. The moving object's
        current silhouette is exactly the region about to be revealed, and
        therefore exactly where the rest of the target may turn out to be. It
        costs nothing to include: that ground is being vacated anyway, and
        putting the object back on it is already forbidden.

        Pass `moving_id` whenever you have it; without one a conservative 20 cm
        column is assumed, which over-blocks rather than under-blocks.
        """
        if self.hmap is None or self.plane is None or self._K is None:
            raise RuntimeError("registry has no observation yet; call update_* first")
        target = self.get(target_id)

        keep = np.zeros(self.hmap.shape, dtype=bool)
        if target.footprint is not None:
            keep |= pl.footprint_keep_out(self.hmap, [target.footprint], dilate_m=dilate_m)

        mover = self.get(moving_id) if moving_id is not None else None
        object_height, object_radius = 0.20, 0.06
        protected = target.mask
        if mover is not None:
            protected = protected | mover.mask
            if mover.footprint is not None:
                object_height = max(mover.footprint.top_m, 0.02)
                object_radius = max(mover.footprint.radius_m, 0.01)

        keep |= pl.projected_occlusion_keep_out(
            self.hmap,
            self.plane,
            self._K,
            protected,
            target_depth_m=float(target.centroid_cam[2]),
            object_height_m=object_height,
            object_radius_m=object_radius,
        )

        if mover is not None and mover.footprint is not None:
            keep |= pl.footprint_keep_out(self.hmap, [mover.footprint], dilate_m=dilate_m)
        return keep

    # -- reporting ---------------------------------------------------------

    def as_prompt_table(self, target_id: Optional[str] = None) -> str:
        """A compact text table of the scene, for the planner prompt."""
        if not self.instances:
            return "(no objects detected)"
        rows = ["| id | object | position (table x,y in cm) | size cm | height cm | visible |",
                "|---|---|---|---|---|---|"]
        for inst in sorted(self.instances.values(), key=lambda i: i.id):
            fp = inst.footprint
            size = (
                f"{fp.half_extent[0]*200:.0f}x{fp.half_extent[1]*200:.0f}" if fp else "?"
            )
            height = f"{fp.height_m*100:.0f}" if fp else "?"
            mark = " **(target)**" if inst.id == target_id else ""
            rows.append(
                f"| {inst.id} | {inst.descriptor}{mark} | "
                f"{inst.centroid_table[0]*100:.0f}, {inst.centroid_table[1]*100:.0f} | "
                f"{size} | {height} | {inst.visibility:.0%} |"
            )
        return "\n".join(rows)

    def describe(self) -> dict:
        return {
            "iteration": self.iteration,
            "n_instances": len(self.instances),
            "plane": self.plane.describe() if self.plane else None,
            "instances": [i.describe() for i in sorted(self.instances.values(), key=lambda x: x.id)],
        }


def _gripper_geometry(gripper_name: str) -> Tuple[float, float]:
    """(jaw aperture, base->fingertip depth) in metres, from the gripper config.

    Reads the same two config fields as `ImagePatch._gripper_geometry` rather
    than calling it: importing `image_patch` loads GroundingDINO, SAM, VLPart
    and MiDaS at module scope, which this module never needs and which the
    offline tests must not pay for.
    """
    import json
    import os

    d = col.gripper_asset_dir(gripper_name)
    if d:
        try:
            with open(os.path.join(d, "config.json")) as f:
                cfg = json.load(f)
            return float(cfg["sweep_volume"]["extents"][0]), float(cfg["fingertip"][-1])
        except Exception as exc:
            logger.debug("could not read geometry for %s: %s", gripper_name, exc)
    return 0.08, 0.11
