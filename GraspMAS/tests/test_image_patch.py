"""image_patch: the 6-DoF grasp_detection path and the CPU/RAM budget.

Importing this module loads the whole perception stack (GroundingDINO, SAM,
VLPart, DPT), so the import is module-scoped and its cost is itself part of what
we assert: the point of dropping BLIP2 and OWLv2 was to fit in 15 GB on CPU, and
a regression that re-adds them would be invisible without a test.

The GraspGen-X client is mocked throughout — no server, no GPU.
"""

import resource
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import image_patch as ip
from perception3d import SceneContext


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


class FakeClient:
    """Stands in for the ZMQ client; records what it was asked for."""

    def __init__(self, grasps=None, scores=None, raise_exc=None):
        self.grasps = grasps
        self.scores = scores
        self.raise_exc = raise_exc
        self.calls = []

    def infer(self, cloud, gripper_name=None, num_grasps=200,
              grasp_threshold=-1.0, topk_num_grasps=100, planner="graspmoe"):
        self.calls.append({
            "cloud": cloud, "gripper_name": gripper_name,
            "num_grasps": num_grasps, "planner": planner,
        })
        if self.raise_exc:
            raise self.raise_exc
        if self.grasps is None:
            return np.zeros((0, 4, 4), np.float32), np.zeros((0,), np.float32)
        return self.grasps, self.scores


def grasps_at(positions, scores):
    g = np.tile(np.eye(4, dtype=np.float32), (len(positions), 1, 1))
    for i, p in enumerate(positions):
        g[i, :3, 3] = p
    return g, np.asarray(scores, np.float32)


@pytest.fixture
def patch_env(monkeypatch, rgb, depth_map, K):
    """A configured ImagePatch plus the fake client it will talk to."""
    scene = SceneContext(depth=depth_map, intrinsics=K, image_shape=rgb.shape[:2])
    ip.configure_scene(scene=scene, gripper_name="franka_panda",
                       num_grasps=50, planner="diffusion")
    ip._GRASP_CACHE.clear()

    client = FakeClient()
    monkeypatch.setattr(ip, "get_graspgen_client", lambda: client)
    # Gripper geometry should not depend on the assets being present in a unit
    # test. fingertip_depth is 0 here so the fixture's expected projections stay
    # simple; TestGripperGeometry covers the real lookup.
    monkeypatch.setattr(ip.ImagePatch, "_gripper_geometry",
                        staticmethod(lambda g: (0.08, 0.0)))
    return client


@pytest.fixture
def image_patch(rgb):
    return ip.ImagePatch(rgb)


@pytest.fixture
def object_patch(rgb, box_mask):
    p = ip.ImagePatch(rgb, name="box")
    p.mask = box_mask
    return p


class TestModuleFootprint:
    """These encode the CPU-only budget decisions; a regression is a real bug."""

    def test_blip2_is_gone(self):
        assert not hasattr(ip, "blip_model"), "BLIP2 costs 15.8 GB download / ~8 GB RSS"
        assert not hasattr(ip, "blip_processor")

    def test_owlv2_is_gone(self):
        assert not hasattr(ip, "owl_model"), "OWLv2 had zero call sites"
        assert not hasattr(ip, "owl_processor")

    def test_ragt_2d_grasp_model_is_gone(self):
        assert not hasattr(ip, "grasp_model")
        assert "grasp.unit_grasp_pose_generation" not in sys.modules

    def test_no_paid_openai_client(self):
        assert not hasattr(ip, "OPENAI_CLIENT")
        assert not hasattr(ip, "EMBEDDING_MODEL")

    def test_live_models_are_present(self):
        for name in ("object_detector", "segmentator", "vlpart", "depth_model"):
            assert hasattr(ip, name), f"{name} should still be loaded"

    def test_device_is_cpu_here(self):
        assert ip.device == "cpu"

    def test_resident_memory_within_budget(self):
        # The whole point of the trimming: comfortably inside 15 GB.
        assert rss_gb() < 8.0, f"perception stack RSS {rss_gb():.1f} GB is too large"


class TestGraspDetection:
    def test_returns_none_for_none_or_list(self, image_patch, patch_env):
        assert image_patch.grasp_detection(None) is None
        assert image_patch.grasp_detection([]) is None

    def test_returns_none_when_patch_has_no_mask(self, image_patch, patch_env, rgb):
        assert image_patch.grasp_detection(ip.ImagePatch(rgb)) is None

    def test_returns_none_for_empty_mask(self, image_patch, patch_env, rgb):
        p = ip.ImagePatch(rgb, name="empty")
        p.mask = np.zeros(rgb.shape[:2], bool)
        assert image_patch.grasp_detection(p) is None

    def test_produces_a_well_formed_grasp(self, image_patch, object_patch, patch_env):
        g, s = grasps_at([[0.0, 0.0, 0.6]], [0.85])
        patch_env.grasps, patch_env.scores = g, s

        out = image_patch.grasp_detection(object_patch)
        assert out is not None
        for key in ("pose", "position", "approach", "closing",
                    "score", "width", "gripper", "rect_2d"):
            assert key in out
        assert np.asarray(out["pose"]).shape == (4, 4)
        assert out["score"] == pytest.approx(0.85)
        assert out["gripper"] == "franka_panda"

    def test_sends_a_metric_cloud_from_the_mask(self, image_patch, object_patch, patch_env):
        g, s = grasps_at([[0.0, 0.0, 0.6]], [0.9])
        patch_env.grasps, patch_env.scores = g, s
        image_patch.grasp_detection(object_patch)

        cloud = patch_env.calls[0]["cloud"]
        assert cloud.shape[1] == 3
        # The fixture's box sits at 0.6 m; the cloud must reflect that, in metres.
        assert cloud[:, 2].min() == pytest.approx(0.6, abs=1e-4)
        assert len(cloud) == int(np.sum(object_patch.mask))

    def test_forwards_gripper_and_planner_settings(self, image_patch, object_patch, patch_env):
        g, s = grasps_at([[0.0, 0.0, 0.6]], [0.9])
        patch_env.grasps, patch_env.scores = g, s
        image_patch.grasp_detection(object_patch, gripper_name="robotiq_2f_85")
        call = patch_env.calls[0]
        assert call["gripper_name"] == "robotiq_2f_85"
        assert call["num_grasps"] == 50
        assert call["planner"] == "diffusion"

    def test_picks_the_highest_scoring_in_mask_grasp(self, image_patch, object_patch, patch_env):
        # All three project inside the mask; the best score must win.
        g, s = grasps_at(
            [[0.0, 0.0, 0.6], [0.005, 0.0, 0.6], [-0.005, 0.0, 0.6]],
            [0.3, 0.95, 0.6],
        )
        patch_env.grasps, patch_env.scores = g, s
        out = image_patch.grasp_detection(object_patch)
        assert out["score"] == pytest.approx(0.95)

    def test_mask_filter_rejects_out_of_mask_grasps(self, image_patch, object_patch, patch_env):
        """This is what keeps the referring expression attached to the 6-DoF
        output: a high-scoring grasp somewhere else must lose to a lower-scoring
        one on the referred region."""
        g, s = grasps_at(
            [[0.0, 0.0, 0.6],     # inside the mask, lower score
             [0.30, 0.20, 0.6]],  # far outside, higher score
            [0.40, 0.99],
        )
        patch_env.grasps, patch_env.scores = g, s
        out = image_patch.grasp_detection(object_patch)
        assert out["score"] == pytest.approx(0.40)

    def test_falls_back_when_every_grasp_is_out_of_mask(self, image_patch, object_patch, patch_env):
        """Returning nothing would be worse than returning the best available."""
        g, s = grasps_at([[0.30, 0.20, 0.6], [0.35, 0.25, 0.6]], [0.5, 0.8])
        patch_env.grasps, patch_env.scores = g, s
        out = image_patch.grasp_detection(object_patch)
        assert out is not None
        assert out["score"] == pytest.approx(0.8)

    def test_returns_none_when_model_finds_nothing(self, image_patch, object_patch, patch_env):
        patch_env.grasps, patch_env.scores = None, None
        assert image_patch.grasp_detection(object_patch) is None

    def test_server_failure_degrades_instead_of_crashing(self, image_patch, object_patch,
                                                         patch_env, monkeypatch):
        """A transient socket problem must not take down the agent loop."""
        from graspgen import GraspGenUnavailable

        client = FakeClient(raise_exc=GraspGenUnavailable("server down"))
        monkeypatch.setattr(ip, "get_graspgen_client", lambda: client)
        assert image_patch.grasp_detection(object_patch) is None

    def test_too_few_3d_points_is_rejected(self, image_patch, patch_env, rgb):
        """A mask over a region with no depth return yields a useless cloud."""
        p = ip.ImagePatch(rgb, name="tiny")
        m = np.zeros(rgb.shape[:2], bool)
        m[10:12, 10:12] = True  # 4 px, well under MIN_OBJECT_POINTS
        p.mask = m
        assert image_patch.grasp_detection(p) is None
        assert patch_env.calls == []  # never even asked the server

    def test_caches_repeat_calls(self, image_patch, object_patch, patch_env):
        """The loop re-plans with the same mask often; on CPU a cache hit is
        worth seconds."""
        g, s = grasps_at([[0.0, 0.0, 0.6]], [0.9])
        patch_env.grasps, patch_env.scores = g, s
        first = image_patch.grasp_detection(object_patch)
        second = image_patch.grasp_detection(object_patch)
        assert first == second
        assert len(patch_env.calls) == 1

    def test_rect_2d_is_usable_by_the_2d_metric(self, image_patch, object_patch, patch_env):
        from utils import eval_grasp

        g, s = grasps_at([[0.0, 0.0, 0.6]], [0.9])
        patch_env.grasps, patch_env.scores = g, s
        out = image_patch.grasp_detection(object_patch)
        rect = out["rect_2d"]
        assert rect is not None and len(rect) == 6
        # Scoring against itself in the OCID layout must be a hit.
        gt = [rect[1], rect[2], rect[3], rect[4], rect[5], 0]
        assert eval_grasp(rect, [gt], dataset_type="OCID") is True


class TestFindById:
    """The tool the whole duplicate-object design routes through.

    It had no test at all: it was implemented, advertised in the Coder prompt,
    and only ever reached when a live model chose to call it. The first live run
    that did raised `'Image' object has no attribute 'shape'` on every call, and
    the Coder silently fell back to `find()` — which is the ambiguity ids exist
    to remove.
    """

    @staticmethod
    def _registry(mask, label="bottle", object_id="obj_003"):
        inst = SimpleNamespace(id=object_id, label=label, mask=mask)
        return SimpleNamespace(get=lambda oid: inst if oid == object_id else None)

    def test_returns_a_patch_covering_the_registered_mask(
        self, monkeypatch, rgb, box_mask
    ):
        monkeypatch.setattr(ip, "REGISTRY", self._registry(box_mask))
        patch = ip.ImagePatch(rgb).find_by_id("obj_003")

        assert patch.name == "obj_003"
        assert patch.mask is not None
        ys, xs = np.nonzero(box_mask)
        # The crop must contain every masked pixel; `crop` flips rows, so an
        # off-by-height error here silently returns a patch of empty background.
        assert patch.left <= xs.min() and patch.right >= xs.max()
        assert patch.mask.sum() > 0

    def test_the_patch_is_usable_by_grasp_detection(
        self, monkeypatch, rgb, box_mask, patch_env
    ):
        """The actual call chain: find_by_id -> grasp_detection."""
        monkeypatch.setattr(ip, "REGISTRY", self._registry(box_mask))
        patch_env.grasps, patch_env.scores = grasps_at([(0.0, 0.0, 0.6)], [0.9])

        root = ip.ImagePatch(rgb)
        result = root.grasp_detection(root.find_by_id("obj_003"))
        assert result is not None and "pose" in result

    def test_raises_a_useful_error_outside_the_declutter_loop(
        self, monkeypatch, rgb
    ):
        monkeypatch.setattr(ip, "REGISTRY", None)
        with pytest.raises(RuntimeError, match="registry"):
            ip.ImagePatch(rgb).find_by_id("obj_003")

    def test_rejects_an_empty_mask(self, monkeypatch, rgb):
        empty = np.zeros(rgb.shape[:2], dtype=bool)
        monkeypatch.setattr(ip, "REGISTRY", self._registry(empty))
        with pytest.raises(ValueError, match="empty mask"):
            ip.ImagePatch(rgb).find_by_id("obj_003")


class TestGripperGeometry:
    """Reads the real gripper config when the assets are present."""

    def test_reads_jaw_and_fingertip_from_config(self):
        import os
        if not os.environ.get("GRASPGENX_GRIPPER_CFG_DIR"):
            pytest.skip("gripper_descriptions not configured")
        width, depth = ip.ImagePatch._gripper_geometry("franka_panda")
        assert 0.05 < width < 0.12, f"implausible Panda jaw width {width}"
        assert 0.05 < depth < 0.20, f"implausible fingertip depth {depth}"

    def test_falls_back_for_an_unknown_gripper(self):
        width, depth = ip.ImagePatch._gripper_geometry("no_such_gripper_xyz")
        assert (width, depth) == (0.08, 0.11)

    def test_grippers_differ(self):
        import os
        if not os.environ.get("GRASPGENX_GRIPPER_CFG_DIR"):
            pytest.skip("gripper_descriptions not configured")
        panda = ip.ImagePatch._gripper_geometry("franka_panda")
        g1 = ip.ImagePatch._gripper_geometry("unitree_g1")
        assert panda != g1, "different hands must not report identical geometry"


class TestComputeDepth:
    def test_returns_metres_from_the_scene(self, patch_env, rgb, box_mask):
        p = ip.ImagePatch(rgb, name="box")
        p.mask = box_mask
        # The fixture puts the box at 0.6 m and the background plane at 1.0 m.
        assert p.compute_depth() == pytest.approx(0.6, abs=0.05)

    def test_smaller_is_closer(self, patch_env, rgb, box_mask):
        """The sign convention flipped when depth became metric; sorting by
        compute_depth must now put the nearest object first."""
        near = ip.ImagePatch(rgb, name="near")
        near.mask = box_mask
        far_mask = np.zeros(rgb.shape[:2], bool)
        far_mask[50:100, 400:500] = True  # background plane at 1.0 m
        far = ip.ImagePatch(rgb, name="far")
        far.mask = far_mask
        assert near.compute_depth() < far.compute_depth()
