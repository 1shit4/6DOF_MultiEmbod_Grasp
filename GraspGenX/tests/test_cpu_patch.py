"""The CPU patch applied to GraspGen-X for this project.

Two things here are silent-failure risks, which is why they get dedicated tests:

* A missed `.cuda()` raises only when that code path runs — and
  `load_gripper_input` runs on *every* request, so one missed line breaks
  everything but only at inference time.
* `enable_flash` defaults to True in the released checkpoints and routes
  attention through an fp16 CUDA-oriented path. On CPU that is wrong, but it
  fails as bad numbers or an obscure dtype error rather than a clear message.

Run inside the `graspgenx` env:
    pytest tests/test_cpu_patch.py
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from graspgenx.grasp_server import (
    disable_flash_attention,
    resolve_device,
    sampler_device,
)

REPO = Path(__file__).resolve().parent.parent


class TestDeviceResolution:
    def test_env_var_overrides_autodetect(self, monkeypatch):
        monkeypatch.setenv("GRASPGENX_DEVICE", "cpu")
        assert resolve_device().type == "cpu"

    def test_falls_back_to_cpu_without_cuda(self, monkeypatch):
        monkeypatch.delenv("GRASPGENX_DEVICE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device().type == "cpu"

    def test_prefers_cuda_when_available_and_unset(self, monkeypatch):
        monkeypatch.delenv("GRASPGENX_DEVICE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device().type == "cuda"

    def test_sampler_device_follows_the_weights(self):
        model = torch.nn.Linear(2, 2)
        sampler = type("S", (), {"model": model})()
        assert sampler_device(sampler).type == "cpu"


class TestNoResidualCudaCalls:
    """A source-level guard: any `.cuda()` reintroduced into grasp_server.py
    would break CPU inference at request time, far from the edit."""

    def test_grasp_server_has_no_hardcoded_cuda(self):
        src = (REPO / "graspgenx" / "grasp_server.py").read_text()
        offenders = [
            (i, line)
            for i, line in enumerate(src.splitlines(), 1)
            if ".cuda()" in line and not line.strip().startswith("#")
        ]
        assert not offenders, f"hardcoded .cuda() at {offenders}"

    def test_load_gripper_input_moves_every_tensor(self):
        """There were seven `.cuda()` calls in this one function; all seven must
        now route through the sampler's device."""
        src = (REPO / "graspgenx" / "grasp_server.py").read_text()
        start = src.index("def load_gripper_input")
        body = src[start : src.index("def get_gripper_info")]
        assert "sampler_device(self)" in body
        assert body.count(".to(device)") >= 7, "a tensor was left behind"


class TestFlashAttention:
    def _cfg(self, flash=True):
        from omegaconf import OmegaConf

        return OmegaConf.create({
            "diffusion": {"ptv3vanilla": {"enable_flash": flash}},
            "discriminator": {"ptv3vanilla": {"enable_flash": flash}},
        })

    def test_disables_both_sides(self):
        cfg = self._cfg(True)
        disable_flash_attention(cfg)
        assert cfg.diffusion.ptv3vanilla.enable_flash is False
        assert cfg.discriminator.ptv3vanilla.enable_flash is False

    def test_is_idempotent(self):
        cfg = self._cfg(False)
        disable_flash_attention(cfg)
        assert cfg.diffusion.ptv3vanilla.enable_flash is False

    def test_tolerates_missing_blocks(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"diffusion": {}})
        disable_flash_attention(cfg)  # must not raise

    def test_released_checkpoints_really_do_enable_flash(self):
        """Documents *why* the override exists — if upstream ever ships
        enable_flash: false, this test tells us the patch can be dropped."""
        ckpt = os.environ.get("GRASPGENX_CHECKPOINT_DIR")
        if not ckpt or not (Path(ckpt) / "release" / "gen" / "config.yaml").exists():
            pytest.skip("checkpoints not available")
        from omegaconf import OmegaConf

        gen = OmegaConf.load(Path(ckpt) / "release" / "gen" / "config.yaml")
        assert gen.diffusion.ptv3vanilla.enable_flash is True
        assert str(gen.diffusion.object_backbone) == "ptv3vanilla"


@pytest.mark.integration
class TestCpuInference:
    """End-to-end on CPU. Needs the checkpoints; marked so the default run
    stays fast."""

    def test_inference_produces_valid_grasps(self):
        ckpt_root = Path(os.environ.get("GRASPGENX_CHECKPOINT_DIR", "")) / "release"
        if not (ckpt_root / "gen" / "config.yaml").exists():
            pytest.skip("checkpoints not available")

        os.environ["GRASPGENX_DEVICE"] = "cpu"
        from graspgenx.grasp_server import GraspGenXSampler, load_grasp_gen_model
        from graspgenx.utils.checkpoint_io import load_model_cfg

        cfg = load_model_cfg(f"{ckpt_root}/gen", f"{ckpt_root}/dis", None, None)
        model = load_grasp_gen_model(cfg)

        assert next(model.parameters()).device.type == "cpu"
        assert cfg.diffusion.ptv3vanilla.enable_flash is False
        assert cfg.discriminator.ptv3vanilla.enable_flash is False

        sampler = GraspGenXSampler(
            cfg, gripper_name="franka_panda",
            assets_dir=str(REPO / "assets"), model=model,
        )

        rng = np.random.default_rng(0)
        half = np.array([0.03, 0.03, 0.06], np.float32)
        pts = rng.uniform(-1, 1, (2000, 3)).astype(np.float32) * half
        axis = rng.integers(0, 3, 2000)
        sign = rng.choice([-1.0, 1.0], 2000).astype(np.float32)
        pts[np.arange(2000), axis] = sign * half[axis]

        grasps, conf = GraspGenXSampler.run_inference(
            pts, sampler, grasp_threshold=-1.0, num_grasps=20
        )
        grasps = grasps.cpu().numpy()
        conf = conf.cpu().numpy()

        assert grasps.shape[1:] == (4, 4)
        assert np.isfinite(grasps).all()
        assert ((conf >= 0) & (conf <= 1)).all()
        # Rotation blocks must be proper rotations, or downstream IK is garbage.
        for R in grasps[:5, :3, :3]:
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-3)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-3)
