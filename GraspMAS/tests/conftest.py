"""Shared fixtures.

Design rule for this suite: the default `pytest` run must be **fully offline**
and must spend **zero LLM requests**. Anything needing the network, the ZMQ
server or a model download is marked and deselected by default, so the suite
stays usable as a fast feedback loop while the free-tier request budget is
reserved for real evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs a live GraspGen-X server")
    config.addinivalue_line("markers", "llm: spends free-tier LLM requests")
    config.addinivalue_line("markers", "slow: downloads model weights or is slow")


@pytest.fixture
def K():
    """A realistic 640x480 pinhole matrix."""
    return np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


@pytest.fixture
def depth_map():
    """A 480x640 depth map: a flat plane at 1.0 m with a 0.6 m box in the middle.

    Includes a band of zeros to exercise invalid-depth handling, since real
    sensors return 0 for "no measurement" and that must never become a point.
    """
    d = np.full((480, 640), 1.0, dtype=np.float32)
    d[200:280, 280:360] = 0.6
    d[:, :40] = 0.0  # invalid strip
    return d


@pytest.fixture
def box_mask():
    """Boolean mask selecting the 0.6 m box region."""
    m = np.zeros((480, 640), dtype=bool)
    m[200:280, 280:360] = True
    return m


@pytest.fixture
def rgb():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[200:280, 280:360] = (200, 120, 60)
    return img


@pytest.fixture
def identity_grasp():
    """A grasp at (0, 0, 0.6) m approaching straight along +Z."""
    from perception3d import Grasp6D

    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 0.6]
    return Grasp6D(pose=pose, score=0.9, gripper="franka_panda", width=0.08)


@pytest.fixture
def synthetic_box_cloud():
    """Surface points of a 6x6x12 cm box at 60 cm, deterministic."""
    rng = np.random.default_rng(0)
    half = np.array([0.03, 0.03, 0.06], np.float32)
    n = 2000
    pts = rng.uniform(-1, 1, (n, 3)).astype(np.float32) * half
    axis = rng.integers(0, 3, n)
    sign = rng.choice([-1.0, 1.0], n).astype(np.float32)
    pts[np.arange(n), axis] = sign * half[axis]
    return pts + np.array([0.0, 0.0, 0.60], np.float32)
