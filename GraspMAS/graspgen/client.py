"""Minimal ZMQ client for a GraspGen-X inference server.

This is a deliberate re-implementation of the wire protocol rather than an
import of ``graspgenx.serving.zmq_client``. That module is itself torch-free,
but importing it executes ``graspgenx/serving/__init__.py``, which pulls in
``zmq_server`` -> ``grasp_server`` -> **torch** plus GraspGen-X's pinned
``diffusers``/``huggingface-hub``. Those pins conflict with the
``transformers==4.48`` stack GraspMAS needs, which is the whole reason the two
frameworks live in separate conda environments.

Wire protocol (see GraspGenX/client-server/README.md): msgpack with
``msgpack_numpy.patch()`` over a ZMQ REQ/REP socket. We only use three actions:
``health``, ``metadata`` and ``infer``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

DEFAULT_HOST = os.environ.get("GRASPGEN_SERVER_HOST", "localhost")
DEFAULT_PORT = int(os.environ.get("GRASPGEN_SERVER_PORT", "5556"))


class GraspGenUnavailable(RuntimeError):
    """The server is unreachable, timed out, or returned an error.

    Raised as a single type so callers (notably ``ImagePatch.grasp_detection``)
    can degrade gracefully into "no grasp found" rather than crashing the whole
    agent loop over a transient socket problem.
    """


class GraspGenClient:
    """Thin, synchronous REQ client. One socket, reconnected after timeouts."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_ms: int = 180_000,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self._sock: Optional[zmq.Socket] = None
        self._metadata_cache: Optional[dict] = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def connect(self) -> None:
        if self._sock is not None:
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.address)
        self._sock = sock
        logger.info("Connected to GraspGen-X server at %s", self.address)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def __enter__(self) -> "GraspGenClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- transport ---------------------------------------------------------

    def _request(self, payload: dict) -> dict:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as exc:
            # A REQ socket is stuck in the wrong state after a timeout; drop it
            # so the next call starts from a clean socket.
            self.close()
            raise GraspGenUnavailable(
                f"GraspGen-X server at {self.address} did not respond within "
                f"{self.timeout_ms} ms"
            ) from exc
        except zmq.ZMQError as exc:
            self.close()
            raise GraspGenUnavailable(f"ZMQ error talking to {self.address}: {exc}") from exc

        try:
            response = msgpack.unpackb(raw, raw=False)
        except Exception as exc:  # malformed frame
            self.close()
            raise GraspGenUnavailable(f"Malformed reply from {self.address}: {exc}") from exc

        if not isinstance(response, dict):
            raise GraspGenUnavailable(
                f"Expected a dict reply, got {type(response).__name__}"
            )
        if "error" in response:
            raise GraspGenUnavailable(f"Server error: {response['error']}")
        return response

    # -- actions -----------------------------------------------------------

    def health(self) -> dict:
        return self._request({"action": "health"})

    def is_alive(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except GraspGenUnavailable:
            return False

    @property
    def metadata(self) -> dict:
        if self._metadata_cache is None:
            self._metadata_cache = self._request({"action": "metadata"})
        return self._metadata_cache

    def infer(
        self,
        point_cloud: np.ndarray,
        gripper_name: Optional[str] = None,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        planner: str = "graspmoe",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 6-DoF grasps for a segmented object point cloud.

        Args:
            point_cloud: (N, 3) float32, **metres**, in whatever frame you want
                the grasps back in (GraspMAS passes a camera-frame cloud).
            gripper_name: any gripper under the server's assets root, e.g.
                ``franka_panda``. ``None`` uses the server default.
            num_grasps: diffusion samples to draw.
            grasp_threshold: discriminator score cutoff; ``-1.0`` disables it
                and falls back to top-k.
            topk_num_grasps: cap on returned grasps; ``-1`` keeps everything.
            planner: ``graspmoe`` (diffusion union OBB candidates, all scored by
                the discriminator) or ``diffusion`` (diffusion only).

        Returns:
            ``(grasps, confidences)`` — ``(K, 4, 4)`` float32 SE(3) poses and
            ``(K,)`` float32 scores in [0, 1], sorted by the server. Poses are
            anchored at the **gripper base**, +Z approach, +X closing.
            Both arrays are empty when nothing was found.
        """
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3); got {pc.shape}")
        if not np.isfinite(pc).all():
            raise ValueError("point_cloud contains non-finite values")
        if planner not in ("diffusion", "graspmoe"):
            raise ValueError(f"planner must be 'diffusion' or 'graspmoe'; got {planner!r}")

        payload = {
            "action": "infer",
            "point_cloud": pc,
            "num_grasps": int(num_grasps),
            "grasp_threshold": float(grasp_threshold),
            "topk_num_grasps": int(topk_num_grasps),
            "planner": planner,
        }
        if gripper_name is not None:
            payload["gripper_name"] = str(gripper_name)

        response = self._request(payload)

        grasps = np.asarray(response.get("grasps"), dtype=np.float32)
        conf = np.asarray(response.get("confidences"), dtype=np.float32)
        if grasps.size == 0:
            return (
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        grasps = grasps.reshape(-1, 4, 4)
        conf = conf.reshape(-1)
        if len(grasps) != len(conf):
            raise GraspGenUnavailable(
                f"Server returned {len(grasps)} grasps but {len(conf)} scores"
            )
        return grasps, conf
