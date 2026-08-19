"""graspgen.client: the ZMQ wire protocol, tested against a mock server.

Runs a real ZMQ REP socket in a thread rather than mocking the socket object, so
the msgpack framing and numpy round-trip are exercised for real — those are
exactly where a wire protocol breaks — while needing neither the model nor a GPU.
"""

import threading
import time

import msgpack
import msgpack_numpy
import numpy as np
import pytest
import zmq

from graspgen import GraspGenClient, GraspGenUnavailable

msgpack_numpy.patch()


class MockServer:
    """Minimal REP server. `handler(request) -> response` decides the reply."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list = []
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if dict(poller.poll(50)).get(self._sock) != zmq.POLLIN:
                continue
            request = msgpack.unpackb(self._sock.recv(), raw=False)
            self.requests.append(request)
            reply = self.handler(request)
            if reply is not None:
                self._sock.send(msgpack.packb(reply, use_bin_type=True))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close(linger=0)
        self._ctx.term()


def make_grasps(k=3):
    grasps = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    for i in range(k):
        grasps[i, :3, 3] = [0.01 * i, 0.0, 0.6]
    scores = np.linspace(0.2, 0.9, k).astype(np.float32)
    return grasps, scores


def ok_handler(request):
    action = request.get("action")
    if action == "health":
        return {"status": "ok"}
    if action == "metadata":
        return {"default_gripper": "franka_panda",
                "model": {"generator_backbone": "ptv3vanilla"}}
    if action == "infer":
        g, s = make_grasps()
        return {"grasps": g, "confidences": s, "gripper_name": "franka_panda",
                "planner": request.get("planner"), "branch_tags": ["diff"] * len(g),
                "timing": {"infer_ms": 1.0}}
    return {"error": f"unknown action {action}"}


class TestBasicActions:
    def test_health(self):
        with MockServer(ok_handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                assert c.health() == {"status": "ok"}

    def test_is_alive_true(self):
        with MockServer(ok_handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                assert c.is_alive() is True

    def test_is_alive_false_when_no_server(self):
        c = GraspGenClient(port=1, timeout_ms=300)
        assert c.is_alive() is False

    def test_metadata_is_cached(self):
        with MockServer(ok_handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                assert c.metadata["default_gripper"] == "franka_panda"
                _ = c.metadata
                assert sum(r["action"] == "metadata" for r in srv.requests) == 1


class TestInferWireFormat:
    def test_payload_matches_protocol(self, synthetic_box_cloud):
        with MockServer(ok_handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                c.infer(synthetic_box_cloud, gripper_name="robotiq_2f_85",
                        num_grasps=50, grasp_threshold=0.7,
                        topk_num_grasps=10, planner="diffusion")
        req = srv.requests[-1]
        assert req["action"] == "infer"
        assert req["gripper_name"] == "robotiq_2f_85"
        assert req["num_grasps"] == 50
        assert req["grasp_threshold"] == pytest.approx(0.7)
        assert req["topk_num_grasps"] == 10
        assert req["planner"] == "diffusion"
        # numpy must survive msgpack intact, not degrade to nested lists.
        assert isinstance(req["point_cloud"], np.ndarray)
        assert req["point_cloud"].shape == synthetic_box_cloud.shape
        assert req["point_cloud"].dtype == np.float32

    def test_gripper_name_omitted_when_none(self, synthetic_box_cloud):
        with MockServer(ok_handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                c.infer(synthetic_box_cloud)
        assert "gripper_name" not in srv.requests[-1]

    def test_returns_shaped_arrays(self, synthetic_box_cloud):
        with MockServer(ok_handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                grasps, scores = c.infer(synthetic_box_cloud)
        assert grasps.shape == (3, 4, 4)
        assert scores.shape == (3,)
        assert grasps.dtype == np.float32

    def test_empty_result_has_correct_shape(self, synthetic_box_cloud):
        """An empty result must still be (0,4,4), so callers can len() it."""
        def handler(req):
            if req["action"] == "infer":
                return {"grasps": np.zeros((0, 4, 4), np.float32),
                        "confidences": np.zeros((0,), np.float32)}
            return ok_handler(req)

        with MockServer(handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                grasps, scores = c.infer(synthetic_box_cloud)
        assert grasps.shape == (0, 4, 4)
        assert scores.shape == (0,)


class TestValidation:
    def test_rejects_wrong_shape(self):
        c = GraspGenClient(port=1)
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            c.infer(np.zeros((10, 4), np.float32))

    def test_rejects_non_finite_cloud(self):
        pc = np.zeros((10, 3), np.float32)
        pc[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            GraspGenClient(port=1).infer(pc)

    def test_rejects_unknown_planner(self, synthetic_box_cloud):
        with pytest.raises(ValueError, match="planner"):
            GraspGenClient(port=1).infer(synthetic_box_cloud, planner="nonsense")


class TestErrorHandling:
    def test_server_error_becomes_unavailable(self, synthetic_box_cloud):
        with MockServer(lambda r: {"error": "ValueError: bad gripper"}) as srv:
            with GraspGenClient(port=srv.port) as c:
                with pytest.raises(GraspGenUnavailable, match="bad gripper"):
                    c.infer(synthetic_box_cloud)

    def test_timeout_becomes_unavailable(self, synthetic_box_cloud):
        def slow(req):
            time.sleep(2.0)
            return {"status": "ok"}

        with MockServer(slow) as srv:
            c = GraspGenClient(port=srv.port, timeout_ms=200)
            with pytest.raises(GraspGenUnavailable, match="did not respond"):
                c.health()

    def test_socket_is_reset_after_timeout(self):
        """A REQ socket is unusable after a timeout; the client must drop it,
        otherwise every later call fails even once the server recovers."""
        state = {"slow": True}

        def handler(req):
            if state["slow"]:
                state["slow"] = False
                time.sleep(1.5)
            return {"status": "ok"}

        with MockServer(handler) as srv:
            c = GraspGenClient(port=srv.port, timeout_ms=300)
            with pytest.raises(GraspGenUnavailable):
                c.health()
            assert c._sock is None  # dropped, so the retry starts clean

    def test_malformed_reply_becomes_unavailable(self, synthetic_box_cloud):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        stop = threading.Event()

        def serve():
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while not stop.is_set():
                if dict(poller.poll(50)).get(sock) == zmq.POLLIN:
                    sock.recv()
                    sock.send(b"this is not msgpack \xff\xfe")
        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            c = GraspGenClient(port=port, timeout_ms=1000)
            with pytest.raises(GraspGenUnavailable, match="Malformed"):
                c.health()
        finally:
            stop.set()
            t.join(timeout=2)
            sock.close(linger=0)
            ctx.term()

    def test_score_count_mismatch_is_caught(self, synthetic_box_cloud):
        def handler(req):
            if req["action"] == "infer":
                g, _ = make_grasps(3)
                return {"grasps": g, "confidences": np.zeros((2,), np.float32)}
            return ok_handler(req)

        with MockServer(handler) as srv:
            with GraspGenClient(port=srv.port) as c:
                with pytest.raises(GraspGenUnavailable, match="scores"):
                    c.infer(synthetic_box_cloud)
