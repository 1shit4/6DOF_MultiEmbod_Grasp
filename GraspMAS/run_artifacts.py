"""Structured capture of everything a run produces, under `outputs/`.

Terminal scrollback is not evidence. Every stage that matters writes here, so
plots, failure triage and the SUMMARY.md evaluation can be regenerated later
without re-running anything (which on a rate-limited free LLM tier is not just
slow but budget-limited).

Layout — see outputs/README.md for the full schema:

    outputs/runs/<UTC>_<name>/
        config.json     versions, flags, model ids, git SHAs
        inputs/         the exact rgb / depth / K used
        masks/          every mask the grounding stack produced
        clouds/         every point cloud sent to the server
        grasps/         ALL candidates + scores, not just the winner
        images/         overlays, candidate fan, depth colormap
        llm_trace.jsonl one line per LLM call
        timings.json    per-stage wall time
        result.json     the final Grasp6D, or the failure
        log.txt
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _git_sha(path: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class JsonNumpyEncoder(json.JSONEncoder):
    """numpy types are not JSON-serializable; every artifact here has them."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


class RunRecorder:
    """One run = one directory. Thread-safe appends; safe to disable."""

    def __init__(
        self,
        name: str = "run",
        root: Optional[str | os.PathLike] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.name = name
        self._lock = threading.Lock()
        self._t_start = time.time()
        self._stage_times: dict[str, float] = {}
        self._counters: dict[str, int] = {}

        if not enabled:
            self.dir = Path(os.devnull).parent
            return

        base = Path(root) if root else _repo_root() / "outputs" / "runs"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
        self.dir = base / f"{stamp}_{safe}"
        for sub in ("inputs", "masks", "clouds", "grasps", "images"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self._log_path = self.dir / "log.txt"
        logger.info("Recording run artifacts to %s", self.dir)

    # -- helpers -----------------------------------------------------------

    def _write_json(self, rel: str, obj: Any) -> None:
        if not self.enabled:
            return
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, cls=JsonNumpyEncoder)

    def _append_jsonl(self, rel: str, obj: Any) -> None:
        if not self.enabled:
            return
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(path, "a") as f:
            f.write(json.dumps(obj, cls=JsonNumpyEncoder) + "\n")

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        line = f"[{time.time() - self._t_start:8.2f}s] {message}"
        with self._lock, open(self._log_path, "a") as f:
            f.write(line + "\n")

    # -- config ------------------------------------------------------------

    def write_config(self, args: dict, extra: Optional[dict] = None) -> None:
        """Snapshot of *how* this run was produced. Without it results are
        unattributable once flags or model ids drift."""
        root = _repo_root()
        cfg = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "name": self.name,
            "args": args,
            "git": {
                "GraspMAS": _git_sha(root / "GraspMAS"),
                "GraspGenX": _git_sha(root / "GraspGenX"),
            },
            "platform": {
                "python": platform.python_version(),
                "system": platform.system(),
                "release": platform.release(),
                "cpu_count": os.cpu_count(),
            },
            "env": {
                k: os.environ.get(k)
                for k in (
                    "GRASPGENX_DEVICE",
                    "GRASPGENX_CHECKPOINT_DIR",
                    "GRASPGENX_GRIPPER_CFG_DIR",
                    "GRASPGEN_SERVER_HOST",
                    "GRASPGEN_SERVER_PORT",
                    "OMP_NUM_THREADS",
                )
            },
        }
        try:
            import torch
            cfg["platform"]["torch"] = torch.__version__
            cfg["platform"]["cuda_available"] = torch.cuda.is_available()
        except Exception:
            pass
        if extra:
            cfg.update(extra)
        self._write_json("config.json", cfg)

    # -- inputs ------------------------------------------------------------

    def save_inputs(
        self,
        image: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
    ) -> None:
        if not self.enabled:
            return
        import cv2

        if image is not None:
            cv2.imwrite(str(self.dir / "inputs" / "rgb.png"),
                        np.asarray(image)[:, :, ::-1])
        if depth is not None:
            np.save(self.dir / "inputs" / "depth.npy", np.asarray(depth, np.float32))
            try:
                from vis6d import depth_colormap
                depth_colormap(depth, self.dir / "images" / "depth_colormap.png")
            except Exception as exc:
                logger.debug("depth colormap failed: %s", exc)
        if intrinsics is not None:
            self._write_json("inputs/K.json", np.asarray(intrinsics).tolist())
        if meta:
            self._write_json("inputs/meta.json", meta)

    # -- per-call artifacts ------------------------------------------------

    def save_mask(self, mask: np.ndarray, label: str, round_idx: int = 0) -> None:
        if not self.enabled:
            return
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        np.save(self.dir / "masks" / f"round{round_idx}_{safe}.npy",
                np.asarray(mask).astype(bool))

    def save_cloud(self, cloud: np.ndarray, label: str, round_idx: int = 0) -> None:
        if not self.enabled:
            return
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        np.save(self.dir / "clouds" / f"round{round_idx}_{safe}.npy",
                np.asarray(cloud, np.float32))

    def save_grasps(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        label: str,
        round_idx: int = 0,
        kept_indices: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
    ) -> None:
        """Save the FULL candidate set, not only the winner.

        Score distributions, approach diversity and the effect of the
        mask-containment filter can only be plotted if the rejects survive.
        """
        if not self.enabled:
            return
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        payload = {
            "grasps": np.asarray(grasps, np.float32),
            "scores": np.asarray(scores, np.float32),
        }
        if kept_indices is not None:
            payload["kept_indices"] = np.asarray(kept_indices, np.int64)
        if meta:
            payload["meta_json"] = np.array(json.dumps(meta, cls=JsonNumpyEncoder))
        np.savez_compressed(
            self.dir / "grasps" / f"round{round_idx}_{safe}.npz", **payload
        )

    def image_path(self, filename: str) -> Path:
        return self.dir / "images" / filename

    @property
    def images_dir(self) -> Path:
        return self.dir / "images"

    # -- LLM trace ---------------------------------------------------------

    def log_llm_call(
        self,
        agent: str,
        model: str,
        provider: str,
        prompt: str,
        response: str,
        latency_s: float,
        usage: Optional[dict] = None,
        retries: int = 0,
        error: Optional[str] = None,
        has_image: bool = False,
    ) -> None:
        self._counters["llm_calls"] = self._counters.get("llm_calls", 0) + 1
        self._append_jsonl(
            "llm_trace.jsonl",
            {
                "i": self._counters["llm_calls"],
                "t": round(time.time() - self._t_start, 3),
                "agent": agent,
                "provider": provider,
                "model": model,
                "has_image": has_image,
                "latency_s": round(latency_s, 3),
                "retries": retries,
                "usage": usage or {},
                "error": error,
                # Truncated: full prompts are large and mostly static boilerplate.
                "prompt": prompt[:4000],
                "response": response[:4000],
            },
        )

    @property
    def llm_call_count(self) -> int:
        return self._counters.get("llm_calls", 0)

    # -- timings and result ------------------------------------------------

    def time_stage(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._stage_times[stage] = self._stage_times.get(stage, 0.0) + seconds

    def stage(self, name: str):
        """`with recorder.stage("grasp_inference"): ...`"""
        recorder = self

        class _Timer:
            def __enter__(self):
                self.t = time.time()
                return self

            def __exit__(self, *exc):
                recorder.time_stage(name, time.time() - self.t)
                return False

        return _Timer()

    def finish(self, result: Any, status: str = "success", error: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self._write_json(
            "timings.json",
            {
                "total_s": round(time.time() - self._t_start, 3),
                "stages_s": {k: round(v, 3) for k, v in self._stage_times.items()},
                "llm_calls": self.llm_call_count,
            },
        )
        self._write_json(
            "result.json",
            {
                "status": status,
                "error": error,
                "result": result,
                "elapsed_s": round(time.time() - self._t_start, 3),
                "llm_calls": self.llm_call_count,
            },
        )
        logger.info("Run artifacts written to %s", self.dir)


class NullRecorder(RunRecorder):
    """No-op recorder for unit tests and library use."""

    def __init__(self):
        super().__init__(name="null", enabled=False)


def append_eval_record(path: str | os.PathLike, record: dict) -> None:
    """Append one evaluation row as JSONL.

    Append-only on purpose: a full OCID-VLG pass cannot finish inside one
    free-tier daily request budget, so runs must be resumable and must never
    truncate what previous days produced.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, cls=JsonNumpyEncoder) + "\n")


def read_eval_records(path: str | os.PathLike) -> list[dict]:
    """Read a JSONL evaluation log, tolerating a truncated final line."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", p)
    return out
