"""run_artifacts: the evidence trail.

If this module is wrong the failure is silent — a run "succeeds" and its
evidence is missing or unreadable, which only surfaces days later when the
analysis will not reproduce. The append-only JSONL behaviour matters especially:
a full OCID-VLG pass cannot finish within one free-tier daily budget, so a
truncating writer would destroy previous days' work.
"""

import json

import numpy as np
import pytest

from run_artifacts import (
    JsonNumpyEncoder,
    NullRecorder,
    RunRecorder,
    append_eval_record,
    read_eval_records,
)


@pytest.fixture
def recorder(tmp_path):
    return RunRecorder(name="unit test/run", root=tmp_path)


class TestLayout:
    def test_creates_the_documented_subdirectories(self, recorder):
        for sub in ("inputs", "masks", "clouds", "grasps", "images"):
            assert (recorder.dir / sub).is_dir()

    def test_directory_name_is_filesystem_safe(self, recorder):
        assert "/" not in recorder.dir.name
        assert recorder.dir.name.endswith("unit_test_run")

    def test_name_is_timestamped_for_ordering(self, recorder):
        assert recorder.dir.name[:4].isdigit()
        assert recorder.dir.name[8] == "T"


class TestConfigSnapshot:
    def test_records_args_and_environment(self, recorder):
        recorder.write_config({"gripper_name": "franka_panda", "num_grasps": 200})
        cfg = json.loads((recorder.dir / "config.json").read_text())
        assert cfg["args"]["gripper_name"] == "franka_panda"
        assert "git" in cfg and "platform" in cfg and "env" in cfg
        assert "timestamp_utc" in cfg

    def test_extra_fields_are_merged(self, recorder):
        recorder.write_config({}, extra={"sent_id": 42})
        cfg = json.loads((recorder.dir / "config.json").read_text())
        assert cfg["sent_id"] == 42


class TestArtifacts:
    def test_saves_inputs(self, recorder, rgb, depth_map, K):
        recorder.save_inputs(image=rgb, depth=depth_map, intrinsics=K,
                             meta={"query": "grasp the box"})
        assert (recorder.dir / "inputs" / "rgb.png").exists()
        assert (recorder.dir / "inputs" / "depth.npy").exists()
        assert (recorder.dir / "inputs" / "K.json").exists()
        assert (recorder.dir / "inputs" / "meta.json").exists()
        # A colourised depth preview makes bad depth obvious at a glance.
        assert (recorder.dir / "images" / "depth_colormap.png").exists()

    def test_saved_depth_round_trips_exactly(self, recorder, depth_map):
        recorder.save_inputs(depth=depth_map)
        back = np.load(recorder.dir / "inputs" / "depth.npy")
        assert np.array_equal(back, depth_map)

    def test_saves_mask_and_cloud(self, recorder, box_mask, synthetic_box_cloud):
        recorder.save_mask(box_mask, "knife handle", round_idx=2)
        recorder.save_cloud(synthetic_box_cloud, "knife handle", round_idx=2)
        assert (recorder.dir / "masks" / "round2_knife_handle.npy").exists()
        assert (recorder.dir / "clouds" / "round2_knife_handle.npy").exists()

    def test_saves_the_full_candidate_set_not_just_the_winner(self, recorder):
        """Score distributions and the effect of the mask filter can only be
        plotted later if the rejected candidates survive."""
        grasps = np.tile(np.eye(4, dtype=np.float32), (25, 1, 1))
        scores = np.linspace(0, 1, 25).astype(np.float32)
        kept = np.array([3, 7, 11])
        recorder.save_grasps(grasps, scores, "obj", 0, kept_indices=kept,
                             meta={"gripper": "franka_panda"})

        data = np.load(recorder.dir / "grasps" / "round0_obj.npz")
        assert data["grasps"].shape == (25, 4, 4)   # all 25, not 3
        assert np.array_equal(data["kept_indices"], kept)
        assert json.loads(str(data["meta_json"]))["gripper"] == "franka_panda"


class TestLlmTrace:
    def test_appends_one_line_per_call(self, recorder):
        for i in range(3):
            recorder.log_llm_call(
                agent="planner", model="gemini-flash", provider="gemini",
                prompt="p", response="r", latency_s=0.5,
                usage={"total_tokens": 10}, retries=0,
            )
        lines = (recorder.dir / "llm_trace.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["agent"] == "planner"
        assert recorder.llm_call_count == 3

    def test_elides_huge_prompts_from_the_middle(self, recorder):
        """Both ends are kept, because both ends carry information.

        These prompts are `str.format()` templates: the static instructions and
        examples come first, and the scene table, blocking analysis and history
        the model is meant to reason from are interpolated at the very end.
        Head-only truncation kept the boilerplate and dropped the evidence,
        which made a trace useless for deciding whether a bad decision was the
        model's fault or the prompt's.
        """
        head, tail = "HEAD" + "x" * 50_000, "y" * 50_000 + "TAIL"
        recorder.log_llm_call(agent="coder", model="m", provider="p",
                              prompt=head + tail, response=head + tail,
                              latency_s=0.1)
        rec = json.loads((recorder.dir / "llm_trace.jsonl").read_text())

        for field in ("prompt", "response"):
            text = rec[field]
            assert len(text) < 12_000, f"{field} was not elided"
            assert text.startswith("HEAD")
            assert text.endswith("TAIL")
            assert "elided" in text

    def test_short_prompts_are_stored_whole(self, recorder):
        recorder.log_llm_call(agent="coder", model="m", provider="p",
                              prompt="short prompt", response="short reply",
                              latency_s=0.1)
        rec = json.loads((recorder.dir / "llm_trace.jsonl").read_text())
        assert rec["prompt"] == "short prompt"
        assert "elided" not in rec["response"]

    def test_records_errors_and_retries(self, recorder):
        recorder.log_llm_call(agent="observer", model="m", provider="p",
                              prompt="p", response="", latency_s=0.0,
                              retries=3, error="429 rate limit")
        rec = json.loads((recorder.dir / "llm_trace.jsonl").read_text())
        assert rec["retries"] == 3 and "429" in rec["error"]


class TestTimingAndResult:
    def test_stage_context_manager_accumulates(self, recorder):
        with recorder.stage("planner"):
            pass
        with recorder.stage("planner"):
            pass
        recorder.finish(result=None, status="no_grasp")
        timings = json.loads((recorder.dir / "timings.json").read_text())
        assert "planner" in timings["stages_s"]
        assert "total_s" in timings

    def test_finish_writes_result(self, recorder, identity_grasp):
        recorder.finish(result=identity_grasp.as_dict(), status="success")
        res = json.loads((recorder.dir / "result.json").read_text())
        assert res["status"] == "success"
        assert res["result"]["gripper"] == "franka_panda"

    def test_log_writes_lines(self, recorder):
        recorder.log("hello")
        assert "hello" in (recorder.dir / "log.txt").read_text()


class TestNumpyEncoder:
    def test_encodes_numpy_scalars_and_arrays(self):
        payload = {"i": np.int64(3), "f": np.float32(1.5),
                   "a": np.arange(3), "b": np.bool_(True)}
        out = json.loads(json.dumps(payload, cls=JsonNumpyEncoder))
        assert out == {"i": 3, "f": 1.5, "a": [0, 1, 2], "b": True}


class TestNullRecorder:
    def test_is_inert_but_api_compatible(self, rgb, box_mask):
        r = NullRecorder()
        assert not r.enabled
        r.write_config({"a": 1})
        r.save_inputs(image=rgb)
        r.save_mask(box_mask, "x")
        r.log_llm_call(agent="a", model="m", provider="p", prompt="",
                       response="", latency_s=0.0)
        with r.stage("s"):
            pass
        r.finish(result=None)  # must not raise or write anything


class TestEvalJsonl:
    def test_append_and_read(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        append_eval_record(p, {"sent_id": 1, "status": "success"})
        append_eval_record(p, {"sent_id": 2, "status": "no_grasp"})
        records = read_eval_records(p)
        assert [r["sent_id"] for r in records] == [1, 2]

    def test_appending_preserves_previous_runs(self, tmp_path):
        """Resumability depends on this: a truncating writer would silently
        destroy the previous days' evaluation."""
        p = tmp_path / "eval.jsonl"
        append_eval_record(p, {"sent_id": 1})
        append_eval_record(p, {"sent_id": 2})
        assert len(read_eval_records(p)) == 2

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_eval_records(tmp_path / "nope.jsonl") == []

    def test_tolerates_a_truncated_final_line(self, tmp_path):
        """An interrupted run leaves a half-written line; resuming must not die."""
        p = tmp_path / "eval.jsonl"
        append_eval_record(p, {"sent_id": 1})
        with open(p, "a") as f:
            f.write('{"sent_id": 2, "sta')
        records = read_eval_records(p)
        assert len(records) == 1

    def test_numpy_values_are_serializable(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        append_eval_record(p, {"sent_id": np.int64(7), "iou": np.float32(0.42)})
        assert read_eval_records(p)[0]["sent_id"] == 7
