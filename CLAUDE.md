# 6dof_GraspMAS — project memory

> **This file is the operational memory of the project.** It is updated at the end of every work
> session. Read it first; it should be enough to resume work without re-deriving anything.

---

## 1. Goal

Upgrade **GraspMAS** — a language-driven grasp-detection multi-agent system — by replacing its **planar
2D grasp head (RAGT-3-3)**, which emits a rotated rectangle `[quality, x, y, w, h, angle]`, with
**GraspGen-X**, NVIDIA's cross-embodiment diffusion model that emits true **6-DoF SE(3) grasps** for any
of 27+ grippers.

Three hard constraints shape every decision here:

1. **No GPU.** 8 CPU cores, 15 GB RAM, ~50 GB free disk. Everything must run on CPU.
2. **No paid API.** The original code calls GPT-4o / GPT-4o-mini everywhere. Replaced with the
   **Google Gemini Flash free tier** via its OpenAI-compatible endpoint.
3. **Keep what makes GraspMAS valuable** — the Planner/Coder/Observer loop, language grounding, and
   *part-level* reasoning ("grasp the knife **by the handle**"). The 6-DoF sampler must inherit the
   language constraint, not bypass it.

Out of scope (user decision): the ManiSkill demo (`Maniskill_demo.ipynb`, `mani_skill_pick_YCB/`).
SAPIEN needs a Vulkan GPU to render camera images, so it cannot run on this machine. It will be
ported later on a GPU box.

---

## 2. System architecture

Two conda environments, bridged by the ZMQ client/server layer that GraspGen-X already ships.

```
┌─────────────────────── env: graspmas (py3.10) ───────────────────────┐
│  GraspMAS                                                            │
│    Planner ─ Coder ─ Observer   (Gemini Flash, free tier)            │
│    ImagePatch tools: GroundingDINO-tiny · SAM-B · VLPart · DPT       │
│    perception3d.py   depth + intrinsics → metric point cloud         │
│    graspgen/client.py  ── torch-free ZMQ client ──┐                  │
└───────────────────────────────────────────────────┼──────────────────┘
                                                    │ msgpack / ZMQ REQ-REP
                                                    │ tcp://localhost:5556
┌───────────────────────── env: graspgenx (py3.11) ─▼──────────────────┐
│  GraspGen-X server — model loaded ONCE, CPU, fp32                    │
│    PTv3-vanilla encoder → diffusion (20 DDPM steps) → discriminator  │
└──────────────────────────────────────────────────────────────────────┘
```

### Why two environments

1. **Dependency isolation.** GraspMAS needs `transformers==4.48` + detectron2; GraspGen-X pins
   `diffusers==0.11.1` + `huggingface-hub==0.25.2` (the latter because diffusers 0.11.1 imports
   `cached_download`, removed in hub 0.26). Co-installing is possible but brittle.
2. **Model load amortized.** The checkpoint is 1.6 GB. GraspMAS calls `grasp_detection` up to
   `max_round` times per query and `main_batch.py` runs many samples — reloading per call on CPU would
   dominate runtime. A persistent server loads it once.
3. **Memory headroom.** Two processes at ~3 GB each fits comfortably in 15 GB; one process holding both
   stacks would be tight.

### Data flow, one `grasp_detection` call

```
RGB (+ depth?) ──▶ GroundingDINO / SAM / VLPart ──▶ mask  (object OR part)
                                                     │
                 depth: real if supplied, else Depth-Anything-V2-Metric  +  K
                                                     ▼
                   unproject masked pixels  →  (N,3) cloud, METRES, CAMERA frame
                                                     ▼
                  ZMQ "infer" { point_cloud, gripper_name, num_grasps, planner }
                                                     ▼
                       GraspGen-X (CPU)  →  (K,4,4) SE(3) poses + (K,) scores
                                                     ▼
              drop grasps whose projected centre falls outside the mask
                          rank by discriminator score → best Grasp6D
                                                     ▼
                   render projected gripper PNG ──▶ Observer critique loop
```

The mask is the coupling point. When it comes from `find_part`, the cloud *is* the part, so the
language constraint reaches the sampler directly.

---

## 3. Repo map

```
6dof_GraspMAS/
├── CLAUDE.md            ← this file
├── SUMMARY.md           ← robotics-perspective evaluation
├── GraspMAS/            fork of Fsoft-AIC/GraspMAS @ b15f178 (2025-10-26)
├── GraspGenX/           NVlabs/GraspGenX @ b942909 (2026-07-14) + CPU patch
├── assets/
│   ├── graspgenx_checkpoints/release/{gen,dis}/   → $GRASPGENX_CHECKPOINT_DIR
│   └── gripper_descriptions/                      → $GRASPGENX_GRIPPER_CFG_DIR
├── outputs/             all run artifacts (see outputs/README.md)
├── patches/             graspgenx_cpu.patch
├── env/                 environment recipes
└── scripts/             setup.sh, run_server.sh, bench_cpu.py, run_tests.sh, plot_outputs.py
```

### Files by provenance

| Status | Files |
|---|---|
| **New** (ours) | `GraspMAS/perception3d.py`, `GraspMAS/vis6d.py`, `GraspMAS/graspgen/client.py`, `GraspMAS/run_artifacts.py`, `GraspMAS/llm_config.yaml`, `GraspMAS/tests/*` |
| **Forked** (modified upstream) | `GraspMAS/image_patch.py`, `GraspMAS/agents/*.py`, `GraspMAS/agents/prompt/*.py`, `GraspMAS/main_simple.py`, `GraspMAS/main_batch.py`, `GraspMAS/utils.py`, `GraspMAS/vlpart/vlpart.py`, `GraspGenX/graspgenx/grasp_server.py`, `GraspGenX/client-server/graspgenx_server.py`, `GraspGenX/graspgenx/serving/zmq_server.py` |
| **Deleted** | `GraspMAS/grasp/{model,model_config,transformer,grasp_detect_multibox,grasp_detect_singlebox,unit_grasp_pose_generation}.py` (the RAGT-3-3 2D path) |
| **Untouched upstream** | everything else, notably `GraspMAS/detectron2/`, `GraspMAS/vlpart/*` (except the one `map_location` fix), all of `GraspGenX/graspgenx/models/` |

---

## 4. Environment recipe

Reproducible from scratch. `git-lfs` was installed via conda-forge into `base`; `uv` is **not** used.

```bash
conda install -y -c conda-forge git-lfs && git lfs install --skip-repo

git clone --recurse-submodules --depth 1 https://github.com/Fsoft-AIC/GraspMAS.git  GraspMAS
git clone --depth 1                      https://github.com/NVlabs/GraspGenX.git    GraspGenX
```

**git-lfs must be installed before cloning.** GraspGen-X's `.gitattributes` puts `*.obj *.stl *.dae
*.npy` under LFS; without it you get 132-byte pointer files instead of gripper meshes and
`assets/sample_data/*/depth.npy`. The HF `gripper_descriptions` clone additionally needed an explicit
`git lfs pull` afterwards — the smudge filter did not fire on clone.

### env `graspgenx` (Python 3.11)

```bash
conda create -y -n graspgenx python=3.11
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 omegaconf hydra-core trimesh==4.5.3 h5py scikit-learn scipy imageio \
            webdataset yourdfpy==0.0.56 diffusers==0.11.1 huggingface-hub==0.25.2 timm==1.0.15 \
            addict viser pyyaml tqdm tensordict matplotlib pyzmq msgpack msgpack-numpy pytest \
            "setuptools<78"
pip install --no-deps -e ./GraspGenX
```

`--no-deps` is deliberate. `GraspGenX/pyproject.toml` declares packages **no file in the repo
imports**: `urdfpy` (broken — pins `networkx==2.2`, needs `np.float` removed in numpy ≥1.24),
`sharedarray`, `torch-geometric`, plus `pyrender` / `PyOpenGL==3.1.5` / `scene-synthesizer`, which are
only reachable from `dataset/renderer.py` and the USD mesh branch — neither is on the inference path.

### env `graspmas` (Python 3.10)

```bash
conda create -y -n graspmas python=3.10
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.48.0 accelerate numpy==1.26.4 opencv-python shapely scikit-image scipy \
            matplotlib gdown openai jupyter pyzmq msgpack msgpack-numpy pytest pyyaml \
            git+https://github.com/openai/CLIP.git
pip install -e ./GraspMAS/detectron2      # builds CPU-only C++ kernels; needs gcc (11.4 present)
```

`bitsandbytes==0.42.0` is **dropped** from `requirements.txt` — CUDA-only wheel, nothing imports it.

### Environment variables

```bash
export GRASPGENX_CHECKPOINT_DIR=<repo>/assets/graspgenx_checkpoints
export GRASPGENX_GRIPPER_CFG_DIR=<repo>/assets/gripper_descriptions
export GRASPGENX_DEVICE=cpu
export LLM_API_KEY=<free Gemini key from Google AI Studio>   # or GraspMAS/api.key
```

Setting the two asset vars pre-empts the auto-`git clone` in `graspgenx/_setup_dependencies.py`, which
would otherwise pull ~1.7 GB into `GraspGenX/ext/` on first import.

`GRASPGENX_GRIPPER_CFG_DIR` points at the **checkout root**, not the inner package —
`get_gripper_descriptions_assets()` appends `gripper_descriptions/assets/x_grippers`.

---

## 5. Key invariants

Violating any of these silently produces wrong grasps.

* **Units: metres.** Everywhere — point clouds, depth, gripper geometry, grasp translations.
  OCID-VLG depth arrives in millimetres and is divided by 1000 in `OCID_VLG/dataset.py:240-242`.
* **Frame: camera.** GraspGen-X returns poses in the frame of the point cloud it was given. We feed a
  camera-frame cloud, so grasps come back in the camera frame. No world transform is applied.
* **Gripper frame:** the pose is anchored at the **gripper base**, not the fingertips.
  `+Z` = approach axis (along the fingers), `+X` = closing direction. Base→TCP is
  `translation([0, 0, fingertip_depth])`.
* **`Grasp6D` schema** — what `grasp_detection` returns:
  `pose` (4×4 list), `score` (float, discriminator confidence in [0,1]), `gripper` (str),
  `width` (float, jaw aperture m), `position` (3,), `approach` (3, = `pose[:3,2]`),
  `closing` (3, = `pose[:3,0]`), `rect_2d` (`[score,x,y,w,h,angle]`, back-projected for
  visualization and OCID IoU evaluation).
* **ZMQ wire contract:** msgpack with `msgpack_numpy.patch()`, REQ/REP. Actions: `health`, `metadata`,
  `infer`. Documented in `GraspGenX/client-server/README.md`.
* **The Observer needs a PNG.** `agents/observer.py:33-34` reads `execute_results["image"]` and calls
  `encode_image` on it. Any code path that produces a grasp must also write an image, or the Observer
  crashes on `None`.

---

## 6. Known gotchas

Found during the pre-implementation audit of both repos. Each cost real time to discover.

### GraspGen-X

* **`enable_flash: true` in the released checkpoints.** This routes attention through an **fp16**
  `scaled_dot_product_attention` branch (`ptv3_vanilla.py:669-694`). On CPU it fails or silently
  degrades. Must be forced to `False` after config load — this selects the mathematically equivalent
  fp32 matmul+softmax path. Config-only; the state dict is unaffected.
* **9 hard `.cuda()` calls**, all in `graspgenx/grasp_server.py` (`:48`, `:197`, and **seven** in
  `load_gripper_input` at `:454, :457, :464, :474, :481, :492, :503`). Everything else in the repo
  resolves device via `next(model.parameters()).device` and needs no change.
* `scripts/config_xgrasp.yaml` is a **training** config and is never read by inference — it says
  `object_backbone: 'pointnet'`, which is misleading. The real config lives inside the checkpoint
  directories and says `ptv3vanilla`.
* **`pointnet2_ops` is never called.** Its import is soft-failing by design, and `_ext-src/` does not
  exist in the repo, so the JIT fallback cannot fire. Not a CPU blocker.
* **Docs reference files that do not exist:** `scripts/client_server_example_{1,2,3}.py` and
  `demo_scene_pc_fused.py` are cited in `README.md`/`SKILL.md` but absent from the repo.
* `infer_scene_depth` is **params-only** — it rejects `gripper_name`. That is why we use the plain
  `infer` action and unproject client-side.
* **Outlier removal is O(N²) and will OOM-kill the server.**
  `graspgenx/utils/point_cloud.py:knn_points` does `torch.cdist(X, X)` plus a `torch.eye(N)` mask.
  `sample()` calls it on the *raw* cloud before any resampling, so a 40 k-point mask asks for a
  40 k × 40 k matrix (~6 GB) and the kernel kills the process. A close-up object mask on a 1280×720
  depth image reaches that easily. **Mitigation:** `perception3d.downsample_cloud` caps the cloud at
  `MAX_CLOUD_POINTS = 8192` before it leaves the GraspMAS process. Free of cost — the model resamples
  to `cfg.data.num_points` (3500) internally anyway.
* **GraspMoE floods the candidate set with unreachable grasps.** Its OBB branch sweeps every face of
  the fitted box (`36 yaws × 6 Z-offsets = 216 candidates`), including the far side a single depth
  view never observed. Measured on the sample scene the raw candidate median approach elevation is
  **180°** — a hand travelling from behind the object back toward the camera. The discriminator does
  rank these down (top-20 by score had a 76° median), but the mask filter does not remove them, so an
  argmax over in-mask candidates can still select one. **Mitigation:**
  `perception3d.filter_grasps_by_visibility` rejects grasps whose approach points back along the
  viewing ray by more than 100°, applied *before* the mask filter.

### GraspMAS

* **BLIP2-flan-t5-XL** was loaded at import (`image_patch.py:53-57`): a 15.8 GB download, ~7.9 GB
  resident, with a hardcoded `.to(device="cuda")` at `:495`. Its only consumer, `best_image_match`, is
  **never called** by any agent, prompt, or example. Deleted.
* **OWLv2** was loaded at import (`:37-38`) with **zero call sites** anywhere. Deleted.
* `vlpart/vlpart.py:55` — `torch.load(f)` with no `map_location` raises *"Attempting to deserialize
  object on a CUDA device"* on a CPU-only box. Patched.
* `compute_depth` returned `int(np.median(MiDaS_relative_inverse_depth))` — a *relative* sort key that
  cannot be unprojected, and the `int()` cast destroys precision. Note the ordering **inverts** when it
  becomes metric: MiDaS larger = closer, metres smaller = closer.
* **`main_batch.py` cannot import as shipped** — `:16` imports `visualize_grasp_pose` from `utils`, but
  it lives in `grasp/utils.py`.
* `eval_grasp` is imported by `main_batch.py` and **never called**; there was no working quantitative
  harness in the repo.
* `utils.py:28` defaults `dataset_type='GraspAnything'`, selecting the `[_, x, y, w, h, angle]` layout —
  but OCID-VLG rects are `[cx, cy, w, h, theta, target]`, the *other* branch. Wrong default for OCID.
* `agents/planner.py:32` hardcodes `model="gpt-4o"` inside the call, silently ignoring
  `self.model_name`.
* `OCID_VLG/dataset.py` exposes **no camera intrinsics**. OCID ships organized `pcd/` clouds, so `K` is
  recovered by fitting against one and then hardcoded per sub-split.
* `OCID_VLG/dataset.py:126-137` does `os.chdir` inside `_load_dicts` — not async-safe, and
  `main_batch.py:50-55` runs `asyncio.gather`.
* Both notebooks are stale on `main`: they `from prompt import planner_prompt_v3, code_prompt_v2`, a
  package that no longer exists (now `agents/prompt/` with `PLAN`/`CODE`/`EXAMPLES_*`).
* `agents/prompt/coder_prompt.py` — `exists` is advertised with no body (`:39`); `mask` is used by
  `grasp_detection` but missing from the attributes block (`:19-32`); the `find_part` docstring
  (`:238-266`) has a stray `"""`; Example 5 (`:453-466`) references an undefined `building_patches`.
* **The prompt files are `str.format()` templates**, so any literal `{` or `}` in them raises
  `KeyError` at call time. Adding a JSON example to the `grasp_detection` spec broke the Coder this
  way. `tests/test_prompts.py::TestTemplatesFormat` now guards all three templates.
* `ImagePatch.__init__` wrote to a hardcoded relative `imgs/` and crashed if it did not exist; it also
  used the raw object name in the filename, so a name containing `/` produced an invalid path. It
  accepted `name=` but never stored it. All fixed.
* `vlpart_checkpoint` was the bare relative path `weights/swinbase_part_0a0000.pth`, so importing
  `image_patch` only worked when the process started from the GraspMAS directory. Now anchored to
  `BASE_PATH` with an actionable error, overridable via `$VLPART_CHECKPOINT`.
* `timm` is a hard runtime dependency of `vlpart/swintransformer.py` but is absent from
  `requirements.txt`.

### This machine

* **A ROS 2 Humble workspace is sourced**, which puts `/opt/ros/humble/.../python3.10/site-packages`
  on `PYTHONPATH`. Conda does not override `PYTHONPATH`, so those packages leak into both envs — ROS
  ships its own numpy/cv2 and a `launch_testing` pytest plugin that is incompatible with modern
  pytest and aborts collection. `scripts/env.sh` clears `PYTHONPATH` for our processes only; the
  user's ROS shell is untouched. `GraspMAS/pytest.ini` also passes `-p no:launch_testing`.
* **detectron2 must be built with `--no-build-isolation`.** Its `setup.py` imports `torch` at build
  time, and pip's isolated build env does not have it.
* **pip will silently replace CPU torch with a CUDA build.** Installing `timm` pulls `torchvision`
  from the default index, which drags in the latest CUDA torch (2.13.0+cu130 here). Install
  `torch` *and* `torchvision` together from the CPU index, and re-pin `numpy==1.26.4` afterwards —
  a `--force-reinstall` also bumps numpy to 2.x.

### LLM / free tier

* Free tier is ~10-15 RPM and ~1,500 requests/day. One GraspMAS query costs **15-25 requests**
  (3 per round × up to 5 rounds, plus calls from generated code) → roughly **60-100 queries/day**.
  Interactive use is fine; a full OCID-VLG evaluation is not, hence `--limit` and checkpoint/resume.
* Gemini's OpenAI-compat layer does not honour every OpenAI parameter. `agents/coder.py:38-45` passes
  `frequency_penalty` and `presence_penalty=2.0`; these are stripped per provider.
* The upstream response parsers assume GPT-4o formatting and break on other models —
  `planner.py:44-45` (`.split('<thought>')[1]`), `observer.py:64-72` (`json.loads` on possibly
  fenced JSON), `coder.py:47` (`split('\n')[1:-1]` assumes exactly one fenced block).

---

## 7. Progress log

### 2026-08-12 — session 2: implementation and verification

**Done**
* Patched GraspGen-X for CPU (`patches/graspgenx_cpu.patch`, 3 files): 9 `.cuda()`
  sites device-parameterized, `enable_flash` forced off, `--device`/`--threads` on the
  server, and `planner` accepted on the `infer` action so GraspMoE is reachable with
  name-based grippers.
* Benchmarked CPU inference: **2–13 s per call, 2.6 GB peak RSS, 3.3 s model load** —
  far better than the 30 s–3 min the plan budgeted.
* Built the GraspMAS side: `perception3d.py` (geometry, `Grasp6D`, depth, filters),
  `graspgen/client.py` (vendored torch-free ZMQ client), `vis6d.py` (projected-gripper
  rendering for the Observer), `run_artifacts.py` (the `outputs/` writer),
  `agents/llm.py` (`ChatLLM`: Gemini free tier, pacing, retry/failover, tolerant parsers).
* Rewired `image_patch.py`: BLIP2, OWLv2 and RAGT-3-3 removed; `grasp_detection` returns
  a 6-DoF dict; `compute_depth` is metric metres; `best_image_match` reimplemented on CLIP
  (free, local) instead of BLIP2 + paid embeddings.
* Rewrote all three prompts for 6-DoF, fixed the upstream prompt defects, rewrote
  `main_simple.py` / `main_batch.py` / `simple_demo.ipynb`, deleted `grasp/`.
* **218 offline tests** (`scripts/run_tests.sh`, ~25 s) — no GPU, no network, no LLM spend.
* Verified end-to-end without an LLM (`scripts/verify_pipeline.py`): 3/3 objects grasped
  (scores 0.78–0.88), 3/3 grippers, part-level grasp on a mustard-bottle **cap** moved
  20 cm and landed inside the cap mask, monocular fallback works at ~17% scale error.
  Full evaluation in `SUMMARY.md`.

**Bugs found by running it** (all now covered by tests — see `SUMMARY.md` §7)
* Server OOM-killed by the O(N²) outlier filter → client-side `downsample_cloud`.
* Voxel sizing used the cube root (volume) for what is a *surface* → 16786 points became
  212 and the model returned nothing → size from the square root, then bisect.
* Mask containment tested at the gripper **base**, ~10 cm behind the contact → rejected
  ~90% of valid grasps → thread the real `fingertip_depth` from the gripper config.
* GraspMoE proposes grasps on the object's unobserved far side → visibility filter.
* `find_part` fell back to the whole object silently → warns and sets `part_found`.
* A literal `{` in the coder prompt broke `str.format()` → escaped, plus a template test.

**Next**
* Needs the user: a free Gemini key, then `python -m agents.llm --probe --test`, then
  `main_simple.py` for the first true closed-loop run.
* Needs the user: the OCID-VLG dataset, then `main_batch.py --limit 20`.

### 2026-08-12 — session 1: planning and environment bring-up

**Done**
* Audited both repos end-to-end before writing code. Established that CPU inference is viable and that
  the blockers are small and localized (see §6). Plan approved by the user.
* Installed `git-lfs` (conda-forge, into `base`). Cloned both repos with LFS resolved:
  GraspMAS `b15f178` (+ detectron2 submodule), GraspGenX `b942909`.
* Created conda envs `graspmas` (3.10) and `graspgenx` (3.11).
* Downloaded assets: `gripper_descriptions` (665 MB after an explicit `git lfs pull` — the clone alone
  left pointer files) with all 26 grippers including `franka_panda`; GraspGen-X checkpoints in progress.
* Wrote `CLAUDE.md`, `SUMMARY.md`, `outputs/README.md`.

**Decisions**
* Two envs + ZMQ rather than one merged env — dependency isolation, model-load amortization, memory.
* Use the `infer` wire action (name-based gripper, server-side asset lookup) rather than
  `infer_scene_depth`, which is params-only. Unprojecting client-side is ~10 lines.
* Gemini Flash free tier via the OpenAI-compatible endpoint
  (`https://generativelanguage.googleapis.com/v1beta/openai/`) — native vision for the Observer, and
  GraspMAS already uses `AsyncOpenAI`, so the transport barely changes.

**Next**
* Patch GraspGen-X for CPU, verify standalone inference, benchmark latency.
* Then the GraspMAS side: LLM layer, `perception3d.py`, `grasp_detection`, prompts, entrypoints, tests.

**Blocked / needs the user**
* A free Gemini API key from Google AI Studio must be placed in `GraspMAS/api.key` (or exported as
  `LLM_API_KEY`) before any end-to-end run.
* The OCID-VLG dataset must be downloaded separately for `main_batch.py`; `download.sh` does not fetch it.

---

## 8. Handoff — what needs the user

Everything that can be verified without credentials or extra datasets **has been**
(218 offline tests, plus `scripts/verify_pipeline.py` end-to-end on real RGB-D).
Two things remain and both are blocked on inputs only the user can supply.

### 1. A free Gemini API key — unblocks the agent loop

<https://aistudio.google.com/apikey> (no credit card). Then:

```bash
export LLM_API_KEY=...                       # or: echo "$KEY" > GraspMAS/api.key
conda run -n graspmas python -m agents.llm --probe --test
```

`--probe` lists the models the key can reach; `--test` sends one text and one
**vision** call. Pin the chosen id in `GraspMAS/llm_config.yaml` (the
`model_candidates` list is resolved against the live `/models` response at
startup, so free-tier model churn does not break the run).

Then the first real closed-loop query:

```bash
scripts/run_server.sh --daemon
cd GraspMAS && python main_simple.py \
    --query "grasp the mustard bottle by its cap" \
    --image-path ../GraspGenX/assets/sample_data/real_world/00/rgb.png \
    --depth-path ../GraspGenX/assets/sample_data/real_world/00/depth.npy \
    --intrinsics ../GraspGenX/assets/sample_data/real_world/00/meta_data.json \
    --max-round 2
```

`--max-round 2` keeps one query to roughly 6–10 requests of the ~1500/day budget.

### 2. The OCID-VLG dataset — unblocks the quantitative comparison

Not fetched by `download.sh`; it is a separate multi-GB download. Once present:

```bash
cd GraspMAS && python main_batch.py --dataset-path /path/to/OCID-VLG --limit 20
```

**OCID ships no camera intrinsics** (`OCID_VLG/dataset.py` exposes none), so
without `--intrinsics` a synthesized `K` is used and positions are metrically
approximate — the choice is recorded per sample as `intrinsics_source`. OCID does
ship organized `pcd/` clouds, so the honest fix is to fit `K` against one of
those, verify by reprojection, and pass it. Results append to
`outputs/eval/*.jsonl` and re-running skips completed `sent_id`s, so an
evaluation can legitimately span several days within the free-tier budget.
