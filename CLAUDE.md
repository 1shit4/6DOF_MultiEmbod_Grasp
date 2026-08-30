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
| **New** (ours) | `GraspMAS/perception3d.py`, `GraspMAS/vis6d.py`, `GraspMAS/graspgen/client.py`, `GraspMAS/run_artifacts.py`, `GraspMAS/agents/key_pool.py`, `GraspMAS/llm_config.yaml`, `GraspMAS/tests/*` |
| **New** — decluttering (§7) | `GraspMAS/{placement,collision,synth_scene,scene_registry,session_state,evaluator,declutter,main_declutter}.py`, `GraspMAS/execution/`, `GraspMAS/agents/task_planner.py`, `GraspMAS/agents/prompt/task_planner_prompt.py`, `scripts/verify_declutter.py`, `docs/declutter.md` |
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

### The picture the Observer judges from (found session 8)

Four defects, all in the same place, all invisible in any exit status: the Observer was asked to
judge a hand that was drawn wrong, against a region that was never drawn at all.

* **`fingertip_depth` defaulted to a constant 0.11 m for every gripper.** Both
  `visualize_grasp_6dof` call sites in `agents/graspmas.py` omitted it. Real values:
  franka_panda **0.1034**, robotiq_2f_85 **0.136**, inspire_hand **0.164** — so the yellow fingertip
  line landed 3-6 cm from where the hand actually closes, and 0.11 is not even the Franka's number.
  That line is precisely what the prompt tells the Observer the object should sit between.
* **The mask was never passed**, so the "magenta region … the exact object or part the grasp was
  computed for" that the prompt describes did not exist. The Observer inferred the target from where
  the hand landed. The mask lives in `grasp_detection` and the result dict had no route back to it —
  now `result["mask_key"]`, resolved by `image_patch.grasp_mask()`. Keyed by a monotonic int, not by
  "the last mask seen": generated code routinely computes an object grasp *and* a part grasp in one
  round, so last-seen is wrong whenever it returns the earlier one.
* **Every gripper was drawn as a two-finger sketch.** Four shipped grippers are `revolute_3f`
  (barrett, inspire, unitree_g1, robotiq_3f) and the prompt says "the orange lines are the two
  fingers". `vis6d._gripper_silhouette` now traces the real 10,500-point surface `collision.py` had
  been loading all along. Traced from a **rasterised point cloud, not a convex hull** — the hull
  fills in the jaw opening, which is the one part of the picture the Observer is asked to read
  (measured hull/real area: panda **1.29**, barrett **1.50**, and barrett correctly yields 2 contours).
* **Measuring the fingertips from the point cloud is worse than assuming them.** The extremes of the
  surface within a slab at fingertip depth give the outer edge of the finger *bodies* — 11.7 cm on a
  Panda whose jaw opens 8 cm, and 33 cm across the palm of a Barrett. The wireframe span was always
  right; only its depth was wrong. Tried, measured, deleted.

`collision.gripper_config` / `gripper_geometry` is now the single reader of `config.json`. Three
subsystems opened it three ways, which is exactly how the overlay came to use 0.11 while the mask
filter used the real number.

### What each agent is allowed to know (session 8, phase 2)

The channels between stages were narrower than the information available, everywhere.

* **`error_logs` was only ever populated when there was NO grasp.** On the success path it stayed
  `None`, so every warning raised while returning a *poor* grasp was invisible. Three constraints can
  be silently dropped to return something rather than nothing — all candidates collide, none landed
  inside the mask, the requested part was not found — and each makes the winner the least bad of a
  bad set rather than a good one. One of those fallbacks carries the comment *"keeping them so the
  Observer can see why"*; the Observer was never told. Now `result["selection"]` (the funnel:
  `416 → 47 visible → 29 on target → 12 clear`) and `result["error_logs"]` (COLLISION / OFF-TARGET /
  WRONG REGION).
* **The Observer's decision rule ignored two of its own five checks.** `target_match` and
  `collision_risk` were absent from it, so the model filled them in and they could not change the
  verdict. Both are now in the rule.
* **The Planner was forbidden from reading any of it.** Prompt rule 2 said *"Base your next subgoals
  on the Observer summary only. Do not directly read verdict, checklist, or JSON"*, and
  `graspmas.py` enforced it by passing `observation.get("summary", "")`. It now receives the whole
  judgement, rendered by `_observation_for_planner`.
* **A three-way failure taxonomy, because the three have different owners.** `geometry` and
  `wrong_part` the inner loop can fix by re-planning; `wrong_object` it *cannot*, since the instance
  id is chosen by the outer task planner before the inner loop starts. Telling the Planner to
  "diversify" on a wrong-object failure invites it to grasp something nobody asked for. It is now
  told to return upward instead.
* **`record_observer` was gated on the grasp succeeding**, which threw the failure kind away exactly
  when it was most informative — including every `wrong_object`, which by construction produces no
  usable grasp.
* **The planner's history now carries how hard the grasp was, but only when the iteration failed.**
  A clean move needs no explanation and prompt space is not free. Without it the planner could not
  separate "the grasp was sound and the hand slipped" from "we never found a sound grasp on this
  thing" — opposite responses, identical rendering.

**Collision attribution** (`SceneRegistry.colliding_instances`). `scene_cloud_excluding` flattens the
scene into an anonymous point array, so the filter knew a grasp was fouled and never by what.
Recovered by testing each instance's own cloud, worst first, and only when something has already
gone wrong. Measured on `occluded_target` with the hand driven onto each object: box names **only
itself** (it is isolated), mug names **mug 1019, bottle 479, banana 124**. If everything named
everything the measure would be worthless.

* **Gotcha: `points_inside_sweep` measures proximity to the hand's *surface*, not containment.** A
  correct grasp deliberately does not touch what it holds — an open Panda jaw clears the banana by
  **2.4 cm** — so the object being grasped registers **zero**. That is what makes it a collision test.
  A first attempt at testing attribution asserted "the grasped object is always inside the hand" and
  was simply wrong about the physics.

* **A run launched without `scripts/env.sh` silently loses the gripper descriptions**, and that now
  costs far more than it used to. `GRASPGENX_GRIPPER_CFG_DIR` unset means `load_gripper_points` falls
  back to a box hull, `gripper_geometry` to Franka-like constants and `gripper_morphology` to
  `"unknown"` — so the overlay reverts to the old two-finger sketch at the wrong depth and the
  Observer is handed a hand that does not exist, which is exactly the defect phase 1 removed. The
  only sign is one `WARNING collision: no gripper geometry for 'franka_panda'` in a busy log. Source
  `scripts/env.sh` (or pass the vars explicitly) for **every** run, not just for tests.

**Blocker-of-a-blocker** (`DeclutterLoop._diagnose_obstruction`). `blocking_objects` was only ever
called on the *target*, so a chain — B blocks A, A blocks the target — left B invisible: nothing ever
asked what stood in A's way, and the run just failed to grasp A repeatedly. Now one level deep, and
**gated on a geometric fact rather than a model's opinion**: it runs only when a specific object is
measurably inside the hand's swept volume. A grasp that failed because the object is wider than the
jaw, or because the hand slipped, produces no attribution and no cascade — those are not fixed by
moving anything. One level, deliberately; recursing further turns a stuck run into a tour of the table.

### What VLPart can and cannot do (measured, session 8)

Every `find_part` failure the loop had been reacting to turned out to be the object genuinely not
having that part — synthetic primitives have none at all, and the "mug" in `real_world/00` is a
handle-less yellow cup. VLPart is **open-vocabulary** (`get_text_embeddings` encodes arbitrary text),
so there is no fixed part list to look up; the question is only what a phrasing scores.

Measured on the mustard bottle in `real_world/00`, threshold **0.30**:

| prompt | best score | |
|---|---|---|
| `bottle cap` | **0.590** | pass |
| `bottle neck` | **0.320** | pass, borderline |
| `bottle lid` | 0.058 | fail |
| `bottle top` | 0.068 | fail |
| `cap` (bare) | 0.003 | fail |
| `the cap of the bottle` | 0.062 | fail |

Two things follow, and both matter for any fix:

* **The `"{object} {part}"` two-word format is load-bearing.** A bare part name scores ~0.001-0.008
  and a natural-language phrase ~0.06. `find_part` builds this format already; do not "improve" it
  into a sentence.
* **Within that format, wording swings the score by 10x.** `cap` 0.59 against `lid` 0.058 for the
  same physical feature — not a marginal miss, a different region of embedding space. So a failed
  lookup is often a naming problem, not an absent part, and **retrying with alternative names is the
  fix**. Where those names come from matters: a hardcoded synonym table is the same over-fitting
  trap as an object-specific prompt rule.

Working parts on that scene, for reference: bottle `cap`/`body`/`label`, box `lid`/`side`.

**The bounded retry, and why the Observer is load-bearing.** `find_part` now tries the requested word,
and only if that finds nothing asks the model (via `llm_query` — the tool exists for exactly this)
for at most **two** alternative names for the same region. Two, not "until something matches": a
wider search is slower, spends requests, and mostly raises the chance that a word matching *some*
region is accepted as the region that was asked for.

Measured on the same bottle:

| asked | matched | substituted | region | scores |
|---|---|---|---|---|
| `cap` | `cap` | no | 40% | cap 0.321 |
| `lid` | **`cap`** | yes | 40% | lid 0.094, cap 0.321 |
| `top` | — | no | falls back cleanly | top 0.103 |
| `wing` | **`shoulder`** | yes | 23% | wing 0.084, shoulder 0.302 |

Row 2 is the point of the feature. **Row 4 is the hazard**: a bottle has no wing, the model offered
"shoulder" (a real bottle part), and it matched — a confident answer to an impossible question.

And the two are **0.019 apart** (0.321 against 0.302), so **no threshold separates them.** Raising
the bar for substitutions loses the legitimate `lid → cap` recovery before it rejects the bogus
`wing → shoulder` one. That is why the substitution is *reported* rather than trusted:
`part_substituted` and `part_term` reach the Observer as a **SUBSTITUTED PART** note, and its prompt
tells it to look at the magenta region and decide whether it is the region the query asked for.
A plausible synonym is not always the same piece of an object — "grip" and "handle" coincide on a
mug and not on a pair of scissors. The Observer is the only safeguard here, by construction.

### The candidate set, and giving a rejection somewhere to go (session 8, phase 3)

GraspGen-X generates ~416 poses per call; 16-66 survive every filter, and the runners-up sit within
0.02-0.05 of the winner. All but one were discarded, and `_GRASP_CACHE` held **that one** keyed on
the mask — so a re-plan after an Observer rejection re-ran the identical pipeline and got a
bit-identical pose. Measured across every recorded multi-round iteration: **22 of 22 had identical
scores to three decimals**, because it was literally the same object.

The cache now holds the **whole surviving set** rather than the chosen pose, `grasp_detection`
returns `candidates` (top `MAX_RETURNED_CANDIDATES` = 20, compact: index / score / position /
approach / closing, no 4x4 per entry), and `ImagePatch.select_grasp(grasp, candidates)` rebuilds a
full grasp from any of them.

**Measured on the mustard bottle in `real_world/00`:**

| | time | score | |
|---|---|---|---|
| first call | **10.02 s** | 0.902 | 416 -> 103 visible -> 99 on target -> 45 clear |
| re-selection | **0.000 s** | 0.838 | a different pose, 3.2 cm away |
| filtered by approach | 0.000 s | 0.817 | kept 3 of 20; approach_z **+0.42** against the original **+0.90** |

So a rejection can now change the approach *direction*, for free. That is the actuator the loop never
had — and the reason the Coder is a language model rather than a script, since the filter that picks
the replacement is written on the spot for whatever the Observer objected to.

Both prompts had to move with it, or the affordance stays unreachable: the Coder's API doc said
grasp_detection returns "the best" grasp and nothing else, so writing code to pick a different one
was not a failure of imagination but correct behaviour against the tool it had been shown. The
Planner is now also told that re-issuing "compute the grasp pose" repeats the same computation on the
same mask, which is how a round gets spent for nothing.

**The candidates are working memory, not run history.** They are consumed from an in-memory registry
and never read back from `progress.json`, while `RECORDER.save_grasps` already writes all ~416 to a
`.npz` in a better form. Storing a JSON copy costs ~2.9 KB per grasp, duplicated once per inner
round, on a file that was 51 KB. `SessionState._slim_grasp` drops them at the boundary and keeps
`n_candidates_available` instead.

### Two files, two jobs (session 8, phase 4)

`progress.json` was doing both jobs badly: it stored everything and the planner
was shown 4.1% of it, chosen by a truncation nobody had revisited. The split is
now explicit.

* **`progress.json`** — the archive. Poses, waypoints, generated code, agent
  transcripts, timestamps. Read by `resume`, `build_report.py`,
  `summarize_run.py` and `verify_declutter.py`. Unchanged.
* **`planner_state.json`** — *exactly* what the task planner is told, written
  beside it every save. `planner_context()` renders from `planner_state()`, so
  the prompt and the file cannot drift: whatever the file says is what the
  planner saw. Measured on a real run: **1,798 chars against 17,399**.

**The plan/history join is done in Python now.** `removal_order` carried no
status, so "attempted twice and failed" was neither done, skipped nor pending,
and the planner re-derived its position by cross-referencing two lists on every
call — ambiguous exactly when a run was going badly. `plan_steps()` marks each
step `done` / `attempted` / `next` / `pending` with the iteration it happened,
and `off_plan_actions()` flags iterations that acted on something the plan never
named (which the geometric fallback does routinely). Both are scoped to the same
window as the history, or a long run accumulates unbounded notes.

An emptied plan no longer reads as "not set yet" — that is opposite advice
("you have not planned" against "your plan is complete").

**A rejected grasp is no longer silently replaced.** `_grasp_for` fell straight
through to a nominal geometric grasp whenever the inner loop came back empty,
which is right for an outage and wrong for a verdict: a run reported `success`
carrying `score: 0.0, source: "nominal"` after the Observer had rejected every
round on the target. The two cases are now separated, and the refusal reason
reaches the planner instead — which is what lets it tell "this cannot be picked
up" from "something has to be cleared first".

### What the fault suite caught (session 8, phase 4)

Re-running the six injected faults after removing the answer key found **three**
defects the 833-test offline suite and `verify_declutter` both passed over —
because neither runs the Observer.

* **The clean control aborted.** Two of this session's own changes compounded:
  phase 2 made the WRONG REGION note unconditionally decisive
  (`semantic_alignment = no`), and phase 4 item 17 made any rejection block the
  geometric fallback. On synthetic primitives an object *has* no parts, so
  `find_part` correctly finds none and the Observer correctly says
  `wrong_part` — and the loop then refused a grasp it had itself called
  "physically feasible and collision-free". **The refusal is now phase-aware**,
  which falls out of the phase split rather than being a special case:
  `geometry` and `wrong_object` block in both phases; `wrong_part` blocks only
  when delivering, because moving an object aside only needs the object to
  move, while *how* it is held is the whole request in a handover.
* **An ambiguous four-word phrase cost a run.** The digest said
  `"grasp 0.47; critic: valid; 2 attempts"`, where "2 attempts" meant inner
  rounds *within one iteration*. The planner's prompt has a two-strikes rule
  about failed attempts *on an object*; it applied the rule correctly to the
  wrong quantity and aborted after a single real failure, twice. Now
  `"2 grasp attempts within this one iteration"`, with a test asserting the
  count never appears unqualified. **No test could have caught this** — both
  readings are valid English; only reading a run's reasoning finds it.
* **The loop cleared objects it did not need to clear** (phase 4 addendum).
  `blocking_objects` runs three tests at three fidelities: **occlusion** (mask
  outline in the image, 2D), **approach** (`points_inside_sweep`, the object's
  real 3D cloud against real gripper geometry) and **proximity** (footprint
  centroids on the table plane, the crudest). The run's whole objective is a
  valid grasp on the target, so flagging an object whose removal buys nothing
  costs a grasp, a place, a re-perception and real robot motion.

  Measured on `occluded_target` with one occluder left: **the banana is 78%
  visible, the mug is flagged as blocking, and a collision-free top-down grasp
  is already available.** The loop would have spent a full iteration to reach a
  state it was already in.

  Now: `blocking_objects(level=)` — **strict** keeps occlusion and 3D approach,
  **relaxed** adds proximity — and the loop starts strict, escalating once only
  when a grasp on the target actually fails. Failure to grasp is the only
  evidence that a conservative reading was too conservative.
  `SceneRegistry.reachable_grasp` asks the question directly.

  **The gate is what a blocker blocks *by*, and getting that wrong was
  instructive.** A first version skipped clearing whenever any grasp was
  reachable, and immediately tried to grasp a **32%-visible fragment** — the
  occluders are *in front of* the banana, not in the hand's way, so the nominal
  pose is collision-free while the mask it is computed from is a scrap of the
  object. Occlusion corrupts perception; proximity and approach are only claims
  about reachability, which `reachable_grasp` measures better than they do. So
  the shortcut fires only when **no** remaining blocker cites occlusion.
  "A collision-free grasp exists" is not "the target is graspable well".

  **Validated live** on the case that motivated it (`wrong_object`,
  `occluded_target`): the hand takes the wrong object, the planner diversifies
  to the other occluder, comes back, moves the mug **successfully** — and the
  target is still flagged. That state aborted the run before. Now the blocker
  list at that iteration is **empty**, because proximity is excluded at strict
  level, and the run finishes with a **0.851** sampler grasp in 4 iterations,
  25 LLM calls. Neither the reachability shortcut nor the escalation was needed:
  dropping the crude test was enough.

### What "in the way" means now (session 8, phase 4 second addendum)

Two of the three blocking tests were **deleted rather than tuned**, because each
answered a question it could not answer.

* **approach** swept a single *synthesised top-down* pose and asked what fell
  inside it — one direction out of the many a real grasp can come from, so a hit
  said nothing about the grasps that matter. Across every recorded run it fired
  **zero** times (259 occlusion flags, 5 proximity, 0 approach): it is skipped
  while the target is occluded, and by the time the target is visible the
  occluders have gone. It only ever ran when it had nothing to find.
* **proximity** flagged anything within half the width of the *whole hand*
  (10.4 cm on a Panda) on the assumption the hand must fit broadside between two
  objects. It need not — it can come in from anywhere.

**What is left are two reasons whose removal demonstrably helps:**

* **occlusion** — it hides the target, so any grasp is computed from a fragment.
  Clearing it buys information nothing else can.
* **contact** — it is touching the target: it may resist the pick, and it may
  topple when the target is lifted out from beside it. Neither is about
  reaching; both are about what happens when the hand closes.

**And a third that is measured rather than assumed.** When the target is visible
the sampler proposes ~416 poses and the collision filter rejects some;
`image_patch._attribute_collisions` records **which object's points** fouled the
best ones, and those become blockers (`fouls_grasp`). That is the honest answer
to "make room for the hand", from every direction the sampler tried. It also
replaces the nominal-pose sweep in the blocker-of-a-blocker cascade.

So the loop is: clear occluders and touchers → try the grasp → let the attempt
itself name what else is in the way. No re-grasping the target after every
removal; only when there is nothing obvious left.

**Two measurement traps hit while building it.**

* **Circumscribed radii are not a contact test.** Subtracting each footprint's
  radius treats a 16 cm banana as a circle of radius 8 cm, so a cylinder
  **3.75 cm away measured as −3.5 cm** — apparently interpenetrating. That is
  the same over-flagging proximity was deleted for, in a new disguise.
  `_surface_gap` measures cloud-to-cloud distance instead: the objects' points
  are already held, it costs one KD-tree query, and it is exact for what is
  asked. `TOUCH_MARGIN_M` is **2 cm**, absorbing the 1.0-1.7 cm single-view bias
  (§7.3) that makes touching objects measure slightly apart.
* **Gripper-derived placement clearance cannot be a hard requirement.** Asking
  for `jaw_half + margin` (13.4 cm Panda, 19.6 cm Barrett) made **59 tests fail
  with "no placement available"** on a table that had been solvable for months.
  It is now a *preference*: ask for gripper clearance, settle for
  `DEFAULT_MARGIN_M`. Refusing to set an object down because the *next* grasp
  might be awkward is the wrong trade — and if the tighter spot does foul a
  later grasp, the collision filter names it and the loop moves it again, one
  iteration instead of a deadlock. Note the original motivation is gone anyway:
  with proximity redefined as 2 cm contact, the plain 3 cm margin already
  exceeds it, so nothing the system places can be re-flagged as touching.

### A dropped socket used to end the run (session 8)

`ChatLLM._is_retryable` matched on status codes and keywords. `openai` raises
`APIConnectionError` whose message is the string **"Connection error."** — no
status, no keyword — so it was classified **non-retryable**: the call failed on
attempt 1 of 5, the provider was marked failed, and a run doing manipulation
died on a transient blip. Over one session that aborted **four** runs and
emptied two others' `agent_rounds`, and each one looked like a *planning*
failure until the log was read.

Now matched by **exception type first** (`openai.APIConnectionError`,
`httpx.ConnectError` / `ReadError` / `RemoteProtocolError` / timeouts, builtin
`ConnectionError`), with a string fallback for errors re-raised without their
type. A genuine bug (`ValueError`, bad key) still fails fast — retrying a bad
argument five times only delays the traceback.

**The diagnosis is worth remembering**, because the obvious lead was wrong:
13 of 17 failures hit the two agents that send **images**, which looked like a
payload problem. It is not — the base64 overlay is **45 KB**, the vision path is
`complete()` like everything else, and the vision model candidates are the same
top two. The vision correlation was incidental; the defect was that *any* blip
was fatal. Measured after the fix: 3 connection failures in one run, **0 fatal**,
run succeeded.

* **Placement clears occlusion, not proximity.** Under `wrong_object` the run
  recovered properly — failed, diversified to the other occluder, came back and
  *successfully* moved the mug 28 cm — and the target was still blocked, with
  the reason changing from `occlusion` to `proximity`. `keep_out_for` protects
  the projected sight-line (§7.5) and says nothing about the jaw gap, so an
  object can be legitimately placed where the hand still will not fit. Predates
  this session; exposed by the fault cascade shifting the free space. Not fixed.

**Measured, six modes, `occluded_target`, `--max-round 2`:** `none`, `drop`,
`offset`, `tip`, `collateral` all reach the target in 3-4 iterations.
`wrong_object` recovers the object and then hits the placement gap above.

**A standing hazard, seen repeatedly this session:** transient
`All LLM providers failed for <agent>: Connection error`. It aborted a
`collateral` run that succeeded on retry, and emptied `agent_rounds` in two
smoke runs. The §7.11 guard handles it, but **check the log for it before
reading any failure as a planning failure**.

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

* **Free-tier quota is per *project*, per model, and far tighter than the headline number.**
  Measured against a live key on 2026-08-20: `gemini-3.5-flash` allows **5 requests/minute
  and 20 requests/day** — one declutter run does not finish inside that. The published limits page does not break the
  numbers down per model; the only place they appear is the body of the 429, under
  `quotaId: GenerateRequestsPer{Minute,Day}PerProjectPerModel-FreeTier`. So probe, do not assume.
  The **flash-lite family is the high-volume tier** and is what `llm_config.yaml` now lists first;
  candidates are ordered by quota, not by capability, because a smarter model that runs out after
  20 calls cannot reach iteration 3. Both flash-lite ids handle vision, which the Observer needs.
* **Thinking tokens come out of `max_tokens`.** Gemini 3.x reasons before answering and that
  reasoning is billed and budgeted as output. Measured on `gemini-3.5-flash`: **780-860 thinking
  tokens per call** against agent budgets of 900-1000, so the visible reply got **33-118 tokens**,
  every structured answer was truncated mid-JSON, and the outer planner aborted the run on
  iteration 0 after failing to parse twice running. Nothing in the symptom names the cause — what
  you see is a model that "cannot follow the output format", and the leaked reasoning that lands
  in the content field ("Wait, the prompt says...") reads like a badly behaved model rather than a
  starved one. Fixed in `ChatLLM._prepare` via `min_max_tokens: 2048` (a floor covering reasoning
  *and* a reply) and `reasoning_effort: low` (the compat layer maps it to Gemini's
  `thinking_level`). Reasoning fell to 49-200 tokens and every reply finished with `stop`.
  `llm_trace.jsonl` now records `reasoning_tokens` and `finish_reason` so the next occurrence is
  visible rather than requiring arithmetic.
* **One key is not enough, and the fix is keys, not patience.** `agents/key_pool.py` pools
  several keys and routes each request to the least-used one. Two things it must get right,
  both found by measurement:
  * **A per-minute 429 must cost a key, not a second.** The obvious implementation — fill the
    rejected key's minute window so the ranking skips it — forces a flat **60 s** wait when
    there is only one key, which is strictly worse than the 2/4/8 s ramp it replaces. The
    cooldown is therefore the provider's own `retryDelay`, else the same exponential backoff,
    applied **per key**. A lone key behaves exactly as before; extra keys make the wait vanish
    rather than shorten it. Measured live with 6 keys: 6 calls in 8.2 s, one key each, no pacing
    delay at all.
  * **Persisted state must not be keyed on the fingerprint.** The display handle is the key's
    last four characters, and two keys can share it — which collapses both onto one record, so
    one key's daily exhaustion silently retires the other. `state_id` is a SHA-256 prefix
    instead; `fingerprint` is display-only. Caught by a test, not by reading.
  Daily exhaustion is remembered in `.llm_quota.json` (gitignored), scoped to the **model** that
  returned it — a key out of `gemini-3.5-flash` still has its full flash-lite budget — and keyed
  so nothing in the file authenticates anything. `llm_trace.jsonl` records `key`/`key_fingerprint`
  so a run can be audited for rotation.
* **Only the 429 body says which limit you hit.** `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`
  versus `...PerMinute...`. An unrecognised 429 is read as per-minute deliberately: rotating on a
  misread costs nothing, retiring a key for a day costs a fifth of the budget.
* Gemini's OpenAI-compat layer does not honour every OpenAI parameter. `agents/coder.py:38-45` passes
  `frequency_penalty` and `presence_penalty=2.0`; these are stripped per provider.
* The upstream response parsers assume GPT-4o formatting and break on other models —
  `planner.py:44-45` (`.split('<thought>')[1]`), `observer.py:64-72` (`json.loads` on possibly
  fenced JSON), `coder.py:47` (`split('\n')[1:-1]` assumes exactly one fenced block).

---

## 7. Long-horizon decluttering

Turns the single-shot grasp finder into a sequential manipulation loop: pick and place one
obstructing object per iteration, re-perceive, re-plan, stop when the target is graspable.
Design and open questions: `/home/ishita/.claude/plans/i-want-to-upgrade-parsed-cocoa.md`.

### 7.1 The table frame

A second frame alongside the camera frame, and the one all placement reasoning happens in.
`+Z` is the support-surface normal **pointing back toward the camera**, the origin is the plane
point closest to the camera centre (`-offset * normal`), and `+X` is the camera's `+X` projected
onto the plane. Consequences worth holding onto:

* **A point's height above the table is `normal · p + offset`**, and equals its table-frame Z.
  Positive is the side the objects are on.
* Free-space search is then a 2D problem in table-frame XY.
* The origin is *not* the table centre, so table-frame coordinates look arbitrary
  (`y ≈ 1.0` on the sample scene). That is expected; only differences are meaningful.

### 7.2 New invariants

* **A place pose is a pure translation of the grasp pose.** The object is set down in the
  orientation it was picked up in, so `T_place = Translation(Δ) @ T_grasp` and only Δ is solved,
  in the table plane. This makes the free-space test exact rather than an estimate over an
  unknown post-rotation shape, and removes any question of stability in a new pose.
  Verified: the moved object's lowest point lands at the release gap to within 1e-3 m.
* **Unobserved space is occupied.** A cell of the height map with no depth returns is `NaN`, and
  `free_space` treats `NaN` as blocked unless explicitly told otherwise. A single view sees no
  surface behind an object; "no returns" is not "empty".
* **`points.json`, not meshes, is the gripper's collision geometry.** Every gripper description
  ships `open`/`close` surface clouds of 10,500 points **in the gripper base frame** — the same
  frame grasp poses use, verified against `config.json["bbox"]`. So no mesh loader is needed, and
  neither `trimesh` nor `python-fcl` (neither is installed in `graspmas`, and FCL is unavailable).

### 7.3 Gotchas found while building it

* **A support plane is not the biggest plane, it is the one with nothing underneath it.**
  Scoring RANSAC candidates by inlier count fits a plane tilted 9.9° through the box fronts of
  the `crowded_table` scene, missing the real table by 8.4 cm, because the clutter has more
  visible points than the surface it stands on. Measured on that scene the true tabletop has
  **0.0%** of the cloud below it and the wrong plane has **6.9%**. `fit_support_plane` therefore
  discards any candidate with more than `max_below` (5%) beneath it *before* comparing inlier
  counts. That takes the error to 0.05° / 0.03 mm.
* **Plane fitting on a whole real capture fails, and should.** `real_world/00` contains floor,
  wall, a second bench and a robot; more than 5% lies below every candidate, so the fit raises
  rather than returning a wall. `support_cloud()` derives the cutoff from the objects — everything
  within `margin_m` of the furthest visible object — which takes the same scene from "no plane
  found" to a 50%-inlier fit with all objects 1-15 cm above the surface.
* **Single-view footprints are biased small by 1.0-1.7 cm**, because a depth image sees only the
  front of an object. Too small is the dangerous direction: it invites placing an object into a
  gap it does not fit. The bias is *not* corrected in `object_footprint` — mirroring the points
  about the viewing ray does fix the extent, but it makes the reported centroid view-dependent,
  and `evaluator` compares centroids across observations to decide whether a move worked. It is
  absorbed once, in `placement.DEFAULT_MARGIN_M` (3 cm = 1.7 cm bias + 1.3 cm safety), with a
  test tying the two together so the margin cannot silently stop covering the bias.
* **A heavily occluded object's footprint is simply wrong.** The banana in `occluded_target` is
  19% visible and comes out 11.8 cm short on its long axis. Nothing recovers what another object
  hides. Gate on visibility, not on the footprint looking plausible.
* **Undersampling the gripper is a correctness bug, not a speed trade.** Proximity is measured
  *from* the gripper's sampled points, so anything smaller than the gap between them passes
  through the hand undetected. Measured 95th-percentile spacing on a Panda: 384 points → 12.3 mm,
  1024 → 7.5 mm, 2048 → 5.5 mm. It must stay under the 10 mm collision threshold, so
  `DEFAULT_GRIPPER_POINTS = 1024`. Cost for 200 candidates against a 15k-point scene: 200 ms.
* **"At least N hits" has to count scene points, not gripper points.** Querying the scene from
  each gripper point and counting the gripper points that found something looks equivalent and is
  not: at 3.4 mm spacing and a 10 mm threshold, one stray depth flyer lands within reach of dozens
  of gripper points and trips any threshold instantly — the exact noise the check exists to
  absorb. `pose_collides` prunes the scene to the hand's neighbourhood, transforms it into the
  gripper frame, and queries a cached tree of the gripper itself. Pruning also made it *faster*
  (289 ms → 200 ms for 200 candidates).
* **`np.ceil` on an exact multiple buys a spare row.** `(1.1 - 0.7) / 0.01` is
  `40.000000000000014` in floating point. `build_height_map` rounds to 6 decimals first.

### 7.4 Identity and obstruction (`scene_registry.py`)

**The planner never names objects, it names instances.** `find("bottle")` is silently ambiguous
with two bottles, and the list order is not stable between iterations, so "the bottle" can mean a
different object on iteration 2. Ids are carried by 3D position instead: an object that did not
move keeps its id. This is not a hedge — between the two real sample scenes the Cheez-It box is
`obj_2` in one and `obj_1` in the other while sitting 4 mm from where it was, so label-based
identity demonstrably fails on real data. Matching is nearest-first across all candidate pairs,
so a close pair is never stolen by a worse but earlier-considered one.

**"In the way" has three meanings**, and a loop implementing only the visible one strands itself:
*occlusion* (hides the target), *approach* (inside the swept volume of a candidate grasp), and
*proximity* (the hand does not fit in the gap). `test_scene_registry.py` includes a target with an
object directly **behind** it: occludes nothing, sits squarely inside the jaw.

Gotchas found:

* **Convex-hull visibility does not measure occlusion.** Filling a mask's convex hull and comparing
  looks like it measures the visible fraction, and scores the 19%-visible banana at **98%** —
  because the visible fragment of an occluded object is usually convex too. What *is* observable is
  the outline, and whether it runs against a nearer object or against free space. Visibility is now
  `1 - (outline against nearer objects) / (whole outline)`, which scores the same banana at 32%.
* **The footprint-reliability threshold has to be 0.95, and the number came from measurement.**
  Occluding the banana by varying amounts: visibility 0.32 → 11.8 cm short, 0.78 → 7.9 cm,
  0.81 → 5.7 cm, 1.00 → 0.6 cm. There is no useful middle ground. Not restrictive in practice: a
  target stays occluded until its blockers are gone, and the blockers stand in front and are
  unoccluded, so the footprints needed to *place* them are reliable from iteration 0.
* **Approach and proximity are skipped while the target is occluded**, because both need the
  target's footprint and a footprint fitted to a fragment is the wrong size, place and orientation
  — the tests would return confident nonsense. Occlusion alone decides until the target is
  uncovered, at which point geometry takes over and finds the second-order blockers. The loop
  establishes obstructions in the only order it can.
* **An occluded object's centroid is not a stable quantity.** It is the mean of whatever is
  visible, so *uncovering* an object moves its apparent centre without the object moving —
  measured at **2.3 cm** on the banana when the mug beside it was picked up, past any sane movement
  threshold. `moved_since(min_visibility=...)` exists for this; the evaluator must use it or it
  will report phantom collateral damage.
* `scene_registry` reads the two gripper config fields directly instead of calling
  `ImagePatch._gripper_geometry`: importing `image_patch` loads GroundingDINO, SAM, VLPart and
  MiDaS at module scope, which the offline tests must not pay for.

### 7.5 Where an object may be put down

The single most consequential finding of the whole build, because it is the difference between a
loop that terminates and one that does not.

**Measured, running the geometric core of the loop on `occluded_target`:**

| placement keep-out | objects moved | result |
|---|---|---|
| none | `mug, bottle, bottle, bottle` (4 steps, hit the cap) | banana 81% visible, **still blocked** |
| enabled | `mug, bottle` (2 steps) | banana **100%** visible, no blockers |

Without a keep-out region, `find_placement` picks the nearest cell with enough clearance — and the
nearest clear cell is very often directly in front of the target, so the loop dutifully lifts the
bottle, sets it back down over the banana, and rediscovers it as a blocker next iteration.

Getting the keep-out right took three attempts, and the two failures are instructive:

1. **A wedge on the table, from the target's footprint corners to the camera.** Fails for the exact
   reason the target needs clearing: it is occluded, so its footprint is a fragment (a 16 cm banana
   fits to 4 cm), the wedge is a narrow sliver, and an object released 5 cm to the side lands back
   in front of it. Observed: moved 20 cm away, banana still 40% hidden.
2. **Projecting each candidate cell's centre line into the image.** Better — occlusion is an image
   property, so decide it in the image — but sampling only the axis lets a wide object be released
   just beside the forbidden column and hide the target with its body. `object_radius_m` samples a
   ring instead.
3. **Protecting the target's mask alone.** Still not enough, because *the target's mask is itself
   truncated by the object being moved*. The region protected is therefore
   `target.mask | mover.mask`: the mover's current silhouette is exactly the ground about to be
   revealed, and therefore exactly where the rest of the target may turn out to be. Free to
   include — that ground is being vacated anyway, and putting the object back on it is already
   forbidden.

`placement.projected_occlusion_keep_out` is the result; the table-space wedge was deleted rather
than kept alongside it. Note the blocked region is a truncated cone, not a half-plane: past about
30 cm toward the camera a 20 cm column projects *below* the target and stops occluding it.

### 7.6 Progress and grand plan (`session_state.py`)

**The LLM never writes either file.** Agents return structured JSON; Python validates and applies
it. A model that can rewrite its own success criterion has no success criterion.
`amend_grand_plan` refuses to touch `goal` or `target` at all, refuses an edit with no stated
reason, refuses unknown fields, and caps revisions at 8 — a plan revised more often than that is
oscillating, not converging. Every accepted change records before, after, and why.

Writes are atomic (temp file + `os.replace`) so a run killed mid-write resumes rather than dying on
a truncated file. An iteration left open by a crash is closed and annotated, not dropped.

`is_stalled()` deliberately does **not** mean "the action failed". A move can fail and still help
(the object was nudged out of the way), and succeed and still achieve nothing (it went exactly where
planned and the target is still blocked). Progress is measured by `still_blocking_target`, which is
the distinction the whole evaluator design rests on.

### 7.7 Execution (`execution/`)

One interface — `capture`, `execute_pick_place`, `reset` — with three backends: `MutationExecutor`
(edits a `SceneSpec` and re-renders; deterministic, exact ground truth, no physics),
`ReplayExecutor` (hands back recorded captures, so the loop can be exercised on the real sample
scenes), and `RobotExecutor` (a documented contract that raises on construction).

**A report must say what happened, not what was asked for.** An executor that echoes the plan back
as success makes the evaluator meaningless. So the grasped object is identified *from the pose* —
whichever object is nearest the fingertip midpoint, as a real hand closes on whatever is between
its fingers — which makes "the plan named obj_3 but the pose was over obj_5" a reportable outcome
rather than an invisible one. A grasp reaching nothing fails at the `grasp` stage.

Physics is not modelled; failures are **injected** instead (`offset`, `drop`, `tip`, `collateral`,
`wrong_object`, each seeded). That is an honest trade rather than a hidden one: every failure this
backend produces is one somebody wrote down, so the evaluator is only ever tested against failures
we thought of. A physics backend is the thing that would surprise us, and it is deferred (M8).

### 7.8 Evaluation and the outer loop

**The evaluator is arithmetic, not an LLM** (the user's call, and the right one). Every question
worth asking is measurable: did the object move (centroid displacement), did it land where
intended (distance to the planned position), is it still in the way (re-run the blocking test).
Beyond being free and deterministic, it removes the incentive problem — an agent asked to grade
its own previous decision *and* pick the next one will tend to find that the last one worked,
which is the premature-completion failure the loop exists to resist.

It returns **two independent verdicts**, and the planner acts on the second:

* `moved_off_target` + `still_blocking: false` — the grasp slipped but the object is clear, so
  **nothing needs redoing**;
* `success` + `still_blocking: true` — it went exactly where it was told and changed nothing, so
  **the plan was wrong**.

A VLM call fires only on `unknown`, which in a healthy run is never.

Gotchas found running it:

* **A deliberate move exceeds the identity match radius.** The radius (8 cm) is sized for depth
  noise on a static object, but a pick-and-place travels 20-30 cm, so the object was retired and
  re-registered under a new id, read as `object_missing`, and — because the *old* id was then
  absent from the blocker list — reported as "no longer blocking". That defeated stall detection
  and ran a hopeless case to its iteration cap. Fixed with `expected_moves`: the loop tells the
  registry where it put things. The hint had to be made **additive** rather than a replacement,
  because a *failed* move leaves the object exactly where it was.
* **Collateral damage can move an object out of its own identity.** With the hand closing on the
  wrong object, the bystander it carried away vanished from `moved_since` entirely — that function
  compares ids present in both snapshots, and this was one id gone and another appeared. Objects
  that disappear between snapshots are now reported as collateral with an unknown distance. Silent
  collateral damage is the failure mode; an unexplained disappearance is at least as alarming as a
  measured shift.
* **`{` in a prompt is still a `KeyError`.** Adding a `return {"grasp": ..., "place": ...}` example
  to the coder prompt broke it exactly as the `grasp_detection` JSON example did in session 2.
  `tests/test_prompts.py` caught it immediately, which is what it is for.

### 7.9 What is deliberately not built

* **Physics.** Execution mutates the scene and re-renders; failures are injected. Honest, but it
  means the evaluator is only ever tested against failures we thought of. PyBullet is viable on
  CPU (prebuilt cp310 wheel, no dependencies, software rasteriser with RGB+depth+segmentation) and
  is the obvious next step.
* **Motion planning between waypoints.** This repo decides where to grasp and where to release and
  checks the hand is clear at those poses and along the approach. Getting the arm between them is
  a motion planner's job.
* **Stacking, containers, and joint rearrangement.** Placement is 2.5D onto the support surface,
  and obstructions are cleared one at a time. A situation needing two objects swapped will not be
  solved.

### 7.10 Synthetic scenes

`synth_scene.py` composes primitives on a table and renders by **ray casting**, not point
splatting — splatting leaves gaps indistinguishable from sensor dropout, and "unobserved" is
load-bearing (see §7.2). Ray casting gives hole-free depth, pixel-exact segmentation, and poses
known to machine precision: table pixels land on the ground-truth plane to within 1e-6 m.

The two real sample scenes stay the right test for *perception* — 8 labelled objects each, real
depth, real intrinsics — but their objects are well separated, so **neither contains real
occlusion** and neither has ground truth for where an object should end up after a move.

Scenarios: `occluded_target` (the motivating case), `two_identical_bottles` (instance identity),
`open_table`, `crowded_table` (placement legitimately has no answer).

**A parallel jaw's 20 cm span lies along the object's *short* axis**, because that is the axis it
closes across. So blockers have to sit in front of an elongated target, not beside it — an object
13 cm to the side is harmless and one 9 cm in front is not. Getting this backwards produced a
scenario whose docstring claimed obstruction that measurement showed did not exist. Measured on
`occluded_target`: banana 19% visible, bottle 391 points inside the gripper's swept volume, mug
40, distractor box 0, and the sweep clear once both blockers are gone. `test_synth_scene.py` and
`test_collision.py` assert each of those numbers, so the scenario cannot drift into being easy.

---

### 7.13 Priority against effort, and what real photographs showed

**Stage 1 sets a priority; stage 2 spends it.** `priority` is 1 for the best
answer, and **ties are meaningful** — equal priority says either object serves
equally well, which is what licenses taking whichever is cheaper. Stage 1 is
deliberately *not* shown what is in the way; Python measures blockers per
candidate, then stage 2 sees both and emits `target_order`. Priority leads;
within a tier effort decides freely; across tiers a swap needs a large gap and is
recorded in `run_notes`. Python validates the order is a permutation of the real
candidates, so the unguarded swap removed in §7.12 cannot return.

Verified live, both directions:

| run | stage 1 | outcome |
|---|---|---|
| `c3_tie_effort` (`affordance_choice`) | knife **p1** (1 blocker), scissors **p1** (0) | took the scissors — **1 iteration, nothing moved** |
| `c3_priority_holds` (`affordance_table`) | knife **p1** alone (1 blocker) | kept the knife, cleared the bottle — 2 iterations |

Its words: *"Both are priority 1... the scissors are not blocked, so they require
zero effort, whereas the knife is partially occluded."* Effort saves work where
suitability ties, and never buys a cheaper wrong answer.

**Three fixes this needed.** A declined retarget acted on nothing *and* recorded
no evaluation, so `iterations_without_progress` counted it — **a refusal supplied
the evidence justifying the next request**, and one real failure plus one
declined ask cleared a bar meant to need two. Non-acting actions are now skipped
by the counter, and a decline falls back to the geometric choice and does real
work. Separately, `set_grand_plan` takes three fields and a model emitted a
fourth (`target_id`), killing the run with a `TypeError` from inside the state
writer that named neither the model nor the key; the plan is now filtered.

#### Real photographs: `scripts/probe_target_selection.py`

Synthetic scenarios hand the planner the **literal label the scene author typed**,
so they test reading an abstract goal and **not** coping when perception is
wrong. This probe runs stage 1 alone against GroundingDINO on five real photos:
**20/20 correct** (`outputs/reports/target_selection_real_photos.md`).

The result worth the exercise: it **declined "something to drink from" on a
workbench whose table listed a `bottle`**, because the image showed a
screwdriver. It overrode a wrong label using the photograph.

* **The defect found.** It *improvised* when nothing served — a spoon for a nail,
  a plate when hungry. Telling it an empty candidate list is a correct answer
  fixed all four cases and held. The first wording over-corrected (it then
  refused a real bottle for not being a glass), so the prompt now poses one
  question — *would the person, handed this object, consider their request
  answered?* — with examples on both sides. Both errors are real and opposed.
* **Detection is the weaker half.** Real objects score 0.39-0.93, hallucinations
  0.15-0.30; at 0.15 an "apple" on a photo of a tool set out-competed a real
  hammer, and on 30 dense tools (330 boxes) the hammer was ranked out of the
  table entirely.
* **Three "planner failures" were measurement failures** — scoring `box` as
  not-food when it was a **Cheez-It box**, calling a correct decline a
  regression, and a `kept[:12]` cap in the probe that deleted a whole detected
  class. All from reading labels instead of opening the images: the exact failure
  the probe exists to catch. **When a target choice looks wrong, open the photo
  before blaming the model.**

**A label is not an object.** Everything after stage 1 works on an instance id
and treats suitability as settled, so a confident wrong label propagates
unchecked. The planner caught one. That is not a guarantee, and it is the open
risk in this design.

### 7.12 An abstract goal, and the target as an inference

`--target` was a label matched by substring, so the human had already done the
reasoning. The task planner now reads the goal itself: *"I need something to
cut"*, *"I am hungry"*. Nothing else in the system can — the registry, the
blocking analysis and the placement search all work on an instance id and have
no notion of what an object is *for*.

**Two staged calls, not mid-turn tool calls.** The ordering is forced:
`blocking_objects(target_id)` needs a target, and `GRAND_PLAN` may not name an
object the blocking analysis did not. So stage 1 picks a **ranked** list from the
scene table, Python computes blocking for the top 3, and stage 2 drafts the plan
seeing all of them. Tool calling was considered and rejected: a tool call is
itself a round trip, so it buys back no requests, and it opens a path where a
model's output is consumed without passing `TaskPlanner.validate` — the pattern
the whole subsystem rests on. Feeding stage 2 every candidate's blocking analysis
gets what the tool loop would have given, for **+1 request per run** (measured:
19 calls inferred vs 18 with `--target`, same scenario, same target).

Ranking is not decoration: `retarget` advances it with **no LLM call at all**.

**The goal is the person's words; the target is our inference about them.** That
is the line that makes a revisable target safe. `amend_grand_plan` still refuses
`goal` and `target` outright; a retarget is not an amendment but a *supersession*
— the old plan is archived with the reason beside it, capped at `MAX_RETARGETS`
(1). An explicit `--target` skips stage 1 entirely, so every earlier run,
`verify_declutter.py` and `--no-llm` behave bit-identically; `--no-llm` without a
target is a parse-time error, since nothing offline can read an abstract goal.

**The four defects this cost, none visible in an exit status.** Every one of the
runs that hid them reported `SUCCESS` or a defensible `ABORTED`.

* **The grand plan could choose the target, and that was a second retargeting
  path with none of the guards** — no evidence, no budget, no record, firing at
  iteration 0. Letting the model "confirm" its target after seeing the costs read
  as harmless. Measured on `affordance_choice`: it ranked the knife first
  (*"a primary tool designed for cutting"*) and the plan then switched to the
  scissors because they were *"fully visible and unobstructed, making them a more
  efficient and immediate solution"* — i.e. **because they were easier**, which
  the prompt forbids two sections earlier. Both runs still said `SUCCESS` and
  both grasped a cutting tool; only *which* and *why* gave it away. The grand
  plan no longer names a target at all.
* **The gate counted the wrong evidence.** It wanted two failed attempts *on the
  target* — but a target is normally defeated without ever being touched, by
  repeated failure to clear what stands in front of it. On `affordance_table`
  with a broken gripper the knife accumulated **zero** attempts while the run
  went nowhere, so the gate could never fire. Evidence is now
  `iterations_without_progress()`, which `is_stalled` also delegates to, so the
  two cannot disagree about whether a target is working.
* **Stall detection pre-empted retargeting entirely.** A stall *is* the evidence
  a retarget needs, and it fired on exactly the iteration that made one eligible.
  The loop now defers a stall **once**, and only when an unused ranked
  alternative exists.
* **A premature retarget was fatal.** `validate` fails to `abort` everywhere,
  a convention written for decisions that are *impossible*. Live, the model asked
  to switch after one failure with sound reasoning (*"rather than risk repeated
  failures on the bottle..."*) and the run died. Refusals now split: structural
  ones (no alternatives, budget spent, target gone) abort; timing ones **defer** —
  nothing is acted on, the correction reaches the next decision, and the run
  continues. Bounded, because a deferred iteration records no progress and stall
  detection counts it; a test drives a planner that retargets on *every* call and
  asserts it still ends below the cap.

A fifth, found while writing the deferral: `state.note()` after `end_iteration()`
raises, because no iteration is open — the same trap §7.11 records for the outage
guard. `note_run()` now exists for facts about the run rather than an iteration.

**Two new scenarios**, both with their measurements asserted so they cannot drift
into being easy — `occluded_target` once claimed an obstruction it did not have
(§7.10) and that is the mistake being guarded against:

* `affordance_table` — knife / bottle / apple / mug. Each need has exactly one
  answer. The knife is 81% visible and blocked by the bottle; the apple and mug
  are 100% visible with **no blockers**. The asymmetry is the point: the right
  answer costs an iteration and the wrong answers are free, so a model dodging
  work is measurably wrong.
* `affordance_choice` — knife / bottle / scissors. Both cut; the knife is
  blocked and the scissors are clear. This is the scene the retarget path needs.
  Two things had to be measured rather than assumed: a fourth object fouled the
  scissors' jaw approach at 22 cm, and at **1.2 cm thick the scissors had no
  blockers and no reachable grasp either** — a run retargeting to them looked
  correct and then span to the iteration cap. **Unblocked is not graspable.**

### 7.11 What only a live LLM found

The offline suite (553 tests) and `verify_declutter.py` (53/53) both pass without
an LLM, and neither could have caught any of the following. Each is a path that
exists only when a model chooses to take it.

* **`find_by_id` had never been executed.** It was implemented, documented in the
  Coder prompt, and reachable only when a model decided to call one tool rather
  than another — so it had **no test at all**, and `test_prompts.py` only checked
  that the name existed on the class. Its very first live invocation raised
  `'Image' object has no attribute 'shape'`: the line read `self.original_img.shape[0]`
  and `original_img` is a **PIL Image** (`image_patch.py:382`), not an array. The
  `hasattr` fallback beside it never fired, because the attribute is present — it
  is the wrong *type*. Every call failed, the Coder fell back to `find()`, and the
  ambiguity that instance ids exist to remove came straight back.
  `tests/test_image_patch.py::TestFindById` now exercises the real call chain and
  reproduces the exact `AttributeError` when the fix is reverted.
* **The instance id never reached the Coder.** The registry assigns stable ids and
  `find_by_id` resolves them, but `_grasp_via_agents` asked the inner loop for
  `inst.descriptor or inst.label` — a *label*. The Coder had no id to pass, so it
  emitted `find("mug")`. Threading the id in was half the fix: the inner **Planner**
  then paraphrased `obj_003 (the mug)` back to "Find the mug in the image", and the
  Coder follows the plan, not the query. `planner_prompt.py` rule 5 now requires ids
  to be carried verbatim into every step. Both halves confirmed live — the Coder now
  emits `find_by_id("obj_003")`.
* **A provider outage killed the run rather than ending it.** `_grasp_via_agents`
  had always caught exceptions and fallen back to a geometric grasp; the *outer*
  planner had no such guard. A daily quota exhausted mid-run raised straight through
  `run()`, the process died, and `progress.json` was left saying `in_progress` with a
  half-open iteration for `--resume` to trip over. Both outer LLM calls are now
  guarded: an unreachable planner aborts with a stated reason and a closed record, and
  an unreachable grand-plan draft falls back to the geometric removal order. Note the
  guard reports through `Decision.corrections`, not `state.note` — `_decide` runs
  *before* `begin_iteration`, so there is no open iteration to note against.
* **Long-horizon runs overwrote their own visual record.** The inner loop names images
  by inner round (`round0_overlay.png`), which is unique within one query and repeats
  on every outer iteration, so iteration 1 silently overwrote iteration 0. Only the
  final pick survived. `RunRecorder.set_scope("iterN")` now gives each iteration its
  own `images/iterN/` directory and prefixes array filenames.
* **The Observer misreads synthetic scenes, and correctly reports code failures.**
  Asked whether a grasp is on "the mug", it sees a red cylinder and reports a target
  mismatch — the primitives do not look like their labels. So synthetic scenes
  exercise the *loop*, not the VLM's semantic grounding; the real sample scenes remain
  the test for that. It did diagnose the `AttributeError` correctly and repeatedly,
  which is the behaviour that matters.
* **`main_declutter.py` could not write artifacts at all.** `recorder.run_dir` does
  not exist — the attribute is `.dir`. Any run without `--no-artifacts` raised
  `AttributeError` before reaching the loop, which is why every previous verification
  went through `verify_declutter.py` instead and this never showed up.
* **Head-truncating the LLM trace hid the evidence.** Prompts were stored as
  `prompt[:4000]`, which on a `str.format()` template keeps the static instructions
  and drops the interpolated scene, blocking analysis and history — exactly the part
  needed to judge whether a bad decision was the model's fault or the prompt's.
  `_elide` now keeps both ends.

## 8. Progress log

### 2026-08-27 — session 8, phase 4: two files, and taking the answer key away

Detail in §6 *"Two files, two jobs"* and *"What the fault suite caught"*.
**837 offline tests** (21 new), **75/75** verification.

* **`planner_state.json`** — exactly what the task planner is told, written
  beside `progress.json`, which stays the archive. `planner_context()` renders
  from it so the prompt and the file cannot drift. **1,798 chars against
  17,399** on a real run. The plan/history join is arithmetic and is done in
  Python now: each step marked `done` / `attempted` / `next` / `pending`, plus
  the off-plan actions the geometric fallback takes and nothing used to mark.
* **The executor no longer names the fault it injected.** Reports describe what
  a sensor could observe; the cause goes to `ground_truth`, which `describe()`
  deliberately excludes. Every previously recorded "recovery" had been reading
  `"injected: the object slipped out of the jaw on lift"`.
* **A rejected grasp is not silently replaced**, and the retarget gate is
  `has_acted()` rather than two fruitless iterations.

**The plan said to delete the `defer` path; that was wrong and it stays.**
Making every refusal terminal is tidier and re-creates §7.12 exactly — a run
that aborts at iteration 0 having done nothing. What changed is how rarely it
fires.

**Measured, six injected faults:** `none`, `drop`, `offset`, `tip`,
`collateral` all reach the target in 3-4 iterations. `wrong_object` recovers the
object — failing, diversifying to the other occluder, coming back and clearing
it **without being told what the fault was**, which is the inference the whole
change was for — and then hits the placement gap in §6.

**The lesson of the phase.** The fault suite found three defects that 833
offline tests and 75/75 verification passed over, because neither runs the
Observer. Two were mine, introduced in phases 2 and 4. One of them was a
four-word ambiguity in a digest — `"2 attempts"` read as iterations rather than
rounds — that no test could catch, because both readings are correct English.
**Read why a run did what it did; the exit status will not tell you.**

### 2026-08-27 — session 8, phase 3: giving a rejection somewhere to go

The phase the whole review pointed at. Detail in §6 *"The candidate set…"* and *"What VLPart can and
cannot do"*. **816 offline tests** (29 new), **75/75** verification.

**Before this, an Observer rejection could not change anything.** `grasp_detection` threw away the
15-60 alternatives that survived its own filters and `_GRASP_CACHE` held the single winner keyed on
the mask, so a re-plan re-ran the identical computation and got a bit-identical pose. Now the cache
holds the whole set, the result carries `candidates`, and `select_grasp` rebuilds a full grasp from
any of them — **0.000 s, no server call, a genuinely different approach direction**.

**`find_part` no longer fails on a naming mismatch.** `bottle cap` scores 0.590 and `bottle lid`
0.058 for the same physical feature, so one bounded retry under other names (asked of the model via
`llm_query`, capped at two) recovers a part that is really there. The hazard and the safeguard are
both measured, and are in §6: a bottle has no *wing*, but "shoulder" was offered and matched at
0.302 — against 0.321 for the legitimate `lid → cap`. **No threshold separates them**, which is why
a substitution is reported to the Observer rather than trusted.

**Live**: success, 3 iterations; `(+19 alternatives)` on every grasp; `llm_query` firing for the
synonym retry; `alts=[20]` in the stored state confirming candidates are counted rather than copied,
with `progress.json` down from ~51 KB to ~30 KB.

**Two things this session's runs surfaced that are not code defects.**
* **Three separate provider outages** (`All LLM providers failed for … Connection error`) knocked the
  inner loop out mid-run. The §7.11 guard caught every one, fell back to a geometric grasp and the
  run still reached the target — which is the guard working, and also why two smoke runs could not
  exercise a full rejection-then-reselect cycle. That path is verified by direct probe against the
  live server and by 14 offline tests, **not yet by an Observer rejection in a live run**.
* **All the outages were the coder, none the planner or observer.** The coder prompt is ~30 KB /
  7.7k tokens, 2.2x the next largest, and grew 51 lines this phase. Two samples is not causation, but
  it is the one agent whose request body is big enough for a transient drop to matter. Worth watching,
  and worth trimming if it recurs — most of that bulk is inherited ViperGPT boilerplate.

### 2026-08-27 — session 8, phase 2: what each agent is allowed to know

Phase 1 fixed the instruments; phase 2 widens the channels between stages. Seven changes, detail in
§6 *"What each agent is allowed to know"*. **787 offline tests** (32 new), **75/75** verification.

**The Observer now gets what was already measured** — the selection funnel
(`416 → 89 visible → 88 on target → 22 clear`), the three dropped-constraint warnings
(COLLISION / OFF-TARGET / WRONG REGION), and the gripper's morphology. It is told to trust those
numbers over the picture, because a 2D projection cannot separate a hand in front of an object from
one driven through it, and the 3D answer was already computed.

**The Planner may finally read the judgement.** Prompt rule 2 forbade it and the code enforced the
ban. It now receives verdict, failure kind and every checklist field.

**A three-way failure taxonomy**, because the three have different owners: `geometry` and
`wrong_part` the inner loop can fix; `wrong_object` it cannot, since the id comes from the outer
planner. It is told to return upward rather than "diversify" onto an object nobody asked for.

**Measured live, `occluded_target`, clean:** success, 3 iterations, **20 LLM calls**, final grasp
**0.886 from the sampler**. The pattern to look for in `progress.json` is a rejection carrying a
*specific* reason followed by a pass — `wrong_part → VALID`, `geometry → VALID` — which is the loop
acting on the diagnosis rather than re-rolling.

**Two defects this cost, both introduced in phase 1 and both found only by running it.**

* **Tuning the prompt against a synthetic scene, and why it was wrong twice over.** The Planner
  chased a mug handle VLPart could not find, so the prompt was rewritten to prefer the body, then to
  forbid `find_part` on declutter moves outright. Both edits were **reverted**, for two reasons the
  user identified and the measurements confirmed:
  * **The evidence was worthless.** A synthetic "mug" is a bare `cylinder` and a "banana" is a `box`
    (`synth_scene.py`). There are no parts to find, so VLPart was *correct* every time. This is the
    §7.13 failure — reading the detector's labels instead of the picture — repeated.
  * **Object-specific verdicts in a prompt do not generalise.** "A mug's handle is a weaker hold"
    trades the model's own judgement for a handful of observed cases and breaks on the unseen object
    that is the reason for using a model at all. The branches now state a *principle* (most secure,
    least damaging; on handover, hold what the person should not receive) with any example marked as
    an illustration. `tests/test_prompts.py::TestGraspPurposeIsAPrinciple` fails if object-specific
    verdicts reappear.
  * **Part choice cannot be removed.** The Observer's `wrong_part` correction has nowhere to go if
    `find_part` is banned, and which part to grasp is central to grasping, not a tool-only concern.
* **A run can now report `success` on a grasp the critic rejected.** With phase 1 discarding rejected
  grasps, `_grasp_for` falls through to the geometric fallback and returns `score: 0.0,
  source: "nominal"`. The run before the prompt fix did exactly this. Phase 1 made it *visible*
  rather than causing it — previously the rejected grasp was returned with a real-looking score.
  Turning it into an honest "no grasp found, and why" is phase 4 item 17.

**A process note.** One smoke run was launched without `scripts/env.sh` and silently lost the gripper
descriptions — box-hull collision geometry, Franka-constant fingertip depth, `"unknown"` morphology —
i.e. it re-created the exact defect phase 1 removed, announced by a single `WARNING` in a busy log.
Source `env.sh` for **every** run, not just for tests.

### 2026-08-27 — session 8, phase 1: fixing the instruments

A four-agent architecture review (`/home/ishita/.claude/plans/comments-1-in-phase-enchanted-peacock.md`)
found one mistake repeated at every layer: **each stage computes good information and discards most
of it before the next stage sees it.** 416 grasps → 1 returned. Five Observer checklist fields → 0 the
Planner may read. 27,673 chars of run history → 1,123 the task planner sees. 10,500 gripper surface
points → a 7-point stick figure. Phase 1 repairs the measuring instruments, because until they are
right nothing downstream can be trusted.

**Six changes**, detail in §6 *"The picture the Observer judges from"*:

1. Real `fingertip_depth` per gripper in the overlay (was a constant 0.11 m, wrong for every gripper
   including the Franka).
2. The mask reaches the overlay, so the magenta region the Observer prompt describes now exists.
3. The real gripper silhouette replaces the two-finger sketch.
4. **The two prompts disagreed about which way is up.** `find()` builds a bottom-origin axis, so a
   larger `vertical_center` is *higher* in the picture. The Coder prompt said so; the Planner prompt
   called it "image rows", implying the opposite. The Planner writes the plan the Coder implements, so
   this silently grasps the object at the wrong end of the table with no error anywhere.
   `test_prompts.py::TestVerticalConventionIsConsistent` pins the convention to `find()`'s source, not
   to either prompt.
5. **A rejected grasp is no longer kept.** The loop did `grasp_pose = result["grasp"]` every round with
   no reference to the verdict, so a query that ran out of rounds finished holding the last *rejected*
   pose and handed it to the executor. Now `GraspMAS._keeps_grasp`: keep the **latest** grasp the
   Observer did not reject — latest, not best-scoring, because a rejected grasp was rejected for a
   reason and the next round exists to fix it, while the score is the discriminator's opinion of the
   pose in isolation and knows nothing about the scene or the query. An *unparseable* verdict still
   keeps the grasp: a provider hiccup on the critique is no evidence against the pose.
6. **The inner loop could not tell "move this aside" from "hand this over".** Both paths sent it the
   same bare instance id. On a knife those want opposite grips — shifted off the table it goes by the
   handle, handed to a person it goes by the blade so they receive the handle — and the system held
   three different opinions about this at once (Coder example: blade; Observer rule: blade is fragile,
   auto-reject; live Planner: handle). `_grasp_for(..., phase=)` now carries "declutter" or "deliver"
   plus the person's own words, and all three prompts were made to agree.

**Measured.** 730 offline tests before, **755 after** (25 new, 0 failures). `verify_declutter.py`
**75/75**. One live run, `occluded_target`, clean: **success in 3 iterations, 18 LLM calls** — inside
the 16-20 baseline, so nothing regressed. The new rule fired in that run and is visible in `log.txt`:
`round 0: grasp rejected by the Observer, not kept`, then round 1 produced a grasp that was accepted.

**Confirmed live but not yet fixed** — the same run logged `execute_command returned dict in 0.0s` on
the round *after* a rejection. That is `_GRASP_CACHE` returning the identical pose, which is Phase 3's
subject: a rejection currently has no way to produce a different grasp.


### 2026-08-24 — session 7: many keys, and an offset that does not clear the target

**Change 1 — the LLM layer routes over a pool of keys.** Every call went through one key
and one process-wide 60 s window, so a per-minute limit cost a **60 s sleep
mid-manipulation** — on a real arm that is a fault, not a rate limit.
`agents/key_pool.py` now holds one window **per key** and sends each request to the
least-used one. Detail and the two defects it cost in §6.

**Measured live with 6 keys** (`gemini-3.1-flash-lite`, `occluded_target`, `--max-round 2`):

| run | outcome | iters | grasp | calls | 429s | keys 1-6 |
|---|---|---|---|---|---|---|
| clean | success | 3 | 0.909 | 16 | 0 | 3/3/3/3/2/2 |
| `drop` | success | 4 | 0.844 | 21 | 0 | 4/3/3/3/4/4 |
| `wrong_object` | success | 5 | 0.877 | 32 | 0 | 6/6/5/5/5/5 |
| `offset` (random) | success | 3 | 0.862 | 20 | 0 | 3/3/4/4/3/3 |
| `offset` (short) | success | 4 | 0.893 | 25 | 0 | 4/4/4/4/5/4 |

**114 calls, zero 429s, zero retries, nothing truncated, no key in any trace.** Six bare
calls took 8.2 s with no pacing delay at all. Routing also carries *across runs* — `drop`
opened on key_5/key_6 because the clean run had spent more of key_1-4, read back from
`.llm_quota.json`.

**Change 1b — an injected offset that leaves the target blocked.** The existing `offset`
fault slipped in a *random* direction and cleared the target in every recorded run, so
half the two-verdict design had never been exercised end to end. `--inject-offset-dir short`
releases early *along* the path instead. The pair now brackets it:

| | iter-0 verdict | moved | still blocking | planner |
|---|---|---|---|---|
| random 8 cm | `moved_off_target` | 31.2 cm, 9.8 cm off plan | **false** | moved on, never retried |
| short 25 cm | `moved_off_target` | 4.7 cm, 23.6 cm off plan | **true** | cleared the other blocker, **came back**, cleared it |

Same verdict, opposite behaviour, decided by `still_blocking` — the design being acted on
rather than recited. Its words at iteration 2: *"Although the first attempt at removing it
was only partially successful (moved_off_target), it is still the primary obstruction.
Since I have only attempted to move it once, I should try again."*

Two things measurement corrected in the fault itself: the slip was computed in 3D and then
flattened, losing 7.6 mm of the requested miss (placement is 2.5D, so it is computed in the
table plane); and a large `short` offset lands the object back at its start, which is
indistinguishable from `drop` — capped at 95% of travel so it stays *moved, and still in
the way*.

**A process lesson, learned the hard way.** The IDE died mid-batch and took the GraspGen-X
ZMQ server with it. The three runs that followed had no 6-DoF sampler, fell back to the
geometric nominal grasp — which is tagged `score = 0.0` **by design** — and one of them
aborted. Nothing said "the server is down"; what it looked like was a planner giving up.
Those runs are kept as `outputs/runs/*_INVALID_no_graspgen_server` rather than deleted.
**Check port 5556 before believing a grasp score, and read `score == 0.0` as "no sampler",
not as "a bad grasp".** Long runs are now launched with `setsid` so an IDE exit cannot
take them down.

**Change 2 — the task planner works out what the goal refers to.** `--target` is now
optional; an abstract goal is read by the planner in a stage-1 call that returns a
*ranked* list, and the target becomes revisable — once, on evidence — through
`retarget`. Design and the five defects it cost in §7.12.

**Measured, `affordance_table` / `affordance_choice` / `occluded_target`**

| run | goal | chose | outcome |
|---|---|---|---|
| `c2_hungry` | *"i am hungry"* | banana | success, 3 iters, 0.933, 19 calls |
| `c2_cut` | *"i need something to cut"* | knife | success, 2 iters, 0.953, 12 calls |
| `c2_explicit_target` | `--target banana` | banana (stage 1 skipped) | success, 3 iters, 18 calls |
| `c2_choice_clean3` | *"...to cut"*, working gripper | knife, **no retarget** | success, 2 iters |
| `c2_retarget3` | *"...to cut"*, gripper permanently broken | knife → **scissors** | success, 4 iters, 0.941 |

Its own words on picking the banana: *"The banana is a food item and directly addresses
the user's hunger."* The retarget run is the sequence worth reading: remove the bottle
(fails) → retarget **declined**, *"only 1 iteration(s) of evidence against obj_001, and 2
are needed"* → retarget **allowed** at iteration 2 and recorded with the model's reason →
grasp the scissors. 19 vs 18 calls on the same scenario is the **+1 request** the two-stage
design predicted.

**Change 3 — priority against effort, and change 4 — real photographs.** Stage 1 now
sets a suitability `priority` (ties meaningful); Python measures blockers; stage 2
orders on both (§7.13). Verified live in both directions: a tie went to the free
object in **1 iteration with nothing moved**, and a lone priority-1 target kept its
place and paid an iteration to clear its blocker.

`scripts/probe_target_selection.py` tests stage 1 alone against GroundingDINO on five
real photographs: **20/20 correct**. It found the planner *improvising* when nothing
served (a spoon for a nail, a plate when hungry) — fixed by saying an empty list is a
correct answer — and demonstrated the opposite capability by **declining a `bottle`
that the image showed to be a screwdriver**.

**730 offline tests** (132 new), **75/75** verification checks. `AsyncRateLimiter` was
deleted — nothing paced through it once the pool existed, and a tested-but-unused pacer
reads as live.

**The lesson all four changes taught.** Every defect here was found by reading *why* a
run did what it did; none was visible in an exit status. The runs concealing the worst
one both reported `SUCCESS` and both grasped a cutting tool. And three "planner
failures" were failures of *measurement* — read from detector labels rather than the
photographs — so when a choice looks wrong, open the image before blaming the model.


### 2026-08-20 — session 4: the loop with a live LLM driving it

The decluttering loop had never been run with a model in charge — session 3 verified it
geometrically. Running it exposed **seven defects, none of them reachable offline**, all
detailed in §7.11. **584 offline tests pass** (35 new), `verify_declutter.py` still passes
**53/53**, and five LLM-driven runs are recorded under `outputs/runs/20260820T13*`.

**Measured — five runs, `gemini-3.1-flash-lite`, `occluded_target`, `--max-round 2`**

| injected | outcome | iterations | what the loop did |
|---|---|---|---|
| none | **success** | 3 | grasps 0.51 / 0.73 / **0.89**, placed 2.0 and 1.5 cm from plan |
| `offset` | **success** | 3 | both releases 6.6-9.8 cm off plan, both still cleared; neither retried |
| `collateral` | **success** | 3 | reported `obj_003` nudged 3.1 cm; checked it had not re-blocked |
| `drop` | failed | 2 | nothing moved; switched object after the first failure, then stalled out |
| `wrong_object` | failed | 2 | intended object never moved, bystander reported as collateral; stalled out |

Both failures stopped at **2 iterations via stall detection**, not at the 6-iteration cap,
and both stated why. No run truncated a reply, and 76 of 76 LLM calls parsed.

**The user's hypothesis was right, and measurable.** Three of the evaluator's five verdicts
(`moved_off_target`, `object_missing`, `unknown`) and collateral damage entirely appeared
**zero times** in the task-planner prompt — the planner was handed vocabulary nobody had
defined. A verdict glossary and a worked collateral example were added, and the `offset` run
is the evidence they work: two actions judged `moved_off_target`, neither retried, because
`still_blocking` was false both times. That is the two-verdict design being acted on rather
than recited.

**Also fixed**
* `min_max_tokens` / `reasoning_effort` in `llm_config.yaml` — thinking tokens were eating
  the entire reply budget (§6).
* `model_candidates` reordered **by free-tier quota, not capability**: `gemini-3.5-flash` is
  5 RPM / **20 per day**, which one run exhausts (§6).
* `RunRecorder.set_scope` — long-horizon runs were overwriting their own images.
* `_normalise_removal_order` — an agent-supplied amendment could poison `grand_plan.json`
  and crash every later iteration.
* `scripts/summarize_run.py` — digests a run directory into a report-ready summary.

**Next**
* The loop has still only been driven against synthetic scenes. The Observer misreads them
  (§7.11), so its semantic critique is untested end to end; `real_world/00` is the test for that.
* M8, the PyBullet physics backend, remains deferred and remains the honest gap.

### 2026-08-23 — session 6: injecting a fault once, and recovering from it

Asked why the `drop` and `wrong_object` runs failed. **They failed because the fault
was injected on every attempt** (`execution/mutation.py`), which makes the task
impossible by construction: no object the planner intends to move can ever move, so the
only outcome such a run can demonstrate is that the loop gives up. That is not the
question worth asking. **Injection is now one-shot by default** (`--inject-at`, default
the first pick-and-place); `--inject-at every` keeps the persistent fault as its own
scenario, a gripper that is simply broken.

**Every mode now recovers and reaches the target**, scripted and with the LLM:

| injected | scripted | with the LLM | how it recovered |
|---|---|---|---|
| none | 3 iters | 3 iters, 20 calls | — |
| `drop` | 4 | 4, 25 calls | `not_moved`; diversified to the other blocker, then came back and retried |
| `wrong_object` | 4 | 5, 30 calls | `not_moved` plus the bystander reported gone; retried |
| `offset` | 3 | 3, 18 calls | released 8 cm off plan, still cleared, correctly not retried |
| `collateral` | 3 | 3, 16 calls | intended move fine; the knock is in the executor's report |
| `tip` | 3 | 3, 16 calls | position correct, so geometry sees success |

A clean run is 3 iterations, so recovery costs at most one — two under `wrong_object`
with the LLM. The planner's reasoning is the part worth reading: on `drop` it wrote
*"the first attempt to remove obj_002 failed because the grasp did not take hold... I
will switch to the other occluder"*, and two iterations later *"while the first attempt
to move it failed (grasp slip), it remains the only identified obstruction. I must
attempt to remove it again"* — diversify, then retry when retrying is the only move
left.

**Three defects found on the way, all from measurement.**

* **Progress was measured on the wrong thing.** `is_stalled` asked only whether the
  object the loop *chose* stopped blocking. Replaying `occluded_target` under a
  persistent `wrong_object` fault: at step 1 the hand closed on a real blocker by
  accident and carried it away, taking the target from **32% visible with 2 blockers to
  78% with 1** — and the run declared "no progress" and stopped. `Evaluation` now keeps
  `target_blockers` (the evaluator was computing the whole set and discarding all but
  one membership test), and an iteration counts as progress if the object acted on
  cleared **or** the target's blocker set shrank, whoever shrank it.
* **A phantom shift on the target aborted a run.** `moved_since` gated collateral on how
  well an object is seen *now*, not on whether the stored centroid it is compared against
  was trustworthy. With the mug carried away the banana went **78% → 100% visible and
  appeared to move 3.8 cm without being touched**; it was reported as collateral damage
  *to the target*, and the planner aborted on the strength of it — reasonably, since
  displacing the thing you were sent to fetch is serious. Both ends of the comparison are
  now gated (`previous_visibility`). With the phantom gone, the same LLM run succeeds.
* **The diagnosis was generic where the evidence was specific.** Both stalled runs
  produced the identical "no progress in 2 iterations" while the executor reports carried
  a clear and *different* signature in each. `SessionState.stall_diagnosis` now names the
  pattern — a repeated failure at the same stage (the grasp, not the choice of object),
  the hand reaching a different object than planned (targeting), no placement (the table
  is full), no grasp (unreachable). **A bug in that fix, caught before shipping:** the
  first version filtered empty errors before `all()`, making it vacuously true, and
  reported "no grasp in the last 3 attempts" from a *single* error. Every claim now
  carries its count, and a pattern must cover at least half the attempts or nothing is
  said.

**A limitation this made explicit, and deliberately not "fixed".** Collateral detection
only sees near-fully-visible objects, because the 95% gate that kills the phantom also
hides real nudges: a box genuinely moved 3.3 cm went unreported at 82% visible. The
executor's `disturbed` field caught it, but a real arm cannot report what it brushed
past. Loosening the gate trades a missed nudge for a false alarm on the target — the
worse error, as this session demonstrated.

**606 offline tests** (22 new), **75/75** verification checks (up from 53: the injected
runs now assert recovery rather than mere termination, plus a persistent-fault case).
Seven runs under `outputs/runs/20260823T1*`, index at `outputs/reports/index.md`.

### 2026-08-21 — session 5: making a run explainable

The loop worked but could not be *read*. Three things were being computed and thrown
away, and each is the kind of loss you only notice when you try to write up a run.

* **The inner loop kept only its last round.** `GraspMAS` overwrites
  `thought`/`plan`/`code` every round, so after a query the reasoning that produced the
  grasp was gone and only the round that concluded survived. Now accumulated in
  `GraspMAS.rounds` and stored per iteration as `agent_rounds` — including the
  "return to user" round, which is a decision and reads as "ran out of rounds" when omitted.
* **The post-execution scene was captured and discarded.** `_perceive` runs twice per
  iteration — once to decide, once to evaluate — and both write an image now
  (`scene_iter<N>_before.png` / `_after.png`). The second is the only picture of what an
  action actually did, and the last iteration has no following capture to stand in for it.
  Note the two calls originally wrote the *same* filename, so the "before" view was
  silently overwritten by the "after" one; `_perceive(phase=...)` is what separates them.
* **`llm_trace.jsonl` had no iteration field**, so the trace was a flat list and no report
  could say which decision led to which pick. The recorder scope the loop already sets per
  iteration was exactly the missing attribution.

Also added: `SessionState.snapshot` freezes both state files under `states/iter<N>/` after
every evaluation, because `progress.json` is rewritten in place and the finished file shows
only the end state — a grand plan amended away at iteration 1 was otherwise unrecoverable.
And `IterationRecord.decision` keeps what the outer planner chose and why, alongside any
amendment it proposed and any correction Python applied to it.

**`scripts/build_report.py`** turns all of that into `<run>/report.md`: prompt, gripper and
initial scene, then per iteration → task planner (decision, blocking analysis, amendment)
→ grasping agents round by round (thought, plan, code, grasp, overlay, observer verdict) →
executor (what it reported, scene after) → evaluator (verdict, distances, state snapshot),
then a final result table and LLM cost. `--index` writes `outputs/reports/index.md` mapping
every run to its scenario and injected failure.

**Six runs recorded** under `outputs/runs/20260821T06*` — the five LLM scenarios plus a
no-LLM control on identical machinery, which is what separates a planning failure from a
geometry failure. Outcomes unchanged from session 4. **584 offline tests**, 53/53
verification checks.

### 2026-08-19 — session 3: long-horizon decluttering

Built the whole loop, M1 through M7. **549 tests pass** (331 new, all offline, ~4m45s) and
`scripts/verify_declutter.py` passes **53/53** end-to-end checks with no LLM, no GPU, no server.

**Delivered**
* `placement.py` — support-plane RANSAC, the table frame, height map, free-space search by
  distance transform, oriented footprints, keep-out regions, and the place pose as a pure
  translation of the grasp pose.
* `collision.py` — the gap `SUMMARY.md` §8.5 named: `grasp_detection` never checked candidates
  against the *rest* of the scene, so in clutter it proposes grasps that drive the hand through
  a neighbour. Gripper geometry from `points.json`, proximity by `scipy.spatial.cKDTree`.
* `scene_registry.py` — instance ids that survive re-perception, and the three-way obstruction
  test (occlusion / approach / proximity) that decides both what to move and when to stop.
* `session_state.py` — `progress.json` and `grand_plan.json`, atomic and resumable, with the
  guardrails that stop an agent editing its own success criterion.
* `execution/` — one interface, three backends: mutation, replay, and a documented robot stub.
* `evaluator.py` — geometric verdicts, VLM fallback only where geometry structurally cannot see.
* `declutter.py`, `main_declutter.py`, `agents/task_planner.py` (+ its prompt) — the loop.
* `synth_scene.py` — ray-cast tabletops with exact ground truth, four scenarios.
* Wired into the existing pipeline: `place_detection` and `find_by_id` on `ImagePatch`, scene
  collision filtering in `grasp_detection`, `GraspMAS.reset()`, a widened result gate, and the
  Observer's verdict kept instead of discarded.
* `docs/declutter.md` — the loop contract, state schemas, and the real-robot interface.

**Measured**

| | |
|---|---|
| declutter, clean run | 2 moves, target 32% → 100% visible, 3 iterations |
| placement accuracy | objects landed 1.5-2.0 cm from plan |
| without the placement keep-out | 4 moves, target still blocked (§7.5) |
| plane fit | 0.05° / 0.03 mm synthetic; 47-50% inliers on the real scenes |
| collision filter | 200 ms for 200 candidates against a 15k-point scene |
| every injected failure | terminated with a stated reason; never ran away |

**Found by measuring, not by reading** — detail in §7.3-§7.8
* Inlier-count RANSAC fits the clutter, not the table — 9.9° / 8.4 cm off on `crowded_table`.
* Plane fitting on a whole real capture cannot work; the cutoff has to come from the objects.
* Single-view footprints are biased small by 1.0-1.7 cm, heavily occluded ones by 11.8 cm.
* Convex-hull visibility scores a 19%-visible object at 98%.
* Placement without a keep-out puts the blocker back in front of the target, repeatedly. The
  keep-out then needed three corrections, each found by measurement (§7.5).
* A deliberate move exceeds the identity match radius and reads as a vanished object.
* Collateral damage can move an object out of its own identity, and went unreported.
* 384 gripper sample points leave 12.3 mm gaps — wider than the 10 mm collision threshold.
* `min_hits` counting gripper points instead of scene points defeats its own purpose.
* The first `occluded_target` scenario did not actually obstruct anything; a parallel jaw spans
  the target's *short* axis, so blockers belong in front, not beside.

**Next**
* A physics backend (M8, PyBullet — viable on CPU, deferred deliberately).
* A real closed-loop run with the LLM: `main_declutter.py` without `--no-llm`, once a key is set.
* The offline suite is now ~4m45s, up from 25s — mostly synthetic rendering and registry
  building. Worth session-scoping more fixtures if it grows further.

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
  (Session 3 took this to 549 / ~5 min.)
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

## 9. Handoff — what needs the user

Everything that can be verified without credentials or extra datasets **has been**:
**730 offline tests**, `scripts/verify_pipeline.py` end-to-end on real RGB-D, and
`scripts/verify_declutter.py` running the whole long-horizon loop against ground truth
(75/75 checks).

**The agent loop is no longer blocked.** The key pool works and the loop has run end to
end with Gemini driving it: the injected-failure matrix (§8, sessions 6-7), abstract
goals with no `--target`, and a live retarget. Narratives at
`outputs/reports/index.md`; stage-1 target selection on real photographs at
`outputs/reports/target_selection_real_photos.md`.

**Two open risks, neither a bug.** *A label is not an object* — suitability is decided
once from detector labels and validated only for existence, so a confident wrong label
propagates unchecked (§7.13, SUMMARY §8.1). And execution is scene mutation, not
physics, so the evaluator has only met failures somebody thought of (§7.9).

**One thing remains**, and it is blocked on a dataset only the user can supply.

### 1. Running it yourself

<https://aistudio.google.com/apikey> (no credit card). **Get several, from
separate Google accounts** — quota is per project, so N keys give N times the
rate and N times the daily budget, and one key does not cover a day's work.

```bash
# one key per line; blank lines and # comments ignored
cat >> GraspMAS/api.key <<'EOF'
AIza...key1
AIza...key2
EOF
# or, equivalently:
export LLM_API_KEY="key1,key2,key3"          # also LLM_API_KEY_1, _2, ...

conda run -n graspmas python -m agents.llm --probe --test
```

`--probe` now prints one row per key — models reachable, requests used today,
models known exhausted — so a typo'd or spent key is visible before a run, not
during one. `--test` sends one text and one **vision** call. Pin the chosen id in `GraspMAS/llm_config.yaml` (the
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

`--max-round 2` keeps one query to roughly 6–10 requests.

Then the decluttering loop with the LLM driving it:

```bash
cd GraspMAS && python main_declutter.py \
    --goal "pick up the banana" --target banana --max-round 2 -v
```

`--target` is optional. Leave it out and the task planner works out what the
goal means from the scene, which is what makes an abstract request work:

```bash
# the banana is the only food on the table
python main_declutter.py --goal "i am hungry" --scenario occluded_target
# the knife is the only cutter, and it is the blocked one
python main_declutter.py --goal "i need something to cut" --scenario affordance_table
# two cutters, one blocked: exercises the retarget path
python main_declutter.py --goal "i need something to cut" \
    --scenario affordance_choice --inject drop --inject-at every
```

Inferring the target costs **one extra request**. `--no-llm` still requires
`--target`, since nothing offline can read a goal like that.

Measured cost: **16 requests** for a clean three-iteration run, 11-18 across the
five recorded runs. The evaluator is geometric and spends nothing. Add
`--inject drop,offset,collateral,wrong_object` to exercise the failure paths, and
`--no-llm` to run the whole machinery for free.

**Watch the per-model daily quota, not the headline one** — `gemini-3.5-flash` is
20 requests/day *per key* and one run exhausts it (§6). The key pool spreads load
across keys and remembers which are spent in `.llm_quota.json`;
`python -m agents.llm --probe` shows the standing of each, and
`scripts/summarize_run.py <run_dir>` digests what a run did.

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
