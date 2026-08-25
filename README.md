# 6-DoF GraspMAS

**GraspMAS** with its planar 2D grasp head replaced by **GraspGen-X**, so
language-driven grasping produces true 6-DoF SE(3) poses instead of image-plane
rectangles. Runs entirely on **CPU** and on a **free LLM tier**.

```
"grasp the mustard bottle by its cap"
        │
        ├─ GroundingDINO + SAM + VLPart  ──▶  mask of the cap
        ├─ depth (sensor, or monocular)  ──▶  metric point cloud, camera frame
        ├─ GraspGen-X diffusion (CPU)    ──▶  6-DoF SE(3) pose + confidence
        └─ Planner / Coder / Observer loop critiques and re-plans
```

| | |
|---|---|
| **[`CLAUDE.md`](CLAUDE.md)** | architecture, environment recipe, invariants, every gotcha found |
| **[`SUMMARY.md`](SUMMARY.md)** | robotics evaluation: what works, measured numbers, failure taxonomy, limitations |
| **[`outputs/README.md`](outputs/README.md)** | artifact schema — what each run writes and why |
| **[`docs/declutter.md`](docs/declutter.md)** | the long-horizon loop: contract, state files, robot interface |

## Quick start

```bash
bash scripts/setup.sh          # envs, repos, CPU patch, ~10 GB of assets
scripts/run_server.sh --daemon # GraspGen-X server, loads the model once
scripts/run_tests.sh           # 606 offline tests: no GPU, no network, no LLM spend
```

Then, with no API key needed (grounding runs on local weights):

```bash
conda run -n graspmas python scripts/verify_pipeline.py
```

For the full agent loop, get a free key at <https://aistudio.google.com/apikey>.

**Use several keys.** The free tier is metered per Google Cloud *project*, so
keys from separate accounts have separate buckets and N keys give N times the
rate and N times the daily budget. Put one per line in `GraspMAS/api.key`
(blank lines and `#` comments ignored), or pass them comma-separated in
`LLM_API_KEY`, or as `LLM_API_KEY_1`, `LLM_API_KEY_2`, ... Every source is
merged. Requests go to whichever key has been used least, and a per-minute
rate limit rotates to another key instead of sleeping. Check what a pool can
reach with `python -m agents.llm --probe`.

```bash
export LLM_API_KEY=...          # or several: "key1,key2,key3"
cd GraspMAS && python main_simple.py \
    --query "grasp the mustard bottle by its cap" \
    --image-path  ../GraspGenX/assets/sample_data/real_world/00/rgb.png \
    --depth-path  ../GraspGenX/assets/sample_data/real_world/00/depth.npy \
    --intrinsics  ../GraspGenX/assets/sample_data/real_world/00/meta_data.json \
    --gripper-name franka_panda
```

`--depth-path` and `--intrinsics` are optional — without them the pipeline
estimates metric depth monocularly, at a measured ~17% scale cost.

## Clutter

When the target is *not* reachable, the loop clears whatever is in the way first —
pick, place, re-perceive, re-plan, until it can be grasped.

```
"pick up the banana"   banana is 32% visible behind a bottle and a mug
        │
        ├─ blocking analysis   occlusion · approach · proximity
        ├─ pick + place        where does it go so it stops obstructing?
        ├─ execute             then look again
        └─ evaluate            did it work, and does it matter?
        ▼
   banana 100% visible, approach clear, grasped — 2 moves
```

```bash
conda run -n graspmas python scripts/verify_declutter.py   # 75 checks, no key needed
cd GraspMAS && python main_declutter.py --goal "pick up the banana" --target banana --no-llm

# With Gemini choosing what to move next (needs a key; ~16 requests)
cd GraspMAS && python main_declutter.py --goal "pick up the banana" --target banana -v
```

Verified both ways: 75/75 geometric checks, and seven LLM-driven runs on 2026-08-23 —
**all reaching the target**, each recovering from a different injected execution fault
(`drop`, `offset`, `tip`, `collateral`, `wrong_object`). Per-iteration narratives at
[`outputs/reports/index.md`](outputs/reports/index.md); measurements in
[`SUMMARY.md`](SUMMARY.md) §11.

Execution is **scene mutation, not physics** — objects are moved and the scene
re-rendered, with failures injected explicitly
(`--inject drop,offset,collateral,wrong_object`). See
[`docs/declutter.md`](docs/declutter.md), and `scripts/summarize_run.py` to digest
a finished run.

## What changed from upstream GraspMAS

| | before | after |
|---|---|---|
| grasp output | `[quality, x, y, w, h, angle]` rectangle | 4×4 SE(3) pose, metres, camera frame |
| approach direction | hardcoded top-down | predicted from 3D geometry (measured 3°–72° across three objects) |
| grippers | one | 27, switchable per request |
| depth | MiDaS relative inverse depth, cast to `int` | metric metres, sensor or monocular |
| LLM | GPT-4o / GPT-4o-mini (paid) | Gemini Flash free tier, rate-limited with failover |
| device | CUDA assumed | CPU, 2–13 s per grasp, 2.6 GB peak |
| horizon | one grasp on a reachable object | clears obstructions until the target is reachable |
| collision | object's own mask only | checked against the whole scene |
| tests | none | 549, offline |

## Repository layout

```
GraspMAS/     fork — 6-DoF grasping, free-tier LLM, tests
GraspGenX/    upstream + patches/graspgenx_cpu.patch
assets/       checkpoints and gripper descriptions
outputs/      every run's artifacts (see outputs/README.md)
scripts/      setup.sh, run_server.sh, run_tests.sh, verify_pipeline.py,
              bench_cpu.py, plot_outputs.py, env.sh
```

## Known constraints

* The **ManiSkill demo is out of scope** — SAPIEN needs a Vulkan GPU to render.
* The **free LLM tier allows ~60–100 queries/day**; `main_batch.py` is
  `--limit`-ed and resumable for that reason.
* **No grasp here has been executed** on a robot or in physics. Decluttering runs
  against scene mutation with injected failures. See [`SUMMARY.md`](SUMMARY.md) §8.

Upstream: [Fsoft-AIC/GraspMAS](https://github.com/Fsoft-AIC/GraspMAS) (IROS 2025) ·
[NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX) (CVPR 2026)
