# SUMMARY — how the 6-DoF GraspMAS actually performs

A robotics-perspective evaluation of replacing GraspMAS's planar grasp head with
GraspGen-X. Written for a reader who cares about grasping, not plumbing; the
architecture and setup live in [`CLAUDE.md`](CLAUDE.md).

**Status as of 2026-08-12.** Everything below is measured on this CPU-only
machine (8 cores, 15 GB RAM, no GPU). Raw artifacts:
`outputs/runs/20260812T153645Z_verify_pipeline/`, benchmark
`outputs/bench/cpu_latency.json`. Reproduce with
`conda run -n graspmas python scripts/verify_pipeline.py`.

---

## 1. Does it work?

**Yes for the grasp pipeline; the agent loop is built and unit-tested but has not
been run end-to-end** because that needs a free Gemini API key the user has not
created yet (see §8).

What is verified working, on real RGB-D:

| | |
|---|---|
| Language → mask → metric cloud → 6-DoF SE(3) pose | ✅ 3/3 target objects |
| Multi-embodiment (one model, several hands) | ✅ 3/3 grippers, no restart |
| Part-level grounding reaching the sampler | ✅ demonstrated (§4) |
| RGB-only monocular fallback | ✅ works, with ~17% scale error (§6) |
| CPU inference | ✅ 2–13 s/call, 2.6 GB peak RSS |
| Offline test suite | ✅ 218 tests, ~25 s, no GPU/network/LLM |

The old pipeline returned `[quality, x, y, w, h, angle]` — an image-plane
rectangle. The new one returns a 4×4 SE(3) pose in metres, in the camera frame,
anchored at the gripper base with `+Z` = approach and `+X` = closing.

---

## 2. Grasp quality

Scene: the GraspGen-X sample tabletop (`real_world/00`) — a YCB-style set on a
wooden table, Kinect-class depth, real intrinsics.

| target | score | position (m, camera frame) | approach vs. optical axis | time |
|---|---|---|---|---|
| yellow cup | 0.78 | (−0.221, 0.051, 1.082) | 3° | 32.6 s |
| mustard bottle | 0.86 | (0.053, 0.078, 0.960) | 44° | 18.0 s |
| cracker box | 0.88 | (−0.478, −0.064, 0.974) | 72° | 16.5 s |

Discriminator scores of 0.78–0.88 are in the range GraspGen-X's own demos treat
as confident (their default threshold is 0.7). Visual inspection of the overlays
(`outputs/.../images/rgbd_*.png`) shows the fingers straddling the object with the
closing line across a graspable dimension in all three cases.

**The approach spread is the substantive result.** Three objects produced 3°, 44°
and 72° — a near head-on grasp into the cup's opening, an angled grasp on the
bottle, and a near-side grasp on the box. The planar predecessor could not
express any of this: `Maniskill_demo.ipynb` hardcoded `approaching = [0, 0, -1]`
and took only the in-plane rotation from the rectangle, so every grasp it ever
produced had the same approach direction. That is 4 DoF wearing a 6-DoF label.

Across all candidates the system generated in this run (1348), 385 survived
filtering, with a **median approach of 67°** and mass spread continuously from 0°
to 100°. See `plots/approach_spread.png`.

### Rotation sanity

Every returned pose was checked for `R·Rᵀ = I` and `det(R) = +1` to 1e-3
(`GraspGenX/tests/test_cpu_patch.py`). A pose that is not a proper rotation
silently poisons downstream IK, and nothing else in the stack would catch it.

---

## 3. Embodiment generalization

Same object, same point cloud, three different hands, no server restart and no
retraining — the model is conditioned on the gripper's swept volume, which
travels with the request:

| gripper | score | jaw width | approach |
|---|---|---|---|
| `franka_panda` | 0.93 | 8.0 cm | 20° |
| `robotiq_2f_85` | 0.93 | 8.5 cm | 5° |
| `unitree_g1` | 0.92 | 10.0 cm | 91° |

The jaw widths are read from each gripper's own `config.json`, so they are the
real hand geometry rather than a constant. The `unitree_g1` choosing a
side approach where the two parallel jaws chose near-head-on is the kind of
embodiment-specific behaviour the single-gripper predecessor could not produce.

27 grippers are available locally; any of them works via `--gripper-name`.

---

## 4. Language grounding fidelity — the part-level claim

This is the reason GraspMAS is worth keeping rather than calling GraspGen-X
directly, so it deserves the most scrutiny.

**The mechanism:** `find_part` produces a mask of the named part; that mask
selects which pixels get unprojected; the resulting cloud *is* the part. The
language constraint therefore reaches the sampler through the geometry it sees,
not through post-hoc filtering.

**Measured** — "mustard bottle" vs. "the cap of the mustard bottle":

| | position (m) | |
|---|---|---|
| whole-object grasp | (−0.134, 0.070, 1.031) | |
| part-level grasp | (0.037, −0.009, 0.974) | |
| **displacement** | **0.197 m** | grasp moved 20 cm onto the cap |
| grasp centre inside the cap mask | **True** | |

The cap mask is 1084 px of a 7889 px object (14%). The overlay
(`images/part_level.png`) shows the gripper on the cap, not the body.

### Important caveat: VLPart's vocabulary is narrow, and misses are silent

Probing 14 (object, part) pairs on this scene:

| segments a real sub-region | falls back to the whole object |
|---|---|
| mustard bottle → cap (14%), lid (14%), body (90%) | toy airplane → wing |
| bottle → cap (40%), body (99%) | cup → handle *(the cup has none — correct)* |
| toy airplane → propeller (18%) | bowl → rim, spray bottle → nozzle/handle |
| cracker box → lid (9%) | bottle → neck |

Upstream `find_part` returned the whole object on a miss **with no signal**, so a
caller could not distinguish "grasped the part" as instructed from "the part
concept doesn't exist, grasped the whole thing". Our first part test used
"airplane → wing", got back a mask byte-identical to the whole airplane, and
would have reported a spurious success. The fallback is retained (a whole-object
grasp beats no grasp) but now logs a warning and sets `patch.part_found = False`.

**Practical consequence:** part-level grasping works for parts in VLPart's
PACO/PartImageNet-derived vocabulary — handles, caps, lids, blades, bodies,
propellers — and degrades to object-level otherwise. Any evaluation of
"part-level accuracy" must check `part_found` or it is measuring the wrong thing.

---

## 5. What the filtering does, and does not, contribute

Two filters sit between the sampler and the returned grasp. Their measured
contributions differ sharply:

**Visibility filter (added here).** GraspMoE's OBB branch sweeps candidates over
every face of the box it fits to the object — including the far side a single
depth image never observed. Raw candidate median approach was **180°**: a hand
travelling from behind the object toward the camera, i.e. through the table.
Rejecting approaches more than 100° off the viewing ray removes these. This is a
real filter: it cut 1348 candidates to 385.

**Mask-containment filter.** Once evaluated at the *fingertips* (see §7), this
keeps essentially everything: 58/58, 57/58, 56/56 on the three test objects. That
is expected and correct — the candidates were generated *from* that object's
cloud, so they land on it. It is a safety net for stragglers, **not** the
mechanism that enforces language. Reporting it as the mechanism would overclaim;
the point cloud is the mechanism.

---

## 6. Monocular RGB-only fallback

When no depth is supplied, `Depth-Anything-V2-Metric-Indoor-Small` (99 MB)
estimates metric depth.

| | |
|---|---|
| sensor depth on target | 1.273 m |
| estimated depth on target | 1.494 m |
| **relative error** | **17%** |
| grasp still produced | yes, score 0.93 |

17% scale error propagates directly into grasp position and into whether the jaw
width fits the object: a 6 cm object estimated 17% too far is reconstructed ~7 cm
wide. The grasp *direction* is much less affected than the *scale*.

**Use real depth when you have it.** The monocular path is for demos and for
images where depth simply does not exist; it is not a substitute for a sensor.
The path is fully wired and the discrepancy is logged on every run, so the cost
is always visible rather than hidden.

---

## 7. Failure taxonomy — what actually went wrong

Every one of these was found by running the pipeline, not by reading code, and
each is now covered by a test.

| failure | symptom | cause | fix |
|---|---|---|---|
| **Server OOM-killed** | client timeout after 180 s, server process gone | `point_cloud_outlier_removal` does `torch.cdist(X, X)` + `torch.eye(N)` on the *raw* cloud; a 40 k-point mask asks for ~6 GB | cap the cloud at 8192 points client-side (`downsample_cloud`) |
| **Cracker box returned 0 grasps** | grounded fine, no grasp | my voxel sizing used the cube root (volume), but a depth cloud is a *surface* — 16786 points collapsed to **212**, too sparse for the encoder | size from the square root, then bisect to land just under the cap |
| **~90% of valid grasps rejected** | scores collapsed (cup 0.907 → 0.565) | mask containment tested at the gripper **base**, ~10 cm behind where the hand closes | thread the real `fingertip_depth` from the gripper config; cup went 6/58 → 58/58 kept |
| **Grasps approaching through the table** | plausible-looking poses, physically unreachable | OBB sweep over unobserved faces | visibility filter (§5) |
| **Spurious part-level success** | "wing" mask identical to whole object | `find_part` fell back silently | warn + `part_found` flag |
| **Coder agent broken** | `KeyError` at call time | prompts are `str.format()` templates; a literal `{` in a JSON example | escape, plus a test that formats all three templates |

Categories that did **not** fail: grounding (3/3 objects correctly located by
GroundingDINO+SAM), rotation validity, ZMQ transport, unit conversion.

---

## 8. Limitations

1. **The agent loop is unexercised end-to-end.** Planner/Coder/Observer are
   rewritten for 6-DoF, routed through the free-tier client, and covered by 218
   offline tests against a stub LLM — but no real query has run, because that
   needs an API key. Everything downstream of the Coder is verified.
2. **Free-tier request budget.** ~1500 requests/day at 15–25 per query ≈ 60–100
   queries/day. Fine interactively; `main_batch.py` is `--limit`-ed and
   resumable for exactly this reason.
3. **OCID-VLG numbers are not in yet.** The harness is written, the metric is
   wired (6-DoF back-projected to a rectangle, scored IoU@0.25 / angle@30°
   against the planar ground truth), and OCID intrinsics are a `--intrinsics`
   flag away — but the dataset is a separate multi-GB download the user has not
   fetched. Without it there is no comparison against the published 2D baseline.
4. **Single-view partial clouds.** One depth image sees one surface. GraspGen-X
   completes the far side plausibly but not truthfully; the visibility filter
   exists precisely because that completion cannot be trusted for approach
   planning.
5. **No scene-collision filtering.** Grasps are checked against the target's
   mask and the viewing ray, not against neighbouring objects. GraspGen-X ships
   `filter_collisions` but it needs a gripper collision mesh and a scene cloud;
   in clutter, expect the arm to want a path that another object occupies.
6. **CPU latency.** 2–13 s per grasp call. Acceptable interactively, limiting for
   large sweeps.
7. **ManiSkill is out of scope** — SAPIEN needs a Vulkan GPU to render camera
   images, so the simulation demo cannot run here at all.

---

## 9. Performance on CPU

Measured on 8 cores, fp32, `ptv3vanilla` backbone, 20 DDPM steps
(`outputs/bench/cpu_latency.json`, plot `outputs/bench/cpu_latency.png`):

| | |
|---|---|
| model load (once, amortized by the server) | 3.3 s |
| grasp inference, 50 samples | 1.2–3.5 s |
| grasp inference, 200 samples | 3.6–13.4 s |
| peak RSS, GraspGen-X server | 2.6 GB |
| peak RSS, GraspMAS perception stack | < 8 GB (test-enforced) |
| full verification (3 objects, 3 grippers, part, mono) | 152 s |

This was much better than expected. The original plan budgeted 30 s–3 min per
call; the reality is 2–13 s. Two reasons: the released checkpoint uses the
pure-PyTorch `ptv3vanilla` backbone rather than a CUDA-extension PointNet++, and
only 20 diffusion steps are used at eval.

**GraspMoE vs. diffusion-only:** GraspMoE consistently produced higher top
scores (0.87–0.93 vs. 0.80–0.84) at comparable cost, because its OBB branch
reuses the diffusion pass's object embedding rather than re-encoding. It is the
default here.

---

## 10. What a GPU would change

* **Latency**, by roughly an order of magnitude — and TensorRT support is already
  in the repo behind `--tensorrt`, currently untestable here.
* **The `enable_flash` config override could be dropped** — the fp16
  `scaled_dot_product_attention` path the checkpoints ship is a CUDA path.
* **ManiSkill becomes possible**, closing the loop from grasp to executed motion,
  which is the one claim this evaluation cannot make: every grasp here is judged
  by discriminator score, projected geometry and visual inspection. **No grasp in
  this report has been executed on a robot or in simulation.** Simulated or real
  success rate remains the honest missing number.
* **A full OCID-VLG sweep** becomes practical in hours rather than days —
  though the LLM free tier, not the GPU, is the binding constraint there.
