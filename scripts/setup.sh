#!/usr/bin/env bash
# Reproduce this project from an empty directory.
#
#   bash scripts/setup.sh
#
# Idempotent: every step checks whether it is already done. Roughly 10 GB of
# downloads and ~30 minutes on a warm network.
#
# Assumes conda is on PATH. Everything runs on CPU; no CUDA is required or used.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------
say "git-lfs"
# Must exist BEFORE cloning: GraspGen-X keeps *.obj/*.stl/*.npy in LFS, and a
# clone without it silently yields 132-byte pointer files instead of gripper
# meshes and sample depth.
if ! git lfs version >/dev/null 2>&1; then
  conda install -y -q -c conda-forge git-lfs
fi
git lfs install --skip-repo

# --------------------------------------------------------------------------
say "repositories"
[[ -d GraspMAS ]]  || git clone --recurse-submodules --depth 1 \
  https://github.com/Fsoft-AIC/GraspMAS.git GraspMAS
[[ -d GraspGenX ]] || git clone --depth 1 \
  https://github.com/NVlabs/GraspGenX.git GraspGenX

say "CPU patch for GraspGen-X"
if grep -q "def resolve_device" GraspGenX/graspgenx/grasp_server.py 2>/dev/null; then
  echo "already applied"
else
  git -C GraspGenX apply "${REPO_ROOT}/patches/graspgenx_cpu.patch"
fi

# --------------------------------------------------------------------------
say "conda env: graspgenx (python 3.11)"
if ! conda env list | grep -qE '^graspgenx\s'; then
  conda create -y -q -n graspgenx python=3.11
fi
# torch and torchvision must be installed TOGETHER from the CPU index, before
# anything else: timm pulls torchvision from PyPI, which drags in a CUDA torch.
conda run -n graspgenx pip install -q torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu
conda run -n graspgenx pip install -q \
  "numpy==1.26.4" omegaconf hydra-core "trimesh==4.5.3" h5py scikit-learn scipy \
  imageio webdataset "yourdfpy==0.0.56" "diffusers==0.11.1" \
  "huggingface-hub==0.25.2" "timm==1.0.15" addict viser pyyaml tqdm tensordict \
  matplotlib pyzmq msgpack msgpack-numpy pytest "setuptools<78"
# --no-deps: pyproject declares urdfpy (broken on numpy>=1.24), sharedarray,
# torch-geometric, pyrender, PyOpenGL and scene-synthesizer, none of which any
# file on the inference path imports.
conda run -n graspgenx pip install -q --no-deps -e ./GraspGenX

# --------------------------------------------------------------------------
say "conda env: graspmas (python 3.10)"
if ! conda env list | grep -qE '^graspmas\s'; then
  conda create -y -q -n graspmas python=3.10
fi
conda run -n graspmas pip install -q torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
conda run -n graspmas pip install -q \
  "transformers==4.48.0" accelerate "numpy==1.26.4" opencv-python shapely \
  scikit-image scipy matplotlib gdown openai jupyter pyzmq msgpack msgpack-numpy \
  pytest pytest-asyncio pyyaml "timm==1.0.15" ftfy regex tqdm "setuptools<78" wheel
# bitsandbytes is dropped from requirements.txt: CUDA-only wheel, nothing imports it.
conda run -n graspmas pip install -q "git+https://github.com/openai/CLIP.git"
# --no-build-isolation: detectron2's setup.py imports torch at build time, and
# pip's isolated build environment does not have it.
conda run -n graspmas pip install -q -e ./GraspMAS/detectron2 --no-build-isolation

# --------------------------------------------------------------------------
say "GraspGen-X checkpoints (1.7 GB)"
mkdir -p assets/graspgenx_checkpoints/release/{gen,dis}
HF=https://huggingface.co/adithyamurali/GraspGenXModel/resolve/main
for f in release/gen/config.yaml release/dis/config.yaml \
         release/gen/epoch_736.pth release/dis/epoch_1056.pth; do
  if [[ ! -s "assets/graspgenx_checkpoints/$f" ]]; then
    echo "  fetching $f"
    curl -sSL -o "assets/graspgenx_checkpoints/$f" "$HF/$f"
  fi
done

say "gripper descriptions (665 MB, 27 grippers)"
if [[ ! -d assets/gripper_descriptions ]]; then
  git clone --depth 1 \
    https://huggingface.co/datasets/adithyamurali/gripper_descriptions \
    assets/gripper_descriptions
fi
# The smudge filter does not fire on an HF clone; pull the LFS objects explicitly
# or every gripper mesh is a pointer file.
( cd assets/gripper_descriptions && git lfs pull )

say "VLPart weights (564 MB)"
mkdir -p GraspMAS/weights
if [[ ! -s GraspMAS/weights/swinbase_part_0a0000.pth ]]; then
  curl -sSL -o GraspMAS/weights/swinbase_part_0a0000.pth \
    https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth
fi

# --------------------------------------------------------------------------
say "verifying"
# shellcheck source=./env.sh
source "${REPO_ROOT}/scripts/env.sh"
conda run -n graspgenx python -c "
import torch; assert not torch.cuda.is_available()
print('graspgenx: torch', torch.__version__)
from graspgenx import get_gripper_descriptions_root
print('  grippers:', get_gripper_descriptions_root())"
conda run -n graspmas python -c "
import torch, detectron2, clip
print('graspmas: torch', torch.__version__, '| detectron2', detectron2.__version__)"

cat <<'EOF'

==> Setup complete.

Next:
  1. Free LLM key (needed only for the agent loop):
       https://aistudio.google.com/apikey
       export LLM_API_KEY=...          # or write it to GraspMAS/api.key
       Several keys are supported: one per line in GraspMAS/api.key, or
       comma-separated in LLM_API_KEY. Quota is per project, so keys from
       separate accounts multiply the rate and the daily budget.
       conda run -n graspmas python -m agents.llm --probe --test

  2. Start the GraspGen-X server (holds the model; leave it running):
       scripts/run_server.sh --daemon

  3. Check everything:
       scripts/run_tests.sh            # 218 offline tests, no key needed
       conda run -n graspmas python scripts/verify_pipeline.py

  4. Run a query:
       cd GraspMAS && python main_simple.py \
         --query "grasp the mustard bottle by its cap" \
         --image-path ../GraspGenX/assets/sample_data/real_world/00/rgb.png \
         --depth-path ../GraspGenX/assets/sample_data/real_world/00/depth.npy \
         --intrinsics ../GraspGenX/assets/sample_data/real_world/00/meta_data.json
EOF
