#!/usr/bin/env bash
# GraspMAS model weights.
#
# Only VLPart's swinbase_part_0a0000.pth is actually loaded (see image_patch.py).
# The four other VLPart checkpoints upstream downloaded (~2.2 GB) were never
# referenced, and the RAGT-3-3 gdown was the planar 2D grasp model that
# GraspGen-X now replaces — both removed.
#
# GroundingDINO, SAM, DPT and (lazily) CLIP are fetched automatically by
# transformers on first use.
set -euo pipefail

mkdir -p weights
cd weights

if [[ ! -f swinbase_part_0a0000.pth ]]; then
  wget https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth
else
  echo "swinbase_part_0a0000.pth already present"
fi
