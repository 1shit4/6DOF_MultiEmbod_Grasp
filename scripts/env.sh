# Shared environment for the 6dof_GraspMAS project.
#   source scripts/env.sh
#
# Setting the two asset variables is what stops graspgenx/_setup_dependencies.py
# from git-cloning ~1.7 GB into GraspGenX/ext/ on first import.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# This machine has a ROS 2 Humble workspace sourced, which puts
# /opt/ros/humble/.../python3.10/site-packages on PYTHONPATH. Conda does not
# override PYTHONPATH, so those packages leak into both envs — ROS ships its own
# numpy/cv2 and a `launch_testing` pytest plugin that is incompatible with modern
# pytest. Clear it for our processes only; the user's ROS shell is unaffected.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export GRASPMAS_SAVED_PYTHONPATH="${PYTHONPATH}"
  unset PYTHONPATH
fi

export GRASPGENX_CHECKPOINT_DIR="${REPO_ROOT}/assets/graspgenx_checkpoints"
export GRASPGENX_GRIPPER_CFG_DIR="${REPO_ROOT}/assets/gripper_descriptions"
export GRASPGENX_DEVICE="${GRASPGENX_DEVICE:-cpu}"

# GraspGen-X assets shipped inside the repo (proc_grippers/, sample_data/).
export GRASPGENX_ASSETS_DIR="${REPO_ROOT}/GraspGenX/assets"

# ZMQ bridge
export GRASPGEN_SERVER_HOST="${GRASPGEN_SERVER_HOST:-localhost}"
export GRASPGEN_SERVER_PORT="${GRASPGEN_SERVER_PORT:-5556}"

# Keep BLAS from oversubscribing the 8 cores; torch owns the thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# Headless rendering for matplotlib / pyrender.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Free-tier LLM keys. Several are supported and recommended: quota is per
# Google Cloud project, so keys from separate accounts multiply both the
# per-minute and the per-day limit. Any of these work, and all are merged:
#   export LLM_API_KEY="key1,key2,key3"
#   export LLM_API_KEY_1=... LLM_API_KEY_2=...
#   GraspMAS/api.key, one key per line
# export LLM_API_KEY=...
