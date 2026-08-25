#!/usr/bin/env bash
# Run the test suite across both conda environments.
#
#   scripts/run_tests.sh                 # offline suite (default) — no server,
#                                        #   no network, no LLM requests
#   scripts/run_tests.sh --integration   # also the tests needing a live server
#
# The default run is deliberately free: the LLM tier has a daily request budget
# that belongs to real evaluation, not to CI. It takes ~5 minutes, most of it
# ray-casting synthetic scenes and rebuilding scene registries — the price of
# testing the decluttering loop against ground truth rather than against mocks.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./env.sh
source "${REPO_ROOT}/scripts/env.sh"

MARKERS='not integration and not llm and not slow'
[[ "${1:-}" == "--integration" ]] && MARKERS='not llm and not slow'

fail=0

echo "==================== GraspMAS (env: graspmas) ===================="
conda run -n graspmas --no-capture-output --cwd "${REPO_ROOT}/GraspMAS" \
  python -m pytest -m "${MARKERS}" -p no:launch_testing "${@:2}"
[[ $? -ne 0 ]] && fail=1

echo
echo "==================== GraspGen-X (env: graspgenx) ================="
conda run -n graspgenx --no-capture-output --cwd "${REPO_ROOT}/GraspGenX" \
  python -m pytest tests/test_cpu_patch.py -q -m "${MARKERS}" -p no:launch_testing
[[ $? -ne 0 ]] && fail=1

echo
if [[ $fail -eq 0 ]]; then
  echo "ALL TESTS PASSED"
else
  echo "SOME TESTS FAILED"
fi
exit $fail
