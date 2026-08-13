#!/usr/bin/env bash
# Submit (or dry-run) one run of the DINOv3 LoRA hyperparameter sweep (JIRA TAO-2526).
#
# Sibling of submit_lora_arm.sh, which stays exactly as it was: the Stage-4 arms are a finished
# experiment and their submission path should not move under them. What differs here is where a
# run's configuration comes from. A Stage-4 arm is one of five presets baked into the sbatch; a
# sweep run is an arbitrary point in a hyperparameter space, so its configuration lives in the
# run registry (cluster_scripts/arms.py) and is passed through as TRAIN_OVERRIDES with
# ARM=SWEEP. One definition, one reader -- the lesson arms.py was written for.
#
#   ssh cw-pdx 'bash <repo>/scripts/slurm/submit_lora_sweep.sh sweep_w1_lr1e-4 1'   # DRY_RUN
#   ssh cw-pdx 'bash <repo>/scripts/slurm/submit_lora_sweep.sh sweep_w1_lr1e-4'     # submit
#
# Usage: submit_lora_sweep.sh <RUN_ID> [DRY_RUN] [TIMEOUT_DURATION]
#
#   RUN_ID              a key in the sweep section of the run registry
#   DRY_RUN             1 to validate paths, print the resolved config, and not submit (default 0)
#   TIMEOUT_DURATION    passed to the sbatch (default 3.65h, the requeue-before-the-4h-cap window)
#
# Environment:
#   MAX_JOBS            concurrent *training* jobs permitted (default 1). The sweep plan allows 2
#                       once fairshare >= 0.5 and the W0+W1 checkpoint has approved it.

set -euo pipefail

RUN_ID="${1:?Usage: submit_lora_sweep.sh <RUN_ID> [DRY_RUN] [TIMEOUT_DURATION]}"
DRY_RUN="${2:-0}"
TIMEOUT_DURATION="${3:-3.65h}"

LUSTRE_USER="${LUSTRE_USER:-/lustre/fsw/portfolios/edgeai/users/vpraveen}"
REPO_DIR="${REPO_DIR:-${LUSTRE_USER}/repos/tao-pytorch}"
REGISTRY="${REGISTRY:-${LUSTRE_USER}/cluster_scripts/arms.py}"
JIRA="${JIRA:-TAO-2526}"

if [[ ! -r "${REGISTRY}" ]]; then
    echo "Run registry not found at ${REGISTRY}." >&2
    exit 2
fi

# The registry raises on an unknown key rather than returning empty, so a typo in a run id fails
# here instead of silently submitting the spec's defaults under the wrong name.
if ! RUN_JSON="$(python3 "${REGISTRY}" get "${RUN_ID}" 2>&1)"; then
    echo "Unknown sweep run '${RUN_ID}':" >&2
    echo "${RUN_JSON}" >&2
    exit 2
fi

field() { python3 "${REGISTRY}" get "${RUN_ID}" "$1"; }

KIND="$(field kind)"
if [[ "${KIND}" != "sweep" ]]; then
    echo "Run '${RUN_ID}' is kind='${KIND}', not 'sweep'." >&2
    echo "Stage-4 arms are submitted with submit_lora_arm.sh, which is unchanged." >&2
    exit 2
fi

TRAIN_DIR="$(field train_dir)"
OVERRIDES="$(field overrides)"
STAGE="$(field stage)"
ROLE="$(field role)"
RESULT_DIR="${RESULT_DIR:-${LUSTRE_USER}/outputs/dinov3_lora/${TRAIN_DIR}}"

cd "${REPO_DIR}"

echo "=== submit_lora_sweep.sh ==="
echo "host       : $(hostname)"
echo "date (UTC) : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "run id     : ${RUN_ID}"
echo "stage      : ${STAGE}"
echo "role       : ${ROLE}"
echo "repo       : ${REPO_DIR}"
echo "registry   : ${REGISTRY}"
echo "results    : ${RESULT_DIR}"
echo "timeout    : ${TIMEOUT_DURATION}"
echo "dry run    : ${DRY_RUN}"
echo "jira       : ${JIRA}"
echo "git commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "overrides  : ${OVERRIDES}"
echo

# Fairshare is logged on every submission per the plan's operating rules -- it is the number that
# decides whether we should be queueing at all, so it belongs in the job's own record.
echo "=== fairshare (edgeai) ==="
sshare -al 2>/dev/null | grep -E 'Account|edgeai_tao-ptm.*vpraveen' || echo "(sshare returned nothing)"
echo

# Concurrency guard, counting *training* jobs only. Eval work (convert, RepBench, ImageNet k-NN)
# runs as short srun steps on the interactive partition under different job names; counting those
# would make the guard refuse a training submission because a two-minute k-NN job was in flight.
echo "=== our current queue ==="
squeue -u "${USER}" -o '%.10i %.30j %.9T %.10M %.6D %R' || true
running="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -cE '^(dinov3_sweep_|dinov3_lora_)' || true)"
MAX_JOBS="${MAX_JOBS:-1}"
echo "training jobs in queue: ${running} (max ${MAX_JOBS})"
echo

if [[ "${DRY_RUN}" != "1" && "${running}" -ge "${MAX_JOBS}" ]]; then
    echo "Refusing to submit: ${running} training job(s) already queued/running, limit ${MAX_JOBS}." >&2
    echo "Raise MAX_JOBS to run more concurrently, or wait." >&2
    exit 3
fi

if [[ -f "${RESULT_DIR}/train_done" ]]; then
    echo "Already completed: ${RESULT_DIR}/train_done exists. Nothing to submit."
    exit 0
fi

export ARM=SWEEP
export RUN_ID DRY_RUN TIMEOUT_DURATION REPO_DIR RESULT_DIR
export TRAIN_OVERRIDES="${OVERRIDES}"
export WANDB_NAME="dinov3_${RUN_ID}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== DRY RUN: executing the sbatch body without submitting ==="
    bash scripts/slurm/dinov3_lora_vitb.sbatch
    exit $?
fi

echo "=== submitting ==="
# Run ids already carry the `sweep_` prefix, so the job name is dinov3_sweep_w0_seedC_TAO-2526
# rather than a doubled dinov3_sweep_sweep_w0_seedC. The `dinov3_sweep_` stem is what the
# concurrency guard above and the driver's queue poll both match on.
sbatch --job-name="dinov3_${RUN_ID}_${JIRA}" scripts/slurm/dinov3_lora_vitb.sbatch
