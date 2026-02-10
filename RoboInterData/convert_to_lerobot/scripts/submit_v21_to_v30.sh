#!/bin/bash
#
# Submit SLURM jobs for LeRobot v2.1 to v3.0 conversion
#
# Usage:
#   ./submit_v21_to_v30.sh droid 10       # Convert DROID dataset with 10 shards
#   ./submit_v21_to_v30.sh rh20t 5        # Convert RH20T dataset with 5 shards
#   ./submit_v21_to_v30.sh droid 10 5     # Resume: submit shards 5-9
#   ./submit_v21_to_v30.sh droid 10 0 5   # Submit only shards 0-4
#

set -e

# =============================================================================
# Configuration
# =============================================================================

PARTITION="your_partition"
CPUS_PER_TASK=32
TIME_LIMIT=3600
CONDA_ENV="openpai_new" # should support lerobot

SCRIPT_DIR="./RoboInterData/convert_to_lerobot/scripts"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_v21_to_v30.py"

# Input directories (v2.1 datasets)
DROID_INPUT="RoboInter-Data/Annotation_with_action_lerobotv21/lerobot_droid_anno"
RH20T_INPUT="RoboInter-Data/Annotation_with_action_lerobotv21/lerobot_rh20t_anno"

# Output directories (v3.0 datasets)
DROID_OUTPUT="RoboInter-Data/Annotation_with_action_lerobotv30/lerobot_droid_anno"
RH20T_OUTPUT="RoboInter-Data/Annotation_with_action_lerobotv30/lerobot_rh20t_anno"

# =============================================================================
# Parse arguments
# =============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_type> <num_shards> [start_shard] [end_shard]"
    echo ""
    echo "Arguments:"
    echo "  dataset_type  : 'droid' or 'rh20t'"
    echo "  num_shards    : Total number of shards"
    echo "  start_shard   : Start shard ID (default: 0)"
    echo "  end_shard     : End shard ID, exclusive (default: num_shards)"
    echo ""
    echo "Examples:"
    echo "  $0 droid 10       # Submit all 10 DROID shards (0-9)"
    echo "  $0 rh20t 5        # Submit all 5 RH20T shards (0-4)"
    echo "  $0 droid 10 5     # Resume: submit shards 5-9"
    echo "  $0 droid 10 0 5   # Submit only shards 0-4"
    exit 1
fi

DATASET_TYPE=$1
NUM_SHARDS=$2
START_SHARD=${3:-0}
END_SHARD=${4:-$NUM_SHARDS}

if [ "$DATASET_TYPE" != "droid" ] && [ "$DATASET_TYPE" != "rh20t" ]; then
    echo "Error: dataset_type must be 'droid' or 'rh20t'"
    exit 1
fi

if [ $START_SHARD -ge $END_SHARD ]; then
    echo "Error: start_shard ($START_SHARD) >= end_shard ($END_SHARD)"
    exit 1
fi

# Select input/output directories based on dataset type
if [ "$DATASET_TYPE" == "droid" ]; then
    INPUT_DIR="${DROID_INPUT}"
    OUTPUT_DIR="${DROID_OUTPUT}"
else
    INPUT_DIR="${RH20T_INPUT}"
    OUTPUT_DIR="${RH20T_OUTPUT}"
fi

LOG_DIR="${SCRIPT_DIR}/logs_v30_${DATASET_TYPE}"

# =============================================================================
# Setup
# =============================================================================

mkdir -p ${LOG_DIR}

echo "============================================================"
echo "Submitting ${DATASET_TYPE^^} v2.1 -> v3.0 conversion jobs"
echo "============================================================"
echo "  Dataset:       ${DATASET_TYPE}"
echo "  Input dir:     ${INPUT_DIR}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Total shards:  ${NUM_SHARDS}"
echo "  Submitting:    ${START_SHARD} to $((END_SHARD - 1)) ($((END_SHARD - START_SHARD)) jobs)"
echo "  Partition:     ${PARTITION}"
echo "  CPUs/task:     ${CPUS_PER_TASK}"
echo "  Conda env:     ${CONDA_ENV}"
echo "  Log directory: ${LOG_DIR}"
echo "============================================================"
echo ""

# =============================================================================
# Submit jobs
# =============================================================================

JOB_IDS=""

for SHARD_ID in $(seq ${START_SHARD} $((END_SHARD - 1))); do
    # Submit job using heredoc
    JOB_OUTPUT=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=v30_${DATASET_TYPE}_s${SHARD_ID}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=gpu:0
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/shard_${SHARD_ID}_%j.out
#SBATCH --error=${LOG_DIR}/shard_${SHARD_ID}_%j.err

echo "=============================================="
echo "Converting v2.1 -> v3.0: shard ${SHARD_ID}/${NUM_SHARDS}"
echo "Dataset: ${DATASET_TYPE}"
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo "=============================================="

# Activate conda environment
source ~/.bashrc
conda activate ${CONDA_ENV}

python ${CONVERT_SCRIPT} \\
    --input_dir ${INPUT_DIR} \\
    --output_dir ${OUTPUT_DIR} \\
    --shard_id ${SHARD_ID} \\
    --num_shards ${NUM_SHARDS}

echo ""
echo "Shard ${SHARD_ID} completed at \$(date)"
EOF
    )

    echo "  Submitted shard ${SHARD_ID}/${NUM_SHARDS} -> Job ID: ${JOB_OUTPUT}"
    JOB_IDS="${JOB_IDS} ${JOB_OUTPUT}"
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Successfully submitted $((END_SHARD - START_SHARD)) jobs!"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Cancel all jobs with:"
echo "  scancel${JOB_IDS}"
echo ""
echo "View logs in:"
echo "  ${LOG_DIR}/"
echo "============================================================"
echo ""
echo "After all jobs complete, run merge (if using sharding):"
echo "  python ${SCRIPT_DIR}/merge_v30_shards.py \\"
echo "      --shard_pattern '${OUTPUT_DIR}_shard*' \\"
echo "      --output_dir ${OUTPUT_DIR}"
echo ""

# Save job info
JOB_INFO_FILE="${LOG_DIR}/submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"
echo "# Submitted at $(date)" > ${JOB_INFO_FILE}
echo "# Dataset: ${DATASET_TYPE}" >> ${JOB_INFO_FILE}
echo "# Conversion: v2.1 -> v3.0" >> ${JOB_INFO_FILE}
echo "# Shards: ${START_SHARD}-$((END_SHARD-1)) of ${NUM_SHARDS}" >> ${JOB_INFO_FILE}
echo "# Job IDs:${JOB_IDS}" >> ${JOB_INFO_FILE}
echo ""
echo "Job info saved to: ${JOB_INFO_FILE}"
