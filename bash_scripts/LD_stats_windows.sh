#!/bin/bash
#SBATCH --job-name=ld_stats_array
#SBATCH --array=0-9999           
#SBATCH --output=logs/ld_stats_%A_%a.out
#SBATCH --error=logs/ld_stats_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Base directories
BASE_DIR="/projects/kernlab/akapoor/Demographic_Inference"
SIM_DIR_BASE="${BASE_DIR}/LD_inferences"
SNAKEMAKE_DIR="${BASE_DIR}"  # Root Snakemake directory

# Define the batch size and total tasks
BATCH_SIZE=50
TOTAL_TASKS=500000

# Calculate simulation and window numbers for the current task
for TASK_ID in $(seq $((SLURM_ARRAY_TASK_ID * BATCH_SIZE)) $(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))); do
    SIM_NUMBER=$((TASK_ID / 100))
    WINDOW_NUMBER=$((TASK_ID % 100))
    WORKING_DIR="${SIM_DIR_BASE}/sim_${SIM_NUMBER}/window_${WINDOW_NUMBER}"
    OUTPUT_FILE="${WORKING_DIR}/ld_stats_window.${WINDOW_NUMBER}.pkl"

    echo "Processing ld_stats for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"

    # Ensure the working directory exists
    mkdir -p "$WORKING_DIR"

    # Run Snakemake directly in the working directory for full isolation
    snakemake \
        --cores 1 \
        --verbose \
        --nolock \
        --snakefile "${SNAKEMAKE_DIR}/Snakefile" \
        --directory "$WORKING_DIR" \
        "${OUTPUT_FILE}"

    # Check if the output file was created
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "Snakemake completed successfully for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"
    else
        echo "Snakemake failed for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}" >&2
        exit 1
    fi
done
