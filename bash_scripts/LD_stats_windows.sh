#!/bin/bash
#SBATCH --job-name=ld_stats_array
#SBATCH --array=0-4999           
#SBATCH --output=logs/ld_stats_%A_%a.out
#SBATCH --error=logs/ld_stats_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

BASE_DIR="/projects/kernlab/akapoor/Demographic_Inference"
SIM_DIR_BASE="${BASE_DIR}/LD_inferences"
SNAKEMAKE_DIR="${BASE_DIR}"  # Root Snakemake directory

# Define the batch size
BATCH_SIZE=100
TOTAL_TASKS=500000

for TASK_ID in $(seq $((SLURM_ARRAY_TASK_ID * BATCH_SIZE)) $(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))); do
    SIM_NUMBER=$((TASK_ID / 100))
    WINDOW_NUMBER=$((TASK_ID % 100))
    SIM_DIR="${SIM_DIR_BASE}/sim_${SIM_NUMBER}"
    WORKING_DIR="${SIM_DIR_BASE}/sim_${SIM_NUMBER}/window_${WINDOW_NUMBER}"
    mkdir -p "$WORKING_DIR"

    echo "Processing ld_stats for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"

    # Ensure the simulation directory exists
    mkdir -p "$SIM_DIR"
    if [ ! -d "$SIM_DIR" ]; then
        echo "Failed to create directory: $SIM_DIR"
        exit 1
    fi

    # Navigate to the simulation directory
    pushd "$SIM_DIR" || { echo "Failed to change directory to $SIM_DIR"; exit 1; }

    # Debug information
    echo "Current SIM_DIR: $(pwd)"
    echo "Target Output: ld_stats_window.${WINDOW_NUMBER}.pkl"

    # Run Snakemake from the simulation directory
    snakemake -p --verbose --rerun-incomplete \
        --nolock \
        --snakefile "${SNAKEMAKE_DIR}/Snakefile" \
        --directory "$WORKING_DIR" \
        "${WORKING_DIR}/ld_stats_window.${WINDOW_NUMBER}.pkl"

    # Check if Snakemake succeeded
    if [ $? -eq 0 ]; then
        echo "Snakemake completed successfully for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"
    else
        echo "Snakemake failed for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"

        # Cleanup .snakemake directory for the task
        if [ -d "${WORKING_DIR}/.snakemake" ]; then
            echo "Cleaning up .snakemake directory in ${WORKING_DIR}"
            rm -rf "${WORKING_DIR}/.snakemake"
        fi

        popd  # Ensure we still return to BASE_DIR even if an error occurs
        exit 1
    fi

    # Return to the base directory
    popd || { echo "Failed to return to $BASE_DIR"; exit 1; }
done
