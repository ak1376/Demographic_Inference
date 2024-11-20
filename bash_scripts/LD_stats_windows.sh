#!/bin/bash
#SBATCH --job-name=ld_stats_array
#SBATCH --array=0-9999           
#SBATCH --output=logs/ld_stats_%A_%a.out
#SBATCH --error=logs/ld_stats_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

BASE_DIR="/gpfs/projects/kernlab/akapoor/Demographic_Inference"
SIM_DIR_BASE="${BASE_DIR}/LD_inferences"

# Define the batch size
BATCH_SIZE=1
TOTAL_TASKS=10000

# Process each task in the batch
for TASK_ID in $(seq $((SLURM_ARRAY_TASK_ID * BATCH_SIZE)) $(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))); do
    SIM_NUMBER=$((TASK_ID / 100))
    WINDOW_NUMBER=$((TASK_ID % 100))
    SIM_DIR="${SIM_DIR_BASE}/sim_${SIM_NUMBER}"

    echo "Processing ld_stats for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"

    # Ensure the simulation directory exists
    mkdir -p "$SIM_DIR"

    # Navigate into the simulation directory
    cd "$SIM_DIR" || { echo "Failed to change directory to $SIM_DIR"; exit 1; }
    echo "Directory changed successfully to: $SIM_DIR"

    # Run Snakemake with absolute paths
    snakemake -p --verbose --nolock --rerun-incomplete \
        --snakefile "${BASE_DIR}/Snakefile" \
        --directory "${BASE_DIR}" \
        "LD_inferences/sim_${SIM_NUMBER}/ld_stats_window.${WINDOW_NUMBER}.pkl"

    # Check Snakemake status and log appropriately
    if [ $? -eq 0 ]; then
        echo "Snakemake completed successfully for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"
    else
        echo "Snakemake failed for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"
        exit 1
    fi

    # Return to the base directory after each iteration
    cd "$BASE_DIR" || { echo "Failed to return to base directory: $BASE_DIR"; exit 1; }
    echo "Returned to base directory: $BASE_DIR"
done
