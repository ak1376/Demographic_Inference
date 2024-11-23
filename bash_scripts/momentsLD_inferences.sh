#!/bin/bash
#SBATCH --job-name=moments_ld_array
#SBATCH --array=0-99  # For 100 simulations
#SBATCH --output=logs/moments_ld_%A_%a.out
#SBATCH --error=logs/moments_ld_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

# Base directory
BASE_DIR="/projects/kernlab/akapoor/Demographic_Inference"
COMBINED_LD_DIR="${BASE_DIR}/combined_LD_inferences"
FINAL_LD_DIR="${BASE_DIR}/final_LD_inferences"

# Ensure the necessary directories exist
mkdir -p "${COMBINED_LD_DIR}"
mkdir -p "${FINAL_LD_DIR}"

# Move to base directory
cd $BASE_DIR || { echo "Failed to change to BASE_DIR: $BASE_DIR"; exit 1; }
echo "Current directory: $(pwd)"

# Define batch size and total simulations
BATCH_SIZE=1
TOTAL_SIMS=100

# Calculate batch start and end
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

# Process simulations in the batch
for SIM_NUMBER in $(seq $BATCH_START $BATCH_END); do
    echo "Processing SIM_NUMBER=${SIM_NUMBER}"
    
    # Log paths
    COMBINED_STATS_OUTPUT="${COMBINED_LD_DIR}/sim_${SIM_NUMBER}/combined_LD_stats_sim_${SIM_NUMBER}.pkl"
    MOMENTS_LD_OUTPUT="${FINAL_LD_DIR}/momentsLD_inferences_sim_${SIM_NUMBER}.pkl"
    echo "Expected output for combined_LD_stats: $COMBINED_STATS_OUTPUT"
    echo "Expected output for momentsLD_inferences: $MOMENTS_LD_OUTPUT"

    # Run combined_ld_stats through combined_LD_inferences
    mkdir -p "$(dirname "${COMBINED_STATS_OUTPUT}")"  # Ensure the sim_X directory exists
    snakemake \
        --nolock \
        --snakefile "${BASE_DIR}/Snakefile" \
        --directory "${COMBINED_LD_DIR}" \
        --rerun-incomplete \
        "$COMBINED_STATS_OUTPUT"
        
    # Check if the file exists
    if [ ! -f "$COMBINED_STATS_OUTPUT" ]; then
        echo "Error: combined_LD_stats file missing for SIM_NUMBER=${SIM_NUMBER}"
        continue  # Skip to the next simulation
    fi

    # Run momentsLD inference through final_LD_inferences
    snakemake \
        --nolock \
        --snakefile "${BASE_DIR}/Snakefile" \
        --directory "${FINAL_LD_DIR}" \
        --rerun-incomplete \
        "$MOMENTS_LD_OUTPUT"
    
    # Check if the file exists
    if [ ! -f "$MOMENTS_LD_OUTPUT" ]; then
        echo "Error: momentsLD_inferences file missing for SIM_NUMBER=${SIM_NUMBER}"
        continue  # Skip to the next simulation
    fi

    echo "Completed processing for SIM_NUMBER=${SIM_NUMBER}"
done
