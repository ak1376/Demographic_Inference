#!/bin/bash
#SBATCH --job-name=batched_genome_windows
#SBATCH --array=0-199               # Will be calculated based on TOTAL_TASKS/BATCH_SIZE
#SBATCH --output=logs/genome_windows_%A_%a.out
#SBATCH --error=logs/genome_windows_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

# Define batch parameters
BATCH_SIZE=50
TOTAL_TASKS=10000  # NUM_SIMS * NUM_WINDOWS (e.g., 100 * 100)

# Start timer for the entire job
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    overall_start_time=$(date +%s)
    echo "Overall start time: $overall_start_time"
fi

# Extract config information
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Extract values from JSON config
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)
NUM_WINDOWS=$(jq -r '.num_windows' $EXPERIMENT_CONFIG_FILE)

# Check if NUM_WINDOWS is valid
if [ -z "$NUM_WINDOWS" ] || [ "$NUM_WINDOWS" -eq 0 ]; then
    echo "Error: NUM_WINDOWS is not defined or zero."
    exit 1
fi

# Set up simulation directory
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
echo "Sim directory: $SIM_DIRECTORY"

# Calculate batch indices
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

# Ensure BATCH_END doesn't exceed TOTAL_TASKS
if [ "$BATCH_END" -ge "$TOTAL_TASKS" ]; then
    BATCH_END=$((TOTAL_TASKS - 1))
fi

# Create logs directory
mkdir -p logs

# Process tasks in parallel
for TASK_ID in $(seq $BATCH_START $BATCH_END); do
    # Calculate sim_number and window_number
    SIM_NUMBER=$((TASK_ID / NUM_WINDOWS))
    WINDOW_NUMBER=$((TASK_ID % NUM_WINDOWS))

    echo "Processing sim_number: $SIM_NUMBER, window_number: $WINDOW_NUMBER"

    # Run genome_windows in background
    snakemake \
        --nolock \
        --config sim_directory=$SIM_DIRECTORY \
        --rerun-incomplete \
        "sampled_genome_windows/sim_${SIM_NUMBER}/window_${WINDOW_NUMBER}/samples.txt" \
        "sampled_genome_windows/sim_${SIM_NUMBER}/window_${WINDOW_NUMBER}/flat_map.txt" \
        "sampled_genome_windows/sim_${SIM_NUMBER}/window_${WINDOW_NUMBER}/individual_file_metadata.txt"

    # If this is the last window for a simulation, run combine_metadata
    if [ "$WINDOW_NUMBER" -eq $((NUM_WINDOWS - 1)) ]; then
        echo "Running combine_metadata for simulation ${SIM_NUMBER}"
        snakemake \
            --config sim_directory=$SIM_DIRECTORY \
            --rerun-incomplete \
            "sampled_genome_windows/sim_${SIM_NUMBER}/metadata.txt"
    fi
done

# Wait for all parallel jobs in the batch to complete
wait

# Calculate total time if this is the last batch
TOTAL_BATCHES=$((TOTAL_TASKS / BATCH_SIZE + (TOTAL_TASKS % BATCH_SIZE > 0 ? 1 : 0)))
if [ "$SLURM_ARRAY_TASK_ID" -eq $((TOTAL_BATCHES - 1)) ]; then
    overall_end_time=$(date +%s)
    echo "Overall end time: $overall_end_time"
    overall_elapsed_time=$((overall_end_time - overall_start_time))
    echo "Total time taken: $overall_elapsed_time seconds"
fi