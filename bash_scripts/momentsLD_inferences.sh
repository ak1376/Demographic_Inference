#!/bin/bash
#SBATCH --job-name=batched_momentsLD_job_array
#SBATCH --array=0-7
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

# Store the original directory
current_dir=$(pwd)
echo "Current Directory: ${current_dir}"

# At the start of your script, after defining current_dir
# Create a local .snakemake directory for locks
mkdir -p "${current_dir}/.snakemake"

# Define the batch size
BATCH_SIZE=50
TOTAL_TASKS=400

# Start the timer for the entire job
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    overall_start_time=$(date +%s)
    echo "Overall start time: $overall_start_time"
fi

# Extract config information
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Extract the values from the JSON config using jq
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
DADI_ANALYSIS=$(jq -r '.dadi_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_ANALYSIS=$(jq -r '.moments_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_LD_ANALYSIS=$(jq -r '.momentsLD_analysis' $EXPERIMENT_CONFIG_FILE)
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

# Function to convert lowercase true/false to True/False
capitalize_bool() {
    if [ "$1" == "true" ]; then
        echo "True"
    elif [ "$1" == "false" ]; then
        echo "False"
    else
        echo "$1"
    fi
}

DADI_ANALYSIS=$(capitalize_bool $DADI_ANALYSIS)
MOMENTS_ANALYSIS=$(capitalize_bool $MOMENTS_ANALYSIS)
MOMENTS_LD_ANALYSIS=$(capitalize_bool $MOMENTS_LD_ANALYSIS)

# Set up the simulation directory path using variables from the config
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
echo "Sim directory: $SIM_DIRECTORY"

# Calculate the start and end indices for the current batch
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

# Ensure BATCH_END does not exceed TOTAL_TASKS
if [ "$BATCH_END" -ge "$TOTAL_TASKS" ]; then
    BATCH_END=$((TOTAL_TASKS - 1))
fi

mkdir -p logs

# Run tasks in parallel
for TASK_ID in $(seq $BATCH_START $BATCH_END); do
    SIM_NUMBER=$((TASK_ID / NUM_WINDOWS))
    WINDOW_NUMBER=$((TASK_ID % NUM_WINDOWS))

    echo "Processing sim_number: $SIM_NUMBER, window_number: $WINDOW_NUMBER"

    # Create directory and cd into it
    mkdir -p sampled_genome_windows/sim_${SIM_NUMBER}/window_${WINDOW_NUMBER}/
    cd sampled_genome_windows/sim_${SIM_NUMBER}/window_${WINDOW_NUMBER}/ || { echo "Failed to change directory"; exit 1; }

    # Run Snakemake for window-specific files only
    snakemake \
        --snakefile /gpfs/projects/kernlab/akapoor/Demographic_Inference/Snakefile \
        --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
        --config sim_directory=$SIM_DIRECTORY sim_number=$SIM_NUMBER window_number=$WINDOW_NUMBER \
        --rerun-incomplete \
        --nolock \
        "LD_inferences/sim_${SIM_NUMBER}/ld_stats_window.${WINDOW_NUMBER}.pkl" 

    # Return to original directory
    cd "$current_dir" || { echo "Failed to return to original directory"; exit 1; }

    # Only create metadata file after all windows are done
    if [ "$WINDOW_NUMBER" -eq $((NUM_WINDOWS - 1)) ]; then
        echo "All windows processed for sim_number: $SIM_NUMBER"
        
        # Create metadata file
        snakemake \
            --snakefile /gpfs/projects/kernlab/akapoor/Demographic_Inference/Snakefile \
            --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
            --rerun-incomplete \
            --nolock \
            "sampled_genome_windows/sim_${SIM_NUMBER}/metadata.txt"
            
        # Run other tasks that depend on all windows being complete
        snakemake \
            --snakefile /gpfs/projects/kernlab/akapoor/Demographic_Inference/Snakefile \
            --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
            --rerun-incomplete \
            "combined_LD_inferences/sim_${SIM_NUMBER}/combined_LD_stats_sim_${SIM_NUMBER}.pkl" \
            "final_LD_inferences/momentsLD_inferences_sim_${SIM_NUMBER}.pkl"
    fi
done