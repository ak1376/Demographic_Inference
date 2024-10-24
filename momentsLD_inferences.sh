#!/bin/bash
#SBATCH --job-name=momentsLD_job_array         # Job name
#SBATCH --array=0-20                    # Adjust this to match your simulation range (e.g., 0-9999 for 10000 simulations)
#SBATCH --output=logs/job_%A_%a.out      # Standard output log file (%A is job ID, %a is the array index)
#SBATCH --error=logs/job_%A_%a.err       # Standard error log file
#SBATCH --time=2:00:00                  # Time limit
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --mem=32G                        # Memory per task
#SBATCH --partition=kern,preempt,kerngpu # Partitions to submit the job to
#SBATCH --account=kernlab                # Account to use
#SBATCH --requeue                        # Requeue on preemption

# Set the simulation directory and experiment configuration path
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

# Function to convert lowercase true/false to capitalized True/False
capitalize_bool() {
    if [ "$1" == "true" ]; then
        echo "True"
    elif [ "$1" == "false" ]; then
        echo "False"
    else
        echo "$1"
    fi
}

# Convert lowercase true/false to True/False
DADI_ANALYSIS=$(capitalize_bool $DADI_ANALYSIS)
MOMENTS_ANALYSIS=$(capitalize_bool $MOMENTS_ANALYSIS)
MOMENTS_LD_ANALYSIS=$(capitalize_bool $MOMENTS_LD_ANALYSIS)

# Get the task ID from SLURM (used as sim_number)
TASK_ID=$SLURM_ARRAY_TASK_ID

# Set up the simulation directory
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"

# Start timing the Snakemake execution
start_time=$(date +%s)

# Execute the MomentsLD analysis after ensuring that the necessary files are ready
snakemake \
    --cores 64 \
    --rerun-incomplete \
    --config sim_directory=$SIM_DIRECTORY sim_number=$TASK_ID \
    --rerun-incomplete \
    "${SIM_DIRECTORY}/simulation_results/momentsLD_inferences_sim_${TASK_ID}.pkl"

# End timing
end_time=$(date +%s)

# Calculate and print the elapsed time
elapsed_time=$(($end_time - $start_time))
echo "Total time taken: $elapsed_time seconds"
