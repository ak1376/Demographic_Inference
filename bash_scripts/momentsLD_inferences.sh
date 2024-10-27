#!/bin/bash
#SBATCH --job-name=momentsLD_job_array        # Job name
#SBATCH --array=0-500                        # Adjust the job array size based on total number of jobs
#SBATCH --output=logs/job_%A_%a.out           # Standard output log file
#SBATCH --error=logs/job_%A_%a.err            # Standard error log file
#SBATCH --time=6:00:00                        # Time limit
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mem=32G                             # Memory per task
#SBATCH --partition=kern,preempt,kerngpu      # Partitions to submit the job to
#SBATCH --account=kernlab                     # Account to use
#SBATCH --requeue                             # Requeue on preemption

# Only measure the time for the full execution of the entire job array
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # Start timing at the beginning of the first job array task
    overall_start_time=$(date +%s)
    echo "Overall start time: $overall_start_time"
fi

# Set up the simulation directory and other variables from the experiment config file
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Extract the values from the JSON config using jq
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
DADI_ANALYSIS=$(jq -r '.dadi_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_ANALYSIS=$(jq -r '.moments_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_LD_ANALYSIS=$(jq -r '.momentsLD_analysis' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_WINDOWS=$(jq -r '.num_windows' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)

# Function to convert lowercase true/false to True/False for Python config
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

# Calculate the total number of tasks (simulations * windows)
TOTAL_TASKS=$((NUM_SIMS_PRETRAIN * NUM_WINDOWS))

# Set up the simulation directory path using variables from the config
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
echo "Sim directory: $SIM_DIRECTORY"

# Calculate sim_number and window_number based on SLURM_ARRAY_TASK_ID
TASK_ID=$SLURM_ARRAY_TASK_ID
SIM_NUMBER=$((TASK_ID / NUM_WINDOWS))  # Integer division to get the sim_number
WINDOW_NUMBER=$((TASK_ID % NUM_WINDOWS))  # Modulus to get the window_number

echo "Processing sim_number: $SIM_NUMBER, window_number: $WINDOW_NUMBER"

# Run the Snakemake rule for calculating LD stats for the current simulation and window
snakemake \
    --nolock \
    --config sim_directory=$SIM_DIRECTORY sim_number=$SIM_NUMBER window_number=$WINDOW_NUMBER \
    --rerun-incomplete \
    "${SIM_DIRECTORY}/sampled_genome_windows/sim_${SIM_NUMBER}/ld_stats_window.${WINDOW_NUMBER}.pkl"

# Check if all windows have been processed for the current simulation
if [ "$WINDOW_NUMBER" -eq $((NUM_WINDOWS - 1)) ]; then
    echo "All windows processed for sim_number: $SIM_NUMBER"

    # Gather LD stats after all windows for this simulation are processed
    snakemake \
        --nolock \
        --config sim_directory=$SIM_DIRECTORY sim_number=$SIM_NUMBER \
        --rerun-incomplete \
        "${SIM_DIRECTORY}/sampled_genome_windows/sim_${SIM_NUMBER}/combined_LD_stats_sim_${SIM_NUMBER}.pkl"

    # Check for errors in the snakemake command
    if [ $? -ne 0 ]; then
        echo "Error: Failed to gather combined LD stats for sim_number ${SIM_NUMBER}"
        exit 1
    fi

    # Run the MomentsLD analysis once LD stats have been gathered
    snakemake \
        --nolock \
        --config sim_directory=$SIM_DIRECTORY sim_number=$SIM_NUMBER \
        --rerun-incomplete \
        "${SIM_DIRECTORY}/simulation_results/momentsLD_inferences_sim_${SIM_NUMBER}.pkl"

    if [ $? -ne 0 ]; then
        echo "Error: MomentsLD analysis failed for sim_number ${SIM_NUMBER}"
        exit 1
    fi
fi

# If this is the last task in the array, calculate the total time for all jobs
if [ "$SLURM_ARRAY_TASK_ID" -eq $((TOTAL_TASKS - 1)) ]; then
    # End timing at the end of the last job array task
    overall_end_time=$(date +%s)
    echo "Overall end time: $overall_end_time"

    # Calculate and print the overall elapsed time
    if [ -n "$overall_start_time" ] && [ -n "$overall_end_time" ]; then
        overall_elapsed_time=$((overall_end_time - overall_start_time))
        echo "Total time taken for the entire job array: $overall_elapsed_time seconds"
    else
        echo "Timing error: overall_start_time or overall_end_time is not set."
    fi
fi
