#!/bin/bash
#SBATCH --job-name=sim_job_array         # Job name
#SBATCH --array=0-99                    # Adjusted to match your simulation needs
#SBATCH --output=logs/simulation_%A_%a.out      # Standard output log file (%A is job ID, %a is the array index)
#SBATCH --error=logs/simulation_%A_%a.err       # Standard error log file
#SBATCH --time=3:00:00                  # Time limit
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=1G                         # Memory per task
#SBATCH --partition=kern,preempt,kerngpu # Partitions to submit the job to
#SBATCH --account=kernlab                # Account to use
#SBATCH --requeue                        # Requeue on preemption

# Set up the simulation directory and other variables
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Extract the values from the JSON config
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
DADI_ANALYSIS=$(jq -r '.dadi_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_ANALYSIS=$(jq -r '.moments_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_LD_ANALYSIS=$(jq -r '.momentsLD_analysis' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)

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

# Set up the simulation directory
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
echo $SIM_DIRECTORY

# Only measure the time for the full execution of the entire job array
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # Start timing at the beginning of the first job array task
    overall_start_time=$(date +%s)
    echo "Start time recorded: $overall_start_time"
fi

# Get the task ID from SLURM (used as sim_number)
TASK_ID=$SLURM_ARRAY_TASK_ID

# Run the Snakemake rule for the specific simulation number
snakemake \
    --snakefile /projects/kernlab/akapoor/Demographic_Inference/Snakefile \
    --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
    --rerun-incomplete \
    "simulated_parameters_and_inferences/simulation_results/sampled_params_${TASK_ID}.pkl" \
    "simulated_parameters_and_inferences/simulation_results/sampled_params_metadata_${TASK_ID}.txt" \
    "simulated_parameters_and_inferences/simulation_results/SFS_sim_${TASK_ID}.pkl" \
    "simulated_parameters_and_inferences/simulation_results/ts_sim_${TASK_ID}.trees"

# Calculate elapsed time only for the last task in the array
if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    # End timing at the end of the last job array task
    overall_end_time=$(date +%s)
    echo "End time recorded: $overall_end_time"
    
    # Ensure both start and end times are set before doing the calculation
    if [[ -n "$overall_start_time" && -n "$overall_end_time" ]]; then
        # Calculate and print the overall elapsed time
        overall_elapsed_time=$((overall_end_time - overall_start_time))
        echo "Total time taken for the entire job array: $overall_elapsed_time seconds"
    else
        echo "Error: Start time or end time not set properly."
    fi
fi