#!/bin/bash
#SBATCH --job-name=remaining_rules      # Job name for the remaining tasks
#SBATCH --output=logs/remaining_rules_%j.out  # Standard output log file
#SBATCH --error=logs/remaining_rules_%j.err   # Standard error log file
#SBATCH --time=01:00:00                 # Time limit (adjust as needed)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=16G                       # Memory per task (adjust as needed)
#SBATCH --partition=kern,kerngpu # Partitions to submit the job to
#SBATCH --account=kernlab               # Account to use

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

# Set up the simulation directory
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
echo $SIM_DIRECTORY

# Run Snakemake to complete the remaining rules
snakemake \
    --nolock \
    "${SIM_DIRECTORY}/preprocessing_results_obj.pkl" \
    "${SIM_DIRECTORY}/training_features.npy" \
    "${SIM_DIRECTORY}/training_targets.npy" \
    "${SIM_DIRECTORY}/validation_features.npy" \
    "${SIM_DIRECTORY}/validation_targets.npy" \
    # "${SIM_DIRECTORY}/postprocessing_results.pkl" \
    # "${SIM_DIRECTORY}/features_and_targets.pkl" \
    # "${MODEL_DIRECTORY}/linear_regression_model.pkl" \
    # "${MODEL_DIRECTORY}/snn_results.pkl" \
    # "${MODEL_DIRECTORY}/snn_model.pth"
