#!/bin/bash
#SBATCH --job-name=aggregate_features
#SBATCH --output=logs/aggregate_features_%j.out
#SBATCH --error=logs/aggregate_features_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab

# Load necessary modules or set up the environment
export PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference

# Define paths
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

# Expand file wildcards
SOFTWARE_INFERENCES=("/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/"*.pkl)
MOMENTSLD_INFERENCES=("/projects/kernlab/akapoor/Demographic_Inference/final_LD_inferences/"*.pkl)

# Debugging: Print expanded file lists
echo "Expanded Software Inferences:"
for file in "${SOFTWARE_INFERENCES[@]}"; do
    echo "$file"
done

echo "Expanded MomentsLD Inferences:"
for file in "${MOMENTSLD_INFERENCES[@]}"; do
    echo "$file"
done

# Check for empty arrays and exit with an error if any are empty
if [ ${#SOFTWARE_INFERENCES[@]} -eq 0 ]; then
    echo "Error: No software inferences found!"
    exit 1
fi

if [ ${#MOMENTSLD_INFERENCES[@]} -eq 0 ]; then
    echo "Error: No MomentsLD inferences found!"
    exit 1
fi

# Run the aggregation script
python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/aggregate_all_features.py \
    "${EXPERIMENT_CONFIG_FILE}" \
    "${SIM_DIRECTORY}" \
    "${SOFTWARE_INFERENCES[@]}" "${MOMENTSLD_INFERENCES[@]}"

if [ $? -eq 0 ]; then
    echo "Feature aggregation completed successfully."
else
    echo "Feature aggregation failed."
    exit 1
fi
