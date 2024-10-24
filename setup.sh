#!/bin/bash

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

# Construct the experiment directory and name using the converted values
EXPERIMENT_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}"
EXPERIMENT_NAME="sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
SIM_DIRECTORY="${EXPERIMENT_DIRECTORY}/sims/${EXPERIMENT_NAME}"

# Print the paths for debugging
echo "EXPERIMENT_DIRECTORY: $EXPERIMENT_DIRECTORY"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SIM_DIRECTORY: $SIM_DIRECTORY"

# Ensure that the working directory is correct
cd /projects/kernlab/akapoor/Demographic_Inference  # Adjust this to your Snakemake working directory

# Run Snakemake with the specified outputs
snakemake \
    ${SIM_DIRECTORY}/config.json \
    ${SIM_DIRECTORY}/inference_config_file.json \
    ${SIM_DIRECTORY}/color_shades.pkl \
    ${SIM_DIRECTORY}/main_colors.pkl \
    --cores 1  # Adjust cores as needed

# Check if Snakemake ran successfully
if [ $? -ne 0 ]; then
    echo "Snakemake execution failed"
    exit 1
else
    echo "Snakemake completed successfully"
fi
