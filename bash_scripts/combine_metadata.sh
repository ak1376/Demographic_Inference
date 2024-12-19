#!/bin/bash
#SBATCH --job-name=combine_metadata
#SBATCH --output=logs/combine_metadata_%j.out
#SBATCH --error=logs/combine_metadata_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab

# Load necessary modules or activate environment if required
# Example:
# module load python

EXPERIMENT_CONFIG_FILE='/projects/kernlab/akapoor/Demographic_Inference/experiment_config.json'

# Extract values from JSON config
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)

SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
BASE_DIR="/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows"

echo "Sim directory: $SIM_DIRECTORY"
echo "Base directory: $BASE_DIR"

# Loop over all simulation directories
for SIM_DIR in "${BASE_DIR}/sim_"*; do
    # Check if the simulation directory exists
    if [ -d "$SIM_DIR" ]; then
        echo "Processing simulation directory: $SIM_DIR"
        pushd "$SIM_DIR" || { echo "Failed to change directory to $SIM_DIR"; exit 1; }

        # Run Snakemake to generate metadata.txt for this simulation
        snakemake \
            --nolock \
            --snakefile /projects/kernlab/akapoor/Demographic_Inference/Snakefile \
            --directory "$SIM_DIR" \
            --shadow-prefix "$SIM_DIR/.snakemake" \
            --rerun-incomplete \
            "${SIM_DIR}/metadata.txt"

        if [ $? -eq 0 ]; then
            echo "Successfully generated metadata.txt for $SIM_DIR"
        else
            echo "Failed to generate metadata.txt for $SIM_DIR"
            popd
            exit 1
        fi

        popd || { echo "Failed to return to previous directory"; exit 1; }
    else
        echo "Simulation directory $SIM_DIR does not exist. Skipping."
    fi
done

echo "All simulations processed successfully!"
