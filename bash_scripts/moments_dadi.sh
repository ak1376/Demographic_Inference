#!/bin/bash
#SBATCH --job-name=feature_processing
#SBATCH --array=0-7  # Adjust based on num_sims_pretrain
#SBATCH --output=logs/feature_processing_%A_%a.out
#SBATCH --error=logs/feature_processing_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

# Store the original directory
current_dir=$(pwd)
echo "Current Directory: ${current_dir}"

# Extract config information
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Extract values from JSON config
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
NUM_WINDOWS=$(jq -r '.num_windows' $EXPERIMENT_CONFIG_FILE)
DADI_ANALYSIS=$(jq -r '.dadi_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_ANALYSIS=$(jq -r '.moments_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_LD_ANALYSIS=$(jq -r '.momentsLD_analysis' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)           # Added this
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)  # Added this

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

# Set up the full simulation directory path - added num_replicates and top_values
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
echo "Sim directory: $SIM_DIRECTORY"

# Process single simulation
SIM_NUMBER=$SLURM_ARRAY_TASK_ID

echo "Processing simulation number: $SIM_NUMBER"

# Create and navigate to simulation-specific directory
mkdir -p moments_dadi_features
cd moments_dadi_features || { echo "Failed to change directory"; exit 1; }

# Set PYTHONPATH
export PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference

# Run feature extraction for this simulation
snakemake \
    --snakefile /projects/kernlab/akapoor/Demographic_Inference/Snakefile \
    --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
    --rerun-incomplete \
    "moments_dadi_features/software_inferences_sim_${SIM_NUMBER}.pkl"

# Return to original directory
cd "$current_dir" || { echo "Failed to return to original directory"; exit 1; }

# Print debug information
echo "Current SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "NUM_SIMS_PRETRAIN: $NUM_SIMS_PRETRAIN"
echo "NUM_SIMS_PRETRAIN - 1: $((NUM_SIMS_PRETRAIN - 1))"

if [ "$SLURM_ARRAY_TASK_ID" -eq $((NUM_SIMS_PRETRAIN - 1)) ]; then
    echo "Last simulation completed. Starting feature aggregation..."
    
    # Create the output directory if it doesn't exist
    mkdir -p "${SIM_DIRECTORY}"
    
    # First collect all inference files
    SOFTWARE_INFERENCES=(moments_dadi_features/software_inferences_sim_*.pkl)
    MOMENTSLD_INFERENCES=(final_LD_inferences/momentsLD_inferences_sim_*.pkl)
    
    # Run aggregation directly with Python
    PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference \
    python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/aggregate_all_features.py \
        "${EXPERIMENT_CONFIG_FILE}" \
        "${SIM_DIRECTORY}" \
        ${SOFTWARE_INFERENCES[@]} ${MOMENTSLD_INFERENCES[@]}
    
    echo "Feature aggregation completed"
fi