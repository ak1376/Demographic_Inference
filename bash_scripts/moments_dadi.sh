#!/bin/bash
#SBATCH --job-name=feature_processing
#SBATCH --array=0-11  # Adjust based on (NUM_SIMS_PRETRAIN * NUM_REPLICATES * NUM_ANALYSES)
#SBATCH --output=logs/moments_dadi_%A_%a.out
#SBATCH --error=logs/moments_dadi_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

# Define variables from config file
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Extract config values
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_REPLICATES=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)

# Supported analyses
ANALYSES=("dadi" "moments")  # Add or remove analyses as needed
NUM_ANALYSES=${#ANALYSES[@]}  # Number of analyses (2 in this case)

# Calculate total number of tasks
TOTAL_TASKS=$((NUM_SIMS_PRETRAIN * NUM_REPLICATES * NUM_ANALYSES))

# Get the specific analysis, simulation, and replicate for this task
ANALYSIS_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_ANALYSES))  # Which analysis (0=dadi, 1=moments)
SIM_NUMBER=$((SLURM_ARRAY_TASK_ID / (NUM_REPLICATES * NUM_ANALYSES)))  # Simulation number
REPLICATE_NUMBER=$(((SLURM_ARRAY_TASK_ID / NUM_ANALYSES) % NUM_REPLICATES))  # Replicate number
ANALYSIS=${ANALYSES[$ANALYSIS_INDEX]}  # dadi or moments

echo "Running analysis: $ANALYSIS for SIM_NUMBER=$SIM_NUMBER, REPLICATE_NUMBER=$REPLICATE_NUMBER"

# Define the replicate directory within the analysis directory
REPLICATE_DIR="/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/sim_${SIM_NUMBER}/${ANALYSIS}/replicate_${REPLICATE_NUMBER}"
mkdir -p "$REPLICATE_DIR"

# Run Snakemake in the replicate directory
snakemake -p --rerun-incomplete --snakefile "/projects/kernlab/akapoor/Demographic_Inference/Snakefile" \
    --directory "$REPLICATE_DIR" \
    "${REPLICATE_DIR}/replicate_${REPLICATE_NUMBER}.pkl"

if [ $? -eq 0 ]; then
    echo "Snakemake completed successfully for $ANALYSIS, sim_${SIM_NUMBER}, replicate_${REPLICATE_NUMBER}"
else
    echo "Snakemake failed for $ANALYSIS, sim_${SIM_NUMBER}, replicate_${REPLICATE_NUMBER}"
    exit 1
fi

# Run aggregation only for the last replicate of each simulation
if [[ "$REPLICATE_NUMBER" -eq "$((NUM_REPLICATES - 1))" && "$ANALYSIS_INDEX" -eq "0" ]]; then
    echo "Starting aggregation for SIM_NUMBER=$SIM_NUMBER..."

    snakemake --nolock -p --rerun-incomplete \
        --snakefile "/projects/kernlab/akapoor/Demographic_Inference/Snakefile" \
        --directory "/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/" \
        "/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/software_inferences_sim_${SIM_NUMBER}.pkl"

    if [ $? -eq 0 ]; then
        echo "Aggregation completed successfully for SIM_NUMBER=$SIM_NUMBER"
    else
        echo "Aggregation failed for SIM_NUMBER=$SIM_NUMBER"
        exit 1
    fi
fi