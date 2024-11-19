#!/bin/bash
#SBATCH --job-name=feature_processing
#SBATCH --array=0-4999  # Adjust based on num_sims_pretrain
#SBATCH --output=logs/moments_dadi_%A_%a.out
#SBATCH --error=logs/moments_dadi_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

# Define the batch size
BATCH_SIZE=1 # let's do 1 for now just to see each individual operation 
TOTAL_TASKS=5000 # 10 simulations, 5 replicates per simulation 

# Store the original directory
current_dir=$(pwd)
echo "Current Directory: ${current_dir}"

# Set PYTHONPATH
export PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference

# Extract config information
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Extract values from JSON config
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
DADI_ANALYSIS=$(jq -r '.dadi_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_ANALYSIS=$(jq -r '.moments_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_LD_ANALYSIS=$(jq -r '.momentsLD_analysis' $EXPERIMENT_CONFIG_FILE)
NUM_REPLICATES=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)           # Added this
# TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)  # Added this

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

# Calculate batch indices
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

if [ "$BATCH_END" -ge "$TOTAL_TASKS" ]; then
    BATCH_END=$((TOTAL_TASKS - 1))
fi

mkdir -p logs

current_dir=$(pwd)
echo "Current Directory: ${current_dir}"

# Process tasks in parallel
for TASK_ID in $(seq $BATCH_START $BATCH_END); do

    # Calculate simulation number (0-9) and replicate number (0-2)
    SIM_NUMBER=$((TASK_ID / NUM_REPLICATES))
    REPLICATE_NUMBER=$((TASK_ID % NUM_REPLICATES))

    echo "Processing simulation number: $SIM_NUMBER and replicate number: $REPLICATE_NUMBER"

    # Create and navigate to simulation-specific directory
    # Create base directory
    BASE_DIR="moments_dadi_features"
    mkdir -p ${BASE_DIR}/sim_${SIM_NUMBER}  # Create the base directory first

    cd ${BASE_DIR}/sim_${SIM_NUMBER} || { echo "Failed to change directory"; exit 1; }

    # Create directories for both dadi and moments
    mkdir -p ${BASE_DIR}/sim_${SIM_NUMBER}/dadi
    mkdir -p ${BASE_DIR}/sim_${SIM_NUMBER}/moments

    # Define output files
    # DADI_OUT="${BASE_DIR}/sim_${SIM_NUMBER}/dadi/replicate_${REPLICATE_NUMBER}.pkl"
    # MOMENTS_OUT="${BASE_DIR}/sim_${SIM_NUMBER}/moments/replicate_${REPLICATE_NUMBER}.pkl"

    # Part 1: Run feature extraction for this simulation (replicates for both moments and dadi)
    snakemake \
        --nolock \
        --snakefile /projects/kernlab/akapoor/Demographic_Inference/Snakefile \
        --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
        --rerun-incomplete \
        "moments_dadi_features/sim_${SIM_NUMBER}/dadi/replicate_${REPLICATE_NUMBER}.pkl" \
        "moments_dadi_features/sim_${SIM_NUMBER}/moments/replicate_${REPLICATE_NUMBER}.pkl"

done

# Return to original directory
cd "$current_dir" || { echo "Failed to return to original directory"; exit 1; }

# Part 2: Aggregate the results to get the top k 
# Only run aggregation if this is the last replicate for the current simulation
LAST_REPLICATE=$((NUM_REPLICATES - 1))
if [ "$REPLICATE_NUMBER" -eq "$LAST_REPLICATE" ]; then
    echo "Processing last replicate for simulation ${SIM_NUMBER}, starting aggregation..."
    
    snakemake \
        --nolock \
        --snakefile /projects/kernlab/akapoor/Demographic_Inference/Snakefile \
        --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
        --rerun-incomplete \
        "moments_dadi_features/software_inferences_sim_${SIM_NUMBER}.pkl"
fi