#!/bin/bash
#SBATCH --job-name=feature_processing
#SBATCH --array=0-9999  # Adjust based on TOTAL_TASKS / BATCH_SIZE
#SBATCH --output=logs/moments_dadi_batch_%A_%a.out
#SBATCH --error=logs/moments_dadi_batch_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Load base variables
BASE_DIR="/projects/kernlab/akapoor/Demographic_Inference"
FEATURES_DIR="${BASE_DIR}/moments_dadi_features"
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'

# Ensure necessary directories exist
mkdir -p "${FEATURES_DIR}"

# Move to base directory
cd $BASE_DIR || { echo "Failed to change to BASE_DIR: $BASE_DIR"; exit 1; }
echo "Current directory: $(pwd)"

# Load experiment configuration
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_REPLICATES=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)

# Supported analyses
ANALYSES=("dadi" "moments")
NUM_ANALYSES=${#ANALYSES[@]}  # Number of analyses

# Define batch size
BATCH_SIZE=3  # Number of tasks to run in each job
TOTAL_TASKS=$((NUM_SIMS_PRETRAIN * NUM_REPLICATES * NUM_ANALYSES))  # Total tasks
NUM_BATCHES=$(((TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE))  # Calculate total batches

# Calculate start and end indices for this batch
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

# Adjust end index for the last batch
if [ "$BATCH_END" -ge "$TOTAL_TASKS" ]; then
    BATCH_END=$((TOTAL_TASKS - 1))
fi

echo "Processing tasks from $BATCH_START to $BATCH_END"

# Process each task in this batch
for TASK_ID in $(seq $BATCH_START $BATCH_END); do
    # Extract task details
    ANALYSIS_INDEX=$((TASK_ID % NUM_ANALYSES))
    SIM_NUMBER=$((TASK_ID / (NUM_REPLICATES * NUM_ANALYSES)))
    REPLICATE_NUMBER=$(((TASK_ID / NUM_ANALYSES) % NUM_REPLICATES))
    ANALYSIS=${ANALYSES[$ANALYSIS_INDEX]}

    echo "Running analysis: $ANALYSIS for SIM_NUMBER=$SIM_NUMBER, REPLICATE_NUMBER=$REPLICATE_NUMBER"

    # Define output paths
    REPLICATE_DIR="${FEATURES_DIR}/sim_${SIM_NUMBER}/${ANALYSIS}/replicate_${REPLICATE_NUMBER}"
    AGGREGATION_OUTPUT="${FEATURES_DIR}/software_inferences_sim_${SIM_NUMBER}.pkl"

    mkdir -p "$REPLICATE_DIR"

    # Process the replicate
    snakemake \
        --nolock \
        --rerun-incomplete \
        --latency-wait 60 \
        --snakefile "${BASE_DIR}/Snakefile" \
        --directory "$REPLICATE_DIR" \
        "${REPLICATE_DIR}/replicate_${REPLICATE_NUMBER}.pkl"

    if [ $? -eq 0 ]; then
        echo "Snakemake completed successfully for $ANALYSIS, sim_${SIM_NUMBER}, replicate_${REPLICATE_NUMBER}"
    else
        echo "Snakemake failed for $ANALYSIS, sim_${SIM_NUMBER}, replicate_${REPLICATE_NUMBER}"
        continue  # Skip to the next task
    fi

    # Aggregation step (run only for the last replicate of each simulation, for `dadi`)
    if [[ "$REPLICATE_NUMBER" -eq "$((NUM_REPLICATES - 1))" && "$ANALYSIS_INDEX" -eq "0" ]]; then
        echo "Running aggregation for SIM_NUMBER=${SIM_NUMBER}"

        snakemake \
            --nolock \
            --rerun-incomplete \
            --latency-wait 60 \
            --snakefile "${BASE_DIR}/Snakefile" \
            --directory "${FEATURES_DIR}" \
            "$AGGREGATION_OUTPUT"

        if [ $? -eq 0 ]; then
            echo "Aggregation completed successfully for SIM_NUMBER=${SIM_NUMBER}"
        else
            echo "Aggregation failed for SIM_NUMBER=${SIM_NUMBER}"
            continue
        fi
    fi
done
