#!/bin/bash
#SBATCH --job-name=batched_genome_windows
#SBATCH --array=0-999          
#SBATCH --output=logs/genome_windows_%A_%a.out
#SBATCH --error=logs/genome_windows_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

BATCH_SIZE=10
TOTAL_TASKS=10000

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    overall_start_time=$(date +%s)
    echo "Overall start time: $overall_start_time"
fi

EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)
NUM_WINDOWS=$(jq -r '.num_windows' $EXPERIMENT_CONFIG_FILE)

if [ -z "$NUM_WINDOWS" ] || [ "$NUM_WINDOWS" -eq 0 ]; then
    echo "Error: NUM_WINDOWS is not defined or zero."
    exit 1
fi

SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"
echo "Sim directory: $SIM_DIRECTORY"

BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

if [ "$BATCH_END" -ge "$TOTAL_TASKS" ]; then
    BATCH_END=$((TOTAL_TASKS - 1))
fi

mkdir -p logs
current_dir=$(pwd)
echo "Current Directory: ${current_dir}"

for TASK_ID in $(seq $BATCH_START $BATCH_END); do
    SIM_NUMBER=$((TASK_ID / NUM_WINDOWS))
    WINDOW_NUMBER=$((TASK_ID % NUM_WINDOWS))

    SIM_DIR="/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_${SIM_NUMBER}"
    WINDOW_DIR="${SIM_DIR}/window_${WINDOW_NUMBER}"
    mkdir -p "$WINDOW_DIR"

    echo "Processing sim_number: $SIM_NUMBER, window_number: $WINDOW_NUMBER in $WINDOW_DIR"
    export PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference

    pushd "$WINDOW_DIR" || { echo "Failed to change directory to $WINDOW_DIR"; exit 1; }
    snakemake \
        --snakefile /projects/kernlab/akapoor/Demographic_Inference/Snakefile \
        --directory "$WINDOW_DIR" \
        --rerun-incomplete \
        --nolock \
        "${WINDOW_DIR}/samples.txt" \
        "${WINDOW_DIR}/flat_map.txt" \
        "${WINDOW_DIR}/individual_file_metadata.txt" \
        "${WINDOW_DIR}/window.${WINDOW_NUMBER}.vcf.gz"

    if [ $? -eq 0 ]; then
        echo "Snakemake completed successfully for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"
    else
        echo "Snakemake failed for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"
        popd
        exit 1
    fi
    popd || { echo "Failed to return to previous directory"; exit 1; }
done

wait

# After all array tasks complete (which means all windows are processed), 
# run another separate job (not part of this array) to combine_metadata for each simulation.
# This can be done in a separate script or by adding job dependencies in SLURM.

TOTAL_BATCHES=$((TOTAL_TASKS / BATCH_SIZE + (TOTAL_TASKS % BATCH_SIZE > 0 ? 1 : 0)))
if [ "$SLURM_ARRAY_TASK_ID" -eq $((TOTAL_BATCHES - 1)) ]; then
    overall_end_time=$(date +%s)
    echo "Overall end time: $overall_end_time"
    overall_elapsed_time=$((overall_end_time - overall_start_time))
    echo "Total time taken: $overall_elapsed_time seconds"
fi
