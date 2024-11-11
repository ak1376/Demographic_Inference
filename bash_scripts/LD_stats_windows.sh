#!/bin/bash
#SBATCH --job-name=ld_stats_array
#SBATCH --array=0-99
#SBATCH --output=logs/ld_stats_%A_%a.out
#SBATCH --error=logs/ld_stats_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

BASE_DIR="/gpfs/projects/kernlab/akapoor/Demographic_Inference"
cd $BASE_DIR

# Define batch size
BATCH_SIZE=10
TOTAL_TASKS=1000

# Calculate batch indices
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

# Process each task in the batch
for TASK_ID in $(seq $BATCH_START $BATCH_END); do
    SIM_NUMBER=$((TASK_ID / 100))
    WINDOW_NUMBER=$((TASK_ID % 100))
    
    echo "Processing ld_stats for sim ${SIM_NUMBER}, window ${WINDOW_NUMBER}"
    
    mkdir -p $BASE_DIR/LD_inferences/sim_${SIM_NUMBER}/
    
    snakemake \
        --snakefile $BASE_DIR/Snakefile \
        --rerun-incomplete \
        --nolock \
        "LD_inferences/sim_${SIM_NUMBER}/ld_stats_window.${WINDOW_NUMBER}.pkl"
done