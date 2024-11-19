#!/bin/bash
#SBATCH --job-name=moments_ld_array
#SBATCH --array=0-999  # For 10 simulations
#SBATCH --output=logs/moments_ld_%A_%a.out
#SBATCH --error=logs/moments_ld_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

BASE_DIR="/gpfs/projects/kernlab/akapoor/Demographic_Inference"
cd $BASE_DIR

# Define batch size for simulations
BATCH_SIZE=1
TOTAL_SIMS=1000

# Calculate which simulations to process
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

# Process each simulation in the batch
for SIM_NUMBER in $(seq $BATCH_START $BATCH_END); do
    echo "Processing aggregation and momentsLD for sim ${SIM_NUMBER}"
    
    # First combined_ld_stats
    snakemake \
        --snakefile $BASE_DIR/Snakefile \
        --rerun-incomplete \
        --nolock \
        "combined_LD_inferences/sim_${SIM_NUMBER}/combined_LD_stats_sim_${SIM_NUMBER}.pkl"
        
    # Then momentsLD inference
    snakemake \
        --snakefile $BASE_DIR/Snakefile \
        --rerun-incomplete \
        --nolock \
        "final_LD_inferences/momentsLD_inferences_sim_${SIM_NUMBER}.pkl"
done