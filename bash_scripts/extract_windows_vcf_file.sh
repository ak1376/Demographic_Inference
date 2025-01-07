#!/bin/bash
#SBATCH --job-name=extract_windows
#SBATCH --array=0-99                   # Array of 100 tasks, one for each window
#SBATCH --output=logs/extract_windows_%A_%a.out
#SBATCH --error=logs/extract_windows_%A_%a.err
#SBATCH --time=6:00:00                 # Adjust time as needed
#SBATCH --cpus-per-task=1              # Adjust CPU resources per task
#SBATCH --mem=16G                      # Adjust memory per task
#SBATCH --partition=kern,preempt       # Adjust partition as needed
#SBATCH --account=kernlab              # Adjust SLURM account
#SBATCH --requeue

# Configuration variables
EXPERIMENT_CONFIG="/projects/kernlab/akapoor/Demographic_Inference/experiment_config.json"
VCF_FILE="/projects/kernlab/akapoor/Demographic_Inference/GHIST-split-isolation.vcf.gz"
OUTPUT_DIR="/projects/kernlab/akapoor/Demographic_Inference/inference_windows/"

# Ensure output directories exist
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# SLURM_ARRAY_TASK_ID corresponds to the window number (0 to 99)
WINDOW_NUMBER=${SLURM_ARRAY_TASK_ID}

# Print job information
echo "Starting task for window ${WINDOW_NUMBER}..."
echo "VCF file: ${VCF_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"

# Run the Python script for the specified window
PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference \
python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/inference_window_generation.py \
    --vcf_file "${VCF_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_config "${EXPERIMENT_CONFIG}" \
    --window_number "${WINDOW_NUMBER}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Window ${WINDOW_NUMBER} processed successfully."
else
    echo "Error processing window ${WINDOW_NUMBER}."
    exit 1
fi
