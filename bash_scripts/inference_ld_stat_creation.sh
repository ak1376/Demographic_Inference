#!/bin/bash
#SBATCH --job-name=ld_stat_creation
#SBATCH --array=0-99                   # Array of 100 tasks, one for each window
#SBATCH --output=logs/inf_ld_stats_%A_%a.out
#SBATCH --error=logs/inf_ld_stats_%A_%a.err
#SBATCH --time=6:00:00                 # Adjust time as needed
#SBATCH --cpus-per-task=1              # Adjust CPU resources per task
#SBATCH --mem=16G                      # Adjust memory per task
#SBATCH --partition=kern,preempt       # Adjust partition as needed
#SBATCH --account=kernlab              # Adjust SLURM account
#SBATCH --requeue

# Configuration variables
INFERENCE_WINDOWS_DIR="/projects/kernlab/akapoor/Demographic_Inference/inference_windows"
OUTPUT_DIR="/projects/kernlab/akapoor/Demographic_Inference/inference_LD_stats/"

# Ensure output directories exist
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# SLURM_ARRAY_TASK_ID corresponds to the window number (0 to 99)
WINDOW_NUMBER=${SLURM_ARRAY_TASK_ID}
WINDOW_DIR="${INFERENCE_WINDOWS_DIR}/window_${WINDOW_NUMBER}"

# Define input files for the current window
VCF_FILE="${WINDOW_DIR}/window.${WINDOW_NUMBER}.vcf.gz"
FLAT_MAP_PATH="${WINDOW_DIR}/flat_map.txt"
POP_FILE_PATH="${WINDOW_DIR}/samples.txt"

# Print job information
echo "Starting LD stats task for window ${WINDOW_NUMBER}..."
echo "VCF file: ${VCF_FILE}"
echo "Flat map path: ${FLAT_MAP_PATH}"
echo "Population file path: ${POP_FILE_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Check if input files exist
if [[ ! -f "${VCF_FILE}" || ! -f "${FLAT_MAP_PATH}" || ! -f "${POP_FILE_PATH}" ]]; then
    echo "Error: Missing input files for window ${WINDOW_NUMBER}. Skipping..."
    exit 1
fi

# Run the Python script for the specified window
PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference \
python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/inference_ld_stats.py \
    --vcf_filepath "${VCF_FILE}" \
    --flat_map_path "${FLAT_MAP_PATH}" \
    --pop_file_path "${POP_FILE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --window_number "${WINDOW_NUMBER}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "LD stats task for window ${WINDOW_NUMBER} completed successfully."
else
    echo "Error processing LD stats for window ${WINDOW_NUMBER}."
    exit 1
fi
