#!/bin/bash

#SBATCH --job-name=combine_and_infer
#SBATCH --output=logs/combine_and_infer_%j.out
#SBATCH --error=logs/combine_and_infer_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue

################################################################################
# 1) Hard-code your paths and parameters here
################################################################################
LD_STATS_DIR="/projects/kernlab/akapoor/Demographic_Inference/inference_LD_stats"
EXPERIMENT_CONFIG_FILEPATH="/projects/kernlab/akapoor/Demographic_Inference/experiment_config.json"
OUTPUT_DIR="/projects/kernlab/akapoor/Demographic_Inference/inference_MomentsLD_results"

# Name of your second Python script (the one that does MomentsLD inference)
MOMENTS_LD_SCRIPT="/projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/inference_MomentsLD_analysis.py"

################################################################################
# 2) Combine LD stats: calls `combine_ld_stats.py`
################################################################################
echo "=================================================================="
echo "Combining LD stats with 'combine_ld_stats.py'..."
echo "LD_STATS_DIR: $LD_STATS_DIR"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "=================================================================="

python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/inference_combine_ld_stats.py \
    --ld_stats_dir "$LD_STATS_DIR" \
    --output_dir "$OUTPUT_DIR"

# If combining failed, exit
if [ $? -ne 0 ]; then
  echo "[Error] combine_ld_stats.py failed. Exiting."
  exit 1
fi

COMBINED_LD_STATS_PATH="$OUTPUT_DIR/combined_LD_stats.pkl"
echo "Combined LD stats created at: $COMBINED_LD_STATS_PATH"

################################################################################
# 3) MomentsLD Inference: calls `your_momentsLD_script.py`
################################################################################
echo "=================================================================="
echo "Running MomentsLD inference with '$MOMENTS_LD_SCRIPT'..."
echo "COMBINED_LD_STATS_PATH: $COMBINED_LD_STATS_PATH"
echo "EXPERIMENT_CONFIG_FILEPATH: $EXPERIMENT_CONFIG_FILEPATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "=================================================================="

export PYTHONPATH="/projects/kernlab/akapoor/Demographic_Inference:$PYTHONPATH"
python "$MOMENTS_LD_SCRIPT" \
    --combined_ld_stats_path "$COMBINED_LD_STATS_PATH" \
    --experiment_config_filepath "$EXPERIMENT_CONFIG_FILEPATH" \
    --output_dir "$OUTPUT_DIR"

# If inference failed, exit
if [ $? -ne 0 ]; then
  echo "[Error] $MOMENTS_LD_SCRIPT failed. Exiting."
  exit 1
fi

echo "MomentsLD inference completed successfully."
echo "Results saved in: $OUTPUT_DIR"
