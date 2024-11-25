#!/bin/bash
#SBATCH --job-name=remaining_rules      
#SBATCH --output=logs/remaining_rules_%j.out  
#SBATCH --error=logs/remaining_rules_%j.err   
#SBATCH --time=05:00:00                 
#SBATCH --cpus-per-task=4               
#SBATCH --mem=32G                       
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue     

# Set config file paths
EXPERIMENT_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'
MODEL_CONFIG_FILE='/home/akapoor/kernlab/Demographic_Inference/model_config.json'

# Extract experiment config values
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
DADI_ANALYSIS=$(jq -r '.dadi_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_ANALYSIS=$(jq -r '.moments_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_LD_ANALYSIS=$(jq -r '.momentsLD_analysis' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_INFERENCE=$(jq -r '.num_sims_inference' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)

# Extract model config values
HIDDEN_SIZE=$(jq -r '.neural_net_hyperparameters.hidden_size' $MODEL_CONFIG_FILE)
NUM_LAYERS=$(jq -r '.neural_net_hyperparameters.num_layers' $MODEL_CONFIG_FILE)
NUM_EPOCHS=$(jq -r '.neural_net_hyperparameters.num_epochs' $MODEL_CONFIG_FILE)
DROPOUT_RATE=$(jq -r '.neural_net_hyperparameters.dropout_rate' $MODEL_CONFIG_FILE)
WEIGHT_DECAY=$(jq -r '.neural_net_hyperparameters.weight_decay' $MODEL_CONFIG_FILE)
BATCH_SIZE=$(jq -r '.neural_net_hyperparameters.batch_size' $MODEL_CONFIG_FILE)
EARLY_STOPPING=$(jq -r '.neural_net_hyperparameters.EarlyStopping' $MODEL_CONFIG_FILE)

# Function to convert lowercase true/false to capitalized True/False
capitalize_bool() {
   if [ "$1" == "true" ]; then
       echo "True"
   elif [ "$1" == "false" ]; then
       echo "False"
   else
       echo "$1"
   fi
}

# Convert boolean values
DADI_ANALYSIS=$(capitalize_bool $DADI_ANALYSIS)
MOMENTS_ANALYSIS=$(capitalize_bool $MOMENTS_ANALYSIS)
MOMENTS_LD_ANALYSIS=$(capitalize_bool $MOMENTS_LD_ANALYSIS)
EARLY_STOPPING=$(capitalize_bool $EARLY_STOPPING)

# Set up simulation directory
SIM_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/sims/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}"

# Set up model directory with neural net parameters
MODEL_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}/models/sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${K}_top_values_${TOP_VALUES_K}/num_hidden_neurons_${HIDDEN_SIZE}_num_hidden_layers_${NUM_LAYERS}_num_epochs_${NUM_EPOCHS}_dropout_value_${DROPOUT_RATE}_weight_decay_${WEIGHT_DECAY}_batch_size_${BATCH_SIZE}_EarlyStopping_${EARLY_STOPPING}"

echo "SIM_DIRECTORY: $SIM_DIRECTORY"
echo "MODEL_DIRECTORY: $MODEL_DIRECTORY"

echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Snakemake path: $(which snakemake)"
module list

# Run Snakemake to complete the remaining rules
snakemake \
    --snakefile /gpfs/projects/kernlab/akapoor/Demographic_Inference/Snakefile \
    --directory /gpfs/projects/kernlab/akapoor/Demographic_Inference \
    --rerun-incomplete \
    "${SIM_DIRECTORY}/postprocessing_results.pkl" \
    "${SIM_DIRECTORY}/features_and_targets.pkl" \
    "${MODEL_DIRECTORY}/linear_regression_model.pkl" \
    "${MODEL_DIRECTORY}/snn_results.pkl" \
    "${MODEL_DIRECTORY}/snn_model.pth"