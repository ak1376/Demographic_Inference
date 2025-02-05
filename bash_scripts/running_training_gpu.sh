#!/bin/bash
#SBATCH --job-name=amodel_building
#SBATCH --partition=kerngpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/job_output_%j.log
#SBATCH --error=logs/job_error_%j.log
#SBATCH --verbose

# Define main directory
MAIN_FILEPATH="/projects/kernlab/akapoor/Demographic_Inference"

# Load JSON configuration files
CONFIG_FILEPATH="$MAIN_FILEPATH/experiment_config.json"
MODEL_CONFIG_FILEPATH="$MAIN_FILEPATH/model_config.json"

# Read JSON configuration files
EXPERIMENT_CONFIG=$(cat "$CONFIG_FILEPATH")
MODEL_CONFIG=$(cat "$MODEL_CONFIG_FILEPATH")

# Extract values from JSON using jq
DEMOGRAPHIC_MODEL=$(echo "$EXPERIMENT_CONFIG" | jq -r '.demographic_model')
DADI_ANALYSIS=$(echo "$EXPERIMENT_CONFIG" | jq -r '.dadi_analysis | if . then "True" else "False" end')
MOMENTS_ANALYSIS=$(echo "$EXPERIMENT_CONFIG" | jq -r '.moments_analysis | if . then "True" else "False" end')
MOMENTS_LD_ANALYSIS=$(echo "$EXPERIMENT_CONFIG" | jq -r '.momentsLD_analysis | if . then "True" else "False" end')
SEED=$(echo "$EXPERIMENT_CONFIG" | jq -r '.seed')
NUM_SIMS_PRETRAIN=$(echo "$EXPERIMENT_CONFIG" | jq -r '.num_sims_pretrain')
NUM_SIMS_INFERENCE=$(echo "$EXPERIMENT_CONFIG" | jq -r '.num_sims_inference')
NUM_REPLICATES=$(echo "$EXPERIMENT_CONFIG" | jq -r '.k')
TOP_VALUES_K=$(echo "$EXPERIMENT_CONFIG" | jq -r '.top_values_k')

EXPERIMENT_DIRECTORY="${DEMOGRAPHIC_MODEL}_dadi_analysis_${DADI_ANALYSIS}_moments_analysis_${MOMENTS_ANALYSIS}_momentsLD_analysis_${MOMENTS_LD_ANALYSIS}_seed_${SEED}"
EXPERIMENT_NAME="sims_pretrain_${NUM_SIMS_PRETRAIN}_sims_inference_${NUM_SIMS_INFERENCE}_seed_${SEED}_num_replicates_${NUM_REPLICATES}_top_values_${TOP_VALUES_K}"
SIM_DIRECTORY="$MAIN_FILEPATH/$EXPERIMENT_DIRECTORY/sims/$EXPERIMENT_NAME"

# Extract neural network hyperparameters
HIDDEN_SIZE=$(echo "$MODEL_CONFIG" | jq -c '.neural_net_hyperparameters.hidden_size')
NUM_LAYERS=$(echo "$MODEL_CONFIG" | jq -r '.neural_net_hyperparameters.num_layers')
NUM_EPOCHS=$(echo "$MODEL_CONFIG" | jq -r '.neural_net_hyperparameters.num_epochs')
DROPOUT_RATE=$(echo "$MODEL_CONFIG" | jq -r '.neural_net_hyperparameters.dropout_rate')
WEIGHT_DECAY=$(echo "$MODEL_CONFIG" | jq -r '.neural_net_hyperparameters.weight_decay')
BATCH_SIZE=$(echo "$MODEL_CONFIG" | jq -r '.neural_net_hyperparameters.batch_size')
EARLY_STOPPING=$(echo "$MODEL_CONFIG" | jq -r '.neural_net_hyperparameters.EarlyStopping')

# Convert hidden_size (list) into a string format
if [[ $HIDDEN_SIZE == *"["* ]]; then
    HIDDEN_SIZE_STR=$(echo "$HIDDEN_SIZE" | jq -r 'join("_")')
else
    HIDDEN_SIZE_STR="$HIDDEN_SIZE"
fi

MODEL_DIRECTORY="$MAIN_FILEPATH/$EXPERIMENT_DIRECTORY/models/$EXPERIMENT_NAME/num_hidden_neurons_${HIDDEN_SIZE_STR}_num_hidden_layers_${NUM_LAYERS}_num_epochs_${NUM_EPOCHS}_dropout_value_${DROPOUT_RATE}_weight_decay_${WEIGHT_DECAY}_batch_size_${BATCH_SIZE}_EarlyStopping_${EARLY_STOPPING}"

# Print directories for debugging
echo "EXPERIMENT_DIRECTORY: $EXPERIMENT_DIRECTORY"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SIM_DIRECTORY: $SIM_DIRECTORY"
echo "MODEL_DIRECTORY: $MODEL_DIRECTORY"

# Check if the features file exists before running
FEATURES_FILE="$SIM_DIRECTORY/features_and_targets.pkl"
if [[ ! -f "$FEATURES_FILE" ]]; then
    echo "ERROR: The file $FEATURES_FILE does not exist!"
    ls -lh "$SIM_DIRECTORY"  # List the directory contents for debugging
    exit 1
fi

# Run Python script with the constructed parameters
python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/setup_trainer.py \
    --experiment_directory "$EXPERIMENT_DIRECTORY" \
    --model_config_file "$MODEL_CONFIG_FILEPATH" \
    --features_file "$FEATURES_FILE" \
    --color_shades "$SIM_DIRECTORY/color_shades.pkl" \
    --main_colors "$SIM_DIRECTORY/main_colors.pkl"

echo "Job completed successfully!"
