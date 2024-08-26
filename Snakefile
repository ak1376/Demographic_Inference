import os

# Define the experiment name
EXPERIMENT_NAME = "snakemake" # Really need to change this later. I don't want to have to be creating a 
EXPERIMENT_DIRECTORY = f"experiments/{EXPERIMENT_NAME}"
# Define the output files
CONFIG_FILE = f"experiments/{EXPERIMENT_NAME}/config.json"
MODEL_CONFIG_FILE = f"experiments/{EXPERIMENT_NAME}/model_config.json"
EXPERIMENT_OBJ_FILE = f"experiments/{EXPERIMENT_NAME}/experiment_obj.pkl"
PREPROCESSING_RESULTS = f"experiments/{EXPERIMENT_NAME}/preprocessing_results_obj.pkl"
LINEAR_MODEL = f"experiments/{EXPERIMENT_NAME}/linear_regression_model.pkl"
FEATURES_OUTPUT = f"experiments/{EXPERIMENT_NAME}/features_and_targets.pkl"
SNN_RESULTS = f"experiments/{EXPERIMENT_NAME}/snn_results.pkl"
SNN_MODEL = f"experiments/{EXPERIMENT_NAME}/snn_model.pth"
USE_FIM = False

# Get the current working directory
CWD = os.getcwd()

rule all:
    input:
        CONFIG_FILE,
        MODEL_CONFIG_FILE,
        EXPERIMENT_OBJ_FILE,
        PREPROCESSING_RESULTS,
        LINEAR_MODEL,
        FEATURES_OUTPUT,
        SNN_RESULTS,
        SNN_MODEL


rule create_experiment:
    # This will save the experiment object and the config file
    output:
        config_file = CONFIG_FILE,
        experiment_obj_file = EXPERIMENT_OBJ_FILE,
        model_config_file = MODEL_CONFIG_FILE
    shell:
        "PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_experiment.py --config_file {output.config_file} --experiment_obj_file {output.experiment_obj_file}"

rule obtain_features:
    input:
        config_file = rules.create_experiment.output.config_file,
        experiment_obj_file = rules.create_experiment.output.experiment_obj_file
    output:
        preprocessing_results = PREPROCESSING_RESULTS,
        linear_model = LINEAR_MODEL
    params:
        experiment_name = EXPERIMENT_NAME,
        num_sims_pretrain = 100,  # Adjust as needed
        num_sims_inference = 100,  # Adjust as needed
        normalization = False,  # Adjust as needed
        remove_outliers = True  # Adjust as needed
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/obtaining_features.py \
        --experiment_config {input.config_file} \
        --experiment_directory experiments/{params.experiment_name} \
        --num_sims_pretrain {params.num_sims_pretrain} \
        --num_sims_inference {params.num_sims_inference} \
        --normalization {params.normalization} \
        --remove_outliers {params.remove_outliers}
        """

rule get_features:
    input:
        preprocessing_results = PREPROCESSING_RESULTS
    output:
        features_output = FEATURES_OUTPUT
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/extracting_features.py \
        --preprocessing_results_filepath {input.preprocessing_results} \
        --experiment_directory experiments/{EXPERIMENT_NAME} \
        """

rule train_and_predict:
    input:
        model_config_file = MODEL_CONFIG_FILE,
        features_file = FEATURES_OUTPUT  # Add this line to ensure correct order of execution
    params:
        use_FIM = False,
        experiment_directory = EXPERIMENT_DIRECTORY

    output:
        model_results = SNN_RESULTS,
        trained_model = SNN_MODEL
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_trainer.py \
        --model_config {input.model_config_file} \
        --experiment_directory {params.experiment_directory} \
        --features_file {input.features_file} \
        --use_FIM {params.use_FIM}
        """