import os

# Define the experiment name
EXPERIMENT_NAME = "snakemake_test"
# Define the output files
CONFIG_FILE = f"experiments/{EXPERIMENT_NAME}/config.json"
EXPERIMENT_OBJ_FILE = f"experiments/{EXPERIMENT_NAME}/experiment_obj.pkl"
PREPROCESSING_RESULTS = f"experiments/{EXPERIMENT_NAME}/preprocessing_results_obj.pkl"
LINEAR_MODEL = f"experiments/{EXPERIMENT_NAME}/linear_regression_model.pkl"
FEATURES_OUTPUT = f"experiments/{EXPERIMENT_NAME}/features_and_targets.pkl"
# Get the current working directory
CWD = os.getcwd()
rule all:
    input:
        CONFIG_FILE,
        EXPERIMENT_OBJ_FILE,
        PREPROCESSING_RESULTS,
        LINEAR_MODEL,
        FEATURES_OUTPUT

rule create_experiment:
    # This will save the experiment object and the config file
    output:
        config_file = CONFIG_FILE,
        experiment_obj_file = EXPERIMENT_OBJ_FILE
    shell:
        "PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_experiment.py --config_file {output.config_file} --experiment_obj_file {output.experiment_obj_file}"

rule obtain_features:
    input:
        config_file = rules.create_experiment.output.config_file,
        experiment_obj_file = rules.create_experiment.output.experiment_obj_file
    output:
        preprocessing_results = "experiments/{EXPERIMENT_NAME}/preprocessing_results_obj.pkl",
        linear_model = "experiments/{EXPERIMENT_NAME}/linear_regression_model.pkl"
    params:
        experiment_name = "{EXPERIMENT_NAME}",
        num_sims_pretrain = 1000,  # Adjust as needed
        num_sims_inference = 1000,  # Adjust as needed
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