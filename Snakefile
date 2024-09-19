import os
import json

# Load the configuration file
CONFIG_FILEPATH = '/sietch_colab/akapoor/Demographic_Inference/experiment_config.json'
with open(CONFIG_FILEPATH, 'r') as f:
    experiment_config = json.load(f)

CWD = os.getcwd()

# Use double quotes for the dictionary keys inside the f-string
EXPERIMENT_DIRECTORY = f'bottleneck_experiments'
EXPERIMENT_NAME = f'sims_pretrain_{experiment_config["num_sims_pretrain"]}_sims_inference_{experiment_config["num_sims_inference"]}_seed_{experiment_config["seed"]}_num_replicates_{experiment_config["k"]}'
SIM_DIRECTORY = f"{EXPERIMENT_DIRECTORY}/{EXPERIMENT_NAME}"
MODEL_DIRECTORY = f"{EXPERIMENT_DIRECTORY}/{EXPERIMENT_NAME}/num_hidden_neurons_{experiment_config['neural_net_hyperparameters']['hidden_size']}_num_hidden_layers_{experiment_config['neural_net_hyperparameters']['num_layers']}_num_epochs_{experiment_config['neural_net_hyperparameters']['num_epochs']}_dropout_value_{experiment_config['neural_net_hyperparameters']['dropout_rate']}_weight_decay_{experiment_config['neural_net_hyperparameters']['weight_decay']}"

rule all:
    input:
        f"{MODEL_DIRECTORY}/experiment_obj.pkl",
        f"{MODEL_DIRECTORY}/model_config.json",
        f"{MODEL_DIRECTORY}/inference_config_file.json",
        f"{SIM_DIRECTORY}/color_shades.pkl",
        f"{SIM_DIRECTORY}/main_colors.pkl",
        f"{SIM_DIRECTORY}/preprocessing_results_obj.pkl",
        f"{MODEL_DIRECTORY}/linear_regression_model.pkl",
        f"{SIM_DIRECTORY}/features_and_targets.pkl",
        f"{SIM_DIRECTORY}/additional_features.pkl",
        f"{MODEL_DIRECTORY}/snn_results.pkl",
        f"{MODEL_DIRECTORY}/snn_model.pth",
        f"{MODEL_DIRECTORY}/inference_results_obj.pkl",
        f"{MODEL_DIRECTORY}/inferred_params_GHIST_bottleneck.txt"



rule setup_folders:
    shell:
        """
        mkdir -p {EXPERIMENT_DIRECTORY} 
        mkdir -p {SIM_DIRECTORY} 
        mkdir -p {MODEL_DIRECTORY} 
        """

rule create_experiment:
    params:
        CONFIG_FILEPATH = CONFIG_FILEPATH,
        EXPERIMENT_NAME = EXPERIMENT_NAME,
        EXPERIMENT_DIRECTORY = EXPERIMENT_DIRECTORY,
        SIM_DIRECTORY = SIM_DIRECTORY,
        MODEL_DIRECTORY = MODEL_DIRECTORY
    output:
        config_file = f"{SIM_DIRECTORY}/config.json",
        experiment_obj_file = f"{MODEL_DIRECTORY}/experiment_obj.pkl",
        model_config_file = f"{MODEL_DIRECTORY}/model_config.json",
        inference_config_file = f"{MODEL_DIRECTORY}/inference_config_file.json",
        colors_shades_file = f"{SIM_DIRECTORY}/color_shades.pkl",
        main_colors_file = f"{SIM_DIRECTORY}/main_colors.pkl"

    # conda: 
    #     "myenv"
    run:
        with open(output.config_file, 'w') as f:
            json.dump(experiment_config, f, indent=4)
        
        # Call shell command inside the Python 'run' block
        shell(f"""
            PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_experiment.py \
            --config_file {params.CONFIG_FILEPATH} \
            --experiment_name {params.EXPERIMENT_NAME} \
            --experiment_directory {EXPERIMENT_DIRECTORY} \
            --sim_directory {params.SIM_DIRECTORY} \
            --model_directory {params.MODEL_DIRECTORY}
        """)

rule obtain_features:
    input:
        config_file = rules.create_experiment.output.config_file,
        experiment_obj_file = rules.create_experiment.output.experiment_obj_file,
        colors_shades_file = rules.create_experiment.output.colors_shades_file,
        main_colors_file = rules.create_experiment.output.main_colors_file
    output:
        preprocessing_results = f"{SIM_DIRECTORY}/preprocessing_results_obj.pkl",
        linear_model = f"{MODEL_DIRECTORY}/linear_regression_model.pkl"

    params:
        model_directory = MODEL_DIRECTORY,
        sim_directory = SIM_DIRECTORY

    # conda: 
    #     "myenv"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/obtaining_features.py \
        --experiment_config {input.config_file} \
        --model_directory {params.model_directory} \
        --sim_directory {params.sim_directory} \
        --color_shades_file {input.colors_shades_file} \
        --main_colors_file {input.main_colors_file} \
        """

rule get_features:
    input:
        preprocessing_results = rules.obtain_features.output.preprocessing_results,
        config_file = rules.create_experiment.output.config_file
    output:
        features_output = f"{SIM_DIRECTORY}/features_and_targets.pkl",
        additional_features_output = f"{SIM_DIRECTORY}/additional_features.pkl"

    params:
        sim_directory = SIM_DIRECTORY

    # conda: 
    #     "myenv"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/extracting_features.py \
        --preprocessing_results_filepath {input.preprocessing_results} \
        --config_filepath {input.config_file} \
        --sim_directory {params.sim_directory} \
        """

rule train_and_predict:
    input:
        model_config_file = rules.create_experiment.output.model_config_file, #TODO: Replace this with model config file saved to Demographic_Inference folder 
        features_file = rules.get_features.output.features_output,
        colors_shades_file = rules.create_experiment.output.colors_shades_file,
        main_colors_file = rules.create_experiment.output.main_colors_file,
        additional_features_file = rules.get_features.output.additional_features_output
    params:
        use_FIM = False,
        experiment_directory = MODEL_DIRECTORY

    output:
        model_results = f"{MODEL_DIRECTORY}/snn_results.pkl",
        trained_model = f"{MODEL_DIRECTORY}/snn_model.pth"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_trainer.py \
        --model_config {input.model_config_file} \
        --experiment_directory {params.experiment_directory} \
        --features_file {input.features_file} \
        --color_shades {input.colors_shades_file} \
        --main_colors {input.main_colors_file} \
        --additional_features_file {input.additional_features_file} \
        --use_FIM {params.use_FIM}
        """

rule get_inferred_params:
    input:
        config = rules.create_experiment.output.config_file,
        trained_weights = rules.train_and_predict.output.trained_model,

    params:
        experiment_directory = MODEL_DIRECTORY,
        inference_obj_path = f"{MODEL_DIRECTORY}/inference_results_obj.pkl"


    output:
        f"{MODEL_DIRECTORY}/inference_results_obj.pkl",
        f"{MODEL_DIRECTORY}/inferred_params_GHIST_bottleneck.txt"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/inference_snakemake.py \
        --config {input.config} \
        --trained_weights {input.trained_weights} \
        --inference_obj_path {params.inference_obj_path} \
        --experiment_directory {params.experiment_directory}
        """
