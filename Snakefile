import os
import json

upper_bound_params = {
    "N0": 10000,
    "Nb": 5000,
    "N_recover": 7000,
    "t_bottleneck_end": 1000,
    "t_bottleneck_start": 2000,
}
lower_bound_params = {
    "N0": 8000,
    "Nb": 4000,
    "N_recover": 6000,
    "t_bottleneck_end": 800,
    "t_bottleneck_start": 1500,
}
model_config = {
    "input_size": 10,
    "hidden_size": 1000,
    "output_size": 5,
    "num_epochs": 1000,
    "learning_rate": 3e-4,
    "num_layers": 1,
    "dropout_rate": 0.1,
    "weight_decay": 1e-4,
    "parameter_names": ["N0", "Nb", "N_recover", "t_bottleneck_end", "t_bottleneck_start"], # these should be a list of parameters that we want to optimize 

}

config = {
    "upper_bound_params": upper_bound_params,
    "lower_bound_params": lower_bound_params,
    "num_sims_pretrain": 10,
    "num_sims_inference": 1,
    "num_samples": 20,
    "experiment_name": "dadi_moments_analysis_new_analysis_optimize_big_length_check",
    "dadi_analysis": True,
    "moments_analysis": True,
    "momentsLD_analysis": False,
    "num_windows": 50,
    "window_length": 1e4,
    "maxiter": 100,
    "genome_length": 1e8,
    "mutation_rate": 1.26e-8,
    "recombination_rate": 1.007e-8,
    "seed": 42,
    "normalization": False,
    "remove_outliers": True,
    "use_FIM": False,
    "neural_net_hyperparameters": model_config,
    "demographic_model": "bottleneck_model",
    "parameter_names": ["N0", "Nb", "N_recover", "t_bottleneck_end", "t_bottleneck_start"], # these should be a list of parameters that we want to optimize 
    "optimization_initial_guess": [0.25, 0.75, 0.1, 0.05],
    "vcf_filepath": "/sietch_colab/akapoor/GHIST-bottleneck.vcf.gz",
    "txt_filepath": "/sietch_colab/akapoor/wisent.txt",
    "popname": "wisent"
    
}

    # inference = Inference(
    #     vcf_filepath=config['vcf_filepath'],
    #     txt_filepath=config['txt_filepath'],
    #     popname=config['popname'],
    #     config=config,
    #     experiment_directory=config['experiment_directory']
    # )

# Define the experiment name
EXPERIMENT_NAME = config['experiment_name']
EXPERIMENT_DIRECTORY = f"experiments/{EXPERIMENT_NAME}"
# Get the current working directory
CWD = os.getcwd()

rule all:
    input:
        f"{EXPERIMENT_DIRECTORY}/config.json",
        f"{EXPERIMENT_DIRECTORY}/model_config.json",
        f"{EXPERIMENT_DIRECTORY}/experiment_obj.pkl",
        f"{EXPERIMENT_DIRECTORY}/inference_config_file.json",
        f"{EXPERIMENT_DIRECTORY}/preprocessing_results_obj.pkl",
        f"{EXPERIMENT_DIRECTORY}/linear_regression_model.pkl",
        f"{EXPERIMENT_DIRECTORY}/features_and_targets.pkl",
        f"{EXPERIMENT_DIRECTORY}/snn_results.pkl",
        f"{EXPERIMENT_DIRECTORY}/snn_model.pth",
        f"{EXPERIMENT_DIRECTORY}/inference_results_obj.pkl",
        f"{EXPERIMENT_DIRECTORY}/inferred_params_GHIST_bottleneck.txt",
        f'{EXPERIMENT_DIRECTORY}/color_shades.pkl',
        f'{EXPERIMENT_DIRECTORY}/main_colors.pkl',

rule create_experiment:
    output:
        config_file = f"{EXPERIMENT_DIRECTORY}/config.json",
        experiment_obj_file = f"{EXPERIMENT_DIRECTORY}/experiment_obj.pkl",
        model_config_file = f"{EXPERIMENT_DIRECTORY}/model_config.json",
        inference_config_file = f"{EXPERIMENT_DIRECTORY}/inference_config_file.json",
        colors_shades_file = f"{EXPERIMENT_DIRECTORY}/color_shades.pkl",
        main_colors_file = f"{EXPERIMENT_DIRECTORY}/main_colors.pkl"

    # conda: 
    #     "myenv"
    run:
        print(f'Experiment Directory: {EXPERIMENT_DIRECTORY}')
        os.makedirs(EXPERIMENT_DIRECTORY, exist_ok=True)
        with open(output.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        shell(f"PYTHONPATH={{CWD}} python {{CWD}}/snakemake_scripts/setup_experiment.py --config_file {{output.config_file}} --experiment_obj_file {{output.experiment_obj_file}}")

rule obtain_features:
    input:
        config_file = rules.create_experiment.output.config_file,
        experiment_obj_file = rules.create_experiment.output.experiment_obj_file,
        colors_shades_file = rules.create_experiment.output.colors_shades_file,
        main_colors_file = rules.create_experiment.output.main_colors_file
    output:
        preprocessing_results = f"{EXPERIMENT_DIRECTORY}/preprocessing_results_obj.pkl",
        linear_model = f"{EXPERIMENT_DIRECTORY}/linear_regression_model.pkl",
    params:
        experiment_name = EXPERIMENT_NAME,

    # conda: 
    #     "myenv"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/obtaining_features.py \
        --experiment_config {input.config_file} \
        --experiment_directory {EXPERIMENT_DIRECTORY} \
        --color_shades_file {input.colors_shades_file} \
        --main_colors_file {input.main_colors_file} \
        """

rule get_features:
    input:
        preprocessing_results = rules.obtain_features.output.preprocessing_results
    output:
        features_output = f"{EXPERIMENT_DIRECTORY}/features_and_targets.pkl"

    # conda: 
    #     "myenv"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/extracting_features.py \
        --preprocessing_results_filepath {input.preprocessing_results} \
        --experiment_directory {EXPERIMENT_DIRECTORY} \
        """

rule train_and_predict:
    input:
        model_config_file = rules.create_experiment.output.model_config_file,
        features_file = rules.get_features.output.features_output,
        colors_shades_file = rules.create_experiment.output.colors_shades_file,
        main_colors_file = rules.create_experiment.output.main_colors_file
    params:
        use_FIM = False,
        experiment_directory = EXPERIMENT_DIRECTORY

    output:
        model_results = f"{EXPERIMENT_DIRECTORY}/snn_results.pkl",
        trained_model = f"{EXPERIMENT_DIRECTORY}/snn_model.pth"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_trainer.py \
        --model_config {input.model_config_file} \
        --experiment_directory {params.experiment_directory} \
        --features_file {input.features_file} \
        --color_shades {input.colors_shades_file} \
        --main_colors {input.main_colors_file} \
        --use_FIM {params.use_FIM}
        """

rule get_inferred_params:
    input:
        config = rules.create_experiment.output.config_file,
        trained_weights = rules.train_and_predict.output.trained_model,
    params:
        experiment_directory = EXPERIMENT_DIRECTORY,
        inference_obj_path = f"{EXPERIMENT_DIRECTORY}/inference_results_obj.pkl"

    output:
        f"{EXPERIMENT_DIRECTORY}/inference_results_obj.pkl",
        f"{EXPERIMENT_DIRECTORY}/inferred_params_GHIST_bottleneck.txt"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/inference_snakemake.py \
        --config {input.config} \
        --trained_weights {input.trained_weights} \
        --inference_obj_path {params.inference_obj_path} \
        --experiment_directory {params.experiment_directory}
        """
