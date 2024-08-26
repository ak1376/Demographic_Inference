import pickle
from experiment_manager import Experiment_Manager
import os
import argparse
import json

def save_config(experiment_directory, experiment_config):
    # Open a file in write mode and save the dictionary as JSON
    with open(f"{experiment_directory}/config.json", "w") as json_file:
        json.dump(experiment_config, json_file, indent=4)  # indent=4 makes the JSON file more readable

    with open(f"{experiment_directory}/model_config.json", "w") as json_file:
        json.dump(experiment_config["neural_net_hyperparameters"], json_file, indent=4)  # indent=4 makes the JSON file more readable        

def create_experiment(config):
    return Experiment_Manager(config)

def main(output_dir=None):
    upper_bound_params = {
    "N0": 10000,
    "Nb": 5000,
    "N_recover": 7000,
    "t_bottleneck_end": 1000,
    "t_bottleneck_start": 2000
    }
    lower_bound_params = {
    "N0": 8000,
    "Nb": 4000,
    "N_recover": 6000,
    "t_bottleneck_end": 800,
    "t_bottleneck_start": 1500
    }
    model_config = {
    "input_size": 8,
    "hidden_size": 1000,
    "output_size": 4,
    "num_epochs": 1000,
    "learning_rate": 3e-4,
    "num_layers": 3,
    "dropout_rate": 0,
    "weight_decay": 0
    }

    config = {
        "upper_bound_params": upper_bound_params,
        "lower_bound_params": lower_bound_params,
        "num_sims_pretrain": 1000,
        "num_sims_inference": 1000,
        "num_samples": 20,
        "experiment_name": "snakemake",
        "dadi_analysis": True,
        "moments_analysis": True,
        "momentsLD_analysis": False,
        "num_windows": 50,
        "window_length": 1e5,
        "maxiter": 100,
        "genome_length": 1e7,
        "mutation_rate": 1.26e-8,
        "recombination_rate": 1.007e-8,
        "seed": 295,
        "normalization": False,
        "remove_outliers": True,
        "neural_net_hyperparameters": model_config
    }
    # with open(output_path, 'wb') as f:
    #     pickle.dump(config, f)

    # Create the Experiment_Manager object
    linear_experiment = create_experiment(config)

    linear_experiment.create_directory(config['experiment_name'])
    
    # Define file paths
    config_file = os.path.join(f"{linear_experiment.experiment_directory}", "config.json")
    print(f'CONFIG FILE PATH: {config_file}')
    experiment_obj_file = os.path.join(f"{linear_experiment.experiment_directory}", "experiment_obj.pkl")
    print("YIKES")
    # Save the config
    save_config(linear_experiment.experiment_directory, config)
    
    # Save the experiment object
    with open(experiment_obj_file, "wb") as f:
        pickle.dump(linear_experiment, f)
    
    return config_file, experiment_obj_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create experiment configuration and object")
    parser.add_argument("--config_file", help="Path to save the config file")
    parser.add_argument("--experiment_obj_file", help="Path to save the experiment object file")
    args = parser.parse_args()

    if args.config_file and args.experiment_obj_file:
        # Running through Snakemake
        output_dir = os.path.dirname(args.config_file)
        config_file, experiment_obj_file = main(output_dir)
        
        # Verify that the output files match the expected paths
        print(config_file)
        print(args.config_file)
        assert config_file == args.config_file, f"Config file mismatch: {config_file} != {args.config_file}"
        assert experiment_obj_file == args.experiment_obj_file, f"Experiment object file mismatch: {experiment_obj_file} != {args.experiment_obj_file}"
    else:
        # Fallback for running script directly (not through Snakemake)
        config_file, experiment_obj_file = main()
    
    print(f"Config saved to: {config_file}")
    print(f"Experiment object saved to: {experiment_obj_file}")






# import pickle
# from experiment_manager import Experiment_Manager

# def save_config(output_path):
#     upper_bound_params = {
#         "N0": 10000,
#         "Nb": 2000,
#         "N_recover": 8000,
#         "t_bottleneck_end": 1000,
#         "t_bottleneck_start": 2000
#     }
#     lower_bound_params = {
#         "N0": 8000,
#         "Nb": 1000,
#         "N_recover": 4000,
#         "t_bottleneck_end": 800,
#         "t_bottleneck_start": 1500
#     }
#     model_config = {
#         "input_size": 8,
#         "hidden_size": 1000,
#         "output_size": 4,
#         "num_epochs": 1000,
#         "learning_rate": 3e-4,
#         "num_layers": 3,
#         "dropout_rate": 0,
#         "weight_decay": 0
#     }
#     config = {
#         "upper_bound_params": upper_bound_params,
#         "lower_bound_params": lower_bound_params,
#         "num_sims_pretrain": 1000,
#         "num_sims_inference": 1000,
#         "num_samples": 20,
#         "experiment_name": "without_FIM",
#         "dadi_analysis": True,
#         "moments_analysis": True,
#         "momentsLD_analysis": False,
#         "num_windows": 50,
#         "window_length": 1e5,
#         "maxiter": 100,
#         "genome_length": 1e7,
#         "mutation_rate": 1.26e-8,
#         "recombination_rate": 1.007e-8,
#         "seed": 295,
#         "normalization": False,
#         "remove_outliers": True,
#         "neural_net_hyperparameters": model_config
#     }
#     with open(output_path, 'wb') as f:
#         pickle.dump(config, f)

#     return config

# if __name__ == "__main__":
#     # Fallback for running script directly (not through Snakemake)
#     config_file = "output/config.pkl"
#     experiment_obj_file = "output/experiment_obj.pkl"
    
#     config = save_config(config_file)
    
#     # Create the Experiment_Manager object and save it
#     linear_experiment = Experiment_Manager(config)
#     with open(experiment_obj_file, "wb") as f:
#         pickle.dump(linear_experiment, f)