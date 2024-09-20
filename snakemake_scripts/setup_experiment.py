import pickle
from experiment_manager import Experiment_Manager
import os
import argparse
import json
from utils import create_color_scheme

def save_config(experiment_directory, experiment_config):
# def save_config(experiment_directory, experiment_config, model_config):
    # Save the full config
    with open(f"{experiment_directory}/config.json", "w") as json_file:
        json.dump(experiment_config, json_file, indent=4)


def create_experiment(config, experiment_name, experiment_directory):
# def create_experiment(config, model_config_file, experiment_name, experiment_directory):
    return Experiment_Manager(config, experiment_name, experiment_directory)


def main(config_file, model_config_file, sim_directory, experiment_name, experiment_directory):
    '''
    This function will take in the model directory and the simulation directory. I want to save the following in the experiment directory:
    1. The config file
    2. The experiment object
    3. The model config file

    I want to save the following to the sim directory:
    1. The color shades
    2. The main colors
    '''

    print("=====================================================")
    print(f'Experiment name: {experiment_name}')
    print(f'Experiment directory: {experiment_directory}')
    # print(f'Model config file: {model_config_file}')
    print("=====================================================")

    # Load the config file
    with open(config_file, "r") as f:
        config = json.load(f)


    # Create the Experiment_Manager object
    # experiment_directory will store the results for that particular experiment: 
    # 1. neural network specific results
    experiment = create_experiment(config, experiment_name = experiment_name, experiment_directory=experiment_directory)
    # experiment = create_experiment(config, model_config_file=model_config, experiment_name = experiment_name, experiment_directory=experiment_directory)

    # Save the config files
    save_config(sim_directory, config)

    # Now create the color scheme we will use for visualizing. Save them as pkl files
    color_shades, main_colors = create_color_scheme(len(config["parameter_names"]))

    with open(f'{sim_directory}/color_shades.pkl', "wb") as f:
        pickle.dump(color_shades, f)

    with open(f'{sim_directory}/main_colors.pkl', "wb") as f:
        pickle.dump(main_colors, f)

    # Define file paths
    config_file = os.path.join(sim_directory, "config.json")
    experiment_obj_file = os.path.join(sim_directory, "experiment_obj.pkl")
    # model_config_file = os.path.join(model_directory, "model_config.json")

    # Save the inference config file
    inference_config = config.copy()
    inference_config["experiment_directory"] = sim_directory
    with open(f"{sim_directory}/inference_config_file.json", "w") as f:
        json.dump(inference_config, f, indent=4)

    # Save the experiment object
    with open(experiment_obj_file, "wb") as f:
        pickle.dump(experiment, f)

    return config_file, experiment_obj_file, model_config_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create experiment configuration and object"
    )
    # parser.add_argument("--config_file", help="Path to the config file")
    # parser.add_argument(
    #     "--experiment_obj_file", help="Path to save the experiment object file"
    # )

    parser.add_argument("--config_file", help="Path to the config file")
    parser.add_argument("--model_config_file", help="Path to the model config file")
    parser.add_argument("--experiment_name", help="Name of the experiment")
    parser.add_argument("--experiment_directory", help="Directory to save the experiment object")
    parser.add_argument("--sim_directory", help="Directory containing the simulation data")

    args = parser.parse_args()


    # Running through Snakemake
    # output_dir = os.path.dirname(args.model_directory)
    config_file, experiment_obj_file, model_config_file = main(args.config_file, args.model_config_file, args.sim_directory, experiment_name=args.experiment_name, experiment_directory=args.experiment_directory)

    # Verify that the output files match the expected paths
    # assert (
    #     config_file == args.config_file
    # ), f"Config file mismatch: {config_file} != {args.config_file}"
    # assert (
    #     experiment_obj_file == args.experiment_obj_file
    # ), f"Experiment object file mismatch: {experiment_obj_file} != {args.experiment_obj_file}"

    print(f"Config saved to: {config_file}")
    print(f"Model config saved to: {model_config_file}")
    print(f"Experiment object saved to: {experiment_obj_file}")
