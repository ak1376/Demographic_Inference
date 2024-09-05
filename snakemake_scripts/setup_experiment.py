import pickle
from experiment_manager import Experiment_Manager
import os
import argparse
import json


def save_config(experiment_directory, experiment_config):
    # Save the full config
    with open(f"{experiment_directory}/config.json", "w") as json_file:
        json.dump(experiment_config, json_file, indent=4)

    # Save the model config
    with open(f"{experiment_directory}/model_config.json", "w") as json_file:
        json.dump(experiment_config["neural_net_hyperparameters"], json_file, indent=4)


def create_experiment(config):
    return Experiment_Manager(config)


def main(config_path):
    # The config_path is now the directory, so we need to construct the full path to the config file
    config_file = os.path.join(config_path, "config.json")

    # Load the config file
    with open(config_file, "r") as f:
        config = json.load(f)

    # Create the Experiment_Manager object
    linear_experiment = create_experiment(config)

    # The experiment directory is already created, so we don't need to create it again
    experiment_directory = config_path

    # Save the config files
    save_config(experiment_directory, config)

    # Define file paths
    config_file = os.path.join(experiment_directory, "config.json")
    experiment_obj_file = os.path.join(experiment_directory, "experiment_obj.pkl")
    model_config_file = os.path.join(experiment_directory, "model_config.json")

    # Save the inference config file
    inference_config = config.copy()
    inference_config["experiment_directory"] = experiment_directory
    with open(f"{experiment_directory}/inference_config_file.json", "w") as f:
        json.dump(inference_config, f, indent=4)

    # Save the experiment object
    with open(experiment_obj_file, "wb") as f:
        pickle.dump(linear_experiment, f)

    return config_file, experiment_obj_file, model_config_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create experiment configuration and object"
    )
    parser.add_argument("--config_file", help="Path to the config file")
    parser.add_argument(
        "--inference_config_file", help="Path to the inference config file"
    )
    parser.add_argument(
        "--experiment_obj_file", help="Path to save the experiment object file"
    )
    args = parser.parse_args()

    # Running through Snakemake
    output_dir = os.path.dirname(args.config_file)
    config_file, experiment_obj_file, model_config_file = main(output_dir)

    # Verify that the output files match the expected paths
    assert (
        config_file == args.config_file
    ), f"Config file mismatch: {config_file} != {args.config_file}"
    assert (
        experiment_obj_file == args.experiment_obj_file
    ), f"Experiment object file mismatch: {experiment_obj_file} != {args.experiment_obj_file}"

    print(f"Config saved to: {config_file}")
    print(f"Model config saved to: {model_config_file}")
    print(f"Experiment object saved to: {experiment_obj_file}")
