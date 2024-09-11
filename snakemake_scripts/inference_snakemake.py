# run_inference.py

import json
from inference import Inference
import torch 
from models import ShallowNN
import pickle

def load_trained_weights(mdl, path):
    mdl.load_state_dict(torch.load(path))
    return mdl

def initialize_model(config):
    mdl = ShallowNN(
        input_size=config["neural_net_hyperparameters"]["input_size"],
        output_size=config["neural_net_hyperparameters"]["output_size"],
        hidden_size=config["neural_net_hyperparameters"]["hidden_size"],
        dropout_rate=config["neural_net_hyperparameters"]["dropout_rate"],
        weight_decay=config["neural_net_hyperparameters"]["weight_decay"],
        num_layers=config["neural_net_hyperparameters"]["num_layers"]
    )
    return mdl

def load_inference_obj(path):
    with open(path, "rb") as file:
        inference_obj = pickle.load(file)
    return inference_obj


# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_directory", type=str, required=True)
    parser.add_argument("--trained_weights", type=str, required=True)
    parser.add_argument("--inference_obj_path", type=str, required=True)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    inference_obj = Inference(
        vcf_filepath=config["vcf_filepath"],
        txt_filepath=config["txt_filepath"],
        popname=config["popname"],
        config=config,
        experiment_directory=args.experiment_directory
    )

    inference_obj.obtain_features()


    # Initialize the model
    mdl = initialize_model(config)

    # Load the trained weights
    mdl = load_trained_weights(mdl, args.trained_weights)

    # Load in the inference object 
    inference_obj_results = load_inference_obj(args.inference_obj_path)

    additional_features = None

    if config["use_FIM"]:
        additional_features = {}
        additional_features['upper_triangular_FIM'] = inference_obj_results['upper_triangular_FIM']

    # Evaluate the model
    inference_obj.evaluate_model(mdl, inference_obj_results, additional_features = additional_features)

