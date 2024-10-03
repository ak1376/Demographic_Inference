from src.models import ShallowNN
import torch
import pickle
import numpy as np
import json


def initialize_model(config):
    mdl = ShallowNN(
        input_size=config["neural_net_hyperparameters"]["input_size"],
        output_size=config["neural_net_hyperparameters"]["output_size"],
        hidden_size=config["neural_net_hyperparameters"]["hidden_size"],
        dropout_rate=config["neural_net_hyperparameters"]["dropout_rate"],
        weight_decay=config["neural_net_hyperparameters"]["weight_decay"],
    )
    return mdl


def load_trained_weights(mdl, path):
    mdl.load_state_dict(torch.load(path))
    return mdl


def load_inference_obj(path):
    with open(path, "rb") as file:
        inference_obj = pickle.load(file)
    return inference_obj


def evaluate_model(mdl, inference_obj):
    inference_features = torch.tensor(inference_obj["features"], dtype=torch.float32)
    mdl.eval()
    with torch.no_grad():
        predictions = mdl(inference_features)
    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trained_weights", type=str, required=True)
    parser.add_argument("--inference_obj", type=str, required=True)
    parser.add_argument("--experiment_directory", type=str, required=True)

    args = parser.parse_args()

    # Load in the config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize the model
    mdl = initialize_model(config)

    # Load the trained weights
    mdl = load_trained_weights(mdl, args.trained_weights)

    # Load the inference object

    inference_obj = load_inference_obj(args.inference_obj)

    # Evaluate the model
    inferred_params = evaluate_model(mdl, inference_obj)

    print(inferred_params)
    # Save the array as a text file
    np.savetxt(
        f"{args.experiment_directory}/inferred_params_GHIST_bottleneck.txt",
        inferred_params,
        delimiter=" ",
        fmt="%.5f",
    )


if __name__ == "__main__":
    main()
