import pickle
import argparse
from models import ShallowNN
import os
from utils import visualizing_results, calculate_model_errors
import json
import torch
from train import Trainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main(experiment_directory, model_config_file, features_file, color_shades, main_colors, additional_features_file, use_FIM=True):
    # Load model config
    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    # Load features
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    # Load the list back
    with open(color_shades, 'rb') as f:
        color_shades = pickle.load(f)

    with open(main_colors, 'rb') as f:
        main_colors = pickle.load(f)

    with open(additional_features_file, 'rb') as f:
        additional_features = pickle.load(f)



    trainer = Trainer(experiment_directory, model_config, color_shades, main_colors, param_names=model_config['neural_net_hyperparameters']['parameter_names'], use_FIM=use_FIM)

    # Train the model
    snn_model, train_losses, val_losses = trainer.train(
        training_data = features["training"]["features"],
        training_targets = features["training"]["targets"],
        validation_data = features["validation"]["features"],
        validation_targets = features["validation"]["targets"],
        additional_features=additional_features,
        visualize=True
    )
    
    snn_results = trainer.predict(
        snn_model = snn_model,
        training_data = features["training"]["features"],
        validation_data = features["validation"]["features"],
        training_targets = features["training"]["targets"],
        validation_targets = features["validation"]["targets"],
        additional_features=additional_features,
        visualize=True
    )

    # Save the trained model
    torch.save(snn_model.state_dict(), f"{experiment_directory}/snn_model.pth")

    snn_results["train_losses"] = train_losses
    snn_results["val_losses"] = val_losses

    with open(f"{experiment_directory}/snn_results.pkl", "wb") as f:
        pickle.dump(snn_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_directory", type=str, required=True)
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--features_file", type=str, required=True)
    parser.add_argument("--color_shades", type=str, required=True)
    parser.add_argument("--main_colors", type=str, required=True)
    parser.add_argument("--additional_features_file", type=str, required=True)
    parser.add_argument("--use_FIM", type=str2bool, default=True)
    args = parser.parse_args()

    main(
        args.experiment_directory,
        args.model_config_file,
        args.features_file,
        args.color_shades,
        args.main_colors,
        args.additional_features_file,
        args.use_FIM
    )
