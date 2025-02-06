# setup_trainer.py

import pickle
import argparse
import json
import torch

from src.models import ShallowNN
from src.utils import plot_loss_curves
from src.train import MLPTrainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(experiment_directory, model_config_file, features_file, color_shades, main_colors):
    # Load model config (JSON)
    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    # Load preprocessed data (Pickle)
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    # Load color palettes/shades
    with open(color_shades, "rb") as f:
        color_shades = pickle.load(f)
    with open(main_colors, "rb") as f:
        main_colors = pickle.load(f)

    # Unpack hyperparameters
    nn_hyperparams = model_config["neural_net_hyperparameters"]

    # Instantiate the ShallowNN model
    mdl = ShallowNN(
        input_size=nn_hyperparams["input_size"],
        hidden_sizes=nn_hyperparams["hidden_size"],
        num_layers=nn_hyperparams["num_layers"],
        output_size=nn_hyperparams["output_size"],
        learning_rate=nn_hyperparams["learning_rate"],
        weight_decay=nn_hyperparams["weight_decay"],
        dropout_rate=nn_hyperparams["dropout_rate"],
        BatchNorm=nn_hyperparams["BatchNorm"],
    )

    # Create our trainer object
    trainer = MLPTrainer(
        experiment_directory,
        model_config,
        color_shades,
        main_colors,
        param_names=nn_hyperparams["parameter_names"],
    )

    # Optional: print debug info
    # print("Max Values in the dataset:")
    # print(f"  Training features max: {features['training']['features'].max()}")
    # print(f"  Training targets max:  {features['training']['targets'].max()}")
    # print(f"  Validation features max: {features['validation']['features'].max()}")
    # print(f"  Validation targets max:  {features['validation']['targets'].max()}")

    # Train
    snn_model, train_losses, val_losses = trainer.train(
        model=mdl,
        X_train=features["training"]["features"],
        y_train=features["training"]["targets"],
        X_val=features["validation"]["features"],
        y_val=features["validation"]["targets"],
    )

    # Predict + Visualization
    snn_results = trainer.predict(
        model=snn_model,
        training_data=features["training"]["features"],
        validation_data=features["validation"]["features"],
        training_targets=features["training"]["targets"],
        validation_targets=features["validation"]["targets"],
        visualize=True,
    )

    # Save the trained model
    torch.save(snn_model.state_dict(), f"{experiment_directory}/snn_model.pth")

    # Attach loss history to results
    snn_results["train_losses"] = train_losses
    snn_results["val_losses"] = val_losses

    # Plot and save loss curves
    plot_loss_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=f"{experiment_directory}/loss_curves.png"
    )

    # Save final results (Pickle)
    with open(f"{experiment_directory}/snn_results.pkl", "wb") as f:
        pickle.dump(snn_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_directory", type=str, required=True)
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--features_file", type=str, required=True)
    parser.add_argument("--color_shades", type=str, required=True)
    parser.add_argument("--main_colors", type=str, required=True)
    args = parser.parse_args()

    main(
        args.experiment_directory,
        args.model_config_file,
        args.features_file,
        args.color_shades,
        args.main_colors,
    )