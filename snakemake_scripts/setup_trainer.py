import pickle
import argparse
from src.models import ShallowNN
from src.utils import plot_loss_curves
import json
import torch
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


def main(
    experiment_directory, model_config_file, features_file, color_shades, main_colors
):
    # Load model config
    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    # Load features
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    # Load the list back
    with open(color_shades, "rb") as f:
        color_shades = pickle.load(f)

    with open(main_colors, "rb") as f:
        main_colors = pickle.load(f)

    mdl = ShallowNN(
        input_size=model_config["neural_net_hyperparameters"]["input_size"],
        hidden_sizes=model_config["neural_net_hyperparameters"]["hidden_size"],
        num_layers=model_config["neural_net_hyperparameters"]["num_layers"],
        output_size=model_config["neural_net_hyperparameters"]["output_size"],
        learning_rate=model_config["neural_net_hyperparameters"]["learning_rate"],
        weight_decay=model_config["neural_net_hyperparameters"]["weight_decay"],
        dropout_rate=model_config["neural_net_hyperparameters"]["dropout_rate"],
        BatchNorm=model_config["neural_net_hyperparameters"]["BatchNorm"],
    )

    
    trainer = MLPTrainer(
        experiment_directory,
        model_config,
        color_shades,
        main_colors,
        param_names=model_config["neural_net_hyperparameters"]["parameter_names"],
    )

    snn_model, train_losses, val_losses = trainer.train(
        model = mdl,
        X_train = features["training"]["features"],
        y_train = features["training"]["targets"],
        X_val = features["validation"]["features"],
        y_val = features["validation"]["targets"],
    )

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

    snn_results["train_losses"] = train_losses
    snn_results["val_losses"] = val_losses


    plot_loss_curves(train_losses = train_losses, val_losses = val_losses, save_path = f'{experiment_directory}/loss_curves.png')

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
