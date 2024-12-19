import pickle 
import joblib
import json
from src.utils import visualizing_results, mean_squared_error
from src.models import LinearReg
import os

def linear_evaluation(features_and_targets_filepath, model_config_path, color_shades_path, main_colors_path, experiment_config_filepath = None, model_directory = None):

    if model_directory is None:
        experiment_config = json.load(open(experiment_config_filepath, "r"))
        model_config = json.load(open(model_config_path, "r"))

        # Build the experiment directory
        EXPERIMENT_DIRECTORY = f"{experiment_config['demographic_model']}_dadi_analysis_{experiment_config['dadi_analysis']}_moments_analysis_{experiment_config['moments_analysis']}_momentsLD_analysis_{experiment_config['momentsLD_analysis']}_seed_{experiment_config['seed']}"

        # Build the experiment name
        EXPERIMENT_NAME = f"sims_pretrain_{experiment_config['num_sims_pretrain']}_sims_inference_{experiment_config['num_sims_inference']}_seed_{experiment_config['seed']}_num_replicates_{experiment_config['k']}_top_values_{experiment_config['top_values_k']}"
        
        # Process the hidden size parameter
        hidden_size = model_config['neural_net_hyperparameters']['hidden_size']
        if isinstance(hidden_size, list):
            hidden_size_str = "_".join(map(str, hidden_size))
        else:
            hidden_size_str = str(hidden_size)

        # Build the model directory
        model_directory = (
            f"{EXPERIMENT_DIRECTORY}/models/{EXPERIMENT_NAME}/"
            f"num_hidden_neurons_{hidden_size_str}_"
            f"num_hidden_layers_{model_config['neural_net_hyperparameters']['num_layers']}_"
            f"num_epochs_{model_config['neural_net_hyperparameters']['num_epochs']}_"
            f"dropout_value_{model_config['neural_net_hyperparameters']['dropout_rate']}_"
            f"weight_decay_{model_config['neural_net_hyperparameters']['weight_decay']}_"
            f"batch_size_{model_config['neural_net_hyperparameters']['batch_size']}_"
            f"EarlyStopping_{model_config['neural_net_hyperparameters']['EarlyStopping']}"
        )

    # Ensure the model directory exists
    os.makedirs(model_directory, exist_ok=True)

    print(f"Model directory created/verified: {model_directory}")
    
    features_and_targets = pickle.load(open(features_and_targets_filepath, "rb"))
    model_config = json.load(open(model_config_path, "r"))
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors = pickle.load(open(main_colors_path, "rb"))

    ## LINEAR REGRESSION

    linear_mdl = LinearReg(
        training_features=features_and_targets["training"]["features"],
        training_targets=features_and_targets["training"]["targets"],
        validation_features=features_and_targets["validation"]["features"],
        validation_targets=features_and_targets["validation"]["targets"],
    )

    training_predictions, validation_predictions = (
        linear_mdl.train_and_validate()
    )

    print(f'PREDICTIONS SHAPE TRAINING: {training_predictions.shape}')

    linear_mdl_obj = linear_mdl.organizing_results(
        features_and_targets,
        training_predictions,
        validation_predictions
    )

    linear_mdl_obj["param_names"] = model_config['neural_net_hyperparameters']['parameter_names']

    # Now calculate the linear model error

    rrmse_dict = {}
    rrmse_dict["training"] = mean_squared_error(
        y_true=linear_mdl_obj["training"]["targets"], y_pred=training_predictions
    )
    rrmse_dict["validation"] = mean_squared_error(
        y_true=linear_mdl_obj["validation"]["targets"], y_pred=validation_predictions
    )

    # Open a file to save the object
    with open(
        f"{model_directory}/linear_mdl_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(linear_mdl_obj, file)

    # Save rrmse_dict to a JSON file
    with open(f"{model_directory}/linear_model_error.json", "w") as json_file:
        json.dump(rrmse_dict, json_file, indent=4)

    # targets
    visualizing_results(
        linear_mdl_obj,
        "linear_results",
        save_loc=model_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    joblib.dump(linear_mdl, f"{model_directory}/linear_regression_model.pkl")

    print("Linear model trained LFG")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Linear Evaluation")
    parser.add_argument(
        "--features_and_targets_filepath",
        type=str,
        help="Path to the features and targets results object",
    )

    parser.add_argument(
        "--model_config_path",
        type=str,
        help="Path to the model configuration file",
    )

    parser.add_argument(
        "--color_shades_file",
        type=str,
        help="Path to the color shades file",
    )

    parser.add_argument(
        "--main_colors_file",
        type=str,
        help="Path to the main colors file",
    )

    # Optional arguments
    parser.add_argument(
        "--experiment_config_filepath",
        type=str,
        default=None,
        help="Path to the experiment configuration file (optional)."
    )
    parser.add_argument(
        "--model_directory",
        type=str,
        default=None,
        help="Path to the model directory (optional)."
    )

    args = parser.parse_args()

    linear_evaluation(
        features_and_targets_filepath=args.features_and_targets_filepath,
        model_config_path=args.model_config_path,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file,
        experiment_config_filepath = args.experiment_config_filepath,
        model_directory = args.model_directory
    )
