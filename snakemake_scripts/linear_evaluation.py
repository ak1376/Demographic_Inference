import pickle 
import joblib
import json
from src.utils import visualizing_results, mean_squared_error
from src.models import LinearReg
import os
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet

def linear_evaluation(features_and_targets_filepath, model_config_path, color_shades_path, main_colors_path, experiment_config_filepath = None, model_directory = None, regression_type = "standard", alpha = 0.0, l1_ratio = 0.5):

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

    # Hyperparameter optimization for ridge, lasso, or elastic net
    if regression_type in ["ridge", "lasso", "elasticnet"]:
        X_train = features_and_targets["training"]["features"]
        y_train = features_and_targets["training"]["targets"]

        if regression_type == "ridge":
            model = Ridge()
            param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
        elif regression_type == "lasso":
            model = Lasso()
            param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
        elif regression_type == "elasticnet":
            model = ElasticNet()
            param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0], "l1_ratio": [0.1, 0.5, 0.9]}

        grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {regression_type}: {grid_search.best_params_}")

        # Update alpha and l1_ratio based on the best parameters
        alpha = grid_search.best_params_.get("alpha", alpha)
        l1_ratio = grid_search.best_params_.get("l1_ratio", l1_ratio)

    ## LINEAR REGRESSION
    linear_mdl = LinearReg(
        training_features=features_and_targets["training"]["features"],
        training_targets=features_and_targets["training"]["targets"],
        validation_features=features_and_targets["validation"]["features"],
        validation_targets=features_and_targets["validation"]["targets"],
        regression_type=regression_type, 
        alpha=alpha,
        l1_ratio=l1_ratio
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

    linear_mdl_obj["param_names"] = experiment_config['parameters_to_estimate']

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
        f"{model_directory}/linear_mdl_obj_{regression_type}.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(linear_mdl_obj, file)

    # Save rrmse_dict to a JSON file
    with open(f"{model_directory}/linear_model_error.json_{regression_type}", "w") as json_file:
        json.dump(rrmse_dict, json_file, indent=4)

    # targets
    visualizing_results(
        linear_mdl_obj,
        f"linear_results_{regression_type}",
        save_loc=model_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    joblib.dump(linear_mdl, f"{model_directory}/linear_regression_model_{regression_type}.pkl")

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

    parser.add_argument(
        "--regression_type",
        type=str,
        default="standard",
        help="Type of regression to perform (standard, ridge, lasso, elastic net)."
    )

    parser.add_argument(
    "--alpha",
    type=float,
    default=0.0,  # Set to 0.0 to reflect no regularization by default
    help="Regularization strength for Ridge, Lasso, or ElasticNet. Set to 0.0 for no regularization (standard linear regression)."
    )

    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,  # Default for ElasticNet; irrelevant for other types unless alpha > 0
        help="Mixing parameter for ElasticNet (only used if regression_type='elasticnet'). 0 <= l1_ratio <= 1. Irrelevant if alpha=0.0."
    )

    args = parser.parse_args()

    linear_evaluation(
        features_and_targets_filepath=args.features_and_targets_filepath,
        model_config_path=args.model_config_path,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file,
        experiment_config_filepath = args.experiment_config_filepath,
        model_directory = args.model_directory,
        regression_type = args.regression_type, 
        alpha = args.alpha,
        l1_ratio = args.l1_ratio
    )
