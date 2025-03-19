import pickle
import joblib
import json
from src.utils import visualizing_results, mean_squared_error
from src.models import LinearReg
import os
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet

def linear_evaluation(
    features_and_targets_filepath,
    model_config_path,
    color_shades_path,
    main_colors_path, 
    experiment_config_filepath=None,
    model_directory=None, 
    regression_type="standard",
    alpha=0.0,
    l1_ratio=0.5
):
    """
    Train a linear regression model (standard, ridge, lasso, or elasticnet)
    on the features & targets, optionally do a hyperparameter search,
    then compute MSE (overall + per-parameter), save results, and create plots.
    """

    # 1) Determine model_directory if not provided
    if model_directory is None:
        experiment_config = json.load(open(experiment_config_filepath, "r"))
        model_config = json.load(open(model_config_path, "r"))

        # Build the experiment directory
        EXPERIMENT_DIRECTORY = (
            f"{experiment_config['demographic_model']}_"
            f"dadi_analysis_{experiment_config['dadi_analysis']}_"
            f"moments_analysis_{experiment_config['moments_analysis']}_"
            f"momentsLD_analysis_{experiment_config['momentsLD_analysis']}_"
            f"seed_{experiment_config['seed']}"
        )

        # Build the experiment name
        EXPERIMENT_NAME = (
            f"sims_pretrain_{experiment_config['num_sims_pretrain']}"
            f"_sims_inference_{experiment_config['num_sims_inference']}"
            f"_seed_{experiment_config['seed']}"
            f"_num_replicates_{experiment_config['k']}"
            f"_top_values_{experiment_config['top_values_k']}"
        )
        
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

    os.makedirs(model_directory, exist_ok=True)
    print(f"Model directory created/verified: {model_directory}")

    # 2) Load data and configs
    features_and_targets = pickle.load(open(features_and_targets_filepath, "rb"))
    model_config = json.load(open(model_config_path, "r"))
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors = pickle.load(open(main_colors_path, "rb"))

    # Possibly do hyperparameter optimization if ridge/lasso/elasticnet
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
            param_grid = {
                "alpha": [0.1, 1.0, 10.0, 100.0],
                "l1_ratio": [0.1, 0.5, 0.9]
            }

        grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {regression_type}: {grid_search.best_params_}")

        # Update alpha/l1_ratio based on best params
        alpha = grid_search.best_params_.get("alpha", alpha)
        l1_ratio = grid_search.best_params_.get("l1_ratio", l1_ratio)

    # 3) Instantiate + train the linear model
    linear_mdl = LinearReg(
        training_features=features_and_targets["training"]["features"],
        training_targets=features_and_targets["training"]["targets"],
        validation_features=features_and_targets["validation"]["features"],
        validation_targets=features_and_targets["validation"]["targets"],
        regression_type=regression_type,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

    training_predictions, validation_predictions = linear_mdl.train_and_validate()
    print(f'PREDICTIONS SHAPE TRAINING: {training_predictions.shape}')

    # 4) Organize results and store param names
    experiment_config = json.load(open(experiment_config_filepath, "r"))
    linear_mdl_obj = linear_mdl.organizing_results(
        features_and_targets,
        training_predictions,
        validation_predictions
    )
    linear_mdl_obj["param_names"] = experiment_config['parameters_to_estimate']  # e.g. ['Na','N1','N2','t_split']

    # 5) Compute overall MSE (RMSE) for entire vector
    rrmse_dict = {}
    rrmse_dict["training"] = mean_squared_error(
        y_true=linear_mdl_obj["training"]["targets"],
        y_pred=training_predictions
    )
    rrmse_dict["validation"] = mean_squared_error(
        y_true=linear_mdl_obj["validation"]["targets"],
        y_pred=validation_predictions
    )

    # 6) Compute per-parameter MSE
    param_list = linear_mdl_obj["param_names"]
    # We'll store param-based MSE in sub-dicts
    rrmse_dict["training_mse"] = {}
    rrmse_dict["validation_mse"] = {}

    true_train = linear_mdl_obj["training"]["targets"]  # shape (N, #params)
    true_valid = linear_mdl_obj["validation"]["targets"]  # shape (M, #params)

    for i, param in enumerate(param_list):
        # Training
        param_mse_train = ((true_train[:, i] - training_predictions[:, i])**2).mean()
        rrmse_dict["training_mse"][param] = param_mse_train
        # Validation
        param_mse_valid = ((true_valid[:, i] - validation_predictions[:, i])**2).mean()
        rrmse_dict["validation_mse"][param] = param_mse_valid

    # 7) Plot + Save artifacts
    # 7a) Save the linear_mdl_obj
    with open(f"{model_directory}/linear_mdl_obj_{regression_type}.pkl", "wb") as file:
        pickle.dump(linear_mdl_obj, file)

    # 7b) Save the overall + param-based MSE to JSON
    with open(f"{model_directory}/linear_model_error_{regression_type}.json", "w") as json_file:
        json.dump(rrmse_dict, json_file, indent=4)

    # 7c) Visualize results
    visualizing_results(
        linear_mdl_obj,
        f"linear_results_{regression_type}",
        save_loc=model_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    # 7d) Finally, save the regression model as well
    joblib.dump(linear_mdl, f"{model_directory}/linear_regression_model_{regression_type}.pkl")
    print("Linear model trained. LFG!")


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
        default=0.0,
        help="Regularization strength for Ridge, Lasso, or ElasticNet. "
             "Set to 0.0 for no regularization (standard linear regression)."
    )

    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,
        help="Mixing parameter for ElasticNet (only used if regression_type='elasticnet'). "
             "0 <= l1_ratio <= 1. Irrelevant if alpha=0.0."
    )

    args = parser.parse_args()

    linear_evaluation(
        features_and_targets_filepath=args.features_and_targets_filepath,
        model_config_path=args.model_config_path,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file,
        experiment_config_filepath=args.experiment_config_filepath,
        model_directory=args.model_directory,
        regression_type=args.regression_type,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio
    )
