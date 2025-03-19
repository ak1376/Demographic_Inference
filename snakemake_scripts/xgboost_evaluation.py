"""
xgboost_evaluation.py

An example script to train, evaluate, and optionally perform hyperparameter optimization
using Random Search on an XGBoost model. Follows a similar structure as the Random Forest example.

Now includes per-parameter MSE for both training and validation.
"""

import pickle
import joblib
import json
import os
import argparse

from src.utils import visualizing_results, mean_squared_error
from src.models import XGBoostReg  # Your XGBoost wrapper class (or rename as needed)
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error as mse_sklearn
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def xgboost_evaluation(
    features_and_targets_filepath,
    model_config_path,
    color_shades_path,
    main_colors_path,
    experiment_config_filepath=None,
    model_directory=None,
    feature_names=None,  # might be used for plotting importances
    **xgb_kwargs
):
    """
    Train and evaluate an XGBoost model using the XGBoostReg wrapper class.
    If no hyperparameters are specified, random search will be used to find
    the best hyperparameters within a predefined parameter space.

    Now also computes per-parameter MSE for training & validation.
    """

    # ------------------------
    # 1) Build or use model directory
    # ------------------------
    if model_directory is None:
        experiment_config = json.load(open(experiment_config_filepath, "r"))
        model_config = json.load(open(model_config_path, "r"))

        # Build the experiment directory (customize as needed)
        EXPERIMENT_DIRECTORY = (
            f"{experiment_config['demographic_model']}_dadi_analysis_{experiment_config['dadi_analysis']}"
            f"_moments_analysis_{experiment_config['moments_analysis']}"
            f"_momentsLD_analysis_{experiment_config['momentsLD_analysis']}"
            f"_seed_{experiment_config['seed']}"
        )

        # Build the experiment name (customize as needed)
        EXPERIMENT_NAME = (
            f"sims_pretrain_{experiment_config['num_sims_pretrain']}_"
            f"sims_inference_{experiment_config['num_sims_inference']}_"
            f"seed_{experiment_config['seed']}_"
            f"num_replicates_{experiment_config['k']}_"
            f"top_values_{experiment_config['top_values_k']}"
        )

        # Process the hidden size parameter (example usage from model_config)
        hidden_size = model_config["neural_net_hyperparameters"]["hidden_size"]
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

    # ------------------------
    # 2) Load data and configs
    # ------------------------
    features_and_targets = pickle.load(open(features_and_targets_filepath, "rb"))

    # Possibly set feature_names from data if not provided
    if feature_names is None:
        if "feature_names" in features_and_targets:
            feature_names = features_and_targets["feature_names"]
        else:
            X_train_temp = features_and_targets["training"]["features"]
            y_train_temp = features_and_targets["training"]["targets"]
            feature_names = X_train_temp.columns.tolist()
            target_names = y_train_temp.columns.tolist()
    else:
        # If we do have feature_names passed in,
        # (or you can handle target_names similarly)
        target_names = ["Unknown_Param"]

    model_config = json.load(open(model_config_path, "r"))
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors = pickle.load(open(main_colors_path, "rb"))

    X_train = features_and_targets["training"]["features"]
    y_train = features_and_targets["training"]["targets"]
    X_val = features_and_targets["validation"]["features"]
    y_val = features_and_targets["validation"]["targets"]

    # Possibly load experiment_config for param names
    if experiment_config_filepath is not None:
        experiment_config = json.load(open(experiment_config_filepath, "r"))
    else:
        experiment_config = None

    # ------------------------
    # 3) Optional: Hyperparameter Optimization via Random Search
    #    If user didn't specify ANY XGBoost hyperparams -> random search
    # ------------------------
    user_provided_params = any(value is not None for value in xgb_kwargs.values())

    if not user_provided_params:
        print("\nNo XGBoost hyperparameters specified. Running RandomizedSearchCV...\n")

        # Define the parameter distributions
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [2, 3, 4, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1, 2, 5],
            "reg_alpha": [0, 0.1, 0.5],
        }

        from xgboost import XGBRegressor
        from sklearn.metrics import make_scorer, mean_squared_error as mse_sklearn
        from sklearn.model_selection import RandomizedSearchCV

        base_xgb = XGBRegressor(objective="reg:squarederror")
        mse_scorer = make_scorer(mse_sklearn, greater_is_better=False)

        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring=mse_scorer,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        print(f"\nBest hyperparameters found via RandomizedSearchCV: {best_params}\n")
        xgb_kwargs.update(best_params)

    # ------------------------
    # 4) Instantiate your XGBoostReg wrapper
    # ------------------------
    xgb_model = XGBoostReg(
        training_features=X_train,
        training_targets=y_train,
        validation_features=X_val,
        validation_targets=y_val,
        **xgb_kwargs
    )

    # ------------------------
    # 5) Train and predict
    # ------------------------
    training_predictions, validation_predictions = xgb_model.train_and_validate()
    print(f"\nXGBoost predictions shape (training): {training_predictions.shape}")
    print(f"XGBoost predictions shape (validation): {validation_predictions.shape}\n")

    # ------------------------
    # 6) Organize results
    # ------------------------
    xgb_mdl_obj = xgb_model.organizing_results(
        features_and_targets,
        training_predictions,
        validation_predictions
    )

    # If param names exist in the experiment config
    if experiment_config is not None and "parameters_to_estimate" in experiment_config:
        xgb_mdl_obj["param_names"] = experiment_config["parameters_to_estimate"]
    else:
        xgb_mdl_obj["param_names"] = ["Unknown_Param"]

    # ------------------------
    # 7) Calculate training & validation errors
    # ------------------------
    xgb_error_dict = {}

    # a) Overall MSE for entire vector
    xgb_error_dict["training"] = mean_squared_error(
        y_true=xgb_mdl_obj["training"]["targets"],
        y_pred=training_predictions
    )
    xgb_error_dict["validation"] = mean_squared_error(
        y_true=xgb_mdl_obj["validation"]["targets"],
        y_pred=validation_predictions
    )

    # b) Per-parameter MSE
    param_list = xgb_mdl_obj["param_names"]
    xgb_error_dict["training_mse"] = {}
    xgb_error_dict["validation_mse"] = {}

    true_train = np.array(xgb_mdl_obj["training"]["targets"])
    true_valid = np.array(xgb_mdl_obj["validation"]["targets"])
    train_preds_arr = np.array(training_predictions)
    val_preds_arr = np.array(validation_predictions)

    for i, param in enumerate(param_list):
        param_mse_train = ((true_train[:, i] - train_preds_arr[:, i]) ** 2).mean()
        xgb_error_dict["training_mse"][param] = param_mse_train

        param_mse_valid = ((true_valid[:, i] - val_preds_arr[:, i]) ** 2).mean()
        xgb_error_dict["validation_mse"][param] = param_mse_valid

    # ------------------------
    # 8) Save artifacts
    # ------------------------
    # 8a) Save the xgb_mdl_obj
    with open(f"{model_directory}/xgb_mdl_obj.pkl", "wb") as file:
        pickle.dump(xgb_mdl_obj, file)

    # 8b) Save the overall + param-based MSE to JSON
    with open(f"{model_directory}/xgb_model_error.json", "w") as json_file:
        json.dump(xgb_error_dict, json_file, indent=4)

    # 8c) Visualize results
    visualizing_results(
        xgb_mdl_obj,
        analysis="xgboost_results",
        save_loc=model_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors
    )

    # 8d) (Optional) plot feature importances if your XGBoostReg supports it
    xgb_model.plot_feature_importances(
        feature_names,
        target_names,
        max_num_features=10,
        save_path=f'{model_directory}/xgb_feature_importances.png'
    )

    # 8e) Save the entire trained wrapper with joblib
    joblib.dump(xgb_model, f"{model_directory}/xgb_model.pkl")

    print("XGBoost model trained and saved. LFG!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Evaluation (with optional random search)")

    # Required arguments
    parser.add_argument(
        "--features_and_targets_filepath",
        type=str,
        required=True,
        help="Path to the features and targets results object (pickle file)."
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="Path to the model configuration file (JSON)."
    )
    parser.add_argument(
        "--color_shades_file",
        type=str,
        required=True,
        help="Path to the color shades file (pickle)."
    )
    parser.add_argument(
        "--main_colors_file",
        type=str,
        required=True,
        help="Path to the main colors file (pickle)."
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

    # XGBoost hyperparameters
    # If these remain None, random search is triggered for all params.
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Number of boosting rounds (trees). (None => random search pick.)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum tree depth for base learners. (None => random search pick.)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Boosting learning rate. (None => random search pick.)"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Subsample ratio of the training instances. (None => random search pick.)"
    )
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=None,
        help="Subsample ratio of columns when constructing each tree. (None => random search pick.)"
    )
    parser.add_argument(
        "--min_child_weight",
        type=float,
        default=None,
        help="Minimum sum of instance weight(hessian) needed in a child. (None => random search pick.)"
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=None,
        help="L2 regularization term on weights (None => random search pick.)"
    )
    parser.add_argument(
        "--reg_alpha",
        type=float,
        default=None,
        help="L1 regularization term on weights (None => random search pick.)"
    )

    args = parser.parse_args()

    # Collect XGBoost kwargs
    xgb_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
    }

    # Run the evaluation
    xgboost_evaluation(
        features_and_targets_filepath=args.features_and_targets_filepath,
        model_config_path=args.model_config_path,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file,
        experiment_config_filepath=args.experiment_config_filepath,
        model_directory=args.model_directory,
        **xgb_kwargs
    )
