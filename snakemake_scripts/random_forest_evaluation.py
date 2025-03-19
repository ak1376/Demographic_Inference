import pickle
import joblib
import json
import os
import argparse

from src.utils import visualizing_results, mean_squared_error
from src.models import RandomForest  # Your RandomForest wrapper class
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error as mse_sklearn
import numpy as np


def random_forest_evaluation(
    features_and_targets_filepath,
    model_config_path,
    color_shades_path,
    main_colors_path,
    experiment_config_filepath=None,
    model_directory=None,
    feature_names=None,
    **rf_kwargs
):
    """
    Train and evaluate a Random Forest model using the RandomForest wrapper class.
    If no hyperparameters are specified (e.g., all are None), random search will be used
    to find the best hyperparameters within a defined parameter space.

    Now also computes per-parameter MSE for training and validation.
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
    # Suppose we store feature_names in the data or pass via argument
    if feature_names is None:
        if "feature_names" in features_and_targets:
            feature_names = features_and_targets["feature_names"]
        else:
            X_train = features_and_targets["training"]["features"]
            y_train = features_and_targets["training"]["targets"]
            feature_names = X_train.columns.tolist()
            target_names = y_train.columns.tolist()
    else:
        target_names = ["Unknown_Target"]

    model_config = json.load(open(model_config_path, "r"))
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors = pickle.load(open(main_colors_path, "rb"))

    X_train = features_and_targets["training"]["features"]
    y_train = features_and_targets["training"]["targets"]
    X_val = features_and_targets["validation"]["features"]
    y_val = features_and_targets["validation"]["targets"]

    # Optionally load experiment_config for param names
    experiment_config = None
    if experiment_config_filepath is not None:
        experiment_config = json.load(open(experiment_config_filepath, "r"))

    # ------------------------
    # 3) Optional: Hyperparameter Optimization via Random Search
    # ------------------------
    user_provided_params = any(value is not None for value in rf_kwargs.values())
    if not user_provided_params:
        print("\nNo hyperparameters specified. Running RandomizedSearchCV to find best hyperparameters...\n")
        param_dist = {
            "n_estimators": [20, 50, 100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10, 15, 20],
            "random_state": [42, 123, 2023, 295],
        }

        mse_scorer = make_scorer(mse_sklearn, greater_is_better=False)
        base_model = RandomForestRegressor()

        random_search = RandomizedSearchCV(
            estimator=base_model,
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
        rf_kwargs.update(best_params)

    # ------------------------
    # 4) Train the RandomForest wrapper
    # ------------------------
    random_forest_mdl = RandomForest(
        training_features=X_train,
        training_targets=y_train,
        validation_features=X_val,
        validation_targets=y_val,
        **rf_kwargs
    )

    # ------------------------
    # 5) Train and get predictions
    # ------------------------
    training_predictions, validation_predictions = random_forest_mdl.train_and_validate()
    print(f"\nRandom Forest predictions shape (training): {training_predictions.shape}")
    print(f"Random Forest predictions shape (validation): {validation_predictions.shape}\n")

    # ------------------------
    # 6) Organize results
    # ------------------------
    random_forest_mdl_obj = random_forest_mdl.organizing_results(
        features_and_targets,
        training_predictions,
        validation_predictions
    )

    # If param names exist in the experiment config
    if experiment_config is not None and "parameters_to_estimate" in experiment_config:
        random_forest_mdl_obj["param_names"] = experiment_config["parameters_to_estimate"]
    else:
        random_forest_mdl_obj["param_names"] = ["Unknown_Param"]

    # ------------------------
    # 7) Calculate MSE (rolled-up + per-parameter)
    # ------------------------
    rf_error_dict = {}
    # Overall MSE for entire (targets) vector
    rf_error_dict["training"] = mean_squared_error(
        y_true=random_forest_mdl_obj["training"]["targets"],
        y_pred=training_predictions
    )
    rf_error_dict["validation"] = mean_squared_error(
        y_true=random_forest_mdl_obj["validation"]["targets"],
        y_pred=validation_predictions
    )

    # Per-parameter MSE
    param_list = random_forest_mdl_obj["param_names"]
    rf_error_dict["training_mse"] = {}
    rf_error_dict["validation_mse"] = {}

    # For convenience, random_forest_mdl_obj["training"]["targets"] is shape (N, P)
    true_train = random_forest_mdl_obj["training"]["targets"]
    true_valid = random_forest_mdl_obj["validation"]["targets"]

    for i, param in enumerate(param_list):
        # training MSE for param i
        param_mse_train = ((true_train[:, i] - training_predictions[:, i])**2).mean()
        rf_error_dict["training_mse"][param] = param_mse_train

        # validation MSE for param i
        param_mse_valid = ((true_valid[:, i] - validation_predictions[:, i])**2).mean()
        rf_error_dict["validation_mse"][param] = param_mse_valid

    # ------------------------
    # 8) Save artifacts
    # ------------------------
    # 8a) Save the random_forest_mdl_obj
    with open(f"{model_directory}/random_forest_mdl_obj.pkl", "wb") as file:
        pickle.dump(random_forest_mdl_obj, file)

    # 8b) Save the overall + param-based MSE to JSON
    with open(f"{model_directory}/random_forest_model_error.json", "w") as json_file:
        json.dump(rf_error_dict, json_file, indent=4)

    # 8c) Visualize results
    visualizing_results(
        random_forest_mdl_obj,
        "random_forest_results",
        save_loc=model_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors
    )

    # 8d) Optionally, plot feature importances if your wrapper supports it
    random_forest_mdl.plot_feature_importances(
        feature_names, 
        target_names, 
        max_num_features=10, 
        save_path=f'{model_directory}/random_forest_feature_importances.png'
    )

    # 8e) Save the entire trained wrapper (including scikit-learn model) with joblib
    joblib.dump(random_forest_mdl, f"{model_directory}/random_forest_model.pkl")
    print("Random Forest model trained and saved. LFG!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Evaluation (with optional random search)")

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

    # Random Forest hyperparameters
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Number of trees in the forest. (Set None => random search pick.)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum depth of the tree. (Set None => random search pick.)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Seed used by the random number generator. (None => random search pick.)"
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=None,
        help="Minimum number of samples required to split an internal node. (None => random search pick.)"
    )

    args = parser.parse_args()

    # Collect RF-specific kwargs
    rf_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "min_samples_split": args.min_samples_split,
    }

    # Run the evaluation
    random_forest_evaluation(
        features_and_targets_filepath=args.features_and_targets_filepath,
        model_config_path=args.model_config_path,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file,
        experiment_config_filepath=args.experiment_config_filepath,
        model_directory=args.model_directory,
        **rf_kwargs
    )
