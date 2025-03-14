"""
xgboost_evaluation.py

An example script to train, evaluate, and optionally perform hyperparameter optimization
using Random Search on an XGBoost model. Follows a similar structure as the Random Forest example.

Usage examples:
1) Fixed hyperparameters (no search):
   python xgboost_evaluation.py \
       --features_and_targets_filepath path/to/features_and_targets.pkl \
       --model_config_path path/to/model_config.json \
       --color_shades_file path/to/color_shades.pkl \
       --main_colors_file path/to/main_colors.pkl \
       --experiment_config_filepath path/to/experiment_config.json \
       --model_directory path/to/desired/model/directory \
       --n_estimators 300 \
       --max_depth 4 \
       --learning_rate 0.05

2) Automatic random-search hyperparameter optimization (by omitting all XGBoost hyperparameters):
   python xgboost_evaluation.py \
       --features_and_targets_filepath path/to/features_and_targets.pkl \
       --model_config_path path/to/model_config.json \
       --color_shades_file path/to/color_shades.pkl \
       --main_colors_file path/to/main_colors.pkl \
       --experiment_config_filepath path/to/experiment_config.json \
       --model_directory path/to/desired/model/directory
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


def xgboost_evaluation(
    features_and_targets_filepath,
    model_config_path,
    color_shades_path,
    main_colors_path,
    experiment_config_filepath=None,
    model_directory=None,
    feature_names=None,              # <- NEW
    **xgb_kwargs
):
    """
    Train and evaluate an XGBoost model using the XGBoostReg wrapper class.
    If no hyperparameters are specified, random search will be used to find
    the best hyperparameters within a predefined parameter space.

    Parameters
    ----------
    features_and_targets_filepath : str
        Path to the pickled features and targets object.
    model_config_path : str
        Path to the model configuration JSON file.
    color_shades_path : str
        Path to the pickled color shades file.
    main_colors_path : str
        Path to the pickled main colors file.
    experiment_config_filepath : str, optional
        Path to the experiment configuration file (if any).
    model_directory : str, optional
        Where to save the trained model and results. If None, it will be built using
        the experiment config (like in the linear evaluation).
    xgb_kwargs : dict
        Additional keyword arguments for the XGBoostReg wrapper, e.g.:
            n_estimators, max_depth, learning_rate, verbosity, subsample, ...
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
        # If none provided, make some up or check if they're in the pickle
        if "feature_names" in features_and_targets:
            feature_names = features_and_targets["feature_names"]
        else:
            X_train = features_and_targets["training"]["features"]
            y_train = features_and_targets['training']['targets']
            feature_names = X_train.columns.tolist()
            target_names = y_train.columns.tolist()

    model_config = json.load(open(model_config_path, "r"))
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors = pickle.load(open(main_colors_path, "rb"))

    X_train = features_and_targets["training"]["features"]
    y_train = features_and_targets["training"]["targets"]
    X_val = features_and_targets["validation"]["features"]
    y_val = features_and_targets["validation"]["targets"]

    # ------------------------
    # 3) Optional: Hyperparameter Optimization via Random Search
    #    If user didn't specify ANY XGBoost hyperparams, run random search.
    # ------------------------
    user_provided_params = any(value is not None for value in xgb_kwargs.values())

    if not user_provided_params:
        print("\nNo XGBoost hyperparameters specified. "
              "Running RandomizedSearchCV to find best hyperparameters...\n")

        # Define the parameter distributions
        # Feel free to expand these lists or change the ranges
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [2, 3, 4, 5, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.2, 0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1, 2, 5],  # L2 reg
            "reg_alpha": [0, 0.1, 0.5],  # L1 reg
        }

        # Create a base model for searching
        base_xgb = XGBRegressor(objective="reg:squarederror")

        # Use negative MSE for scoring
        mse_scorer = make_scorer(mse_sklearn, greater_is_better=False)

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=10,            # Number of random configurations
            cv=3,                 # 3-fold cross-validation
            scoring=mse_scorer,   # Use neg MSE
            random_state=42,      # Reproducibility
            n_jobs=-1,            # Use all available CPUs
            verbose=1
        )

        # Fit on training data
        random_search.fit(X_train, y_train)

        # Get best hyperparams from the search
        best_params = random_search.best_params_
        print(f"\nBest hyperparameters found via RandomizedSearchCV: {best_params}\n")

        # Update xgb_kwargs with the best found parameters
        xgb_kwargs.update(best_params)

    # ------------------------
    # 4) Train model (user-provided or best from random search)
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

    # Attach parameter names if available in model_config
    if (
        "neural_net_hyperparameters" in model_config 
        and "parameter_names" in model_config["neural_net_hyperparameters"]
    ):
        xgb_mdl_obj["param_names"] = experiment_config["parameters_to_estimate"]
    else:
        xgb_mdl_obj["param_names"] = ["Unknown_Param"]

    # ------------------------
    # 7) Calculate training & validation errors
    # ------------------------
    xgb_error_dict = {}
    xgb_error_dict["training"] = mean_squared_error(
        y_true=xgb_mdl_obj["training"]["targets"],
        y_pred=training_predictions
    )
    xgb_error_dict["validation"] = mean_squared_error(
        y_true=xgb_mdl_obj["validation"]["targets"],
        y_pred=validation_predictions
    )

    # ------------------------
    # 8) Save artifacts
    # ------------------------
    with open(f"{model_directory}/xgb_mdl_obj.pkl", "wb") as file:
        pickle.dump(xgb_mdl_obj, file)

    with open(f"{model_directory}/xgb_model_error.json", "w") as json_file:
        json.dump(xgb_error_dict, json_file, indent=4)

    # Optional: visualize results if your function supports it
    visualizing_results(
        xgb_mdl_obj,
        "xgboost_results",
        save_loc=model_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors
    )

    xgb_model.plot_feature_importances(feature_names, target_names, max_num_features=10, save_path=f'{model_directory}/xgb_feature_importances.png')


    # Save the entire trained wrapper (including XGBoost model) with joblib
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

    # Parse arguments
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
