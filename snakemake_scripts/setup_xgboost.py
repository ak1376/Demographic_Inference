from src.models import XGBoost
import argparse
import json
import pickle 
from src.utils import plot_loss_curves, visualizing_results
import numpy as np

def main(experiment_directory, xgb_model_config_file, features_file, color_shades, main_colors):

    # load in the model config json file

    with open(xgb_model_config_file, "r") as f:
        model_config = json.load(f)

    print(model_config.keys())

    # load in the features pickle file
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    with open(color_shades, "rb") as f:
        color_shades = pickle.load(f)
    
    with open(main_colors, "rb") as f:
        main_colors = pickle.load(f)

    
    n_estimators = model_config["xgb_hyperparameters"]["n_estimators"]
    learning_rate = model_config["xgb_hyperparameters"]["learning_rate"]
    max_depth = model_config["xgb_hyperparameters"]["max_depth"]
    verbosity = model_config["xgb_hyperparameters"]["verbosity"]
    train_percentage = model_config["xgb_hyperparameters"]["train_percentage"]
    
    # TODO: Have a separate config file for xgboost hyperparameters
    xgb_model = XGBoost(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        verbosity=verbosity,
        train_percentage=train_percentage
    )

    X_train = features['training']['features']
    y_train = features['training']['targets']

    X_val = features['validation']['features']
    y_val = features['validation']['targets']

    train_error, validation_error, y_pred_train, y_pred_test = xgb_model.train_and_validate(X_train, y_train, X_test=X_val, y_test=y_val)

    # Save the train error and validation error in a txt file
    with open(f"{experiment_directory}/xgb_model_errors.txt", "w") as f:
        f.write(f"Training error: {train_error}\n")
        f.write(f"Validation error: {validation_error}\n")

    train_losses_per_target, val_losses_per_target = xgb_model.get_epoch_losses()

    train_losses = np.mean(np.array(train_losses_per_target), axis = 0)
    val_losses = np.mean(np.array(val_losses_per_target), axis = 0)

    xgb_obj = {}
    xgb_obj['model'] = xgb_model
    xgb_obj['param_names'] = model_config['xgb_hyperparameters']['parameter_names']


    xgb_obj['training'] = {}
    xgb_obj['training']['features'] = X_train
    xgb_obj['training']['targets'] = y_train
    xgb_obj['training']['predictions'] = y_pred_train
    xgb_obj['train_losses'] = train_losses_per_target

    xgb_obj['validation'] = {}
    xgb_obj['validation']['features'] = X_val
    xgb_obj['validation']['targets'] = y_val
    xgb_obj['validation']['predictions'] = y_pred_test
    xgb_obj['val_losses'] = val_losses_per_target

    plot_loss_curves(train_losses, val_losses, f'{experiment_directory}/xgb_loss_curves.png')

    visualizing_results(xgb_obj, analysis = 'xgboost', save_loc = experiment_directory, color_shades = color_shades, main_colors = main_colors, stages = ['training', 'validation'])

    # save the xgb_obj as a pickle
    with open(f"{experiment_directory}/xgb_model_obj.pkl", "wb") as f:
        pickle.dump(xgb_obj, f)

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
