import pickle 
import joblib
import json
from src.utils import visualizing_results, root_mean_squared_error
from src.models import LinearReg

def linear_evaluation(features_and_targets_filepath, model_directory, experiment_config_path, color_shades_path, main_colors_path):
    
    features_and_targets = pickle.load(open(features_and_targets_filepath, "rb"))
    experiment_config = json.load(open(experiment_config_path, "r"))
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



    linear_mdl_obj = linear_mdl.organizing_results(
        features_and_targets,
        training_predictions,
        validation_predictions
    )

    linear_mdl_obj["param_names"] = experiment_config["parameter_names"]

    # Now calculate the linear model error

    rrmse_dict = {}
    rrmse_dict["training"] = root_mean_squared_error(
        y_true=linear_mdl_obj["training"]["targets"], y_pred=training_predictions
    )
    rrmse_dict["validation"] = root_mean_squared_error(
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
        "--model_directory",
        type=str,
        help="Directory where the model object will be saved",
    )

    parser.add_argument(
        "--experiment_config_filepath",
        type=str,
        help="Path to the experiment configuration file",
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

    args = parser.parse_args()

    linear_evaluation(
        features_and_targets_filepath=args.features_and_targets_filepath,
        model_directory=args.model_directory,
        experiment_config_path=args.experiment_config_filepath,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file
    )
