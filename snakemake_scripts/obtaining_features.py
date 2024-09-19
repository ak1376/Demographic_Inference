# obtain_features.py

import numpy as np
import time
import pickle
import joblib
from preprocess import Processor
from utils import (
    visualizing_results,
    root_mean_squared_error,
    calculate_and_save_rrmse
)

from models import LinearReg
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def obtain_features(
    experiment_config,
    model_directory,
    sim_directory,
    color_shades_file,
    main_colors_file,
): 

    # Load the experiment config
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    # Load in the color schemes and main colors
    with open(color_shades_file, "rb") as f:
        color_shades = pickle.load(f)

    with open(main_colors_file, "rb") as f:
        main_colors = pickle.load(f)

    processor = Processor(
        experiment_config,
        model_directory,
        recombination_rate=experiment_config["recombination_rate"],
        mutation_rate=experiment_config["mutation_rate"],
        window_length=experiment_config["window_length"],
    )

    # Now I want to define training, validation, and testing indices:

    # Generate all indices and shuffle them
    all_indices = np.arange(experiment_config["num_sims_pretrain"])
    np.random.shuffle(all_indices)

    # Split into training and validation indices
    n_train = int(
        experiment_config["training_percentage"]
        * experiment_config["num_sims_pretrain"]
    )

    training_indices = all_indices[:n_train]
    validation_indices = all_indices[n_train:]
    testing_indices = np.arange(experiment_config["num_sims_inference"])

    preprocessing_results_obj = {
        stage: {} for stage in ["training", "validation", "testing"]
    }

    # Create a copy of the preprocessing results object to have the normalized features (only used for plotting)
    preprocessing_results_obj_normalized = {
        stage: {} for stage in ["training", "validation", "testing"]
    }

    for stage, indices in [
        ("training", training_indices),
        ("validation", validation_indices),
        ("testing", testing_indices),
    ]:

        # Your existing process_and_save_data function

        # Call the remote function and get the ObjectRef
        print(f"Processing {stage} data")
        features, normalized_features, targets, upper_triangle_features = processor.pretrain_processing(
            indices
        )

        preprocessing_results_obj[stage][
            "predictions"
        ] = features  # This is for input to the ML model
        preprocessing_results_obj[stage]["targets"] = targets
        preprocessing_results_obj[stage][
            "upper_triangular_FIM"
        ] = upper_triangle_features

        preprocessing_results_obj_normalized[stage][
            "predictions"
        ] = normalized_features  # This is for plotting
        preprocessing_results_obj_normalized[stage]["targets"] = targets
        preprocessing_results_obj_normalized[stage][
            "upper_triangular_FIM"
        ] = upper_triangle_features

    preprocessing_results_obj["param_names"] = experiment_config["parameter_names"]
    preprocessing_results_obj_normalized["param_names"] = experiment_config["parameter_names"]

    # TODO: Calculate and save the rrmse_dict but removing the outliers from analysis
    rrmse_dict = calculate_and_save_rrmse(
        preprocessing_results_obj, save_path=f"{sim_directory}/rrmse_dict.json"
    )

    # Open a file to save the object
    print("SIMDIRECTORY", sim_directory)
    with open(
        f"{sim_directory}/preprocessing_results_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(preprocessing_results_obj, file)


    visualizing_results(
        preprocessing_results_obj_normalized,
        save_loc=sim_directory,
        analysis=f"preprocessing_results",
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    visualizing_results(
        preprocessing_results_obj_normalized,
        save_loc=sim_directory,
        analysis=f"preprocessing_results_testing",
        stages=["testing"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    ## LINEAR REGRESSION

    linear_mdl = LinearReg(
        training_features=preprocessing_results_obj["training"]["predictions"],
        training_targets=preprocessing_results_obj["training"]["targets"],
        validation_features=preprocessing_results_obj["validation"]["predictions"],
        validation_targets=preprocessing_results_obj["validation"]["targets"],
        testing_features=preprocessing_results_obj["testing"]["predictions"],
        testing_targets=preprocessing_results_obj["testing"]["targets"],
    )

    if experiment_config["use_FIM"]:

        upper_triangular_features = {}
        upper_triangular_features["training"] = preprocessing_results_obj["training"][
            "upper_triangular_FIM"
        ]
        upper_triangular_features["validation"] = preprocessing_results_obj[
            "validation"
        ]["upper_triangular_FIM"]
        upper_triangular_features["testing"] = preprocessing_results_obj["testing"][
            "upper_triangular_FIM"
        ]

        training_predictions, validation_predictions, testing_predictions = (
            linear_mdl.train_and_validate(upper_triangular_features)
        )

    else:
        training_predictions, validation_predictions, testing_predictions = (
            linear_mdl.train_and_validate()
        )

    linear_mdl_obj = linear_mdl.organizing_results(
        preprocessing_results_obj,
        training_predictions,
        validation_predictions,
        testing_predictions,
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
    rrmse_dict["testing"] = root_mean_squared_error(
        y_true=linear_mdl_obj["testing"]["targets"], y_pred=testing_predictions
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
    # torch.save(
    #     snn_model.state_dict(),
    #     f"{self.model_directory}/neural_network_model.pth",
    # )

    # Save the color shades and main colors for usage with the neural network

    print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--model_directory", type=str, required=True)
    parser.add_argument("--color_shades_file", type=str, required=True)
    parser.add_argument("--main_colors_file", type=str, required=True)
    args = parser.parse_args()

    obtain_features(
        experiment_config=args.experiment_config,
        model_directory=args.model_directory,
        sim_directory=args.sim_directory,
        color_shades_file=args.color_shades_file,
        main_colors_file=args.main_colors_file,
    )
