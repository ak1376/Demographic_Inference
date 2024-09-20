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
        experiment_directory = sim_directory,
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

    print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--color_shades_file", type=str, required=True)
    parser.add_argument("--main_colors_file", type=str, required=True)
    args = parser.parse_args()

    obtain_features(
        experiment_config=args.experiment_config,
        sim_directory=args.sim_directory,
        color_shades_file=args.color_shades_file,
        main_colors_file=args.main_colors_file,
    )
