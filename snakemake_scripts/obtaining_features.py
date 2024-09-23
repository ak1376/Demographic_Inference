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
    sim_directory
): 

    # Load the experiment config
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

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

    for stage, indices in [
        ("training", training_indices),
        ("validation", validation_indices),
        ("testing", testing_indices),
    ]:

        # Your existing process_and_save_data function

        # Call the remote function and get the ObjectRef
        print(f"Processing {stage} data")
        features, targets, upper_triangle_features = processor.pretrain_processing(
            indices
        )

        preprocessing_results_obj[stage]["predictions"] = features
        preprocessing_results_obj[stage]["targets"] = targets
        preprocessing_results_obj[stage]["upper_triangular_FIM"] = upper_triangle_features

        features = features.reshape(features.shape[0], -1)
        print(f"Features shape: {features.shape}")

        # Concatenate the features and upper triangle features column-wise

        if experiment_config['use_FIM'] == True:
            upper_triangle_features = upper_triangle_features.reshape(upper_triangle_features.shape[0], -1) # type:ignore
            preprocessing_results_obj[stage]["upper_triangular_FIM_reshape"] = upper_triangle_features
            all_features = np.concatenate((features, upper_triangle_features), axis=1) #type:ignore

        else:
            all_features = features
    
        # Save all_features and targets to a file
        np.save(f"{sim_directory}/{stage}_features.npy", all_features)

        #TODO: FIX 
        targets = targets[:,0,0,:] # This extracts the ground truth values. Later we can always tile and get it to match the the features shape. 
        np.save(f"{sim_directory}/{stage}_targets.npy", targets)

    #TODO: Need to add fields to the below object
    # # Open a file to save the object
    with open(
        f"{sim_directory}/preprocessing_results_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(preprocessing_results_obj, file)

    #     if experiment_config["normalization"] == True:
    #         feature_values = normalized_features
    #         targets_values = normalized_targets
    #     else:
    #         feature_values = features
    #         targets_values = targets

    #     preprocessing_results_obj[stage][
    #         "predictions"
    #     ] = feature_values  # This is for input to the ML model
    #     preprocessing_results_obj[stage]["targets"] = targets_values
    #     preprocessing_results_obj[stage][
    #         "upper_triangular_FIM"
    #     ] = upper_triangle_features

        
    #     # Object for plotting

    #     if experiment_config["normalization"] == True:

    #         preprocessing_results_obj_plotting[stage][
    #             "predictions"
    #         ] = normalized_features  # This is for plotting
    #         preprocessing_results_obj_plotting[stage]["targets"] = normalized_targets
    #         preprocessing_results_obj_plotting[stage][
    #             "upper_triangular_FIM"
    #         ] = upper_triangle_features
    #     else:


    # preprocessing_results_obj["param_names"] = experiment_config["parameter_names"]
    # preprocessing_results_obj_normalized["param_names"] = experiment_config["parameter_names"]

    # TODO: ALL THE CODE BELOW SHOULD BE MOVED TO THE EXTRACT_FEATURES RULE
    # rrmse_dict = calculate_and_save_rrmse(
    #     preprocessing_results_obj, save_path=f"{sim_directory}/rrmse_dict.json"
    # )


    # visualizing_results(
    #     preprocessing_results_obj_normalized,
    #     save_loc=sim_directory,
    #     analysis=f"preprocessing_results",
    #     stages=["training", "validation"],
    #     color_shades=color_shades,
    #     main_colors=main_colors,
    # )

    # visualizing_results(
    #     preprocessing_results_obj_normalized,
    #     save_loc=sim_directory,
    #     analysis=f"preprocessing_results_testing",
    #     stages=["testing"],
    #     color_shades=color_shades,
    #     main_colors=main_colors,
    # )

    # print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    args = parser.parse_args()

    obtain_features(
        experiment_config=args.experiment_config,
        sim_directory=args.sim_directory
    )
