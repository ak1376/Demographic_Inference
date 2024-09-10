# obtain_features.py

import numpy as np
import time
import pickle
import joblib
from preprocess import Processor
from utils import (
    visualizing_results,
    root_mean_squared_error,
    calculate_and_save_rrmse,
    find_outlier_indices,
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
    experiment_directory,
    color_shades_file,
    main_colors_file,
):

    # Load the experiment config
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    print(experiment_config.keys())

    # Load the experiment object to get the experiment directory
    with open(f"{experiment_directory}/experiment_obj.pkl", "rb") as f:
        experiment_obj = pickle.load(f)
    experiment_directory = experiment_obj.experiment_directory

    # Load in the color schemes and main colors
    with open(color_shades_file, "rb") as f:
        color_shades = pickle.load(f)
     
    with open(main_colors_file, "rb") as f:
        main_colors = pickle.load(f)

    processor = Processor(
        experiment_config,
        experiment_directory,
        recombination_rate=experiment_config["recombination_rate"],
        mutation_rate=experiment_config["mutation_rate"],
        window_length=experiment_config["window_length"],
    )

    # Now I want to define training, validation, and testing indices:

    # Generate all indices and shuffle them
    all_indices = np.arange(experiment_config['num_sims_pretrain'])
    np.random.shuffle(all_indices)

    # Split into training and validation indices
    n_train = int(0.8 * experiment_config['num_sims_pretrain'])

    training_indices = all_indices[:n_train]
    validation_indices = all_indices[n_train:]
    testing_indices = np.arange(experiment_config['num_sims_inference'])

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
        features, targets = processor.pretrain_processing(indices)

        preprocessing_results_obj[stage]["predictions"] = features # This is for input to the ML model
        preprocessing_results_obj[stage]["targets"] = targets

    preprocessing_results_obj["param_names"] = experiment_config['parameter_names']

    #TODO: Calculate and save the rrmse_dict but removing the outliers from analysis
    rrmse_dict = calculate_and_save_rrmse(
        preprocessing_results_obj,
        save_path=f"{experiment_directory}/rrmse_dict.json",
        dadi_analysis=experiment_config['dadi_analysis'],
        moments_analysis=experiment_config['moments_analysis'],
        momentsLD_analysis=experiment_config['momentsLD_analysis'],
    )

    # Open a file to save the object
    with open(
        f"{experiment_directory}/preprocessing_results_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(preprocessing_results_obj, file)

    # TODO: This function should pass in a list of the demographic parameters for which we want to produce plots.
    
    visualizing_results(
        preprocessing_results_obj,
        save_loc=experiment_directory,
        analysis=f"preprocessing_results",
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors
    )

    visualizing_results(
        preprocessing_results_obj,
        save_loc=experiment_directory,
        analysis=f"preprocessing_results_testing",
        stages=["testing"],
        color_shades=color_shades,
        main_colors=main_colors
    )

    ## LINEAR REGRESSION

    linear_mdl = LinearReg(training_features = preprocessing_results_obj["training"]["predictions"] ,
                            training_targets = preprocessing_results_obj["training"]["targets"],
                                validation_features = preprocessing_results_obj["validation"]["predictions"], 
                                validation_targets = preprocessing_results_obj["validation"]["targets"],
                                testing_features = preprocessing_results_obj["testing"]["predictions"],
                                    testing_targets = preprocessing_results_obj["testing"]["targets"] )
                            
    training_predictions, validation_predictions, testing_predictions = linear_mdl.train_and_validate()

    linear_mdl_obj = linear_mdl.organizing_results(preprocessing_results_obj, training_predictions, validation_predictions, testing_predictions)
    
    linear_mdl_obj["param_names"] = experiment_config['parameter_names']

    # Now calculate the linear model error

    rrmse_dict = {}
    rrmse_dict["training"] = root_mean_squared_error(
        y_true=linear_mdl_obj["training"]["targets"], y_pred=np.squeeze(training_predictions, axis = 1)
    )
    rrmse_dict["validation"] = root_mean_squared_error(
        y_true=linear_mdl_obj["validation"]["targets"], y_pred=np.squeeze(validation_predictions, axis = 1)
    )
    rrmse_dict["testing"] = root_mean_squared_error(
        y_true=linear_mdl_obj["testing"]["targets"], y_pred=np.squeeze(testing_predictions, axis = 1)
    )

    # Open a file to save the object
    with open(
        f"{experiment_directory}/linear_mdl_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(linear_mdl_obj, file)

    # Save rrmse_dict to a JSON file
    with open(
        f"{experiment_directory}/linear_model_error.json", "w"
    ) as json_file:
        json.dump(rrmse_dict, json_file, indent=4)

    # targets
    visualizing_results(
        linear_mdl_obj,
        "linear_results",
        save_loc=experiment_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors
    )

    joblib.dump(
        linear_mdl, f"{experiment_directory}/linear_regression_model.pkl"
    )
    # torch.save(
    #     snn_model.state_dict(),
    #     f"{self.experiment_directory}/neural_network_model.pth",
    # )
    
    # Save the color shades and main colors for usage with the neural network

    print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--experiment_directory", type=str, required=True)
    parser.add_argument("--color_shades_file", type=str, required=True)
    parser.add_argument("--main_colors_file", type=str, required=True)
    args = parser.parse_args()

    print("========================================")
    print(args.color_shades_file)
    print(args.main_colors_file)
    print("========================================")

    obtain_features(

        experiment_config=args.experiment_config,
        experiment_directory=args.experiment_directory,
        color_shades_file = args.color_shades_file,
        main_colors_file = args.main_colors_file,
    )
