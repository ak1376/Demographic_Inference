# obtain_features.py

import numpy as np
import time
import pickle
import joblib
from sklearn.linear_model import LinearRegression
from preprocess import Processor, FeatureExtractor
from utils import (
    process_and_save_data,
    visualizing_results,
    calculate_model_errors,
    root_mean_squared_error,
    calculate_and_save_rrmse,
    find_outlier_indices,
    creating_features_dict,
    concatenating_features,
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
    num_sims_pretrain,
    num_sims_inference,
    normalization=False,
    remove_outliers=True,
):

    # print(f'Normalization option: {normalization}')

    # Load the experiment config
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    print(experiment_config.keys())

    # Load the experiment object to get the experiment directory
    with open(f"{experiment_directory}/experiment_obj.pkl", "rb") as f:
        experiment_obj = pickle.load(f)
    experiment_directory = experiment_obj.experiment_directory

    processor = Processor(
        experiment_config,
        experiment_directory,
        recombination_rate=experiment_config["recombination_rate"],
        mutation_rate=experiment_config["mutation_rate"],
        window_length=experiment_config["window_length"],
    )
    extractor = FeatureExtractor(
        experiment_directory,
        dadi_analysis=experiment_config["dadi_analysis"],
        moments_analysis=experiment_config["moments_analysis"],
        momentsLD_analysis=experiment_config["momentsLD_analysis"],
    )

    all_indices = np.arange(num_sims_pretrain)
    np.random.shuffle(all_indices)
    n_train = int(0.8 * num_sims_pretrain)

    training_indices = all_indices[:n_train]
    validation_indices = all_indices[n_train:]
    testing_indices = np.arange(num_sims_inference)

    print(f"Number of Training Indices: {len(training_indices)}")
    print(f"Number of Validation Indices: {len(validation_indices)}")
    print(f"Number of Testing Indices: {len(testing_indices)}")

    features_dict = {stage: {} for stage in ["training", "validation", "testing"]}
    targets_dict = {stage: {} for stage in ["training", "validation", "testing"]}
    feature_names = {}
    concatenated_features = {
        stage: {} for stage in ["training", "validation", "testing"]
    }
    concatenated_targets = {
        stage: {} for stage in ["training", "validation", "testing"]
    }

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
        merged_dict = processor.pretrain_processing(indices)

        dadi_dict, moments_dict, momentsLD_dict = process_and_save_data(
            merged_dict,
            stage,
            experiment_directory,
            experiment_config["dadi_analysis"],
            experiment_config["moments_analysis"],
            experiment_config["momentsLD_analysis"],
        )

        features_dict, targets_dict = creating_features_dict(
            stage,
            dadi_dict,
            moments_dict,
            momentsLD_dict,
            features_dict,
            targets_dict,
            experiment_config["dadi_analysis"],
            experiment_config["moments_analysis"],
            experiment_config["momentsLD_analysis"]
        )

        concatenated_features, concatenated_targets= concatenating_features( stage, concatenated_features, concatenated_targets, features_dict, targets_dict)


        outlier_indices = find_outlier_indices(concatenated_features)
        print(outlier_indices)

        if remove_outliers:
            concatenated_features = np.delete(concatenated_features, outlier_indices, axis=0)
            concatenated_targets = np.delete(concatenated_targets, outlier_indices, axis=0)

        preprocessing_results_obj[stage]["predictions"] = concatenated_features
        preprocessing_results_obj[stage]["targets"] = concatenated_targets
        preprocessing_results_obj["param_names"] = experiment_config['parameter_names']


    #TODO: Calculate and save the rrmse_dict but removing the outliers from analysis
    rrmse_dict = calculate_and_save_rrmse(
        features_dict,
        targets_dict,
        save_path=f"{experiment_directory}/rrmse_dict.json",
        dadi_analysis=experiment_config["dadi_analysis"],
        moments_analysis=experiment_config["moments_analysis"],
        momentsLD_analysis=experiment_config["momentsLD_analysis"]
    )

    # Open a file to save the object
    with open(
        f"{experiment_directory}/preprocessing_results_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(preprocessing_results_obj, file)

    # TODO: This function should pass in a list of the demographic parameters for which we want to produce plots.
    color_shades, main_colors = visualizing_results(
        preprocessing_results_obj,
        save_loc=experiment_directory,
        analysis=f"preprocessing_results",
        stages=["training", "validation"],
        color_shades=None,
        main_colors=None
    )

    _, _ = visualizing_results(
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
        f"{experiment_directory}/linear_mdl_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(linear_mdl_obj, file)

    # Save rrmse_dict to a JSON file
    with open(
        f"{experiment_directory}/linear_model_error.json", "w"
    ) as json_file:
        json.dump(rrmse_dict, json_file, indent=4)

    # targets
    _, _ = visualizing_results(
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

    file_path = f'{experiment_directory}/color_shades.pkl'

    # Save the list using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(color_shades, f)

    file_path = f'{experiment_directory}/main_colors.pkl'

    # Save the list using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(main_colors, f)

    print("Training complete!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--experiment_directory", type=str, required=True)
    parser.add_argument("--num_sims_pretrain", type=int, required=True)
    parser.add_argument("--num_sims_inference", type=int, required=True)
    parser.add_argument("--normalization", type=str2bool, default=True)
    parser.add_argument("--remove_outliers", type=str2bool, default=True)
    args = parser.parse_args()

    obtain_features(
        args.experiment_config,
        args.experiment_directory,
        args.num_sims_pretrain,
        args.num_sims_inference,
        args.normalization,
        args.remove_outliers,
    )
