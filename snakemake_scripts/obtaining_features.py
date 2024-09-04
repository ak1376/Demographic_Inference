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
)
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

        # Now let's take the information from each dictionary above and put it in the features_dict and, if in pretrain mode, the targets_dict.
        # TODO: Maybe put in a helper function?

        if experiment_config["dadi_analysis"]:
            concatenated_array = np.column_stack(
                [dadi_dict["opt_params"][key] for key in dadi_dict["opt_params"]]
            )
            features_dict[stage]["dadi"] = concatenated_array

            if dadi_dict["simulated_params"]:
                concatenated_array = np.column_stack(
                    [
                        dadi_dict["simulated_params"][key]
                        for key in dadi_dict["simulated_params"]
                    ]
                )
                targets_dict[stage]["dadi"] = concatenated_array

        if experiment_config["moments_analysis"]:
            concatenated_array = np.column_stack(
                [moments_dict["opt_params"][key] for key in moments_dict["opt_params"]]
            )
            features_dict[stage]["moments"] = concatenated_array

            if moments_dict["simulated_params"]:
                concatenated_array = np.column_stack(
                    [
                        moments_dict["simulated_params"][key]
                        for key in moments_dict["simulated_params"]
                    ]
                )
                targets_dict[stage]["moments"] = concatenated_array

        if experiment_config["momentsLD_analysis"]:
            concatenated_array = np.column_stack(
                [
                    momentsLD_dict["opt_params"][key]
                    for key in momentsLD_dict["opt_params"]
                ]
            )
            features_dict[stage]["momentsLD"] = concatenated_array

            if momentsLD_dict["simulated_params"]:
                concatenated_array = np.column_stack(
                    [
                        momentsLD_dict["simulated_params"][key]
                        for key in momentsLD_dict["simulated_params"]
                    ]
                )
                targets_dict[stage]["momentsLD"] = concatenated_array

        # Now columnwise the dadi, moments, and momentsLD inferences to get a concatenated features and targets array
        concat_feats = np.column_stack(
            [features_dict[stage][subkey] for subkey in features_dict[stage]]
        )

        if targets_dict[stage]:
            concat_targets = np.column_stack(
                [targets_dict[stage]["dadi"]]
            )  # dadi because dadi and moments values for the targets are the same.

        concatenated_features[stage] = concat_feats  # type:ignore
        concatenated_targets[stage] = concat_targets  # type:ignore

    training_features, validation_features, testing_features = (
        concatenated_features["training"],
        concatenated_features["validation"],
        concatenated_features["testing"],
    )
    training_targets, validation_targets, testing_targets = (
        concatenated_targets["training"],
        concatenated_targets["validation"],
        concatenated_targets["testing"],
    )

    preprocessing_results_obj = {}

    preprocessing_results_obj["training"] = {}
    preprocessing_results_obj["validation"] = {}
    preprocessing_results_obj["testing"] = {}

    preprocessing_results_obj["training"]["predictions"] = training_features
    preprocessing_results_obj["training"]["targets"] = training_targets

    preprocessing_results_obj["validation"]["predictions"] = validation_features
    preprocessing_results_obj["validation"]["targets"] = validation_targets

    preprocessing_results_obj["testing"]["predictions"] = testing_features
    preprocessing_results_obj["testing"]["targets"] = testing_targets

    # Assuming features_dict and targets_dict are already defined
    rrmse_dict = calculate_and_save_rrmse(
        features_dict,
        targets_dict,
        save_path=f"{experiment_directory}/rrmse_dict.json",
        dadi_analysis=experiment_config["dadi_analysis"],
        moments_analysis=experiment_config["moments_analysis"],
        momentsLD_analysis=experiment_config["momentsLD_analysis"],
    )

    # preprocessing_results_obj['rrmse'] = rrmse_dict

    # Open a file to save the object
    with open(
        f"{experiment_directory}/preprocessing_results_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(preprocessing_results_obj, file)

    # TODO: This function should be modified to properly consider outliers.
    # TODO: This function should pass in a list of the demographic parameters for which we want to produce plots.
    visualizing_results(
        preprocessing_results_obj,
        save_loc=experiment_directory,
        analysis=f"preprocessing_results",
        stages=["training", "validation"],
    )

    visualizing_results(
        preprocessing_results_obj,
        save_loc=experiment_directory,
        analysis=f"preprocessing_results_testing",
        stages=["testing"],
    )

    ## LINEAR REGRESSION
    linear_mdl = LinearRegression()
    linear_mdl.fit(training_features, training_targets)  # type:ignore

    training_predictions = linear_mdl.predict(training_features)  # type:ignore
    validation_predictions = linear_mdl.predict(validation_features)  # type:ignore
    testing_predictions = linear_mdl.predict(testing_features)  # type:ignore

    # TODO: Linear regression should be moded to the models module.
    linear_mdl_obj = {}
    linear_mdl_obj["model"] = linear_mdl

    linear_mdl_obj["training"] = {}
    linear_mdl_obj["validation"] = {}
    linear_mdl_obj["testing"] = {}

    linear_mdl_obj["training"]["predictions"] = training_predictions
    linear_mdl_obj["training"]["targets"] = training_targets

    linear_mdl_obj["validation"]["predictions"] = validation_predictions
    linear_mdl_obj["validation"]["targets"] = validation_targets

    linear_mdl_obj["testing"]["predictions"] = testing_predictions
    linear_mdl_obj["testing"]["targets"] = testing_targets

    # Now calculate the linear model error

    rrmse_dict = {}
    rrmse_dict["training"] = root_mean_squared_error(
        y_true=training_targets, y_pred=training_predictions
    )
    rrmse_dict["validation"] = root_mean_squared_error(
        y_true=validation_targets, y_pred=validation_predictions
    )
    rrmse_dict["testing"] = root_mean_squared_error(
        y_true=testing_targets, y_pred=testing_predictions
    )

    # linear_mdl_obj["rrmse"] = rrmse_dict

    # Open a file to save the object
    with open(
        f"{experiment_directory}/linear_mdl_obj.pkl", "wb"
    ) as file:  # "wb" mode opens the file in binary write mode
        pickle.dump(linear_mdl_obj, file)

    # Save rrmse_dict to a JSON file
    with open(f"{experiment_directory}/linear_model_error.json", "w") as json_file:
        json.dump(rrmse_dict, json_file, indent=4)

    # targets
    visualizing_results(
        linear_mdl_obj,
        "linear_results",
        save_loc=experiment_directory,
        stages=["training", "validation"],
    )

    joblib.dump(linear_mdl, f"{experiment_directory}/linear_regression_model.pkl")
    # torch.save(
    #     snn_model.state_dict(),
    #     f"{self.experiment_directory}/neural_network_model.pth",
    # )

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
