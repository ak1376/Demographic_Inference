"""
The experiment manager should import the following modules:
- Processor object
- delete_vcf_files function
"""

import os
import shutil
import numpy as np
import pickle
import ray
import time
import json

from preprocess import Processor, FeatureExtractor
from utils import (
    visualizing_results,
    process_and_save_data,
    calculate_and_save_rrmse,
    root_mean_squared_error,
    find_outlier_indices
)
from sklearn.linear_model import LinearRegression
import joblib


class Experiment_Manager:
    def __init__(self, config_file):
        # Later have a config file for model hyperparameters
        self.experiment_config = config_file

        self.upper_bound_params = config_file["upper_bound_params"]
        self.lower_bound_params = config_file["lower_bound_params"]
        self.num_sims_pretrain = config_file["num_sims_pretrain"]
        self.num_sims_inference = config_file["num_sims_inference"]
        self.num_samples = config_file["num_samples"]
        self.experiment_name = config_file["experiment_name"]
        self.num_windows = config_file["num_windows"]
        self.window_length = config_file["window_length"]
        self.maxiter = config_file["maxiter"]
        self.mutation_rate = config_file["mutation_rate"]
        self.recombination_rate = config_file["recombination_rate"]
        self.genome_length = config_file["genome_length"]
        self.dadi_analysis = config_file["dadi_analysis"]
        self.moments_analysis = config_file["moments_analysis"]
        self.momentsLD_analysis = config_file["momentsLD_analysis"]
        self.seed = config_file["seed"]
        self.normalization = config_file["normalization"]
        self.remove_outliers = config_file["remove_outliers"]
        self.neural_net_hyperparameters = config_file["neural_net_hyperparameters"]
        self.create_directory(self.experiment_name)
        np.random.seed(self.seed)

        # Open a file in write mode and save the dictionary as JSON
        with open(f"{self.experiment_directory}/config.json", "w") as json_file:
            json.dump(
                self.experiment_config, json_file, indent=4
            )  # indent=4 makes the JSON file more readable

    def create_directory(
        self, folder_name, base_dir="experiments", archive_subdir="archive"
    ):
        # Full paths for base and archive directories
        base_path = os.path.join(base_dir, folder_name)
        archive_path = os.path.join(base_dir, archive_subdir)

        # Ensure the archive subdirectory exists
        os.makedirs(archive_path, exist_ok=True)

        # Check if the folder already exists
        if os.path.exists(base_path):
            # Find a new name for the existing folder to move it to the archive
            i = 1
            new_folder_name = f"{folder_name}_{i}"
            new_folder_path = os.path.join(archive_path, new_folder_name)
            while os.path.exists(new_folder_path):
                i += 1
                new_folder_name = f"{folder_name}_{i}"
                new_folder_path = os.path.join(archive_path, new_folder_name)

            # Rename and move the existing folder to the archive subdirectory
            shutil.move(base_path, new_folder_path)
            print(f"Renamed and moved existing folder to: {new_folder_path}")

        # Create the new directory
        os.makedirs(base_path, exist_ok=True)
        print(f"Created new directory: {base_path}")

        self.experiment_directory = base_path

    def load_features(self, file_name):
        with open(f"{file_name}", "rb") as file:
            loaded_object = pickle.load(file)

        return loaded_object

    def obtaining_features(self):
        """
        This should do the dadi and moments inference (input to the ML models). THIS IS PRETRAINING CODE
        """

        # First step: define the processor
        processor = Processor(self.experiment_config, self.experiment_directory)
        extractor = FeatureExtractor(
            self.experiment_directory,
            dadi_analysis=self.experiment_config["dadi_analysis"],
            moments_analysis=self.experiment_config["moments_analysis"],
            momentsLD_analysis=self.experiment_config["momentsLD_analysis"],
        )

        # Now I want to define training, validation, and testing indices:

        # Generate all indices and shuffle them
        all_indices = np.arange(self.num_sims_pretrain)
        np.random.shuffle(all_indices)

        # Split into training and validation indices
        n_train = int(0.8 * self.num_sims_pretrain)

        training_indices = all_indices[:n_train]
        validation_indices = all_indices[n_train:]
        testing_indices = np.arange(self.num_sims_inference)

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
                self.experiment_directory,
                self.dadi_analysis,
                self.moments_analysis,
                self.momentsLD_analysis,
            )

            # Now let's take the information from each dictionary above and put it in the features_dict and, if in pretrain mode, the targets_dict.
            # TODO: Maybe put in a helper function?

            if self.dadi_analysis:
                concatenated_array = np.column_stack(
                    [dadi_dict["opt_params"][key] for key in dadi_dict["opt_params"]]
                )

                # Need to find the outliers 
                outlier_indices = find_outlier_indices(concatenated_array)

                features_dict[stage]["dadi"] = concatenated_array

                if dadi_dict["simulated_params"]:
                    concatenated_array = np.column_stack(
                        [
                            dadi_dict["simulated_params"][key]
                            for key in dadi_dict["simulated_params"]
                        ]
                    )
                    targets_dict[stage]["dadi"] = concatenated_array

            if self.moments_analysis:
                concatenated_array = np.column_stack(
                    [
                        moments_dict["opt_params"][key]
                        for key in moments_dict["opt_params"]
                    ]
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

            if self.momentsLD_analysis:
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

        for stage in preprocessing_results_obj:

            features = preprocessing_results_obj[stage]["predictions"]
            outlier_indices = find_outlier_indices(features)
            print(outlier_indices)

            if self.remove_outliers:
                features = np.delete(features, outlier_indices, axis=0)

        
        # Assuming features_dict and targets_dict are already defined
        rrmse_dict = calculate_and_save_rrmse(
            features_dict,
            targets_dict,
            save_path=f"{self.experiment_directory}/rrmse_dict.json",
            dadi_analysis=self.dadi_analysis,
            moments_analysis=self.moments_analysis,
            momentsLD_analysis=self.momentsLD_analysis,
        )

        # preprocessing_results_obj['rrmse'] = rrmse_dict

        # Open a file to save the object
        with open(
            f"{self.experiment_directory}/preprocessing_results_obj.pkl", "wb"
        ) as file:  # "wb" mode opens the file in binary write mode
            pickle.dump(preprocessing_results_obj, file)

        # TODO: This function should be modified to properly consider outliers.
        # TODO: This function should pass in a list of the demographic parameters for which we want to produce plots.
        visualizing_results(
            preprocessing_results_obj,
            save_loc=self.experiment_directory,
            analysis=f"preprocessing_results",
            stages=["training", "validation"],
        )

        visualizing_results(
            preprocessing_results_obj,
            save_loc=self.experiment_directory,
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
            f"{self.experiment_directory}/linear_mdl_obj.pkl", "wb"
        ) as file:  # "wb" mode opens the file in binary write mode
            pickle.dump(linear_mdl_obj, file)

        # Save rrmse_dict to a JSON file
        with open(
            f"{self.experiment_directory}/linear_model_error.json", "w"
        ) as json_file:
            json.dump(rrmse_dict, json_file, indent=4)

        # targets
        visualizing_results(
            linear_mdl_obj,
            "linear_results",
            save_loc=self.experiment_directory,
            stages=["training", "validation"],
        )

        joblib.dump(
            linear_mdl, f"{self.experiment_directory}/linear_regression_model.pkl"
        )
        # torch.save(
        #     snn_model.state_dict(),
        #     f"{self.experiment_directory}/neural_network_model.pth",
        # )

        print("Training complete!")
