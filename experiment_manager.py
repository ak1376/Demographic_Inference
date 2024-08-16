"""
The experiment manager should import the following modules:
- Processor object
- delete_vcf_files function


"""

import os
import shutil
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch
import torch.optim as optim
from torch.optim.adam import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


from preprocess import Processor, delete_vcf_files
from utils import (
    visualizing_results,
    visualize_model_predictions,
    # shap_values_plot,
    # partial_dependence_plots,
    extract_features,
    root_mean_squared_error,
    find_outlier_indices,  # FUNCTION FOR DEBUGGING PURPOSES
    resample_to_match_row_count,
)
from models import XGBoost, ShallowNN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import TQDMProgressBar
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer
import joblib


class Experiment_Manager:
    def __init__(self, config_file):
        # Later have a config file for model hyperparameters
        self.experiment_config = config_file

        self.upper_bound_params = config_file["upper_bound_params"]
        self.lower_bound_params = config_file["lower_bound_params"]
        self.num_sims = config_file["num_sims"]
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

        self.create_directory(self.experiment_name)

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

    def pretrain(self):
        """
        This should do the preprocessing, inference, etc.
        """

        # First step: define the processor
        processor = Processor(self.experiment_config, self.experiment_directory)

        dadi_dict, moments_dict, momentsLD_dict = processor.run()

        # Now save the results using pickle -- all three dictionaries
        if self.dadi_analysis:
            with open(
                os.path.join(self.experiment_directory, "dadi_dict.pkl"), "wb"
            ) as f:
                pickle.dump(dadi_dict, f)

        if self.moments_analysis:
            with open(
                os.path.join(self.experiment_directory, "moments_dict.pkl"), "wb"
            ) as f:
                pickle.dump(moments_dict, f)

        if self.momentsLD_analysis:
            with open(
                os.path.join(self.experiment_directory, "momentsLD_dict.pkl"), "wb"
            ) as f:
                pickle.dump(momentsLD_dict, f)

        # TODO: Change this function so that I am not a priori assuming it's dadi
        with open(
            os.path.join(self.experiment_directory, "generative_params.pkl"), "wb"
        ) as f:
            pickle.dump(dadi_dict["simulated_params"], f)

        # EVERYTHING BELOW IS UNOPTIMIZED AND/OR PLACEHOLDER CODE
        # Note that the simulated params in both the dadi_dict and moments_dict are identical

        # Initialize an empty list to hold feature arrays
        features_list = []
        feature_names = []
        targets_list = []

        # I want to now extract the features from the dictionaries (that we will be using for analysis)
        if self.dadi_analysis:
            features_dadi, targets = extract_features(
                dadi_dict["simulated_params"],
                dadi_dict["opt_params"],
                normalization=False,
            )
            error_value = root_mean_squared_error(y_true=targets, y_pred=features_dadi)
            print(f"The error value for dadi is: {error_value}")
            outlier_indices_dadi = find_outlier_indices(features_dadi)

            # TODO: Add a flag in the driver to specify whether we want to remove outliers from the pretraining process
            remove_outliers = True
            if remove_outliers:
                # Remove the outliers from the features and targets
                features_dadi = np.delete(
                    features_dadi, np.unique(outlier_indices_dadi), axis=0
                )
                targets = np.delete(targets, outlier_indices_dadi, axis=0)

            features_list.append(features_dadi)
            targets_list.append(targets)

            # features_dadi_no_outliers = np.delete(features_dadi, outlier_indices_dadi, axis=0)
            # targets_no_outliers = np.delete(targets, outlier_indices_dadi, axis=0)
            # error_value_no_outliers = relative_squared_error(y_true=targets_no_outliers, y_pred=features_dadi_no_outliers)
            # print(f"The error value for dadi without outliers is: {error_value_no_outliers}")

            np.savetxt(
                os.path.join(self.experiment_directory, "outlier_indices_dadi.csv"),
                outlier_indices_dadi,
                delimiter=",",
            )

            visualizing_results(
                dadi_dict,
                "dadi",
                save_loc=self.experiment_directory,
                outlier_indices=outlier_indices_dadi,
            )
            feature_names_dadi = [
                "Nb_opt_dadi",
                "N_recover_opt_dadi",
                "t_bottleneck_start_opt_dadi",
                "t_bottleneck_end_opt_dadi",
            ]

            feature_names.append(feature_names_dadi)

        if self.moments_analysis:
            features_moments, targets = extract_features(
                moments_dict["simulated_params"],
                moments_dict["opt_params"],
                normalization=False,
            )
            error_value = root_mean_squared_error(
                y_true=targets, y_pred=features_moments
            )
            print(f"The error value for moments is: {error_value}")

            outlier_indices_moments = find_outlier_indices(features_moments)

            remove_outliers = True
            if remove_outliers:
                # Remove the outliers from the features and targets
                features_moments = np.delete(
                    features_moments, np.unique(outlier_indices_moments), axis=0
                )
                targets = np.delete(targets, outlier_indices_moments, axis=0)

            features_list.append(features_moments)
            targets_list.append(targets)

            # features_moments_no_outliers = np.delete(features_moments, outlier_indices_moments, axis=0)
            # targets_no_outliers = np.delete(targets, outlier_indices_moments, axis=0)
            # error_value_no_outliers = relative_squared_error(y_true=targets_no_outliers, y_pred=features_moments_no_outliers)
            # print(f"The error value for moments without outliers is: {error_value_no_outliers}")

            np.savetxt(
                os.path.join(self.experiment_directory, "outlier_indices_moments.csv"),
                outlier_indices_moments,
                delimiter=",",
            )

            visualizing_results(
                moments_dict,
                "moments",
                save_loc=self.experiment_directory,
                outlier_indices=outlier_indices_moments,
            )
            feature_names_moments = [
                "Nb_opt_moments",
                "N_recover_opt_moments",
                "t_bottleneck_start_opt_moments",
                "t_bottleneck_end_opt_moments",
            ]
            feature_names.append(feature_names_moments)

        if self.momentsLD_analysis:
            features_momentsLD, targets = extract_features(
                momentsLD_dict["simulated_params"],
                momentsLD_dict["opt_params"],
                normalization=False,
            )

            error_value = root_mean_squared_error(
                y_true=targets, y_pred=features_moments
            )
            print(f"The error value for MomentsLD is: {error_value}")
            outlier_indices_momentsLD = find_outlier_indices(features_momentsLD)

            remove_outliers = True
            if remove_outliers:
                # Remove the outliers from the features and targets
                features_momentsLD = np.delete(
                    features_momentsLD, np.unique(outlier_indices_momentsLD), axis=0
                )
                targets = np.delete(targets, outlier_indices_moments, axis=0)

            features_list.append(features_momentsLD)
            targets_list.append(targets)

            # features_momentsLD_no_outliers = np.delete(features_momentsLD, outlier_indices_momentsLD, axis=0)
            # targets_no_outliers = np.delete(targets, outlier_indices_momentsLD, axis=0)
            # error_value_no_outliers = relative_squared_error(y_true=targets_no_outliers, y_pred=features_momentsLD_no_outliers)
            # print(f"The error value for moments without outliers is: {error_value_no_outliers}")

            np.savetxt(
                os.path.join(
                    self.experiment_directory, "outlier_indices_momentsLD.csv"
                ),
                outlier_indices_momentsLD,
                delimiter=",",
            )

            visualizing_results(
                momentsLD_dict,
                "MomentsLD",
                save_loc=self.experiment_directory,
                outlier_indices=outlier_indices_momentsLD,
            )
            feature_names_momentsLD = [
                "Nb_opt_momentsLD",
                "N_recover_opt_momentsLD",
                "t_bottleneck_start_opt_momentsLD",
                "t_bottleneck_end_opt_momentsLD",
            ]
            feature_names.append(feature_names_momentsLD)

        # I need to do resampling if I am removing outliers
        if remove_outliers:
            # Filter out empty arrays (if we are not using all three dictionaries)
            non_empty_features_list = [arr for arr in features_list if arr.size > 0]

            # Find the maximum number of rows in the non-empty arrays
            if non_empty_features_list:
                max_rows = max(arr.shape[0] for arr in non_empty_features_list)
            else:
                max_rows = 0  # Handle case where all arrays are empty

            # Resample only the non-empty arrays to match the maximum number of rows
            resampled_features_list = []

            for arr in non_empty_features_list:
                if arr.size > 0:
                    # Resample the features array
                    resampled_arr, resampling_indices = resample_to_match_row_count(
                        arr, max_rows, return_indices=True
                    )
                    resampled_features_list.append(resampled_arr)

                if arr.shape[0] != max_rows:
                    resampled_targets = targets[resampling_indices]

        # Concatenate all resampled features along axis 1
        features = (
            np.concatenate(resampled_features_list, axis=1)
            if resampled_features_list
            else np.array([])
        )

        # Reorder the elements by the element number of each sublist
        feature_names_reordered = []

        # Assume all sublists have the same length
        for i in range(len(feature_names[0])):
            for sublist in feature_names:
                feature_names_reordered.append(sublist[i])

        # PARAMETER INFERENCE IS COMPLETE. NOW IT'S TIME TO DO THE MACHINE LEARNING PART.

        # Probably not the most efficient, but placeholder for now

        # Convert the list to a dictionary with 0, 1, 2, ... as keys
        feature_names = {
            i: feature_names_reordered[i] for i in range(len(feature_names_reordered))
        }

        target_names = {
            0: "Nb_sample",
            1: "N_recover_sample",
            2: "t_bottleneck_start_sample",
            3: "t_bottleneck_end_sample",
        }

        # I want to do cross validation rather than a traditional train/test split
        # custom_scorer = make_scorer(relative_squared_error, greater_is_better=False)

        # Define the Leave-One-Out Cross-Validation strategy
        loo = LeaveOneOut()

        # Now do a train test split
        # X_train, X_test, y_train, y_test = train_test_split(
        #     features, targets, train_size=0.8, random_state=295
        # )

        # Feature scaling
        # feature_scaler = StandardScaler()
        # features_scaled = feature_scaler.fit_transform(features)

        # # Target scaling
        # target_scaler = StandardScaler()
        # targets_scaled = target_scaler.fit_transform(resampled_targets)

        # If we don't want to do scaling. #TODO: option for later. 
        features_scaled = features
        try:
            targets_scaled = resampled_targets
        except UnboundLocalError as e:
            targets_scaled = targets            

        # MODEL DEFINITIONS

        ## LINEAR REGRESSION
        lin_mdl = LinearRegression()

        ## XGBOOST
        xgb_model = XGBoost(feature_names, target_names, loo=loo)

        ## SHALLOW NEURAL NETWORK
        # Define model hyperparameters
        input_size = features_scaled.shape[1]  # Number of features
        hidden_size = 100  # Number of neurons in the hidden layer
        output_size = 4  # Number of output classes
        num_epochs = 1000
        learning_rate = 3e-4

        # Instantiate the model
        model = ShallowNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            loo=loo,
            experiment_directory=self.experiment_directory,
        )
        # Calculate the number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")


        # Create a custom scorer using make_scorer
        custom_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

        # Train the linear regression model
        linear_mdl_scores = cross_val_score(
            lin_mdl,
            features_scaled,
            targets_scaled,
            cv=loo,
            scoring=custom_scorer
        )

        mean_lin_mdl_score = -1 * linear_mdl_scores.mean()

        linear_mdl_predictions = cross_val_predict(
            lin_mdl, features_scaled, targets_scaled, cv=loo
        )

        # Optionally, retrain on the entire dataset
        lin_mdl.fit(features_scaled, targets_scaled)

        # Convert the array to a list of dictionaries
        param_names = ["Nb", "N_recover", "t_bottleneck_start", "t_bottleneck_end"]
        opt_params = [
            {
                param_names[j]: linear_mdl_predictions[i, j]
                for j in range(linear_mdl_predictions.shape[1])
            }
            for i in range(linear_mdl_predictions.shape[0])
        ]

        linear_mdl_obj = {}
        linear_mdl_obj["model"] = lin_mdl
        linear_mdl_obj["opt_params"] = opt_params

        # Convert the array to a list of dictionaries
        # Extract the keys (assuming all dictionaries have the same keys)

        # keys = dadi_dict['simulated_params'][0].keys()

        # # Create the array where each column corresponds to values of a specific key
        # array = np.array([[d[key] for d in dadi_dict['simulated_params']] for key in keys]).T

        simulated_params = [
            {
                param_names[j]: targets_scaled[i, j]
                for j in range(linear_mdl_predictions.shape[1])
            }
            for i in range(linear_mdl_predictions.shape[0])
        ]

        linear_mdl_obj["simulated_params"] = simulated_params

        # targets
        visualizing_results(
            linear_mdl_obj, "linear_results", save_loc=self.experiment_directory
        )

        # loss_xgb, predictions_xgb, _ = xgb_model.train_and_validate(
        #     features_scaled, targets_scaled, cross_val=True
        # )
        average_val_loss_snn, predictions_snn,all_train_loss_curves_snn, all_val_loss_curves_snn = model.cross_validate(
            features_scaled,
            targets_scaled,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )

        opt_params = [
            {
                param_names[j]: predictions_snn[i, j]
                for j in range(predictions_snn.shape[1])
            }
            for i in range(predictions_snn.shape[0])
        ]

        model.train_on_full_data(torch.tensor(features_scaled, dtype = torch.float32), torch.tensor(targets_scaled, dtype = torch.float32), num_epochs=num_epochs, learning_rate=learning_rate)
        # Load the saved model state dictionary
        model.load_state_dict(torch.load('/sietch_colab/akapoor/experiments/linear_model_bottleneck/final_model.pth'))

        snn_mdl_obj = {}
        snn_mdl_obj["model"] = model
        snn_mdl_obj["opt_params"] = opt_params
        snn_mdl_obj["simulated_params"] = simulated_params

        visualizing_results(
            snn_mdl_obj, "snn_results", save_loc=self.experiment_directory
        )

        model.plot_loss_curves(all_train_loss_curves_snn, all_val_loss_curves_snn)

        # Let's do a check to see if the model has actually trained properly
        model.eval() 
        features_check = model(torch.tensor(features_scaled[0,:], dtype = torch.float32))
        print(f"Features: {features_scaled[0,:]}")
        print(f"Predictions: {features_check}")
        print(f'Actual: {targets_scaled[0,:]}')
        
        
        # CALCULATE THE RELATIVE LOSS IN PARAMETERS FOR EACH MODEL
        linear_mdl_error = root_mean_squared_error(
            targets_scaled, linear_mdl_predictions
        )
        # xgb_error = relative_squared_error(targets_scaled, predictions_xgb)
        snn_error = root_mean_squared_error(targets_scaled, predictions_snn)

        # Write the errors to a text file
        with open(f"{self.experiment_directory}/model_errors.txt", "w") as file:
            file.write(f"Linear Model Error: {linear_mdl_error}\n")
            file.write(f"SNN Error: {snn_error}\n")

        # print(f"The Linear Regression Cross Validation error is: {mean_lin_mdl_score}")
        # print("==============================")
        # print(f"The XGBoost Cross Validation error is: {loss_xgb}")
        # print("==============================")
        # print(f"The Shallow Neural Network Cross Validation error is: {average_val_loss_snn}")

        joblib.dump(lin_mdl, f"{self.experiment_directory}/linear_regression_model.pkl")
        # torch.save(
        #     model.state_dict(), f"{self.experiment_directory}/neural_network_model.pth"
        # )

        print("Training complete!")

    def inference(self, vcf_file):

        pass 

