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
import ray
import time
import json

from preprocess import Processor, FeatureExtractor
from utils import (
    visualizing_results,
    visualize_model_predictions,
    # shap_values_plot,
    # partial_dependence_plots,
    extract_features,
    root_mean_squared_error,
    find_outlier_indices,  # FUNCTION FOR DEBUGGING PURPOSES
    resample_to_match_row_count,
    save_dict_to_pickle,
    process_and_save_data,
    calculate_model_errors,
)
from models import XGBoost, ShallowNN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import TQDMProgressBar
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer
import joblib
from train import Trainer


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
            json.dump(self.experiment_config, json_file, indent=4)  # indent=4 makes the JSON file more readable

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
        with open(f'{file_name}', "rb") as file:
            loaded_object = pickle.load(file)

        return loaded_object

    def obtaining_features(self):
        """
        This should do the dadi and moments inference (input to the ML models)
        """

        ray.init(
            num_cpus=os.cpu_count(), local_mode=False
        )  # This is a placeholder for now.

        # First step: define the processor
        processor = Processor(self.experiment_config, self.experiment_directory)
        extractor = FeatureExtractor(self.experiment_directory)

        # Now I want to define training, validation, and testing indices:

        # Generate all indices and shuffle them
        all_indices = np.arange(self.num_sims_pretrain)
        np.random.shuffle(all_indices)

        # Split into training and validation indices
        n_train = int(0.8 * self.num_sims_pretrain)

        training_indices = all_indices[:n_train]
        validation_indices = all_indices[n_train:]
        testing_indices = np.arange(self.num_sims_inference)

        for stage, indices in [
            ("training", training_indices),
            ("validation", validation_indices),
            ("testing", testing_indices),
        ]:

            # Your existing process_and_save_data function


            # Call the remote function and get the ObjectRef
            result_ref = process_and_save_data.remote(
                processor, indices, stage, self.experiment_directory
            )

            start = time.time() 

            # Use ray.get() to retrieve the actual result
            dadi_dict, moments_dict, momentsLD_dict = ray.get(result_ref)

            end = time.time()

            print(f"Time taken for processing {stage} data: {end - start}")

            # Process each dictionary
            if extractor.dadi_analysis:
                extractor.process_batch(dadi_dict, "dadi", stage, normalization=self.normalization)
            if extractor.moments_analysis:
                extractor.process_batch(
                    moments_dict, "moments", stage, normalization=self.normalization
                )
            if extractor.momentsLD_analysis:
                extractor.process_batch(
                    momentsLD_dict, "momentsLD", stage, normalization=self.normalization
                )

        # After processing all stages, finalize the processing

        features, targets, feature_names = extractor.finalize_processing(
            remove_outliers=self.remove_outliers
        )
        training_features, validation_features, testing_features = (
            features["training"],
            features["validation"],
            features["testing"],
        )
        training_targets, validation_targets, testing_targets = (
            targets["training"],
            targets["validation"],
            targets["testing"],
        )

        # target_names = {
        #     0: "Nb_sample",
        #     1: "N_recover_sample",
        #     2: "t_bottleneck_start_sample",
        #     3: "t_bottleneck_end_sample",
        # }

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

        # Open a file to save the object
        with open(f"{self.experiment_directory}/preprocessing_results_obj.pkl", "wb") as file:  # "wb" mode opens the file in binary write mode
            pickle.dump(preprocessing_results_obj, file)


        # TODO: This function should be modified to properly consider outliers. 
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
        linear_mdl.fit(training_features, training_targets)

        training_predictions = linear_mdl.predict(training_features)
        validation_predictions = linear_mdl.predict(validation_features)
        # testing_predictions = linear_mdl.predict(testing_features)

        linear_mdl_obj = {}
        linear_mdl_obj["model"] = linear_mdl

        linear_mdl_obj["training"] = {}
        linear_mdl_obj["validation"] = {}
        linear_mdl_obj["testing"] = {}

        linear_mdl_obj["training"]["predictions"] = training_predictions
        linear_mdl_obj["training"]["targets"] = training_targets

        linear_mdl_obj["validation"]["predictions"] = validation_predictions
        linear_mdl_obj["validation"]["targets"] = validation_targets

        # linear_mdl_obj["testing"]["predictions"] = testing_predictions
        linear_mdl_obj["testing"]["targets"] = testing_targets

        # Open a file to save the object
        with open(f"{self.experiment_directory}/linear_mdl_obj.pkl", "wb") as file:  # "wb" mode opens the file in binary write mode
            pickle.dump(linear_mdl_obj, file)

        # targets
        visualizing_results(
            linear_mdl_obj, "linear_results", save_loc=self.experiment_directory, stages=["training", "validation"]
        )

        # ALL THIS BELOW CODE SHOULD BE GOING IN A DIFFERENT OBJECT: TRAINER OBJECT

        ray.shutdown()

        # Calculate errors for each model separately
        preprocessing_errors = calculate_model_errors(
            preprocessing_results_obj, "preprocessing", datasets=["training", "validation"]
        )
        linear_errors = calculate_model_errors(linear_mdl_obj, "linear", datasets=["training", "validation"]) # The reason why we are not passing "testing" because we will pass the trained model along with the testing data to the inference object. 
        # snn_errors = calculate_model_errors(snn_mdl_obj, "snn", datasets=["training", "validation"])

        # Combine all errors if needed
        all_errors = {**preprocessing_errors, **linear_errors}

        # Print results
        with open(f"{self.experiment_directory}/preprocessing_data_error.txt", "w") as f:
            for model, datasets in all_errors.items():
                f.write(f"\n{model.upper()} Model Errors:\n")
                for dataset, error in datasets.items():
                    f.write(f"  {dataset.capitalize()} RMSE: {error:.4f}\n")
        print(
            f"Results have been saved to {self.experiment_directory}/model_errors.txt"
        )

        joblib.dump(
            linear_mdl, f"{self.experiment_directory}/linear_regression_model.pkl"
        )
        # torch.save(
        #     snn_model.state_dict(),
        #     f"{self.experiment_directory}/neural_network_model.pth",
        # )

        print("Training complete!")