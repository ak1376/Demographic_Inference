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

            # Use ray.get() to retrieve the actual result
            dadi_dict, moments_dict, momentsLD_dict = ray.get(result_ref)

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

        target_names = {
            0: "Nb_sample",
            1: "N_recover_sample",
            2: "t_bottleneck_start_sample",
            3: "t_bottleneck_end_sample",
        }

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

        visualizing_results(
            preprocessing_results_obj,
            save_loc=self.experiment_directory,
            analysis=f"preprocessing_results",
        )

        # I want to do cross validation rather than a traditional train/test split
        # custom_scorer = make_scorer(relative_squared_error, greater_is_better=False)

        ## LINEAR REGRESSION
        linear_mdl = LinearRegression()
        linear_mdl.fit(training_features, training_targets)

        training_predictions = linear_mdl.predict(training_features)
        validation_predictions = linear_mdl.predict(validation_features)
        testing_predictions = linear_mdl.predict(testing_features)

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

        # root_mean_squared_error(y_true = linear_mdl_obj['training']['targets'], y_pred = linear_mdl_obj['training']['predictions'])
        # root_mean_squared_error(y_true = linear_mdl_obj['validation']['targets'], y_pred = linear_mdl_obj['validation']['predictions'])
        # root_mean_squared_error(y_true = linear_mdl_obj['testing']['targets'], y_pred = linear_mdl_obj['testing']['predictions'])

        # targets
        visualizing_results(
            linear_mdl_obj, "linear_results", save_loc=self.experiment_directory
        )

        # loss_xgb, predictions_xgb, _ = xgb_model.train_and_validate(
        #     features_scaled, targets_scaled, cross_val=True
        # )

        ray.shutdown()

        # MODEL DEFINITIONS
        # Initialize Ray with both CPU and GPU resources
        ray.init(num_cpus=os.cpu_count(), num_gpus=3, local_mode=False)

        ## SHALLOW NEURAL NETWORK
        # Define model hyperparameters
        # TODO: This should be in the config file
        input_size = training_features.shape[1]  # Number of features
        hidden_size = 500  # Number of neurons in the hidden layer
        output_size = 4  # Number of output classes
        num_epochs = 1000
        learning_rate = 3e-4
        num_layers = 4

        # model_config = {
        #     "input_size": input_size,
        #     "hidden_size": hidden_size,
        #     "output_size": output_size,
        #     "num_layers": num_layers,
        # }

        # Set the device to GPU if available, otherwise use CPU

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import time

        start_time = time.time()
        snn_model, train_losses, val_losses = ShallowNN.train_and_validate(
            X_train=training_features,
            y_train=training_targets,
            X_val=validation_features,
            y_val=validation_targets,
            input_size=self.neural_net_hyperparameters['input_size'],
            hidden_size=self.neural_net_hyperparameters['hidden_size'],
            output_size=self.neural_net_hyperparameters['output_size'],
            num_layers=self.neural_net_hyperparameters['num_layers'],
            num_epochs=self.neural_net_hyperparameters['num_epochs'],
            learning_rate=self.neural_net_hyperparameters['learning_rate']

        )
        end_time = time.time()

        print(f"Time taken for training: {end_time - start_time}")

        training_predictions = snn_model.predict(training_features)
        validation_predictions = snn_model.predict(validation_features)
        testing_predictions = snn_model.predict(testing_features)

        snn_mdl_obj = {}
        snn_mdl_obj["training"] = {}
        snn_mdl_obj["validation"] = {}
        snn_mdl_obj["testing"] = {}

        snn_mdl_obj["training"]["predictions"] = training_predictions
        snn_mdl_obj["training"]["targets"] = training_targets

        snn_mdl_obj["validation"]["predictions"] = validation_predictions
        snn_mdl_obj["validation"]["targets"] = validation_targets

        snn_mdl_obj["testing"]["predictions"] = testing_predictions
        snn_mdl_obj["testing"]["targets"] = testing_targets

        visualizing_results(
            snn_mdl_obj, "snn_results", save_loc=self.experiment_directory
        )

        ShallowNN.plot_loss_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=f"{self.experiment_directory}/loss_curves.png",
        )

        # Calculate errors for each model separately
        preprocessing_errors = calculate_model_errors(
            preprocessing_results_obj, "preprocessing"
        )
        linear_errors = calculate_model_errors(linear_mdl_obj, "linear")
        snn_errors = calculate_model_errors(snn_mdl_obj, "snn")

        # Combine all errors if needed
        all_errors = {**preprocessing_errors, **linear_errors, **snn_errors}

        # Print results
        with open(f"{self.experiment_directory}/model_errors.txt", "w") as f:
            for model, datasets in all_errors.items():
                f.write(f"\n{model.upper()} Model Errors:\n")
                for dataset, error in datasets.items():
                    f.write(f"  {dataset.capitalize()} RMSE: {error:.4f}\n")
        print(
            f"Results have been saved to {self.experiment_directory}/model_errors.txt"
        )

        # print(f"The Linear Regression Cross Validation error is: {mean_lin_mdl_score}")
        # print("==============================")
        # print(f"The XGBoost Cross Validation error is: {loss_xgb}")
        # print("==============================")
        # print(f"The Shallow Neural Network Cross Validation error is: {average_val_loss_snn}")

        joblib.dump(
            linear_mdl, f"{self.experiment_directory}/linear_regression_model.pkl"
        )
        torch.save(
            snn_model.state_dict(),
            f"{self.experiment_directory}/neural_network_model.pth",
        )

        print("Training complete!")

    def inference(self, vcf_file):

        pass
