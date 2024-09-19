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
import colorsys

from preprocess import Processor
from utils import (
    visualizing_results,
    calculate_and_save_rrmse,
    root_mean_squared_error,
    find_outlier_indices
)
from models import LinearReg

import joblib


class Experiment_Manager:
    def __init__(self, config_file, experiment_name, experiment_directory):
        # Later have a config file for model hyperparameters
        self.experiment_config = config_file
        # self.model_config = model_config_file

        self.upper_bound_params = config_file["upper_bound_params"]
        self.lower_bound_params = config_file["lower_bound_params"]
        self.num_sims_pretrain = config_file["num_sims_pretrain"]
        self.num_sims_inference = config_file["num_sims_inference"]
        self.num_samples = config_file["num_samples"]
        # self.experiment_name = config_file["experiment_name"]
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
        # self.neural_net_hyperparameters = model_config_file["neural_net_hyperparameters"]

        self.demographic_model = config_file["demographic_model"]
        self.parameter_names = config_file["parameter_names"]
        self.optimization_initial_guess = config_file["optimization_initial_guess"]

        self.experiment_name = experiment_name # for snakemake ? 
        self.experiment_directory = experiment_directory

        # self.create_directory(self.experiment_name)

        np.random.seed(self.seed)

        self.create_color_scheme()

        # Open a file in write mode and save the dictionary as JSON
        with open(f"{self.experiment_directory}/config.json", "w") as json_file:
            json.dump(
                self.experiment_config, json_file, indent=4
            )  # indent=4 makes the JSON file more readable

    def create_color_scheme(self):

        num_params = len(self.parameter_names)
        main_colors = []
        color_shades = {}
        
        for i in range(num_params):
            # Generate main color using HSV color space
            hue = i / num_params
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            
            # Convert RGB to hex
            main_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            main_colors.append(main_color)
            
            # Generate shades
            shades = []
            for j in range(3):
                # Adjust saturation and value for shades
                sat = 1.0 - (j * 0.3)
                val = 1.0 - (j * 0.2)
                shade_rgb = colorsys.hsv_to_rgb(hue, sat, val)
                shade = '#{:02x}{:02x}{:02x}'.format(int(shade_rgb[0]*255), int(shade_rgb[1]*255), int(shade_rgb[2]*255))
                shades.append(shade)
            
            color_shades[main_color] = shades

        
        self.color_shades = color_shades
        self.main_colors = main_colors

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

        # Now I want to define training, validation, and testing indices:

        # Generate all indices and shuffle them
        all_indices = np.arange(self.num_sims_pretrain)
        np.random.shuffle(all_indices)

        # Split into training and validation indices
        n_train = int(0.8 * self.num_sims_pretrain)

        training_indices = all_indices[:n_train]
        validation_indices = all_indices[n_train:]
        testing_indices = np.arange(self.num_sims_inference)

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
            features, normalized_features, targets, upper_triangle_features = processor.pretrain_processing(indices)

            preprocessing_results_obj[stage]["predictions"] = features # This is for input to the ML model, minus the upper triangular features 
            preprocessing_results_obj[stage]["targets"] = targets
            preprocessing_results_obj[stage]["upper_triangular_FIM"] = upper_triangle_features

        preprocessing_results_obj["param_names"] = self.parameter_names

        #TODO: Calculate and save the rrmse_dict but removing the outliers from analysis
        rrmse_dict = calculate_and_save_rrmse(
            preprocessing_results_obj,
            save_path=f"{self.experiment_directory}/rrmse_dict.json"
        )

        # Open a file to save the object
        with open(
            f"{self.experiment_directory}/preprocessing_results_obj.pkl", "wb"
        ) as file:  # "wb" mode opens the file in binary write mode
            pickle.dump(preprocessing_results_obj, file)

        # TODO: This function should pass in a list of the demographic parameters for which we want to produce plots.
        
        visualizing_results(
            preprocessing_results_obj,
            save_loc=self.experiment_directory,
            analysis=f"preprocessing_results",
            stages=["training", "validation"],
            color_shades=self.color_shades,
            main_colors=self.main_colors
        )

        visualizing_results(
            preprocessing_results_obj,
            save_loc=self.experiment_directory,
            analysis=f"preprocessing_results_testing",
            stages=["testing"],
            color_shades=self.color_shades,
            main_colors=self.main_colors
        )

        ## LINEAR REGRESSION

        linear_mdl = LinearReg(training_features = preprocessing_results_obj["training"]["predictions"] ,
                        training_targets = preprocessing_results_obj["training"]["targets"],
                            validation_features = preprocessing_results_obj["validation"]["predictions"], 
                            validation_targets = preprocessing_results_obj["validation"]["targets"],
                            testing_features = preprocessing_results_obj["testing"]["predictions"],
                                testing_targets = preprocessing_results_obj["testing"]["targets"] )

        if self.experiment_config['use_FIM']:

            upper_triangular_features = {}
            upper_triangular_features['training'] = preprocessing_results_obj['training']['upper_triangular_FIM']
            upper_triangular_features['validation'] = preprocessing_results_obj['validation']['upper_triangular_FIM']
            upper_triangular_features['testing'] = preprocessing_results_obj['testing']['upper_triangular_FIM']
                
            training_predictions, validation_predictions, testing_predictions = linear_mdl.train_and_validate(upper_triangular_features)
        
        else:
            training_predictions, validation_predictions, testing_predictions = linear_mdl.train_and_validate()

        linear_mdl_obj = linear_mdl.organizing_results(preprocessing_results_obj, training_predictions, validation_predictions, testing_predictions)
        
        linear_mdl_obj["param_names"] = self.parameter_names

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
            color_shades=self.color_shades,
            main_colors=self.main_colors
        )

        joblib.dump(
            linear_mdl, f"{self.experiment_directory}/linear_regression_model.pkl"
        )
        # torch.save(
        #     snn_model.state_dict(),
        #     f"{self.experiment_directory}/neural_network_model.pth",
        # )
        
        # Save the color shades and main colors for usage with the neural network

        file_path = f'{self.experiment_directory}/color_shades.pkl'

        print("Training complete!")
