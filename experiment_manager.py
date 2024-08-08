'''
The experiment manager should import the following modules:
- Processor object
- delete_vcf_files function


'''
import os
import shutil
from tqdm import tqdm
import numpy as np

from preprocess import Processor, delete_vcf_files
from utils import visualizing_results, visualize_model_predictions, shap_values_plot, partial_dependence_plots
from models import XGBoost
from sklearn.model_selection import train_test_split


class Experiment_Manager:
    def __init__(self, config_file):
        # Later have a config file for model hyperparameters
        self.experiment_config = config_file

        self.upper_bound_params = config_file['upper_bound_params']
        self.lower_bound_params = config_file['lower_bound_params']
        self.num_sims = config_file['num_sims']
        self.num_samples = config_file['num_samples']
        self.experiment_name = config_file['experiment_name']
        self.num_windows = config_file['num_windows']

        self.create_directory(self.experiment_name)

        
    def create_directory(self, folder_name, archive_subdir='archive'):
        # Create a directory for the experiment

        # Ensure the archive subdirectory exists
        if not os.path.exists(archive_subdir):
            os.makedirs(f'experiments/{archive_subdir}', exist_ok=True)

        # Check if the folder already exists
        if os.path.exists(folder_name):
            # Find a new name for the existing folder
            i = 1
            new_folder_name = f"experiments/{folder_name}_{i}"
            while os.path.exists(new_folder_name):
                i += 1
                new_folder_name = f"{folder_name}_{i}"
            
            # Rename the existing folder
            os.rename(folder_name, new_folder_name)
            
            # Move the renamed folder to the archive subdirectory
            shutil.move(new_folder_name, os.path.join(archive_subdir, new_folder_name))
            print(f"Renamed and moved existing folder to: {os.path.join(archive_subdir, new_folder_name)}")
        
        # Create the new directory
        os.makedirs(f'experiments/{folder_name}', exist_ok=True)
        print(f"Created new directory: {folder_name}")

        self.experiment_directory = folder_name

    def run(self):
        '''
        This should do the preprocessing, inference, etc.
        '''

        # First step: define the processor
        processor = Processor(self.experiment_config, self.experiment_directory)

        dadi_dict, moments_dict = processor.run()

        # I want to save the results as PNG files within the results folder 
        visualizing_results(dadi_dict, "dadi", save_loc = self.experiment_directory)
        visualizing_results(moments_dict, "moments", save_loc = self.experiment_directory)

        # PARAMETER INFERENCE IS COMPLETE. NOW IT'S TIME TO DO THE MACHINE LEARNING PART.

        # Probably not the most efficient, but placeholder for now

        feature_names = {
            0: 'Nb_opt_dadi',
            1: 'Nb_opt_moments',
            2: 'N_recover_opt_dadi',
            3: 'N_recover_opt_moments',
            4: 't_bottleneck_start_opt_dadi',
            5: 't_bottleneck_start_opt_moments',
            6: 't_bottleneck_end_opt_dadi',
            7: 't_bottleneck_end_opt_moments'
        }

        target_names = {
            0: 'Nb_sample',
            1: 'N_recover_sample',
            2: 't_bottleneck_start_sample',
            3: 't_bottleneck_end_sample'
        }

        xgb_model = XGBoost(feature_names, target_names)
        # Note that the simulated params in both the dadi_dict and moments_dict are identical
        features_dadi, targets_dadi = xgb_model.extract_features(dadi_dict['simulated_params'], dadi_dict['opt_params'])
        features_moments, targets_moments = xgb_model.extract_features(moments_dict['simulated_params'], moments_dict['opt_params'])

        features = np.concatenate((features_dadi, features_moments), axis=1)
        targets = targets_dadi # Same as targets_moments


        # Now do a train test split
        X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=xgb_model.train_percentage, random_state=295)    

        train_error, validation_error, y_pred = xgb_model.train_and_validate(X_train, y_train, X_test, y_test)

        visualize_model_predictions(y_test, y_pred, target_names=target_names, folder_loc=f'experiments/{self.experiment_directory}')
        shap_values_plot(X_test, multi_output_model=xgb_model, feature_names=feature_names, target_names=target_names, folder_loc=f'experiments/{self.experiment_directory}')


        return dadi_dict, moments_dict # CHANGE LATER 
    