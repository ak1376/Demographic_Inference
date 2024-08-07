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

        simulation_dict, dadi_dict, moments_dict = processor.run()

        return simulation_dict, dadi_dict, moments_dict # CHANGE LATER 
    