import argparse
from src.preprocess import Processor
import pickle
import numpy as np
import json


def main(experiment_config_file, sim_directory, software_inferences_file_list):
    '''
    Aggregates the software inference for each simulation, ensuring training and validation data are kept separate.
    '''
    # Load the experiment configuration
    with open(experiment_config_file, "r") as f:
        experiment_config = json.load(f)
        
    # Initialize processor
    processor = Processor(
        experiment_config,
        experiment_directory=sim_directory,
        recombination_rate=experiment_config["recombination_rate"],
        mutation_rate=experiment_config["mutation_rate"],
    )

    # Split the data into training and validation sets
    all_indices = np.arange(experiment_config["num_sims_pretrain"])
    np.random.shuffle(all_indices)
    training_indices = all_indices[:int(experiment_config["training_percentage"] * experiment_config["num_sims_pretrain"])]
    validation_indices = all_indices[int(experiment_config["training_percentage"] * experiment_config["num_sims_pretrain"]):]

    preprocessing_results_obj = {
        stage: {} for stage in ["training", "validation"]
    }

    # Separate loop to process each stage (training/validation)
    for stage, indices in [
        ("training", training_indices),
        ("validation", validation_indices)
    ]:
        # Step 1: Initialize lists to collect analysis data
        analysis_data = []
        upper_triangular_data = []
        targets_data = []
        ll_data = []
        
        # Step 2: Dynamically extract and append data for each analysis type
        for analysis_type in ['dadi_analysis', 'moments_analysis', 'momentsLD_analysis']:
            if processor.experiment_config.get(analysis_type):
                analysis_key = 'opt_params_' + analysis_type.split('_')[0]  # e.g., 'opt_params_dadi'

                analysis_type_data = []
                targets_type_data = []
                ll_values_data = []

                # Filter software inference files for the current stage
                software_inferences_list = [software_inferences_file_list[i] for i in indices]

                for result_file in software_inferences_list:
                    with open(result_file, "rb") as f:
                        result = pickle.load(f)

                    ll_values = result["ll_all_replicates_" + analysis_type.split('_')[0]]
                    
                    for index in np.arange(len(result[analysis_key])):  # Iterate over the demographic parameters
                        param_values = list(result[analysis_key][index].values())
                        target_values = list(result['simulated_params'].values())
                    
                        if analysis_type == 'moments_analysis' and processor.experiment_config.get('use_FIM', True):
                            # Store upper triangular FIM separately
                            upper_triangular = result['opt_params_moments'][index].get('upper_triangular_FIM', None)
                            if upper_triangular is not None:
                                upper_triangular_data.append(upper_triangular)
                                param_values = [value for value in param_values if not isinstance(value, np.ndarray)]

                        # Collect analysis and target values
                        analysis_type_data.append(param_values)
                        targets_type_data.append(target_values)

                    ll_values_data.append(ll_values)

                # Append collected data to the lists
                analysis_data.append(analysis_type_data)
                targets_data.append(targets_type_data)

        # Step 3: Convert data into NumPy arrays
        analysis_arrays = np.array(analysis_data)
        targets_arrays = np.array(targets_data)

        # Determine array dimensions
        num_analyses = processor.experiment_config['dadi_analysis'] + processor.experiment_config['moments_analysis'] + processor.experiment_config['momentsLD_analysis']
        num_sims = len(software_inferences_list)
        num_reps = len(analysis_data[0]) // num_sims
        num_params = len(analysis_data[0][0])

        # Reshape arrays to the desired format
        analysis_arrays = analysis_arrays.reshape((num_analyses, num_sims, num_reps, num_params))
        targets_arrays = targets_arrays.reshape((num_analyses, num_sims, num_reps, num_params))

        # Transpose arrays to match the desired output shape
        features = np.transpose(analysis_arrays, (1, 2, 0, 3))
        targets = np.transpose(targets_arrays, (1, 2, 0, 3))

        # Handle upper triangular data if it exists
        if upper_triangular_data:
            upper_triangular_array = np.array(upper_triangular_data).reshape((1, num_sims, num_reps, upper_triangular_data[0].shape[0]))
            upper_triangular_array = np.transpose(upper_triangular_array, (1, 2, 0, 3))
        else:
            upper_triangular_array = None

        # Store features and targets in the preprocessing results object
        preprocessing_results_obj[stage]["predictions"] = features
        preprocessing_results_obj[stage]["targets"] = targets
        preprocessing_results_obj[stage]["upper_triangular_FIM"] = upper_triangular_array

        # Reshape features for saving
        features = features.reshape(features.shape[0], -1)

        # Concatenate features with FIM data if applicable
        if experiment_config['use_FIM']:
            upper_triangle_features = upper_triangular_array.reshape(upper_triangular_array.shape[0], -1) #type:ignore 
            preprocessing_results_obj[stage]["upper_triangular_FIM_reshape"] = upper_triangle_features
            all_features = np.concatenate((features, upper_triangle_features), axis=1)
        else:
            all_features = features

        # Save the features and targets
        np.save(f"{sim_directory}/{stage}_features.npy", all_features)
        targets = targets[:, 0, 0, :]  # Extract ground truth values for the first replicate
        np.save(f"{sim_directory}/{stage}_targets.npy", targets)

    # Save the preprocessing results object
    with open(f"{sim_directory}/preprocessing_results_obj.pkl", "wb") as file:
        pickle.dump(preprocessing_results_obj, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config_file", type=str, help="Path to the experiment config file")
    parser.add_argument("sim_directory", type=str, help="Path to the simulation directory")
    parser.add_argument("software_inferences_file_list", type=str, nargs='+', help="List of filepaths to the software inference results")
    args = parser.parse_args()
    main(args.experiment_config_file, args.sim_directory, args.software_inferences_file_list)