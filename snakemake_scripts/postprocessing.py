'''
This rule should do any specified post processing: 
1. outlier removal 
2. normalization

This should also calculate the rmse and do plotting

'''

import numpy as np
import pickle
from scipy.stats import zscore
import json


def postprocessing(experiment_config, preprocessing_results_obj, training_features,training_targets, validation_features, validation_targets):

    # Load in the training features, training targets, validation features, and validation targets
    training_features = np.load(training_features)
    validation_features = np.load(validation_features)
    training_targets = np.load(training_targets)
    validation_targets = np.load(validation_targets)

    # Load in the experiment config json 
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    with open(preprocessing_results_obj, "rb") as file:
        preprocessing_results_obj = pickle.load(file)


    # Postprocessing dict has the features in a way that we can directly input into the ML model
    # Preprocessing dict has all the features unshaped and organized so we can do plotting and RMSE calculations

    postprocessing_dict = {}

    features_dict = {
        'training': training_features,
        'validation': validation_features
    }

    targets_dict = {
        'training': training_targets,
        'validation': validation_targets
    }

    for stage in ['training', 'validation']:
        num_sims, num_reps, num_analyses, num_params = preprocessing_results_obj[stage]['predictions'].shape
        
        features = features_dict[stage]
        targets = targets_dict[stage]

        if experiment_config['remove_outliers'] == True:
            # Remove outliers
            
            print("===> Removing outliers and imputing with median values.")

            # NOW LET'S DO OUTLIER REMOVAL AND MEDIAN IMPUTATION. 
            reshaped = features.copy()
            # Step 2: Calculate Z-scores for the entire array
            z_scores = np.abs(zscore(features, axis=0))
            # Define the threshold for outliers (Grubbs test Z-score = 3)
            threshold = 3
            outliers = z_scores > threshold

            print(f"Number of outliers: {np.sum(outliers)}")

            # Step 3: Replace outliers with the median of the non-outlier values
            # Compute the median of the values that are not outliers
            median_value = np.median(reshaped[~outliers])

            # Replace outliers with the median
            reshaped[outliers] = median_value

            features = reshaped.copy()

        normalized_features = []

        if experiment_config['normalization'] == True:
            print("===> Normalizing the data.")
            
            # Extract upper bound values based on the keys in 'parameters_to_estimate'
            upper_bound_values = np.array([experiment_config['upper_bound_params'][key] for key in experiment_config['parameters_to_estimate']])

            # Extract lower bound values based on the keys in 'parameters_to_estimate'
            lower_bound_values = np.array([experiment_config['lower_bound_params'][key] for key in experiment_config['parameters_to_estimate']])

            # Calculate mean and standard deviation vectors
            mean_vector = 0.5 * (upper_bound_values + lower_bound_values)
            std_vector = (upper_bound_values - lower_bound_values) / np.sqrt(12)  # Correct std deviation for uniform distribution

            # Normalize the targets
            normalized_targets = (targets - mean_vector) / (std_vector)

            # Check for zero values in the normalized targets
            zero_target_indices = np.where(normalized_targets == 0)
            if zero_target_indices[0].size > 0:  # If any zero values are found
                print("Warning: Zero values found in the normalized targets!")
                # Extract raw target values where normalized target values are 0
                raw_target_values = targets[zero_target_indices]
                print("Raw target values corresponding to zero normalized targets:", raw_target_values)

                # Add 1 to the normalized targets that are zero
                normalized_targets[zero_target_indices] += 1
                print("Added 1 to zero normalized target values.")
            else:
                print("No zero values found in the normalized targets.")

            targets = normalized_targets.copy()

            # NORMALIZE THE FEATURES TOO (FOR THE PREPROCESSING PLOTTING)
            mean_vector = np.mean(features, axis=0)
            std_vector = np.std(features, axis=0)

            normalized_features = (features - mean_vector) / (std_vector)

            # Check for zero values in the normalized features
            zero_feature_indices = np.where(normalized_features == 0)
            if zero_feature_indices[0].size > 0:  # If any zero values are found
                print("Warning: Zero values found in the normalized features!")
                # Extract raw feature values where normalized feature values are 0
                raw_feature_values = features[zero_feature_indices]
                print("Raw feature values corresponding to zero normalized features:", raw_feature_values)
            else:
                print("No zero values found in the normalized features.")  


        targets = targets.reshape(targets.shape[0], -1)

        postprocessing_dict[stage] = {
            "normalization": experiment_config['normalization'],
            'predictions': features,
            "normalized_predictions": normalized_features,
            'targets': targets
        }
    
    return postprocessing_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--preprocessing_results_obj_filepath", type=str, required=True)
    parser.add_argument("--training_features_filepath", type=str, required=True)
    parser.add_argument("--validation_features_filepath", type=str, required=True)
    parser.add_argument("--training_targets_filepath", type=str, required=True)
    parser.add_argument("--validation_targets_filepath", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)

    args = parser.parse_args()

    postprocessing_dict = postprocessing(args.config_file, args.preprocessing_results_obj_filepath, args.training_features_filepath, args.training_targets_filepath, args.validation_features_filepath, args.validation_targets_filepath)
    # Save the postprocessing dict
    with open(f'{args.sim_directory}/postprocessing_results.pkl', "wb") as f:
        pickle.dump(postprocessing_dict, f)

    print(f"Postprocessing dict keys: {postprocessing_dict['training'].keys()}")

    print("Postprocessing complete!")

