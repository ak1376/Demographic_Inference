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
import pandas as pd


def postprocessing(experiment_config, training_features, training_targets, validation_features, validation_targets):

    # Load data and config
    training_features = pd.read_csv(training_features, index_col=0)
    validation_features = pd.read_csv(validation_features, index_col=0)
    training_targets = pd.read_csv(training_targets, index_col=0)
    validation_targets = pd.read_csv(validation_targets, index_col=0)

    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    postprocessing_dict = {
        'parameter_names': training_features.columns.tolist(),
        'target_names': training_targets.columns.tolist()
    }

    features_dict = {'training': training_features, 'validation': validation_features}
    targets_dict = {'training': training_targets, 'validation': validation_targets}

    # Define parameters to process
    param_types = ['Na', 'N1', 'N2', 't_split', 'm']

    for stage in ['training', 'validation']:
        features = features_dict[stage]
        targets = targets_dict[stage]
        outliers_imputed = None
        

        print(f"\nProcessing {stage} data:")

        if experiment_config['remove_outliers']:
            for param in param_types:
                # Get bounds
                lower_bound = experiment_config['lower_bound_params'].get(param)
                upper_bound = experiment_config['upper_bound_params'].get(param)

                if lower_bound is None or upper_bound is None:
                    continue

                # Find all columns ending with this parameter
                param_cols = [col for col in features.columns if col.endswith('_' + param)]
                
                for col in param_cols:
                    # Identify outliers
                    outlier_mask = (features[col] < lower_bound) | (features[col] > upper_bound)
                    valid_mask = ~outlier_mask

                    if outlier_mask.any():
                        # Use median of valid values only
                        valid_median = features.loc[valid_mask, col].median() #type:ignore
                        features.loc[outlier_mask, col] = valid_median
                        # print(f"{col}: Replaced {outlier_mask.sum()} outliers. New range: [{features[col].min():.3f}, {features[col].max():.3f}]")

            outliers_imputed = features.copy()

        if experiment_config['normalization']:
            print("===> Normalizing the data.")
            
            # Normalize targets column by column using corresponding bounds
            normalized_targets = targets.copy()
            for param in experiment_config['parameter_names']:
                lower_bound = experiment_config['lower_bound_params'][param]
                upper_bound = experiment_config['upper_bound_params'][param]
                mean = 0.5 * (upper_bound + lower_bound)
                std = (upper_bound - lower_bound) / np.sqrt(12)
                target_col = f'simulated_params_{param}'
                normalized_targets[target_col] = (targets[target_col] - mean) / std
            
            # Normalize parameter features using same bounds
            features_copy = features.copy()
            for param in ['Na', 'N1', 'N2', 't_split', 'm']:
                param_cols = [col for col in features.columns if col.endswith('_' + param)]
                lower_bound = experiment_config['lower_bound_params'][param]
                upper_bound = experiment_config['upper_bound_params'][param]
                mean = 0.5 * (upper_bound + lower_bound)
                std = (upper_bound - lower_bound) / np.sqrt(12)
                
                for col in param_cols:
                    features_copy[col] = (features[col] - mean) / std

            postprocessing_dict[stage] = { #type:ignore
                "normalization": experiment_config['normalization'],
                "predictions": outliers_imputed if outliers_imputed is not None else features_dict[stage],
                "normalized_predictions": features_copy,
                "targets": targets,
                "normalized_targets": normalized_targets
            }
            
        else:
            postprocessing_dict[stage] = { #type:ignore
                "normalization": experiment_config['normalization'],
                "predictions": outliers_imputed if outliers_imputed is not None else features_dict[stage],
                "normalized_predictions": None,
                "targets": targets,
                "normalized_targets": None
            }

    return postprocessing_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--training_features_filepath", type=str, required=True)
    parser.add_argument("--validation_features_filepath", type=str, required=True)
    parser.add_argument("--training_targets_filepath", type=str, required=True)
    parser.add_argument("--validation_targets_filepath", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)

    args = parser.parse_args()

    postprocessing_dict = postprocessing(args.config_file, args.training_features_filepath, args.training_targets_filepath, args.validation_features_filepath, args.validation_targets_filepath)
    # Save the postprocessing dict
    with open(f'{args.sim_directory}/postprocessing_results.pkl', "wb") as f:
        pickle.dump(postprocessing_dict, f)

    print(f"Postprocessing dict keys: {postprocessing_dict['training'].keys()}") #type:ignore

    print("Postprocessing complete!")

