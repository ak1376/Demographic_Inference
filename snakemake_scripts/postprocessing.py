import numpy as np
import pickle
from scipy.stats import zscore
import json
import pandas as pd

def postprocessing(experiment_config, training_features, training_targets, validation_features, validation_targets):
    # Load data and config
    training_features = pd.read_csv(training_features)
    validation_features = pd.read_csv(validation_features)
    training_targets = pd.read_csv(training_targets)
    validation_targets = pd.read_csv(validation_targets)

    with open(experiment_config, "r") as f:
        experiment_cfg = json.load(f)

    postprocessing_dict = {
        'parameter_names': training_features.columns.tolist(),
        'target_names': training_targets.columns.tolist()
    }

    features_dict = {'training': training_features, 'validation': validation_features}
    targets_dict = {'training': training_targets, 'validation': validation_targets}

    param_types = experiment_cfg['parameters_to_estimate']
    lower_bounds = experiment_cfg['lower_bound_params']
    upper_bounds = experiment_cfg['upper_bound_params']

    for stage in ['training', 'validation']:
        features = features_dict[stage].copy()
        targets = targets_dict[stage].copy()

        print(f"\nProcessing {stage} data:")

        # ====== 1. Outlier Removal ======
        mask = np.ones(len(features), dtype=bool)
        for param in param_types:
            lb = lower_bounds[param]
            ub = upper_bounds[param]
            param_cols = [
                col for col in features.columns
                if col.endswith(f'_{param}')
                and 'FIM_element' not in col
                and 'SFS' not in col
            ]
            for col in param_cols:
                mask &= (features[col] >= lb) & (features[col] <= ub)

        n_outliers = (~mask).sum()
        print(f"Removed {n_outliers} outliers from {stage} set.")
        features = features[mask].reset_index(drop=True)
        targets = targets[mask].reset_index(drop=True)

        # ====== 2. Normalization (if requested) ======
        if experiment_cfg['normalization']:
            print("===> Normalizing the data.")
            normalized_targets = targets.copy()
            normalized_features = features.copy()

            for param in param_types:
                lb = lower_bounds[param]
                ub = upper_bounds[param]
                mean = 0.5 * (lb + ub)
                std = (ub - lb) / np.sqrt(12)

                target_col = f'simulated_params_{param}'
                if target_col in normalized_targets.columns:
                    normalized_targets[target_col] = (normalized_targets[target_col] - mean) / std

                param_cols = [
                    col for col in features.columns
                    if col.endswith(f'_{param}')
                    and 'FIM_element' not in col
                    and 'SFS' not in col
                ]
                for col in param_cols:
                    normalized_features[col] = (features[col] - mean) / std

            postprocessing_dict[stage] = {
                "normalization": experiment_cfg['normalization'],
                "predictions": features,
                "normalized_predictions": normalized_features,
                "targets": targets,
                "normalized_targets": normalized_targets
            }
        else:
            postprocessing_dict[stage] = {
                "normalization": experiment_cfg['normalization'],
                "predictions": features,
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

    postprocessing_dict = postprocessing(
        args.config_file,
        args.training_features_filepath,
        args.training_targets_filepath,
        args.validation_features_filepath,
        args.validation_targets_filepath
    )

    # Save the postprocessing dict
    with open(f'{args.sim_directory}/postprocessing_results.pkl', "wb") as f:
        pickle.dump(postprocessing_dict, f)

    print(f"Postprocessing dict keys: {postprocessing_dict['training'].keys()}")
    print("Postprocessing complete!")