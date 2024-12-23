import argparse
import pickle
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main(experiment_config_file, sim_directory, software_inferences_dir, momentsLD_inferences_dir):

    # Load configuration
    with open(experiment_config_file, "r") as f:
        experiment_config = json.load(f)

    parameters = ["Na", "N1", "N2", "t_split"]
    replicates = experiment_config['top_values_k']

    # Containers for predictions and targets
    software_predictions_data = []
    momentsLD_predictions_data = []
    targets_data = []

    # Outlier tracking
    outlier_counts = {method: {param: 0 for param in parameters} for method in ['momentsLD', 'dadi', 'moments']}
    outlier_values = {method: {param: [] for param in parameters} for method in ['momentsLD', 'dadi', 'moments']}

    # Process software inference files
    for idx, filepath in enumerate(software_inferences_dir):
        with open(filepath, 'rb') as f:
            sim_data = pickle.load(f)
        
        row = {}
        
        # Verify dadi and moments predictions
        for replicate in range(1, replicates + 1):
            for param in parameters:
                dadi_val = sim_data['opt_params_dadi'][replicate - 1][param]
                moments_val = sim_data['opt_params_moments'][replicate - 1][param]

                row[f"dadi_rep{replicate}_{param}"] = dadi_val
                row[f"moments_rep{replicate}_{param}"] = moments_val

                # Check for outliers
                lower = experiment_config['lower_bound_params'][param]
                upper = experiment_config['upper_bound_params'][param]

                if not (lower <= dadi_val <= upper):
                    outlier_counts['dadi'][param] += 1
                    outlier_values['dadi'][param].append(dadi_val)

                if not (lower <= moments_val <= upper):
                    outlier_counts['moments'][param] += 1
                    outlier_values['moments'][param].append(moments_val)

                # Extract FIM if it exists
                if experiment_config.get('use_FIM', False):
                    if 'upper_triangular_FIM' in sim_data['opt_params_moments'][replicate - 1]:
                        fim = sim_data['opt_params_moments'][replicate - 1]['upper_triangular_FIM']
                        for i, fim_val in enumerate(fim):
                            row[f"moments_rep{replicate}_FIM_element_{i}"] = fim_val

        software_predictions_data.append(row)
        targets_data.append({f"simulated_params_{param}": sim_data['simulated_params'][param] 
                            for param in parameters})

    # Process MomentsLD inference files
    for idx, filepath in enumerate(momentsLD_inferences_dir):
        with open(filepath, 'rb') as f:
            momentsLD_sim_data = pickle.load(f)

        row = {}

        for param in parameters:
            val = momentsLD_sim_data['opt_params_momentsLD'][0][param]

            row[f"momentsLD_{param}"] = val

            # Check for outliers
            lower = experiment_config['lower_bound_params'][param]
            upper = experiment_config['upper_bound_params'][param]

            if not (lower <= val <= upper):
                outlier_counts['momentsLD'][param] += 1
                outlier_values['momentsLD'][param].append(val)

        momentsLD_predictions_data.append(row)

    # Create DataFrames
    software_df = pd.DataFrame(software_predictions_data)
    momentsLD_df = pd.DataFrame(momentsLD_predictions_data)
    targets_df = pd.DataFrame(targets_data)

    # Print outlier counts, values, and median
    for method, param_data in outlier_counts.items():
        print(f"Outliers for {method}:")
        for param, count in param_data.items():
            median_outlier = np.median(outlier_values[method][param]) if outlier_values[method][param] else None
            print(f"  Parameter {param}: {count} outliers")
            print(f"    Outlier values: {outlier_values[method][param]}")
            print(f"    Median outlier value: {median_outlier}")

    # Combine software and momentsLD predictions
    combined_predictions_df = pd.concat([software_df, momentsLD_df], axis=1)

    print(f'Shape of the combined predictions df before NA removal: {combined_predictions_df.shape}')

    # Drop any row that has at least one NaN value
    combined_predictions_df = combined_predictions_df.dropna()
    valid_indices = combined_predictions_df.dropna().index
    combined_predictions_df = combined_predictions_df.loc[valid_indices].reset_index(drop=True)
    targets_df = targets_df.loc[valid_indices].reset_index(drop=True)

    print(f'Shape of the combined predictions df AFTER NA removal: {combined_predictions_df.shape}')

    # Filter based on bounds for all methods and parameters
    mask = pd.Series(True, index=combined_predictions_df.index)

    # Filter for each parameter and method
    methods = ['momentsLD', 'dadi_rep1', 'dadi_rep2', 'moments_rep1', 'moments_rep2']

    for param in parameters:
        lower = experiment_config['lower_bound_params'][param]
        upper = experiment_config['upper_bound_params'][param]

        for method in methods:
            col_name = f"{method}_{param}"
            param_mask = (combined_predictions_df[col_name] >= lower) & (combined_predictions_df[col_name] <= upper)
            mask &= param_mask

    combined_predictions_df = combined_predictions_df[mask].reset_index(drop=True)
    targets_df = targets_df[mask].reset_index(drop=True)

    # Final NaN check
    combined_predictions_df = combined_predictions_df.dropna()
    valid_indices = combined_predictions_df.dropna().index
    combined_predictions_df = combined_predictions_df.loc[valid_indices].reset_index(drop=True)
    targets_df = targets_df.loc[valid_indices].reset_index(drop=True)

    # Z-score the FIM elements
    fim_columns = [col for col in combined_predictions_df.columns if 'FIM_element' in col]
    if fim_columns:
        scaler = StandardScaler()
        combined_predictions_df[fim_columns] = scaler.fit_transform(combined_predictions_df[fim_columns])

    # Print shapes to verify
    print("Final shapes:")
    print(f"Predictions shape: {combined_predictions_df.shape}")
    print(f"Targets shape: {targets_df.shape}")

    # Generate train/validation split indices
    train_indices, val_indices = train_test_split(
        range(len(combined_predictions_df)), 
        test_size=0.2, 
        random_state=42
    )

    preprocessing_results_obj = {
        "training": {
            "predictions": combined_predictions_df.iloc[train_indices].reset_index(drop=True),
            "targets": targets_df.iloc[train_indices].reset_index(drop=True),
            "indices": train_indices,
        },
        "validation": {
            "predictions": combined_predictions_df.iloc[val_indices].reset_index(drop=True),
            "targets": targets_df.iloc[val_indices].reset_index(drop=True),
            "indices": val_indices,
        },
        "parameter_names": parameters
    }

    preprocessing_results_obj['training']['predictions'].to_csv(f'{sim_directory}/training_features.csv', index=False)
    preprocessing_results_obj['training']['targets'].to_csv(f'{sim_directory}/training_targets.csv', index=False)
    preprocessing_results_obj['validation']['predictions'].to_csv(f'{sim_directory}/validation_features.csv', index=False)
    preprocessing_results_obj['validation']['targets'].to_csv(f'{sim_directory}/validation_targets.csv', index=False)

    with open(f'{sim_directory}/preprocessing_results_obj.pkl', 'wb') as file:
        pickle.dump(preprocessing_results_obj, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config_file", type=str, help="Path to the experiment config file")
    parser.add_argument("sim_directory", type=str, help="Path to the simulation directory")
    parser.add_argument("--software_inferences_dir", nargs='+', required=True, help="List of Moments/Dadi inferences")
    parser.add_argument("--momentsLD_inferences_dir", nargs='+', required=True, help="List of MomentsLD inferences")
    args = parser.parse_args()

    main(
        args.experiment_config_file,
        args.sim_directory,
        args.software_inferences_dir,
        args.momentsLD_inferences_dir
    )
