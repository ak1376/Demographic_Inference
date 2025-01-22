import argparse
import pickle
import numpy as np
import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def main(experiment_config_file, sim_directory, software_inferences_dir, momentsLD_inferences_dir):
    print("Starting preprocessing pipeline...")

    # Ensure output directory exists
    os.makedirs(sim_directory, exist_ok=True)
    print(f"Output directory created or verified: {sim_directory}")

    # Extract simulation indices
    print("\nExtracting indices from file paths...")
    software_indices = {int(path.split("_sim_")[-1].split(".pkl")[0]) for path in software_inferences_dir}
    momentsLD_indices = {int(path.split("_sim_")[-1].split(".pkl")[0]) for path in momentsLD_inferences_dir}
    common_indices = software_indices.intersection(momentsLD_indices)

    # Filter paths by common indices
    software_inferences_dir = [path for path in software_inferences_dir if int(path.split("_sim_")[-1].split(".pkl")[0]) in common_indices]
    momentsLD_inferences_dir = [path for path in momentsLD_inferences_dir if int(path.split("_sim_")[-1].split(".pkl")[0]) in common_indices]
    print(f"Filtered software inference files: {len(software_inferences_dir)}")
    print(f"Filtered momentsLD inference files: {len(momentsLD_inferences_dir)}")

    # Load experiment configuration
    print("\nLoading experiment configuration...")
    with open(experiment_config_file, "r") as f:
        experiment_config = json.load(f)
    print("Configuration loaded.")

    parameters = ["N0", "Nb", "N_recover", "t_bottleneck_start", "t_bottleneck_end"]
    replicates = experiment_config['top_values_k']
    print(f"Parameters: {parameters}")
    print(f"Replicates: {replicates}")

    software_predictions_data = []
    momentsLD_predictions_data = []
    targets_data = []

    # Process software inference files (raw values retained)
    print("\nProcessing software inference files...")
    for idx, filepath in enumerate(software_inferences_dir):
        print(f"Processing file {idx + 1}/{len(software_inferences_dir)}: {filepath}")
        with open(filepath, 'rb') as f:
            sim_data = pickle.load(f)

        row = {}
        for replicate in range(1, replicates + 1):
            for param in parameters:
                # Get raw values (no clipping)
                dadi_val = sim_data['opt_params_dadi'][replicate - 1][param]
                moments_val = sim_data['opt_params_moments'][replicate - 1][param]

                row[f"dadi_rep{replicate}_{param}"] = dadi_val
                row[f"moments_rep{replicate}_{param}"] = moments_val

                # Extract and handle FIM if exists
                if experiment_config.get('use_FIM', False):
                    fim = sim_data['opt_params_moments'][replicate - 1].get('upper_triangular_FIM', [])
                    # Check if FIM is not iterable or contains NaN(s)
                    if not isinstance(fim, (list, tuple, np.ndarray)) or np.isnan(fim).any():
                        print(f"  Invalid FIM detected for {filepath}. Replacing with NaNs.")
                        fim_length = (len(parameters) * (len(parameters) - 1)) // 2
                        fim = [np.nan] * fim_length
                    for i, fim_val in enumerate(fim):
                        row[f"moments_rep{replicate}_FIM_element_{i}"] = fim_val

        software_predictions_data.append(row)
        targets_data.append({f"simulated_params_{param}": sim_data['simulated_params'][param] for param in parameters})

    print("Finished processing software inference files.")

    # Process MomentsLD inference files (raw values retained)
    print("\nProcessing MomentsLD inference files...")
    for idx, filepath in enumerate(momentsLD_inferences_dir):
        print(f"Processing file {idx + 1}/{len(momentsLD_inferences_dir)}: {filepath}")
        with open(filepath, 'rb') as f:
            momentsLD_sim_data = pickle.load(f)

        row = {}
        for param in parameters:
            # Get raw value (no clipping)
            val = momentsLD_sim_data['opt_params_momentsLD'][0][param]
            row[f"momentsLD_{param}"] = val

        momentsLD_predictions_data.append(row)

    print("Finished processing MomentsLD inference files.")

    # Create DataFrames
    print("\nCombining results into DataFrames...")
    software_df = pd.DataFrame(software_predictions_data)
    momentsLD_df = pd.DataFrame(momentsLD_predictions_data)
    targets_df = pd.DataFrame(targets_data)
    print(f"Software DataFrame shape: {software_df.shape}")
    print(f"MomentsLD DataFrame shape: {momentsLD_df.shape}")
    print(f"Targets DataFrame shape: {targets_df.shape}")

    # Combine the software and momentsLD data
    combined_predictions_df = pd.concat([software_df, momentsLD_df], axis=1)
    print(f"Combined DataFrame shape: {combined_predictions_df.shape}")

    # Handle missing values using KNN Imputer
    print("\nHandling missing values using KNN Imputer...")
    imputer = KNNImputer(n_neighbors=5)
    combined_predictions_df[:] = imputer.fit_transform(combined_predictions_df)
    print("Missing value handling complete.")

    # Normalize FIM columns (if present)
    fim_columns = [col for col in combined_predictions_df.columns if 'FIM_element' in col]
    if fim_columns:
        print("\nNormalizing FIM elements...")
        scaler = StandardScaler()
        combined_predictions_df[fim_columns] = scaler.fit_transform(combined_predictions_df[fim_columns])
        print("FIM normalization complete.")

    # Split data into training and validation sets
    print("\nSplitting data into training and validation sets...")
    train_indices, val_indices = train_test_split(range(len(combined_predictions_df)), test_size=0.2, random_state=42)
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

    # Save results
    print("\nSaving processed results...")
    training_pred_path = os.path.join(sim_directory, 'training_features.csv')
    training_target_path = os.path.join(sim_directory, 'training_targets.csv')
    validation_pred_path = os.path.join(sim_directory, 'validation_features.csv')
    validation_target_path = os.path.join(sim_directory, 'validation_targets.csv')
    print(f"Saving training predictions to: {training_pred_path}")
    print(f"Saving training targets to: {training_target_path}")
    print(f"Saving validation predictions to: {validation_pred_path}")
    print(f"Saving validation targets to: {validation_target_path}")
    
    preprocessing_results_obj['training']['predictions'].to_csv(training_pred_path, index=False)
    preprocessing_results_obj['training']['targets'].to_csv(training_target_path, index=False)
    preprocessing_results_obj['validation']['predictions'].to_csv(validation_pred_path, index=False)
    preprocessing_results_obj['validation']['targets'].to_csv(validation_target_path, index=False)
    with open(os.path.join(sim_directory, 'preprocessing_results_obj.pkl'), 'wb') as file:
        pickle.dump(preprocessing_results_obj, file)
    print("\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config_file", type=str, help="Path to the experiment config file")
    parser.add_argument("sim_directory", type=str, help="Path to the simulation directory")
    parser.add_argument("--software_inferences_dir", nargs='+', required=True, help="List of software inference files")
    parser.add_argument("--momentsLD_inferences_dir", nargs='+', required=True, help="List of momentsLD inference files")
    args = parser.parse_args()

    main(args.experiment_config_file, args.sim_directory, args.software_inferences_dir, args.momentsLD_inferences_dir)
