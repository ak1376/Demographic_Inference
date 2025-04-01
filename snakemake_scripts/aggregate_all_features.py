import argparse
import pickle
import numpy as np
import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


# def remove_outliers_isolation_forest(df, contamination=0.05, random_state=42):
#     """
#     Remove outliers using Isolation Forest.

#     Args:
#         df (pd.DataFrame): DataFrame containing the data.
#         contamination (float): The proportion of outliers in the data set.
#         random_state (int): Random state for reproducibility.

#     Returns:
#         pd.DataFrame: DataFrame with outliers removed.
#         pd.Series: Mask indicating which rows are kept.
#     """
#     print("\nApplying Isolation Forest for outlier detection...")
#     iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
#     iso_forest.fit(df)
#     predictions = iso_forest.predict(df)
#     mask = predictions == 1  # 1 for inliers, -1 for outliers
#     n_outliers = (predictions == -1).sum()
#     print(f"Isolation Forest detected {n_outliers} outliers out of {len(df)} samples.")
#     return df[mask], mask


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
    software_inferences_dir = [
        path for path in software_inferences_dir
        if int(path.split("_sim_")[-1].split(".pkl")[0]) in common_indices
    ]
    momentsLD_inferences_dir = [
        path for path in momentsLD_inferences_dir
        if int(path.split("_sim_")[-1].split(".pkl")[0]) in common_indices
    ]
    print(f"Filtered software inference files: {len(software_inferences_dir)}")
    print(f"Filtered momentsLD inference files: {len(momentsLD_inferences_dir)}")

    # Load experiment configuration
    print("\nLoading experiment configuration...")
    with open(experiment_config_file, "r") as f:
        experiment_config = json.load(f)
    print("Configuration loaded.")

    parameters = experiment_config['parameters_to_estimate']
    replicates = experiment_config.get('top_values_k', 1)  # Default to 1 if not specified
    lower_bounds = experiment_config.get('lower_bound_params', {})
    upper_bounds = experiment_config.get('upper_bound_params', {})
    print(f"Parameters: {parameters}")
    print(f"Replicates: {replicates}")

    software_predictions_data = []
    momentsLD_predictions_data = []
    targets_data = []

    # Process software inference files
    print("\nProcessing software inference files...")
    for idx, filepath in enumerate(software_inferences_dir):
        print(f"Processing file {idx + 1}/{len(software_inferences_dir)}: {filepath}")
        with open(filepath, 'rb') as f:
            sim_data = pickle.load(f)

        row = {}
        for replicate in range(1, replicates + 1):
            for param in parameters:
                dadi_val = sim_data['opt_params_dadi'][replicate - 1].get(param, np.nan)
                moments_val = sim_data['opt_params_moments'][replicate - 1].get(param, np.nan)
                row[f"dadi_rep{replicate}_{param}"] = dadi_val
                row[f"moments_rep{replicate}_{param}"] = moments_val

                if experiment_config.get('use_FIM', False):
                    fim = sim_data['opt_params_moments'][replicate - 1].get('upper_triangular_FIM', [])
                    # Validate FIM
                    if not isinstance(fim, (list, tuple, np.ndarray)) or (isinstance(fim, (list, tuple, np.ndarray)) and any(np.isnan(fim))):
                        print(f"  Invalid FIM detected for {filepath}. Replacing with NaNs.")
                        fim_length = (len(parameters) * (len(parameters) - 1)) // 2
                        fim = [np.nan] * fim_length
                    for i, fim_val in enumerate(fim):
                        row[f"moments_rep{replicate}_FIM_element_{i}"] = fim_val

        software_predictions_data.append(row)
        targets_data.append({f"simulated_params_{param}": sim_data['simulated_params'].get(param, np.nan) for param in parameters})

    print("Finished processing software inference files.")

    # Process MomentsLD inference files
    print("\nProcessing MomentsLD inference files...")
    for idx, filepath in enumerate(momentsLD_inferences_dir):
        print(f"Processing file {idx + 1}/{len(momentsLD_inferences_dir)}: {filepath}")
        with open(filepath, 'rb') as f:
            momentsLD_sim_data = pickle.load(f)

        row = {}
        for param in parameters:
            row[f"momentsLD_{param}"] = momentsLD_sim_data['opt_params_momentsLD'][0].get(param, np.nan)

        momentsLD_predictions_data.append(row)

    print("Finished processing MomentsLD inference files.")

    # Combine results into DataFrames
    print("\nCombining results into DataFrames...")
    software_df = pd.DataFrame(software_predictions_data)
    momentsLD_df = pd.DataFrame(momentsLD_predictions_data)
    targets_df = pd.DataFrame(targets_data)

    combined_predictions_df = pd.concat([software_df, momentsLD_df], axis=1)
    print(f"Combined DataFrame shape: {combined_predictions_df.shape}")

    # DEBUG STUFF
    print("\nChecking for NaNs before normalization...")
    nan_counts = combined_predictions_df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]

    if nan_columns.empty:
        print("No NaNs found in combined_predictions_df.")
    else:
        print("Columns with NaNs:")
        print(nan_columns)

    # Print rows that contain NaNs
    if not combined_predictions_df.isna().any().any():
        print("No rows with NaNs found in combined_predictions_df.")
    else:
        print("Rows with NaNs (showing top 5 rows with any NaN values):")
        print(combined_predictions_df[combined_predictions_df.isna().any(axis=1)].head())

    # Drop rows with NaN values in the combined DataFrame
    print("\nDropping rows with NaN values...")
    combined_predictions_df = combined_predictions_df.dropna()
    targets_df = targets_df.loc[combined_predictions_df.index].reset_index(drop=True)
    print(f"DataFrame shape after dropping NaN rows: {combined_predictions_df.shape}")

    # # Handle missing values using KNN Imputer
    # print("\nHandling missing values using KNN Imputer...")
    # imputer = KNNImputer(n_neighbors=5)
    # combined_predictions_df_imputed = pd.DataFrame(imputer.fit_transform(combined_predictions_df),
    #                                               columns=combined_predictions_df.columns)
    # print("Missing value handling complete.")

    # # Optionally, ensure imputed values respect bounds by clipping
    # # This step can help prevent outliers introduced by imputation
    # print("\nClipping imputed values to parameter bounds...")
    # for param in parameters:
    #     matching_columns = [col for col in combined_predictions_df_imputed.columns if param in col]
    #     lower = lower_bounds.get(param, -np.inf)
    #     upper = upper_bounds.get(param, np.inf)
    #     for col in matching_columns:
    #         combined_predictions_df_imputed[col] = combined_predictions_df_imputed[col].clip(lower, upper)
    # print("Clipping complete.")

    # # Remove extreme outliers using Isolation Forest
    # clean_predictions_df, mask = remove_outliers_isolation_forest(combined_predictions_df_imputed, contamination=0.05)
    # clean_targets_df = targets_df[mask].reset_index(drop=True)

    # n_removed = len(combined_predictions_df_imputed) - len(clean_predictions_df)
    # print(f"Removed {n_removed} outlier samples ({(n_removed / len(combined_predictions_df_imputed)) * 100:.2f}% of data)")

    # # Check if any samples remain
    # if clean_predictions_df.empty:
    #     print("No samples remain after outlier removal. Exiting...")
    #     return

    clean_predictions_df = combined_predictions_df.copy()
    clean_targets_df = targets_df.copy()

    # Normalize FIM columns (if present)
    fim_columns = [col for col in clean_predictions_df.columns if 'FIM_element' in col]
    if fim_columns:
        print("\nNormalizing FIM elements...")
        scaler = StandardScaler()
        clean_predictions_df[fim_columns] = scaler.fit_transform(clean_predictions_df[fim_columns])
        print("FIM normalization complete.")

    # Split data into training and validation sets
    print("\nSplitting data into training and validation sets...")
    train_indices, val_indices = train_test_split(
        range(len(clean_predictions_df)), test_size=0.2, random_state=42
    )

    # Create preprocessing results object
    preprocessing_results_obj = {
        "training": {
            "predictions": clean_predictions_df.iloc[train_indices].reset_index(drop=True),
            "targets": clean_targets_df.iloc[train_indices].reset_index(drop=True),
            "indices": list(train_indices),
        },
        "validation": {
            "predictions": clean_predictions_df.iloc[val_indices].reset_index(drop=True),
            "targets": clean_targets_df.iloc[val_indices].reset_index(drop=True),
            "indices": list(val_indices),
        },
        "parameter_names": parameters,
    }

    # Save results
    print("\nSaving processed results...")
    training_pred_path = os.path.join(sim_directory, 'training_features.csv')
    training_target_path = os.path.join(sim_directory, 'training_targets.csv')
    validation_pred_path = os.path.join(sim_directory, 'validation_features.csv')
    validation_target_path = os.path.join(sim_directory, 'validation_targets.csv')

    preprocessing_results_obj['training']['predictions'].to_csv(training_pred_path, index=False)
    preprocessing_results_obj['training']['targets'].to_csv(training_target_path, index=False)
    preprocessing_results_obj['validation']['predictions'].to_csv(validation_pred_path, index=False)
    preprocessing_results_obj['validation']['targets'].to_csv(validation_target_path, index=False)

    with open(os.path.join(sim_directory, 'preprocessing_results_obj.pkl'), 'wb') as file:
        pickle.dump(preprocessing_results_obj, file)

    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess simulation data by removing outliers and splitting into train/validation sets.")
    parser.add_argument("experiment_config_file", type=str, help="Path to the experiment config file")
    parser.add_argument("sim_directory", type=str, help="Path to the simulation directory")
    parser.add_argument("--software_inferences_dir", nargs='+', required=True, help="List of software inference files")
    parser.add_argument("--momentsLD_inferences_dir", nargs='+', required=True, help="List of momentsLD inference files")
    args = parser.parse_args()

    main(args.experiment_config_file, args.sim_directory, args.software_inferences_dir, args.momentsLD_inferences_dir)
