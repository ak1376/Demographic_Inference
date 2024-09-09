import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import re
import os
from sklearn.preprocessing import StandardScaler
import pickle
import json
import colorsys
import math

def visualizing_results(
    linear_mdl_obj, analysis, save_loc="results", outlier_indices=None, stages=None, color_shades=None, main_colors=None
):
    # Default to all stages if not specified
    if stages is None:
        stages = ["training", "validation", "testing"]

    params = linear_mdl_obj['param_names']
    num_params = len(params)

    rows = math.ceil(math.sqrt(num_params))
    cols = math.ceil(num_params / rows)

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, param in enumerate(params):
        plt.subplot(rows, cols, i + 1)

        all_predictions = []
        all_targets = []

        for j, stage in enumerate(stages):
            predictions = linear_mdl_obj[stage]["reshaped_features"][:, i]
            targets = linear_mdl_obj[stage]["reshaped_targets"][:, i]

            # Append all predictions and targets to determine the global min/max for each plot
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Scatter plot for each stage
            plt.scatter(
                targets,
                predictions,
                alpha=0.5,
                color=color_shades[main_colors[i % len(main_colors)]][j],  # type: ignore
                label=f"{stage.capitalize()}",
            )

        # Set equal axis limits
        max_value = max(max(all_predictions), max(all_targets))
        min_value = min(min(all_predictions), min(all_targets))
        
        plt.xlim([min_value, max_value])
        plt.ylim([min_value, max_value])

        # Add the ideal line y = x
        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="black",
            linestyle="--",
            label="Ideal: Prediction = Target",
        )

        # Set equal aspect ratio
        plt.gca().set_aspect('equal', 'box')

        # Labels and title
        plt.xlabel(f"True {param}")
        plt.ylabel(f"Predicted {param}")
        plt.title(f"{param}: True vs Predicted")

        plt.legend()

    plt.tight_layout()

    # Save the figure
    filename = f"{save_loc}/{analysis}"
    if outlier_indices is not None:
        filename += "_outliers_removed"
    filename += ".png"

    plt.savefig(filename, format="png", dpi=300)
    print(f"Saved figure to: {filename}")
    plt.show()


def calculate_model_errors(model_obj, model_name, datasets):
    """
    PLACEHOLDER FOR NOW


    Calculate RMSE for a single model across training, validation, and testing datasets.

    :param model_obj: Dictionary containing model predictions and targets for each dataset
    :param model_name: String name of the model (for labeling the output)
    :return: Dictionary of RMSE values for the model across datasets
    """
    # datasets = ["training", "validation", "testing"]
    errors = {}

    for dataset in datasets:
        errors[dataset] = root_mean_squared_error(
            model_obj[dataset]["targets"],
            model_obj[dataset][
                "predictions"
            ],  # The :4 is to only consider the first 4 columns which are the parameters of interest. For moments I also get the upper triangle of the FIM, and those aren't parameters we are inferring.
        )

    return {model_name: errors}


def calculate_and_save_rrmse(
    features_dict,
    targets_dict,
    save_path,
    dadi_analysis=False,
    moments_analysis=False,
    momentsLD_analysis=False,
):
    rrmse_dict = {}

    if dadi_analysis:
        rrmse_training_dadi = root_mean_squared_error(
            features_dict["training"]["dadi"], targets_dict["training"]["dadi"]
        )
        rrmse_validation_dadi = root_mean_squared_error(
            features_dict["validation"]["dadi"], targets_dict["validation"]["dadi"]
        )
        rrmse_testing_dadi = root_mean_squared_error(
            features_dict["testing"]["dadi"], targets_dict["testing"]["dadi"]
        )

        rrmse_dict["dadi"] = {
            "training": rrmse_training_dadi,
            "validation": rrmse_validation_dadi,
            "testing": rrmse_testing_dadi,
        }

    if moments_analysis:

        rrmse_training_moments = root_mean_squared_error(
            features_dict["training"]["moments"], targets_dict["training"]["moments"]
        )
        rrmse_validation_moments = root_mean_squared_error(
            features_dict["validation"]["moments"],
            targets_dict["validation"]["moments"],
        )
        rrmse_testing_moments = root_mean_squared_error(
            features_dict["testing"]["moments"], targets_dict["testing"]["moments"]
        )

        rrmse_dict["moments"] = {
            "training": rrmse_training_moments,
            "validation": rrmse_validation_moments,
            "testing": rrmse_testing_moments,
        }

    if momentsLD_analysis:
        rrmse_training_momentsLD = root_mean_squared_error(
            features_dict["training"]["momentsLD"],
            targets_dict["training"]["momentsLD"],
        )
        rrmse_validation_momentsLD = root_mean_squared_error(
            features_dict["validation"]["momentsLD"],
            targets_dict["validation"]["momentsLD"],
        )
        rrmse_testing_momentsLD = root_mean_squared_error(
            features_dict["testing"]["momentsLD"], targets_dict["testing"]["momentsLD"]
        )

        rrmse_dict["momentsLD"] = {
            "training": rrmse_training_momentsLD,
            "validation": rrmse_validation_momentsLD,
            "testing": rrmse_testing_momentsLD,
        }

    # Overall training, validation, and testing RMSE
    rrmse_training = np.mean([rrmse_dict[key]["training"] for key in rrmse_dict])
    rrmse_validation = np.mean([rrmse_dict[key]["validation"] for key in rrmse_dict])
    rrmse_testing = np.mean([rrmse_dict[key]["testing"] for key in rrmse_dict])

    rrmse_dict["overall"] = {
        "training": rrmse_training,
        "validation": rrmse_validation,
        "testing": rrmse_testing,
    }

    # Save rrmse_dict to a JSON file
    with open(save_path, "w") as json_file:
        json.dump(rrmse_dict, json_file, indent=4)

    return rrmse_dict

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the root mean squared error, with special handling for 8-column predictions.
    If y_pred has 8 columns, it concatenates the values in the second group of 4 columns
    with the values in the first group of 4 columns.

    :param y_true: True values (numpy array)
    :param y_pred: Predicted values (numpy array)
    :return: Root mean squared error value
    """
    # Check if y_pred has 8 columns
    if y_pred.shape[1] == 8:
        # Concatenate the second group of 4 columns with the first group
        y_pred = np.concatenate([y_pred[:, :4], y_pred[:, 4:]], axis=0)
        # Repeat y_true to match the new y_pred shape
        y_true = np.tile(y_true, (2, 1))

    # Ensure y_true and y_pred have the same shape
    assert (
        y_true.shape == y_pred.shape
    ), "Shapes of y_true and y_pred must match after processing"

    relative_error = (y_pred - y_true) / y_true
    squared_relative_error = np.square(relative_error)
    rrmse_value = np.sqrt(np.mean(np.sum(squared_relative_error, axis=1)))

    return rrmse_value


def save_windows_to_vcf(windows, prefix="window"):
    """
    Save each windowed tree sequence as a VCF file.

    Parameters:
    - windows: List of tskit.TreeSequence objects containing the random windows
    - prefix: Prefix for the VCF file names
    """
    for i, window_ts in enumerate(windows):
        vcf_file = f"{prefix}_{i+1}.vcf"
        with open(vcf_file, "w") as vcf_output:
            window_ts.write_vcf(vcf_output)
        print(f"Saved window {i+1} to {vcf_file}")


def find_outlier_indices(data, threshold=3):
    """
    Find outliers in the data using the Z-score method.

    Parameters:
    - data: Numpy array containing the data
    - threshold: Z-score threshold for identifying outliers

    Returns:
    - outliers: Numpy array containing the outliers
    """
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    outliers_indices = np.where(z_scores > threshold)[0]  # I only want the rows

    if len(outliers_indices)>0:
        print(f"Found {len(outliers_indices)} outliers")

    return outliers_indices


def resample_to_match_row_count(arr, target_rows, return_indices=False):
    # If the array already has the target number of rows, return it as is
    if arr.shape[0] == target_rows:
        if return_indices:
            return arr, np.arange(arr.shape[0])
        else:
            return arr

    # Calculate the number of repetitions needed
    num_repeats = target_rows // arr.shape[0]
    remainder = target_rows % arr.shape[0]

    # Generate indices for repetition
    indices = np.tile(np.arange(arr.shape[0]), num_repeats)

    # If there is a remainder, add additional indices
    if remainder > 0:
        indices = np.concatenate([indices, np.arange(remainder)])

    # Resample the array using the calculated indices
    resampled_array = arr[indices, :]

    if return_indices:
        return resampled_array, indices
    else:
        return resampled_array


def save_dict_to_pickle(data_dict, filename, directory):
    """Save a dictionary to a pickle file."""
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Saved: {filepath}")


def process_and_save_data(
    merged_dict,
    data_type,
    experiment_directory,
    dadi_analysis,
    moments_analysis,
    momentsLD_analysis,
):
    # TODO: I need to rewrite this so that I can call this for both pretraining mode and inference mode.
    """Process data and save results to pickle files."""
    # merged_dict = processor.pretrain_processing(indices)

    if dadi_analysis:
        dadi_dict = {
            "simulated_params": merged_dict["simulated_params"],
            "sfs": merged_dict["sfs"],
            "model_sfs": merged_dict["model_sfs_dadi"],
            "opt_theta": merged_dict["opt_theta_dadi"],
            "opt_params": merged_dict["opt_params_dadi"],
        }
    else:
        dadi_dict = {}

    if moments_analysis:
        moments_dict = {
            "simulated_params": merged_dict["simulated_params"],
            "sfs": merged_dict["sfs"],
            "model_sfs": merged_dict["model_sfs_moments"],
            "opt_theta": merged_dict["opt_theta_moments"],
            "opt_params": merged_dict["opt_params_moments"],
        }
    else:
        moments_dict = {}

    if momentsLD_analysis:
        momentsLD_dict = {
            "simulated_params": merged_dict["simulated_params"],
            "sfs": merged_dict["sfs"],
            "opt_params": merged_dict["opt_params_momentsLD"],
        }
    else:
        momentsLD_dict = {}

    for name, data in [
        ("dadi", dadi_dict),
        ("moments", moments_dict),
        ("momentsLD", momentsLD_dict),
    ]:
        filename = f"{name}_dict_{data_type}.pkl"
        save_dict_to_pickle(data, filename, experiment_directory)

    # Save ground truth (generative parameters)
    ground_truth = (
        dadi_dict.get("simulated_params")
        or moments_dict.get("simulated_params")
        or momentsLD_dict.get("simulated_params")
    )
    if ground_truth is not None:
        filename = f"ground_truth_{data_type}.pkl"
        save_dict_to_pickle(ground_truth, filename, experiment_directory)
    else:
        print(f"Warning: No ground truth found for {data_type}")

    return dadi_dict, moments_dict, momentsLD_dict


def creating_features_dict(stage, dadi_dict, moments_dict, momentsLD_dict, features_dict, targets_dict, dadi_analysis, moments_analysis, momentsLD_analysis, use_FIM = False): 
    #TODO: This code could be paired down even more. 
    # This function assumed that the parameters of interest are the first num_params keys in the dictionary. This is only applicable to use_FIM = True in moments. 
    
    feature_names = []
    
    if dadi_analysis:
        concatenated_array = np.column_stack(
            [dadi_dict["opt_params"][key] for key in dadi_dict["opt_params"]]
            )

        features_dict[stage]["dadi"] = concatenated_array

        if dadi_dict["simulated_params"]:
            concatenated_array = np.column_stack(
                [
                    dadi_dict["simulated_params"][key]
                    for key in dadi_dict["simulated_params"]
                ]
            )
            targets_dict[stage]["dadi"] = concatenated_array
        
        key_names = [key + "_dadi" for key in list(dadi_dict['opt_params'].keys())]

        feature_names.extend(key_names)


    if moments_analysis:
        if use_FIM == False:
            concatenated_array = np.column_stack(
                [
                    moments_dict["opt_params"][key]
                    for key in moments_dict["opt_params"]
                ]
            )
        else:
            # Concatenate all features except for "upper_triangular_FIM"
            concatenated_array = np.column_stack(
                [
                    np.array(moments_dict["opt_params"][key]).flatten()
                    if isinstance(moments_dict["opt_params"][key], (np.ndarray, list))
                    else np.array([moments_dict["opt_params"][key]])
                    for key in moments_dict["opt_params"]
                    if key != "upper_triangular_FIM"
                ]
            )

            # Concatenate the "upper_triangular_FIM" separately (if it exists)
            if "upper_triangular_FIM" in moments_dict["opt_params"]:
                if isinstance(moments_dict['opt_params']['N0'], np.floating):
                    upper_triangular_FIM_array = np.expand_dims(np.array(moments_dict["opt_params"]["upper_triangular_FIM"]), axis = 0)
                else:
                    upper_triangular_FIM_array = np.array(moments_dict["opt_params"]["upper_triangular_FIM"])
                # Optionally concatenate with the rest of the array or process separately
                concatenated_array = np.column_stack([concatenated_array, upper_triangular_FIM_array])

        features_dict[stage]["moments"] = concatenated_array

        if moments_dict["simulated_params"]:
            concatenated_array = np.column_stack(
                [
                    moments_dict["simulated_params"][key]
                    for key in moments_dict["simulated_params"]
                ]
            )
            targets_dict[stage]["moments"] = concatenated_array

        
        key_names = [key + "_moments" for key in list(moments_dict['opt_params'].keys())]
        feature_names.extend(key_names)

    if momentsLD_analysis:
        concatenated_array = np.column_stack(
            [
                momentsLD_dict["opt_params"][key]
                for key in momentsLD_dict["opt_params"]
            ]
        )
        features_dict[stage]["momentsLD"] = concatenated_array

        if momentsLD_dict["simulated_params"]:
            concatenated_array = np.column_stack(
                [
                    momentsLD_dict["simulated_params"][key]
                    for key in momentsLD_dict["simulated_params"]
                ]
            )
            targets_dict[stage]["momentsLD"] = concatenated_array

        key_names = [key + "_momentsLD" for key in list(momentsLD_dict['opt_params'].keys())]

        feature_names.extend(key_names)

    return features_dict, targets_dict


def concatenating_features(stage, concatenated_features, concatenated_targets, features_dict, targets_dict):

    # Now columnwise the dadi, moments, and momentsLD inferences to get a concatenated features and targets array
    concat_feats = np.column_stack(
        [features_dict[stage][subkey] for subkey in features_dict[stage]]
    )

    if targets_dict[stage]:
        concat_targets = np.column_stack(
        [targets_dict[stage]['dadi'] for subkey in features_dict[stage]] # dadi because dadi, moments, and momentsLD values for the targets are the same.
    )

    concatenated_features = concat_feats
    concatenated_targets = concat_targets


    return concatenated_features, concatenated_targets


def create_color_scheme(num_params):

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

    return color_shades, main_colors