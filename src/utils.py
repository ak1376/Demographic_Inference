import matplotlib.pyplot as plt
import numpy as np
import os
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

    # Loop through each parameter (dimension of the parameters)
    for i, param in enumerate(params):
        plt.subplot(rows, cols, i + 1)

        all_predictions = []
        all_targets = []

        # Loop through each analysis (dimension 1)
        for j, stage in enumerate(stages):
            predictions = linear_mdl_obj[stage]["predictions"]
            targets = linear_mdl_obj[stage]["targets"]
            
            # Flatten along the rows to plot results across analyses for each parameter
            predictions_flat = predictions.reshape(-1)
            targets_flat = targets.reshape(-1)

            # Append all predictions and targets to determine the global min/max for each plot
            all_predictions.extend(predictions_flat)
            all_targets.extend(targets_flat)

            # Scatter plot for each stage
            plt.scatter(
                targets_flat,
                predictions_flat,
                alpha=0.2,
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
                "predictions"]
        )

    return {model_name: errors}


def root_mean_squared_error(y_true, y_pred):

    # Ensure y_true and y_pred have the same shape
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"

    # Check for zeros in y_true to avoid division by zero
    if np.any(y_true == 0):
        zero_indices = np.where(y_true == 0)
        raise ValueError(f"Division by zero encountered in y_true at indices {zero_indices}")

    # Compute relative error
    relative_error = (y_pred - y_true) / y_true
    squared_relative_error = np.square(relative_error)
    
    # Take mean over all parameters
    rrmse_value = np.sqrt(np.mean(squared_relative_error))
    
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


def save_dict_to_pickle(data_dict, filename, directory):
    """Save a dictionary to a pickle file."""
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Saved: {filepath}")



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

def plot_loss_curves(train_losses, val_losses, save_path):

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training and Validation Loss Curves (Epoch-by-Epoch)")
    plt.legend()
    plt.savefig(save_path)
    plt.close()