import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import re
import os
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
import pickle
import ray


def extract_features(simulated_params, opt_params, normalization=True):
    """
    opt_params can come from any of the inference methods.
    """

    # Extracting parameters from the flattened lists
    Nb_opt = [d["Nb"] for d in opt_params]
    N_recover_opt = [d["N_recover"] for d in opt_params]
    t_bottleneck_start_opt = [d["t_bottleneck_start"] for d in opt_params]
    t_bottleneck_end_opt = [d["t_bottleneck_end"] for d in opt_params]

    Nb_sample = [d["Nb"] for d in simulated_params]
    N_recover_sample = [d["N_recover"] for d in simulated_params]
    t_bottleneck_start_sample = [d["t_bottleneck_start"] for d in simulated_params]
    t_bottleneck_end_sample = [d["t_bottleneck_end"] for d in simulated_params]

    # Put all these features into a single 2D array
    opt_params_array = np.column_stack(
        (Nb_opt, N_recover_opt, t_bottleneck_start_opt, t_bottleneck_end_opt)
    )

    # Combine simulated parameters into targets
    targets = np.column_stack(
        (
            Nb_sample,
            N_recover_sample,
            t_bottleneck_start_sample,
            t_bottleneck_end_sample,
        )
    )

    if normalization:
        # Feature scaling
        feature_scaler = StandardScaler()
        features = feature_scaler.fit_transform(opt_params_array)

        # Target scaling
        target_scaler = StandardScaler()
        targets = target_scaler.fit_transform(targets)

    else:
        # Features are the optimized parameters
        features = opt_params_array

    return features, targets


def visualizing_results(
    linear_mdl_obj, analysis, save_loc="results", outlier_indices=None, stages=None
):
    # Default to all stages if not specified
    if stages is None:
        stages = ["training", "validation", "testing"]

    params = ["Nb", "N_recover", "t_bottleneck_start", "t_bottleneck_end"]
    main_colors = ["blue", "green", "red", "purple"]

    # Define color shades for each stage
    color_shades = {
        "blue": [
            "#1e90ff",
            "#4169e1",
            "#0000cd",
        ],  # Dodger blue, Royal blue, Medium blue
        "green": [
            "#90ee90",
            "#32cd32",
            "#006400",
        ],  # Light green, Lime green, Dark green
        "red": ["#ff6347", "#dc143c", "#8b0000"],  # Tomato, Crimson, Dark red
        "purple": ["#da70d6", "#9370db", "#4b0082"],  # Orchid, Medium purple, Indigo
    }

    plt.figure(figsize=(20, 15))

    for i, param in enumerate(params):
        plt.subplot(2, 2, i + 1)

        for j, stage in enumerate(stages):
            predictions = linear_mdl_obj[stage]["predictions"][:, i]
            targets = linear_mdl_obj[stage]["targets"][:, i]

            if outlier_indices is not None:
                predictions = np.delete(predictions, outlier_indices)
                targets = np.delete(targets, outlier_indices)

            plt.scatter(
                targets,
                predictions,
                alpha=0.5,
                color=color_shades[main_colors[i]][j],
                label=f"{stage.capitalize()}",
            )

        plt.xlabel(f"True {param}")
        plt.ylabel(f"Predicted {param}")
        plt.title(f"{param}: True vs Predicted")

        max_value = max(
            max(linear_mdl_obj[stage]["targets"][:, i].max() for stage in stages),
            max(linear_mdl_obj[stage]["predictions"][:, i].max() for stage in stages),
        )
        min_value = min(
            min(linear_mdl_obj[stage]["targets"][:, i].min() for stage in stages),
            min(linear_mdl_obj[stage]["predictions"][:, i].min() for stage in stages),
        )
        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="black",
            linestyle="--",
        )

        plt.legend()

    plt.tight_layout()

    filename = f"{save_loc}/{analysis}"
    if outlier_indices is not None:
        filename += "_outliers_removed"
    filename += ".png"

    plt.savefig(filename, format="png", dpi=300)
    print(f"Saved figure to: {filename}")
    plt.show()


def calculate_model_errors(model_obj, model_name, datasets):
    """
    Calculate RMSE for a single model across training, validation, and testing datasets.

    :param model_obj: Dictionary containing model predictions and targets for each dataset
    :param model_name: String name of the model (for labeling the output)
    :return: Dictionary of RMSE values for the model across datasets
    """
    # datasets = ["training", "validation", "testing"]
    errors = {}

    for dataset in datasets:
        errors[dataset] = root_mean_squared_error(
            model_obj[dataset]["targets"], model_obj[dataset]["predictions"]
        )

    return {model_name: errors}


def feature_importance(
    multi_output_model, model_number, feature_names, target_names, save_loc="results"
):
    # Plot feature importance for each output
    first_output_model = multi_output_model.estimators_[model_number]
    fig, ax = plt.subplots(figsize=(22, 8))
    xgb.plot_importance(first_output_model, ax=ax)
    plt.title(f"Feature importance for output {target_names[model_number]}")

    # Replace the feature indices with their names using the feature_names dictionary
    labels = ax.get_yticklabels()
    new_labels = []
    for label in labels:
        text = label.get_text()
        index = int(re.findall(r"\d+", text)[0])  # Extract the index
        new_labels.append(
            feature_names.get(index, text)
        )  # Use the dictionary to get the name

    # Set the ticks and the new labels
    ax.set_yticks(ax.get_yticks())  # Fix the number of ticks
    ax.set_yticklabels(new_labels)

    # Save the plot as a PDF
    plt.savefig(
        os.path.join(
            os.getcwd(),
            f"{save_loc}/feature_importance_output_{target_names[model_number]}.png",
        ),
        format="png",
    )
    plt.show()


def visualize_model_predictions(
    y_test, y_pred, target_names, folder_loc, fileprefix="model_predictions"
):
    num_outputs = y_test.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create a 2x2 grid of subplots

    for i, ax in enumerate(axes.flatten()):
        if i < num_outputs:
            ax.scatter(y_test[:, i], y_pred[:, i])
            ax.set_xlabel("True values")
            ax.set_ylabel("Predicted values")
            ax.set_title(f"True vs predicted values for output {target_names[i]}")
        else:
            fig.delaxes(ax)  # Remove empty subplot

    # Adjust layout
    plt.tight_layout()

    # Save the figure as a PDF
    file_name = f"{folder_loc}/{fileprefix}.png"
    plt.savefig(file_name, format="png")
    plt.show()

    print(f"Figure saved to {file_name}")


# def shap_values_plot(
#     X_test,
#     multi_output_model,
#     feature_names,
#     target_names,
#     folder_loc,
#     fileprefix="shap_values",
# ):
#     # TODO: Fix this error message: 'XGBoost' object has no attribute 'estimators_'
#     num_outputs = len(multi_output_model.n_estimators)

#     if not os.path.exists("results"):
#         os.makedirs("results")

#     # Save individual SHAP plots as images
#     for i in range(num_outputs):
#         output_model = multi_output_model.estimators_[i]
#         explainer = shap.Explainer(output_model)
#         shap_values = explainer.shap_values(X_test)

#         plt.figure()
#         shap.summary_plot(
#             shap_values,
#             X_test,
#             plot_type="bar",
#             feature_names=feature_names,
#             show=False,
#         )
#         plt.title(f"SHAP values for output {i}", pad=20)
#         plt.savefig(f"results/{fileprefix}_feature_{i}.png", format="png")
#         plt.close()

#     # Create a 2x2 grid of subplots to display the saved images
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     for i, ax in enumerate(axes.flatten()):
#         if i < num_outputs:
#             img = plt.imread(f"{folder_loc}/{fileprefix}_feature_{i}.png")
#             ax.imshow(img)
#             ax.axis("off")  # Hide axes
#             ax.set_title(
#                 f"SHAP values for output {target_names[i]}", fontsize=14, pad=20
#             )
#         else:
#             fig.delaxes(ax)  # Remove empty subplot

#     # Adjust layout and save the combined figure as a PNG
#     plt.tight_layout()
#     combined_file_name = f"results/{fileprefix}.png"
#     plt.savefig(combined_file_name, format="png")
#     plt.show()

#     print(f"Figure saved to {combined_file_name}")


# def partial_dependence_plots(
#     multi_output_model,
#     X_test,
#     index,
#     features,
#     feature_names,
#     target_names,
#     fileprefix="partial_dependence",
# ):
#     output_model = multi_output_model.estimators_[index]

#     # Calculate the number of rows and columns for the grid
#     n_features = len(features)
#     n_cols = 2
#     n_rows = (n_features + 1) // n_cols

#     # Partial dependence plots for the specified output
#     fig, ax = plt.subplots(
#         n_rows, n_cols, figsize=(12, 10), constrained_layout=True
#     )  # Adjust the figure size
#     ax = ax.flatten()  # Flatten the axes array for easy indexing

#     # Create partial dependence plots and label each subplot
#     disp = PartialDependenceDisplay.from_estimator(
#         output_model, X_test, features, ax=ax, feature_names=feature_names
#     )

#     # Set the title for each subplot
#     for i, axi in enumerate(ax):
#         if i < n_features:
#             axi.set_ylabel(feature_names[i])
#         else:
#             axi.set_visible(False)  # Hide any unused subplots

#     plt.suptitle(f"Partial Dependence Plots for output {target_names[index]}")
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

#     # Save the plot as a PNG
#     file_name = f"{fileprefix}_feature_{target_names[index]}.png"
#     plt.savefig(file_name, format="png")

#     # Show the plot
#     plt.show()


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


@ray.remote
def process_and_save_data(processor, indices, data_type, experiment_directory):
    """Process data and save results to pickle files."""
    dadi_dict, moments_dict, momentsLD_dict = processor.run(indices)

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
