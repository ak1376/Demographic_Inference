import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import re
import os
import shap
from sklearn.inspection import PartialDependenceDisplay


def visualizing_results(results_obj, analysis, save_loc="results"):
    # Extract simulated parameters
    simulated_params = results_obj["simulated_params"]

    t_bottleneck_start_sample = [d["t_bottleneck_start"] for d in simulated_params]
    t_bottleneck_end_sample = [d["t_bottleneck_end"] for d in simulated_params]
    Nb_sample = [d["Nb"] for d in simulated_params]
    N_recover_sample = [d["N_recover"] for d in simulated_params]

    # Extract optimized parameters
    opt_params = results_obj["opt_params"]

    Nb_opt = [d["Nb"] for d in opt_params]
    N_recover_opt = [d["N_recover"] for d in opt_params]
    t_bottleneck_start_opt = [d["t_bottleneck_start"] for d in opt_params]
    t_bottleneck_end_opt = [d["t_bottleneck_end"] for d in opt_params]

    # Plotting the results
    plt.figure(figsize=(12, 8))

    # Plot Nb
    plt.subplot(2, 2, 1)
    plt.scatter(Nb_sample, Nb_opt, alpha=0.5, color="blue")
    plt.xlabel("Simulated Nb")
    plt.ylabel("Optimized Nb")
    plt.title("Nb: Simulated vs Optimized")
    # plt.suptitle(f'Relative Squared Error: {sum_relative_squared_errors[0]}', fontsize=16)

    # Plot N_recover
    plt.subplot(2, 2, 2)
    plt.scatter(N_recover_sample, N_recover_opt, alpha=0.5, color="green")
    plt.xlabel("Simulated N_recover")
    plt.ylabel("Optimized N_recover")
    plt.title("N_recover: Simulated vs Optimized")
    # plt.suptitle(f'Relative Squared Error: {sum_relative_squared_errors[1]}', fontsize=16)

    # Plot t_bottleneck_start
    plt.subplot(2, 2, 3)
    plt.scatter(
        t_bottleneck_start_sample, t_bottleneck_start_opt, alpha=0.5, color="red"
    )
    plt.xlabel("Simulated t_bottleneck_start")
    plt.ylabel("Optimized t_bottleneck_start")
    plt.title("t_bottleneck_start: Simulated vs Optimized")
    # plt.suptitle(f'Relative Squared Error: {sum_relative_squared_errors[2]}', fontsize=16)

    # Plot t_bottleneck_end
    plt.subplot(2, 2, 4)
    plt.scatter(
        t_bottleneck_end_sample, t_bottleneck_end_opt, alpha=0.5, color="purple"
    )
    plt.xlabel("Simulated t_bottleneck_end")
    plt.ylabel("Optimized t_bottleneck_end")
    plt.title("t_bottleneck_end: Simulated vs Optimized")
    # plt.suptitle(f'Relative Squared Error: {sum_relative_squared_errors[3]}', fontsize=16)
    plt.tight_layout()
    print(f"/{save_loc}/inference_results_{analysis}.png")
    plt.savefig(f"{save_loc}/inference_results_{analysis}.png", format="png")
    plt.show()


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


def shap_values_plot(
    X_test,
    multi_output_model,
    feature_names,
    target_names,
    folder_loc,
    fileprefix="shap_values",
):
    # TODO: Fix this error message: 'XGBoost' object has no attribute 'estimators_'
    num_outputs = len(multi_output_model.n_estimators)

    if not os.path.exists("results"):
        os.makedirs("results")

    # Save individual SHAP plots as images
    for i in range(num_outputs):
        output_model = multi_output_model.estimators_[i]
        explainer = shap.Explainer(output_model)
        shap_values = explainer.shap_values(X_test)

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_test,
            plot_type="bar",
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"SHAP values for output {i}", pad=20)
        plt.savefig(f"results/{fileprefix}_feature_{i}.png", format="png")
        plt.close()

    # Create a 2x2 grid of subplots to display the saved images
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < num_outputs:
            img = plt.imread(f"{folder_loc}/{fileprefix}_feature_{i}.png")
            ax.imshow(img)
            ax.axis("off")  # Hide axes
            ax.set_title(
                f"SHAP values for output {target_names[i]}", fontsize=14, pad=20
            )
        else:
            fig.delaxes(ax)  # Remove empty subplot

    # Adjust layout and save the combined figure as a PNG
    plt.tight_layout()
    combined_file_name = f"results/{fileprefix}.png"
    plt.savefig(combined_file_name, format="png")
    plt.show()

    print(f"Figure saved to {combined_file_name}")


def partial_dependence_plots(
    multi_output_model,
    X_test,
    index,
    features,
    feature_names,
    target_names,
    fileprefix="partial_dependence",
):
    output_model = multi_output_model.estimators_[index]

    # Calculate the number of rows and columns for the grid
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols

    # Partial dependence plots for the specified output
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(12, 10), constrained_layout=True
    )  # Adjust the figure size
    ax = ax.flatten()  # Flatten the axes array for easy indexing

    # Create partial dependence plots and label each subplot
    disp = PartialDependenceDisplay.from_estimator(
        output_model, X_test, features, ax=ax, feature_names=feature_names
    )

    # Set the title for each subplot
    for i, axi in enumerate(ax):
        if i < n_features:
            axi.set_ylabel(feature_names[i])
        else:
            axi.set_visible(False)  # Hide any unused subplots

    plt.suptitle(f"Partial Dependence Plots for output {target_names[index]}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

    # Save the plot as a PNG
    file_name = f"{fileprefix}_feature_{target_names[index]}.png"
    plt.savefig(file_name, format="png")

    # Show the plot
    plt.show()


def relative_squared_error(y_true, y_pred):

    # Squared difference between prediction and true value, normalized by the true value
    relative_squared_errors = ((y_pred - y_true) / y_true) ** 2

    # Sum the relative squared errors across all columns and take the square root
    sum_relative_squared_errors = np.sqrt(np.sum(relative_squared_errors, axis=0))

    return sum_relative_squared_errors


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
