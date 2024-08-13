"""
The experiment manager should import the following modules:
- Processor object
- delete_vcf_files function


"""

import os
import shutil
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocess import Processor, delete_vcf_files
from utils import (
    visualizing_results,
    visualize_model_predictions,
    # shap_values_plot,
    # partial_dependence_plots,
    extract_features,
    relative_squared_error,
)
from models import XGBoost, ShallowNN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import TQDMProgressBar
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import make_scorer



class Experiment_Manager:
    def __init__(self, config_file):
        # Later have a config file for model hyperparameters
        self.experiment_config = config_file

        self.upper_bound_params = config_file["upper_bound_params"]
        self.lower_bound_params = config_file["lower_bound_params"]
        self.num_sims = config_file["num_sims"]
        self.num_samples = config_file["num_samples"]
        self.experiment_name = config_file["experiment_name"]
        self.num_windows = config_file["num_windows"]
        self.window_length = config_file["window_length"]
        self.maxiter = config_file["maxiter"]
        self.mutation_rate = config_file["mutation_rate"]
        self.recombination_rate = config_file["recombination_rate"]
        self.genome_length = config_file["genome_length"]
        self.dadi_analysis = config_file["dadi_analysis"]
        self.moments_analysis = config_file["moments_analysis"]
        self.momentsLD_analysis = config_file["momentsLD_analysis"]

        self.create_directory(self.experiment_name)

    def create_directory(
        self, folder_name, base_dir="experiments", archive_subdir="archive"
    ):
        # Full paths for base and archive directories
        base_path = os.path.join(base_dir, folder_name)
        archive_path = os.path.join(base_dir, archive_subdir)

        # Ensure the archive subdirectory exists
        os.makedirs(archive_path, exist_ok=True)

        # Check if the folder already exists
        if os.path.exists(base_path):
            # Find a new name for the existing folder to move it to the archive
            i = 1
            new_folder_name = f"{folder_name}_{i}"
            new_folder_path = os.path.join(archive_path, new_folder_name)
            while os.path.exists(new_folder_path):
                i += 1
                new_folder_name = f"{folder_name}_{i}"
                new_folder_path = os.path.join(archive_path, new_folder_name)

            # Rename and move the existing folder to the archive subdirectory
            shutil.move(base_path, new_folder_path)
            print(f"Renamed and moved existing folder to: {new_folder_path}")

        # Create the new directory
        os.makedirs(base_path, exist_ok=True)
        print(f"Created new directory: {base_path}")

        self.experiment_directory = base_path

    def run(self):
        """
        This should do the preprocessing, inference, etc.
        """

        # First step: define the processor
        processor = Processor(self.experiment_config, self.experiment_directory)

        dadi_dict, moments_dict, momentsLD_dict = processor.run()

        feature_names = []

        # I want to save the results as PNG files within the results folder
        if self.dadi_analysis:
            visualizing_results(dadi_dict, "dadi", save_loc=self.experiment_directory)
            feature_names_dadi = [
                "Nb_opt_dadi",
                "N_recover_opt_dadi",
                "t_bottleneck_start_opt_dadi",
                "t_bottleneck_end_opt_dadi",
            ]
            feature_names.append(feature_names_dadi)
        if self.moments_analysis:
            visualizing_results(
                moments_dict, "moments", save_loc=self.experiment_directory
            )
            feature_names_moments = [
                "Nb_opt_moments",
                "N_recover_opt_moments",
                "t_bottleneck_start_opt_moments",
                "t_bottleneck_end_opt_moments",
            ]
            feature_names.append(feature_names_moments)
        if self.momentsLD_analysis:
            visualizing_results(
                momentsLD_dict, "MomentsLD", save_loc=self.experiment_directory
            )
            feature_names_momentsLD = [
                "Nb_opt_momentsLD",
                "N_recover_opt_momentsLD",
                "t_bottleneck_start_opt_momentsLD",
                "t_bottleneck_end_opt_momentsLD",
            ]
            feature_names.append(feature_names_momentsLD)

        # Reorder the elements by the element number of each sublist
        feature_names_reordered = []

        # Assume all sublists have the same length
        for i in range(len(feature_names[0])):
            for sublist in feature_names:
                feature_names_reordered.append(sublist[i])

        # PARAMETER INFERENCE IS COMPLETE. NOW IT'S TIME TO DO THE MACHINE LEARNING PART.

        # Probably not the most efficient, but placeholder for now

        # Convert the list to a dictionary with 0, 1, 2, ... as keys
        feature_names = {
            i: feature_names_reordered[i] for i in range(len(feature_names_reordered))
        }

        # feature_names = {
        #     0: "Nb_opt_dadi",
        #     1: "Nb_opt_moments",
        #     2: "Nb_opt_momentsLD",
        #     3: "N_recover_opt_dadi",
        #     4: "N_recover_opt_moments",
        #     5: "N_recover_opt_momentsLD",
        #     6: "t_bottleneck_start_opt_dadi",
        #     7: "t_bottleneck_start_opt_moments",
        #     8: "t_bottleneck_start_opt_momentsLD",
        #     9: "t_bottleneck_end_opt_dadi",
        #     10: "t_bottleneck_end_opt_moments",
        #     11: "t_bottleneck_end_opt_momentsLD",
        # }

        target_names = {
            0: "Nb_sample",
            1: "N_recover_sample",
            2: "t_bottleneck_start_sample",
            3: "t_bottleneck_end_sample",
        }

        # EVERYTHING BELOW IS UNOPTIMIZED AND/OR PLACEHOLDER CODE

        xgb_model = XGBoost(feature_names, target_names)
        # Note that the simulated params in both the dadi_dict and moments_dict are identical

        # Initialize an empty list to hold feature arrays
        features_list = []

        # Dynamically check and append features if the corresponding analysis is being done
        if self.dadi_analysis:
            features_dadi, targets = extract_features(
                dadi_dict["simulated_params"], dadi_dict["opt_params"]
            )
            features_list.append(features_dadi)
            error_value = relative_squared_error(y_true=targets, y_pred=features_dadi)
            print(f"The error value for dadi is: {error_value}")

        if self.moments_analysis:
            features_moments, targets = extract_features(
                moments_dict["simulated_params"], moments_dict["opt_params"]
            )
            features_list.append(features_moments)
            error_value = relative_squared_error(
                y_true=targets, y_pred=features_moments
            )
            print(f"The error value for moments is: {error_value}")

        if self.momentsLD_analysis:
            features_momentsLD, targets = extract_features(
                momentsLD_dict["simulated_params"], momentsLD_dict["opt_params"]
            )
            features_list.append(features_momentsLD)
            error_value = relative_squared_error(
                y_true=targets, y_pred=features_moments
            )
            print(f"The error value for MomentsLD is: {error_value}")

        # Concatenate all available features along axis 1
        features = (
            np.concatenate(features_list, axis=1) if features_list else np.array([])
        )

        # I want to do cross validation rather than a traditional train/test split 
        custom_scorer = make_scorer(relative_squared_error, greater_is_better=False)

        # Define the Leave-One-Out Cross-Validation strategy
        loo = LeaveOneOut()

        # Now do a train test split
        # X_train, X_test, y_train, y_test = train_test_split(
        #     features, targets, train_size=0.8, random_state=295
        # )

        # Feature scaling
        feature_scaler = StandardScaler()
        features_scaled = feature_scaler.fit_transform(features)

        # Target scaling
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets)

        # MODEL DEFINITIONS

        ## LINEAR REGRESSION
        lin_mdl = LinearRegression()

        ## XGBOOST
        xgb_model = XGBoost(feature_names, target_names)

        ## SHALLOW NEURAL NETWORK
        # Define model hyperparameters
        input_size = features_scaled.shape[1]  # Number of features
        hidden_size = 100  # Number of neurons in the hidden layer
        output_size = 4  # Number of output classes

        # Instantiate the model
        model = ShallowNN(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        )
        # Calculate the number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        # Train the linear regression model
        linear_mdl_scores = scores = cross_val_score(lin_mdl, features_scaled, targets_scaled, cv=loo, scoring=custom_scorer)
        mean_lin_mdl_score = scores.mean()

        loss_xgb, predictions_xgb, _ = xgb_model.train_and_validate(
            features_scaled, targets_scaled, cross_val=True
        )
        print(f"The Linear Regression Cross Validation error is: {mean_lin_mdl_score}")
        print("==============================")
        print(f"The XGBoost Cross Validation error is: {loss_xgb}")
        print("==============================")

        visualize_model_predictions(
            y_test_scaled,
            y_pred_lin_mdl_test,
            target_names=target_names,
            folder_loc=f"{self.experiment_directory}",
        )

        # Convert to PyTorch tensors
        # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        # y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        # X_val_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

        # Create PyTorch datasets
        # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Define DataLoaders
        # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=32)

        # Number of epochs to train for
        num_epochs = 8000
        optimizer = optim.Adam(model.parameters(), lr=3e-4)  # Adam optimizer
        criterion = torch.nn.MSELoss()

        train_losses = []
        val_losses = []
        train_epoch_losses = []
        val_epoch_losses = []

        train_predictions = []
        train_targets = []
        val_predictions = []
        val_targets = []

        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_train_loss = 0.0

            # Training loop
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.unsqueeze(1)  # Add a channel dimension
                targets = targets.unsqueeze(1)  # Add a channel dimension
                optimizer.zero_grad()  # Zero the parameter gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_train_loss += loss.item()
                train_losses.append(loss.item())  # Log batch training loss

                # Store batch predictions and targets for training
                train_predictions.append(outputs.detach().cpu())
                train_targets.append(targets.detach().cpu())

            avg_train_loss = running_train_loss / len(train_loader)
            train_epoch_losses.append(avg_train_loss)  # Log epoch training loss

            # Validation loop
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.unsqueeze(1)  # Add a channel dimension
                    targets = targets.unsqueeze(1)  # Add a channel dimension
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    running_val_loss += loss.item()
                    val_losses.append(loss.item())  # Log batch validation loss

                    # Store batch predictions and targets for validation
                    val_predictions.append(outputs.detach().cpu())
                    val_targets.append(targets.detach().cpu())

            avg_val_loss = running_val_loss / len(val_loader)
            val_epoch_losses.append(avg_val_loss)  # Log epoch validation loss

            # Print average loss for the epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Convert lists of batch predictions and targets to tensors
        train_predictions = torch.cat(train_predictions, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        # Plot the loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss (Batch)")
        plt.plot(val_losses, label="Validation Loss (Batch)")
        plt.xlabel("Batch Number")
        plt.ylabel("Log Loss")
        plt.title("Training and Validation Loss Curves")
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.legend()
        plt.show()
        plt.savefig(f"{self.experiment_directory}/loss_curves_shallow_nn.png")

        # Optionally, plot the epoch average losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_epoch_losses, label="Training Loss (Epoch Average)")
        plt.plot(val_epoch_losses, label="Validation Loss (Epoch Average)")
        plt.xlabel("Epoch Number")
        plt.ylabel("Log Average Loss")
        plt.title("Training and Validation Loss Curves (Epoch Average)")
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.legend()
        plt.show()
        plt.savefig(f"{self.experiment_directory}/epoch_loss_curves_shallow_nn.png")

        print("Training complete!")
