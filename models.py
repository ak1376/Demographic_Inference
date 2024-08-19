from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from utils import root_mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.adam import Adam
from sklearn.model_selection import cross_val_score, cross_val_predict
from torch import optim
from torch.optim.adam import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import ray


class XGBoost:
    def __init__(
        self,
        feature_names,
        target_names,
        objective="reg:squarederror",  # TODO: Investigate whether it makes sense to change this objective function to be trained in the same way as the GHIST is evaluated?
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        verbosity=2,
        train_percentage=0.8,
        loo=None,
    ):

        self.feature_names = feature_names
        self.target_names = target_names
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbosity = verbosity
        self.train_percentage = train_percentage
        self.loo = loo

        # Initialize XGBoost model
        xgb_model = xgb.XGBRegressor(
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            verbosity=self.verbosity,
        )
        # Wrap the XGBoost model with MultiOutputRegressor
        multi_output_model = MultiOutputRegressor(xgb_model)

        self.xgb_model = multi_output_model

    def train_and_validate(
        self, X_train, y_train, X_test=None, y_test=None, cross_val=False
    ):
        """
        Train the model
        """
        if cross_val:
            # Check that X_test and y_test are not None
            if X_test is not None and y_test is not None:
                features = np.concatenate([X_train, X_test], axis=0)
                targets = np.concatenate([y_train, y_test], axis=0)

            scores = cross_val_score(
                self.xgb_model,
                X_train,
                y_train,
                cv=self.loo,
                scoring="neg_mean_squared_error",
            )
            predictions = cross_val_predict(
                self.xgb_model, features, targets, cv=self.loo
            )

            return (
                scores.mean(),
                predictions,
                None,
            )  # Return the mean cross-validation score

        else:
            if X_test is None or y_test is None:
                raise ValueError(
                    "X_test and y_test must be provided if cross_val is False"
                )

            # Train the model on the training data
            self.xgb_model.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred_test = self.xgb_model.predict(X_test)

            # Calculate training and validation errors
            train_error = root_mean_squared_error(
                y_train, self.xgb_model.predict(X_train)
            )
            validation_error = root_mean_squared_error(y_test, y_pred_test)

            return train_error, validation_error, y_pred_test


class ShallowNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, loo, experiment_directory, num_layers = 3, device="cpu"):
        super(ShallowNN, self).__init__()
        self.num_layers = num_layers
        self.loo = loo
        self.experiment_directory = experiment_directory
        self.device = device
        
        # Create a list of layers
        layers = []
        
        # First layer (input to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers (loop through to add layers dynamically)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Use nn.Sequential to combine the layers
        self.network = nn.Sequential(*layers)

        self.network = self.network.to(self.device)

    def forward(self, x):
        return self.network(x)

    @ray.remote(num_gpus=1)
    def train_and_evaluate(
        self, X_train, y_train, X_val, y_val, num_epochs=10, learning_rate=0.001
    ):
        """
        Train the model and return the training and validation loss curves.
        This is a remote function that will be run in parallel using Ray.
        """
        model = ShallowNN(
            input_size=X_train.shape[1],
            hidden_size=self.network[0].cpu().out_features,
            output_size=self.network[-1].cpu().out_features,
            loo=self.loo,
            experiment_directory=self.experiment_directory,
            num_layers=self.num_layers
        )

        model = model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        train_loss_curve = []
        val_loss_curve = []

        model.train()
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            train_loss = criterion(outputs, y_train)
            train_loss.backward()
            optimizer.step()

            # Evaluate on the validation set
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)

            # Record the training and validation loss for this epoch
            train_loss_curve.append(train_loss.cpu().item())
            val_loss_curve.append(val_loss.cpu().item())

            model.train()  # Return to training mode for the next epoch

        # Evaluate on the validation set and store predictions
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            val_loss = criterion(y_pred, y_val).item()

        return train_loss_curve, val_loss_curve, y_pred.cpu().numpy(), val_loss

    def cross_validate(self, X, y, num_epochs=10, learning_rate=0.001):
        """
        Perform Leave-One-Out Cross-Validation using Ray for parallel processing,
        return predictions, and record loss curves.
        """

        num_epochs = num_epochs
        learning_rate = learning_rate

        loo = self.loo
        validation_losses = []
        predictions = np.zeros_like(y)  # To store predictions for each sample
        all_train_loss_curves = []
        all_val_loss_curves = []

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        total_splits = loo.get_n_splits(X)  # Get the total number of splits

        # List to store Ray remote object references
        futures = []

        # Submit Ray tasks for each fold
        for train_index, val_index in loo.split(X):
            X_train, X_val = X_tensor[train_index], X_tensor[val_index]
            y_train, y_val = y_tensor[train_index], y_tensor[val_index]

            # Submit the training and evaluation task to Ray
            futures.append(
                self.train_and_evaluate.remote(
                    self,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                )
            )

        # Gather results from Ray tasks
        results = ray.get(futures)

        for i, (train_loss_curve, val_loss_curve, y_pred, val_loss) in enumerate(
            results
        ):
            # Store the loss curves and validation loss
            all_train_loss_curves.append(train_loss_curve)
            all_val_loss_curves.append(val_loss_curve)
            validation_losses.append(val_loss)

            # Store the predictions
            predictions[i] = y_pred

        # Calculate the average validation loss across all folds
        average_val_loss = np.mean(validation_losses)
        print(f"Average Validation Loss (LOO-CV): {average_val_loss}")

        return average_val_loss, predictions, all_train_loss_curves, all_val_loss_curves
    

    #TODO: Need to add num_epochs and learning_rate as arguments
    def train_on_full_data(self, X, y, num_epochs=1000, learning_rate=3e-4):
        model = ShallowNN(
            input_size=X.shape[1],
            hidden_size=self.network[0].cpu().out_features,
            output_size=self.network[-1].cpu().out_features,
            loo=self.loo,
            experiment_directory=self.experiment_directory,
        )

        model = model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Save the trained model
        model_path = f"{self.experiment_directory}/final_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


    def plot_loss_curves(self, all_train_loss_curves, all_val_loss_curves):
        """
        Plot only the average training and validation loss curves across all folds.
        """
        def average_across_sublists(sublists):
            """
            Calculate the average of elements at each position across multiple sublists.
            """
            return [sum(values) / len(values) for values in zip(*sublists)]

        # Calculate the average loss curves
        avg_train_loss_curve = average_across_sublists(all_train_loss_curves)
        avg_val_loss_curve = average_across_sublists(all_val_loss_curves)

        plt.figure()

        # Plot the average loss curves
        plt.plot(
            avg_train_loss_curve,
            label="Average Train Loss Curve",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            avg_val_loss_curve,
            label="Average Val Loss Curve",
            color="red",
            linewidth=2,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.yscale("log")
        plt.title("Average Training and Validation Loss Curves (LOO-CV)")
        plt.legend()
        plt.show()
        plt.savefig(
            f"{self.experiment_directory}/shallow_nn_loss_curves.png", format="png"
        )