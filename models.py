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
from sklearn.model_selection import LeaveOneOut
import torch.nn.utils as utils


class XGBoost:
    def __init__(
        self,
        feature_names,
        target_names,
        objective="reg:squarederror", 
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


@ray.remote(num_gpus=1)
class ModelActor:
    def __init__(self, model_config):
        self.model = ShallowNN(**model_config) #type: ignore
        self.model = self.model.cuda()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def train_and_evaluate(
        self, X_train, y_train, X_val, y_val, num_epochs, learning_rate
    ):
        criterion = nn.MSELoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        train_loss_curve = []
        val_loss_curve = []

        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_val, y_val = X_val.cuda(), y_val.cuda()

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Evaluate on the validation set
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            # Record the training and validation loss for this epoch
            train_loss_curve.append(loss.item())
            val_loss_curve.append(val_loss)

        # Final evaluation and predictions
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_val)

        return train_loss_curve, val_loss_curve, y_pred.cpu().numpy(), val_loss


class ShallowNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(ShallowNN, self).__init__()
        self.num_layers = num_layers

        layers = []

        # First layer (input to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        # layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm after the first layer
        layers.append(nn.ReLU())

        # Hidden layers (loop through to add layers dynamically)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm after each hidden layer
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine the layers into a sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
        X (numpy.ndarray or torch.Tensor): Input data for prediction.

        Returns:
        numpy.ndarray: Predicted outputs.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if X.device != next(self.parameters()).device:
                X = X.to(next(self.parameters()).device)
            predictions = self(X)

        return predictions.cpu().numpy()

    @staticmethod
    def train_and_validate(
        X_train,
        y_train,
        X_val,
        y_val,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        num_epochs=1000,
        learning_rate=3e-4
        # weight_decay=1e-4
    ):
        model = ShallowNN(input_size, hidden_size, output_size, num_layers).cuda()
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).cuda()
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).cuda()

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_loss.backward()
            # utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                )

        return model, train_losses, val_losses

    @staticmethod
    def plot_loss_curves(train_losses, val_losses, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", color="red", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
