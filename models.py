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


@ray.remote(num_gpus=1)
class ModelActor:
    def __init__(self, model_config):
        self.model = ShallowNN(**model_config)
        self.model = self.model.cuda()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, num_epochs, learning_rate):
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
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    @staticmethod
    def cross_validate(X, y, model_config, num_epochs=10, learning_rate=0.001):
        loo = LeaveOneOut()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        futures = []

        # Initialize lists to store results
        all_predictions = []
        all_train_losses = []
        all_val_losses = []
        
        for train_index, val_index in loo.split(X):
            actor = ModelActor.remote(model_config)
            X_train, X_val = X_tensor[train_index], X_tensor[val_index]
            y_train, y_val = y_tensor[train_index], y_tensor[val_index]
            
            futures.append(actor.train_and_evaluate.remote(X_train, y_train, X_val, y_val, num_epochs, learning_rate))
        
        results = ray.get(futures)
        
        all_train_loss_curves, all_val_loss_curves, predictions, validation_losses = zip(*results)

        # Store results
        all_predictions.append(predictions)
        all_train_losses.append(all_train_loss_curves)
        all_val_losses.append(all_val_loss_curves)
        
        average_val_loss = np.mean(validation_losses)
        print(f"Average Validation Loss (LOOCV): {average_val_loss}")
        
        predictions = np.array(predictions).squeeze()
        
        return average_val_loss, predictions, all_train_loss_curves, all_val_loss_curves

    @staticmethod
    def train_on_full_data(X, y, input_size, hidden_size, output_size, num_layers=3, num_epochs=1000, learning_rate=3e-4):
        model = ShallowNN(input_size, hidden_size, output_size, num_layers).cuda()
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        y_tensor = torch.tensor(y, dtype=torch.float32).cuda()

        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        return model

    @staticmethod
    def plot_loss_curves(all_train_loss_curves, all_val_loss_curves, save_path):
        def average_across_sublists(sublists):
            return [sum(values) / len(values) for values in zip(*sublists)]

        avg_train_loss_curve = average_across_sublists(all_train_loss_curves)
        avg_val_loss_curve = average_across_sublists(all_val_loss_curves)

        plt.figure()
        plt.plot(avg_train_loss_curve, label="Average Train Loss", color="blue", linewidth=2)
        plt.plot(avg_val_loss_curve, label="Average Val Loss", color="red", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Average Training and Validation Loss Curves (LOOCV)")
        plt.legend()
        plt.savefig(save_path)
        plt.close()