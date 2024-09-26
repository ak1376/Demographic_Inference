from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from src.utils import root_mean_squared_error

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from sklearn.model_selection import cross_val_score, cross_val_predict
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from torch.amp.autocast_mode import autocast  # Correct import for autocast
from torch.amp.grad_scaler import GradScaler  # Correct import for GradScaler

class LinearReg:
    def __init__(self, training_features, training_targets, validation_features, validation_targets):

        self.training_features = training_features
        self.training_targets = training_targets
        self.validation_features = validation_features
        self.validation_targets = validation_targets

        self.model = LinearRegression()

    def train_and_validate(self):
        """
        Train the model

        I think the additional features should be a dictionary with keys 'training', 'validation', and 'testing'
        """
        # Train the model on the training data

        training_features = self.training_features.reshape(self.training_features.shape[0], -1)

        self.model.fit(self.training_features, self.training_targets)

        training_predictions = self.model.predict(self.training_features).reshape(self.training_features.shape[0], -1)
        validation_predictions = self.model.predict(self.validation_features).reshape(self.validation_features.shape[0], -1)

        return training_predictions, validation_predictions
    
    def organizing_results(self, preprocessing_results_obj, training_predictions, validation_predictions):

        # TODO: Linear regression should be moded to the models module.
        linear_mdl_obj = {}
        linear_mdl_obj["model"] = self.model

        linear_mdl_obj["training"] = {}
        linear_mdl_obj["validation"] = {}

        linear_mdl_obj["training"]["predictions"] = training_predictions
        linear_mdl_obj["training"]["targets"] = preprocessing_results_obj["training"]["targets"]
        
        # Because the targets for each analysis are the same.

        linear_mdl_obj["validation"]["predictions"] = validation_predictions
        linear_mdl_obj["validation"]["targets"] = preprocessing_results_obj["validation"]["targets"]


        return linear_mdl_obj

class XGBoost:
    def __init__(
        self,
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        verbosity=2,
        train_percentage=0.8,
        loo=None,
    ):

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


# Hook function to inspect BatchNorm outputs
def inspect_batchnorm_output(module, input, output):
    print(f"BatchNorm Output Mean: {output.mean().item()}")
    print(f"BatchNorm Output Variance: {output.var().item()}")


class ShallowNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,  # Can be an integer or a list of hidden layer sizes
        num_layers,
        output_size,
        dropout_rate=0.1,
        weight_decay=1e-4,
        BatchNorm=False,
    ):
        super(ShallowNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        layers = []

        # Check if hidden_sizes is an integer, if so, use num_layers
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_layers  # Create a list with num_layers entries

        # First layer (input to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if BatchNorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Add BatchNorm
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))  # Dropout after the first layer

        # Hidden layers (loop through to add layers dynamically)
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if BatchNorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i]))  # Add BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout after each hidden layer

        # Output layer (no dropout after the output layer)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine the layers into a sequential container
        self.network = nn.Sequential(*layers)

        # Calculate the number of trainable parameters
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print("================================================")
        print(f"Number of trainable parameters: {num_params}")
        print("================================================")

        self.num_params = num_params

    def forward(self, x):
        return self.network(x)

    def predict(self, X, eval_mode = False):
        """
        Make predictions using the trained model.

        Args:
        X (numpy.ndarray or torch.Tensor): Input data for prediction.

        Returns:
        numpy.ndarray: Predicted outputs.
        """

        if eval_mode:
            self.eval()  # Set the model to evaluation mode

        else:
            self.train()

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if X.device != next(self.parameters()).device:
                X = X.to(next(self.parameters()).device)
            predictions = self(X)
            # print(predictions)

        return predictions.cpu().numpy()