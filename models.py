from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from utils import relative_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam


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
    ):

        self.feature_names = feature_names
        self.target_names = target_names
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbosity = verbosity
        self.train_percentage = train_percentage

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

    def train_and_validate(self, X_train, y_train, X_test, y_test):
        """
        Train the model
        """

        self.xgb_model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.xgb_model.predict(X_test)

        train_error = relative_squared_error(y_train, self.xgb_model.predict(X_train))
        validation_error = relative_squared_error(y_test, y_pred)

        return train_error, validation_error, y_pred


class ShallowNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        # x = (self.hidden_layer(x))
        x = self.output_layer(x)
        return x