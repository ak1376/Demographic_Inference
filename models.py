from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from utils import relative_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.adam import Adam
from sklearn.model_selection import cross_val_score, cross_val_predict


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
        custom_scorer = None, 
        loo = None
    ):

        self.feature_names = feature_names
        self.target_names = target_names
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbosity = verbosity
        self.train_percentage = train_percentage
        self.custom_scorer = custom_scorer
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

    def train_and_validate(self, X_train, y_train, X_test = None, y_test = None, cross_val = False):
        """
        Train the model
        """
        if cross_val:
            # Check that X_test and y_test are not None
            if X_test is not None and y_test is not None:
                features = np.concatenate([X_train, X_test], axis=0)
                targets = np.concatenate([y_train, y_test], axis=0)

            scores = cross_val_score(self.xgb_model, features, targets, cv=self.loo, scoring=self.custom_scorer)
            predictions = cross_val_predict(self.xgb_model, features, targets, cv=self.loo)
            
            return scores.mean(), predictions, None  # Return the mean cross-validation score
        

        else:
            if X_test is None or y_test is None:
                raise ValueError("X_test and y_test must be provided if cross_val is False")
            
            # Train the model on the training data
            self.xgb_model.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred_test = self.xgb_model.predict(X_test)

            # Calculate training and validation errors
            train_error = relative_squared_error(y_train, self.xgb_model.predict(X_train))
            validation_error = relative_squared_error(y_test, y_pred_test)

            return train_error, validation_error, y_pred_test

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
