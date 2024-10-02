import xgboost as xgb
import numpy as np
from src.utils import root_mean_squared_error
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from sklearn.linear_model import LinearRegression
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        n_estimators=200,  # Increased for smaller learning rate
        learning_rate=0.05,  # Reduced to prevent overfitting
        max_depth=3,  # Reduced for simpler trees
        verbosity=2,
        alpha=0.01,  # L1 regularization
        lambd=1,  # L2 regularization (lambda is a reserved keyword, hence using lambd)
        subsample=0.8,  # Row subsampling to prevent overfitting
        colsample_bytree=0.8,  # Column subsampling to prevent overfitting
        min_child_weight=5,  # More conservative learning
        train_percentage=0.8,
    ):
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbosity = verbosity
        self.alpha = alpha
        self.lambd = lambd
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.train_percentage = train_percentage
        self.models = []  # To store the individual XGBRegressor models for each target
        self.eval_results = []  # To store evaluation results for each target

    def train_and_validate(
        self, X_train, y_train, X_test=None, y_test=None
    ):
        """
        Train the model and track training/validation losses per epoch.
        """
        # Ensure that test data is provided
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided")

        # Clear previous models and evaluation results
        self.models = []
        self.eval_results = []

        # Train one XGBRegressor for each target (each column in y_train/y_test)
        for i in range(y_train.shape[1]):
            estimator = xgb.XGBRegressor(
                objective=self.objective,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                verbosity=self.verbosity,
                alpha=self.alpha,  # L1 regularization
                lambd=self.lambd,  # L2 regularization (lambda)
                subsample=self.subsample,  # Row subsampling
                colsample_bytree=self.colsample_bytree,  # Column subsampling
                min_child_weight=self.min_child_weight,  # Minimum child weight
                eval_metric="rmse"  # Metric for evaluation
            )
            
            # Fit one target at a time
            estimator.fit(
                X_train,
                y_train[:, i],
                eval_set=[(X_train, y_train[:, i]), (X_test, y_test[:, i])],
                verbose=True  # Set to True if you want to see training progress
            )

            # Store the model and its eval results
            self.models.append(estimator)
            self.eval_results.append(estimator.evals_result())

        # Make predictions on both train and test data
        y_pred_train = np.column_stack([model.predict(X_train) for model in self.models])
        y_pred_test = np.column_stack([model.predict(X_test) for model in self.models])

        # Calculate training and validation errors
        train_error = root_mean_squared_error(y_train, y_pred_train)
        validation_error = root_mean_squared_error(y_test, y_pred_test)

        return train_error, validation_error, y_pred_train, y_pred_test

    def get_epoch_losses(self):
        """
        Get training and validation losses for each epoch.
        Returns two lists: training losses and validation losses for each output regressor.
        """
        if self.eval_results:
            train_losses_per_target = []
            val_losses_per_target = []

            # Extract RMSE values for each target's model
            for result in self.eval_results:
                train_losses = result['validation_0']['rmse']
                val_losses = result['validation_1']['rmse']
                train_losses_per_target.append(train_losses)
                val_losses_per_target.append(val_losses)

            return train_losses_per_target, val_losses_per_target
        else:
            raise ValueError("No evaluation results available. Train the model first.")        

# Hook function to inspect BatchNorm outputs
def inspect_batchnorm_output(module, input, output):
    print(f"BatchNorm Output Mean: {output.mean().item()}")
    print(f"BatchNorm Output Variance: {output.var().item()}")


class ShallowNN(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, num_layers, output_size, 
                 learning_rate=1e-3, weight_decay=1e-4, dropout_rate=0.1, BatchNorm=False):
        super(ShallowNN, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # Initialize lists to store epoch losses
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []

        layers = []

        # Check if hidden_sizes is an integer, if so, create a list
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_layers

        # First layer (input to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if BatchNorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Add BatchNorm
        layers.append(nn.ELU())
        layers.append(nn.Dropout(dropout_rate))  # Dropout after the first layer

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if BatchNorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i]))  # Add BatchNorm
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout after each hidden layer

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine the layers into a sequential container
        self.network = nn.Sequential(*layers)

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = self.criterion(preds, y)

        # Log the training loss per batch without adding to progress bar
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        val_loss = self.criterion(preds, y)

        # Log the validation loss per batch without adding to progress bar
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return val_loss

    def on_validation_epoch_end(self):
        # Retrieve the average training loss for the epoch
        avg_train_loss = self.trainer.callback_metrics.get('train_loss', None)
        # Retrieve the average validation loss for the epoch
        avg_val_loss = self.trainer.callback_metrics.get('val_loss', None)

        # Log the training loss to the progress bar first
        if avg_train_loss is not None:
            self.log('train_loss_epoch', avg_train_loss, prog_bar=True, logger=False, sync_dist=True)

        # Then log the validation loss to the progress bar
        if avg_val_loss is not None:
            self.log('val_loss_epoch', avg_val_loss, prog_bar=True, logger=False, sync_dist=True)

        # Update your own lists if needed
        if avg_train_loss is not None:
            self.train_losses_per_epoch.append(avg_train_loss.item())
        if avg_val_loss is not None:
            self.val_losses_per_epoch.append(avg_val_loss.item())

        # Print losses if you wish
        # current_epoch = self.trainer.current_epoch
        # if avg_train_loss is not None and avg_val_loss is not None:
        #     print(f"Epoch {current_epoch}: Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        # elif avg_val_loss is not None:
        #     print(f"Epoch {current_epoch}: Validation Loss: {avg_val_loss:.4f}")
        # else:
        #     print(f"Epoch {current_epoch}: No losses logged yet.")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        # Return both optimizer and scheduler
        return optimizer
    
    def predict_from_trained_network(self, X, eval_mode=False):
        if eval_mode:
            self.eval()
        else:
            self.train()

        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if X.device != next(self.parameters()).device:
                X = X.to(next(self.parameters()).device)
            predictions = self(X)

        return predictions.cpu().numpy()