# import xgboost as xgb
import numpy as np
from src.utils import root_mean_squared_error
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from sklearn.linear_model import LinearRegression
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor


class LinearReg:
    def __init__(self, training_features, training_targets, validation_features, validation_targets, 
                 regression_type="standard", **kwargs):

        self.training_features = training_features
        self.training_targets = training_targets
        self.validation_features = validation_features
        self.validation_targets = validation_targets

        self.model = None

        # Filter kwargs for Ridge
        if regression_type == "ridge":
            from sklearn.linear_model import Ridge
            ridge_kwargs = {k: v for k, v in kwargs.items() if k in ["alpha", "fit_intercept", "normalize", "solver", "random_state"]}
            print(f"Initializing Ridge with kwargs={ridge_kwargs}")
            self.model = Ridge(**ridge_kwargs)

        # Filter kwargs for Lasso
        elif regression_type == "lasso":
            from sklearn.linear_model import Lasso
            lasso_kwargs = {k: v for k, v in kwargs.items() if k in ["alpha", "fit_intercept", "normalize", "precompute", "max_iter", "tol", "warm_start", "positive", "random_state"]}
            print(f"Initializing Lasso with kwargs={lasso_kwargs}")
            self.model = Lasso(**lasso_kwargs)

        # Filter kwargs for ElasticNet
        elif regression_type == "elasticnet":
            from sklearn.linear_model import ElasticNet
            elasticnet_kwargs = {k: v for k, v in kwargs.items() if k in ["alpha", "l1_ratio", "fit_intercept", "normalize", "precompute", "max_iter", "tol", "warm_start", "positive", "random_state"]}
            print(f"Initializing ElasticNet with kwargs={elasticnet_kwargs}")
            self.model = ElasticNet(**elasticnet_kwargs)

        # Standard Linear Regression
        elif regression_type == "standard":
            from sklearn.linear_model import LinearRegression
            # LinearRegression does not accept 'alpha' or 'l1_ratio', so we exclude them
            linear_kwargs = {k: v for k, v in kwargs.items() if k in ["fit_intercept", "normalize", "copy_X", "n_jobs", "positive"]}
            print(f"Initializing LinearRegression with kwargs={linear_kwargs}")
            self.model = LinearRegression(**linear_kwargs)

        else:
            raise ValueError("Invalid regression type. Please choose from 'standard', 'ridge', 'lasso', or 'elasticnet'.")

    def train_and_validate(self):
        """
        Train the model

        I think the additional features should be a dictionary with keys 'training', 'validation', and 'testing'
        """
        # Train the model on the training data

        # training_features = self.training_features.reshape(self.training_features.shape[0], -1)

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
        linear_mdl_obj["training"]["targets"] = np.asarray(preprocessing_results_obj["training"]["targets"])
        
        # Because the targets for each analysis are the same.

        linear_mdl_obj["validation"]["predictions"] = validation_predictions
        linear_mdl_obj["validation"]["targets"] = np.asarray(preprocessing_results_obj["validation"]["targets"])


        return linear_mdl_obj

class RandomForest:
    def __init__(
        self,
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        **kwargs
    ):
        """
        A random forest regression wrapper.
        Additional **kwargs can include parameters for the RandomForestRegressor such as:
            n_estimators, criterion, max_depth, min_samples_split, etc.
        """
        from sklearn.ensemble import RandomForestRegressor

        self.training_features = training_features
        self.training_targets = training_targets
        self.validation_features = validation_features
        self.validation_targets = validation_targets
        
        valid_params = [
            "n_estimators", "criterion", "max_depth", "min_samples_split",
            "min_samples_leaf", "min_weight_fraction_leaf", "max_features",
            "max_leaf_nodes", "min_impurity_decrease", "bootstrap", "oob_score",
            "n_jobs", "random_state", "verbose", "warm_start", "ccp_alpha",
            "max_samples"
        ]
        
        rf_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        print(f"Initializing RandomForestRegressor with kwargs={rf_kwargs}")
        
        self.model = MultiOutputRegressor(RandomForestRegressor(**rf_kwargs))

    def train_and_validate(self):
        """
        Fit the RandomForestRegressor and return predictions on training and validation sets.
        """
        self.model.fit(self.training_features, self.training_targets)

        training_predictions = self.model.predict(self.training_features)
        validation_predictions = self.model.predict(self.validation_features)

        # Reshape if 1D
        if training_predictions.ndim == 1:
            training_predictions = training_predictions.reshape(-1, 1)
        if validation_predictions.ndim == 1:
            validation_predictions = validation_predictions.reshape(-1, 1)

        return training_predictions, validation_predictions

    def organizing_results(self, preprocessing_results_obj, train_preds, val_preds):
        """
        Returns a dictionary with model, predictions, and targets.
        """
        import numpy as np

        rf_mdl_obj = {
            "model": self.model,
            "training": {
                "predictions": train_preds,
                "targets": np.asarray(preprocessing_results_obj["training"]["targets"]),
            },
            "validation": {
                "predictions": val_preds,
                "targets": np.asarray(preprocessing_results_obj["validation"]["targets"]),
            },
        }
        return rf_mdl_obj

    def plot_feature_importances(self, feature_names, target_names, max_num_features=None, save_path="feature_importances.png"):
        """
        Saves a grid plot of feature importances for each output of the trained RandomForestRegressor.

        :param feature_names: List of feature names corresponding to each column.
        :param max_num_features: If set, only plot that many top features.
        :param save_path: The file path where the plot will be saved (default: 'feature_importances_grid.png').
        """
        num_outputs = len(self.model.estimators_)  # Number of outputs (single-output estimators)
        n_cols = 3  # Number of columns in the grid
        n_rows = (num_outputs + n_cols - 1) // n_cols  # Calculate number of rows needed

        # Create a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
        axes = axes.flatten()  # Flatten axes for easy indexing

        for output_idx, single_estimator in enumerate(self.model.estimators_):
            importances = single_estimator.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]

            if max_num_features is not None:
                sorted_idx = sorted_idx[:max_num_features]

            sorted_importances = importances[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]

            # Plot feature importances for the current output
            ax = axes[output_idx]
            ax.bar(range(len(sorted_importances)), sorted_importances, align="center")
            ax.set_xticks(range(len(sorted_importances)))
            ax.set_xticklabels(sorted_names, rotation=45, ha="right")
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            ax.set_title(f"Feature Importances (Output {target_names[output_idx]})")

        # Hide unused subplots
        for ax in axes[len(self.model.estimators_):]:
            ax.axis("off")

        # Save the grid plot
        plt.suptitle("Feature Importances Across Outputs", fontsize=16)
        plt.savefig(save_path)
        plt.close()

class XGBoostReg:
    def __init__(
        self,
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        **kwargs
    ):
        """
        XGBoost regressor wrapper.
        Additional **kwargs can include parameters for XGBRegressor such as:
            objective, n_estimators, learning_rate, max_depth, verbosity, alpha, etc.
        """
        import xgboost as xgb
        
        self.training_features = training_features
        self.training_targets = training_targets
        self.validation_features = validation_features
        self.validation_targets = validation_targets
        
        valid_params = [
            "objective", "n_estimators", "learning_rate", "max_depth", "verbosity",
            "alpha", "lambda", "subsample", "colsample_bytree", "min_child_weight",
            "eval_metric", "booster", "tree_method", "gamma", "reg_lambda", "reg_alpha"
        ]
        
        xgb_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        print(f"Initializing XGBRegressor with kwargs={xgb_kwargs}")

        # If user passed 'lambda', rename to 'reg_lambda'
        if 'lambda' in xgb_kwargs:
            xgb_kwargs['reg_lambda'] = xgb_kwargs.pop('lambda')

        self.model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_kwargs))

    def train_and_validate(self):
        """
        Fit XGBRegressor and return predictions on training and validation sets.
        """
        self.model.fit(self.training_features, self.training_targets)
        
        train_preds = self.model.predict(self.training_features)
        val_preds = self.model.predict(self.validation_features)

        if train_preds.ndim == 1:
            train_preds = train_preds.reshape(-1, 1)
        if val_preds.ndim == 1:
            val_preds = val_preds.reshape(-1, 1)

        return train_preds, val_preds

    def organizing_results(self, preprocessing_results_obj, train_preds, val_preds):
        import numpy as np

        xgb_mdl_obj = {
            "model": self.model,
            "training": {
                "predictions": train_preds,
                "targets": np.asarray(preprocessing_results_obj["training"]["targets"]),
            },
            "validation": {
                "predictions": val_preds,
                "targets": np.asarray(preprocessing_results_obj["validation"]["targets"]),
            },
        }
        return xgb_mdl_obj

    def plot_feature_importances(self, feature_names, target_names=None, max_num_features=None, save_path="xgb_feature_importances.png"):
        """
        Saves a grid plot of feature importances for each output of the trained XGBoostRegressor.

        :param feature_names: List of feature names corresponding to each column.
        :param target_names: List of target names for labeling the outputs (optional).
        :param max_num_features: If set, only plot that many top features for each output.
        :param save_path: The file path where the plot will be saved (default: 'xgb_feature_importances.png').
        """
        num_outputs = len(self.model.estimators_)  # Number of outputs (single-output estimators)
        n_cols = 3  # Number of columns in the grid
        n_rows = (num_outputs + n_cols - 1) // n_cols  # Calculate the number of rows needed

        # Create a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
        axes = axes.flatten()  # Flatten axes for easy indexing

        for output_idx, single_estimator in enumerate(self.model.estimators_):
            importances = single_estimator.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]

            if max_num_features is not None:
                sorted_idx = sorted_idx[:max_num_features]

            sorted_importances = importances[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]

            # Plot feature importances for the current output
            ax = axes[output_idx]
            ax.bar(range(len(sorted_importances)), sorted_importances, align="center", color="orange")
            ax.set_xticks(range(len(sorted_importances)))
            ax.set_xticklabels(sorted_names, rotation=45, ha="right")
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            title = f"Feature Importances (Output {output_idx})"
            if target_names:
                title = f"Feature Importances ({target_names[output_idx]})"
            ax.set_title(title)

        # Hide unused subplots
        for ax in axes[len(self.model.estimators_):]:
            ax.axis("off")

        # Save the grid plot
        plt.suptitle("XGBoost Feature Importances Across Outputs", fontsize=16)
        plt.savefig(save_path)
        plt.close()

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
        
        # Initialize lists for storing losses
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []
        self.automatic_optimization = True  # Ensure automatic optimization is enabled
        # Save hyperparameters so they're available after distributed training
        self.save_hyperparameters()


        layers = []

        # Check if hidden_sizes is an integer, if so, create a list
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_layers

        # First layer (input to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if BatchNorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Add BatchNorm
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))  # Dropout after the first layer

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if BatchNorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i]))  # Add BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout after each hidden layer

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine the layers into a sequential container
        self.network = nn.Sequential(*layers)

        # Count number of trainable parameters
        self.num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)


        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        val_loss = self.criterion(preds, y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    def on_train_epoch_end(self):
        # Get loss from trainer and ensure it's synced across processes
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None and self.trainer.is_global_zero:
            self.train_losses_per_epoch.append(train_loss.item())
            # Save to state dict to persist across processes
            self.trainer.strategy.barrier()

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None and self.trainer.is_global_zero:
            self.val_losses_per_epoch.append(val_loss.item())
            self.trainer.strategy.barrier()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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