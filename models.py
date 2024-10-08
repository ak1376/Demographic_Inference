from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from utils import root_mean_squared_error

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from sklearn.model_selection import cross_val_score, cross_val_predict
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression

class LinearReg:
    def __init__(self, training_features, training_targets, validation_features, validation_targets, testing_features, testing_targets):

        self.training_features = training_features
        self.training_targets = training_targets
        self.validation_features = validation_features
        self.validation_targets = validation_targets
        self.testing_features = testing_features
        self.testing_targets = testing_targets

        self.model = LinearRegression()

    def train_and_validate(self):
        """
        Train the model
        """
        # Train the model on the training data
        # Reshape dynamically: flatten the second and third dimensions
        
        new_shape = (self.training_features.shape[0], -1) 
        training_features = self.training_features.reshape(new_shape)


        new_shape = (self.validation_features.shape[0], -1) 
        validation_features = self.validation_features.reshape(new_shape)

        new_shape = (self.testing_features.shape[0], -1)
        testing_features = self.testing_features.reshape(new_shape)


        self.model.fit(training_features, self.training_targets)

        training_predictions = np.expand_dims(self.model.predict(training_features) , axis = 1) 
        validation_predictions = np.expand_dims(self.model.predict(validation_features), axis = 1)
        testing_predictions = np.expand_dims(self.model.predict(testing_features), axis = 1)

        return training_predictions, validation_predictions, testing_predictions
    
    def organizing_results(self, preprocessing_results_obj, training_predictions, validation_predictions, testing_predictions):

        # TODO: Linear regression should be moded to the models module.
        linear_mdl_obj = {}
        linear_mdl_obj["model"] = self.model

        linear_mdl_obj["training"] = {}
        linear_mdl_obj["validation"] = {}
        linear_mdl_obj["testing"] = {}

        linear_mdl_obj["training"]["predictions"] = training_predictions
        linear_mdl_obj["training"]["targets"] = preprocessing_results_obj["training"]["targets"].copy()

        linear_mdl_obj["validation"]["predictions"] = validation_predictions
        linear_mdl_obj["validation"]["targets"] = preprocessing_results_obj["validation"]["targets"].copy()

        linear_mdl_obj["testing"]["predictions"] = testing_predictions
        linear_mdl_obj["testing"]["targets"] = preprocessing_results_obj["testing"]["targets"].copy()

        return linear_mdl_obj

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


# Hook function to inspect BatchNorm outputs
def inspect_batchnorm_output(module, input, output):
    print(f"BatchNorm Output Mean: {output.mean().item()}")
    print(f"BatchNorm Output Variance: {output.var().item()}")


class ShallowNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        dropout_rate=0.1,
        weight_decay=1e-4,
    ):
        super(ShallowNN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        layers = []

        # First layer (input to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        # layers.append(nn.BatchNorm1d(hidden_size))  # Add BatchNorm
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))  # Dropout after the first layer

        # Hidden layers (loop through to add layers dynamically)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))  # Add BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout after each hidden layer

        # Output layer (no dropout after the output layer)
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
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout_rate=0.1,
        batch_size=64,
        use_FIM=True,
    ):
        model = ShallowNN(
            input_size,
            hidden_size,
            output_size,
            num_layers,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        ).cuda()
        criterion = nn.MSELoss()
        optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        new_shape = (X_train.shape[0], -1) 
        training_features = X_train.reshape(new_shape)


        new_shape = (X_val.shape[0], -1) 
        validation_features = X_val.reshape(new_shape)


        # Convert training and validation data into PyTorch tensors
        X_train_tensor = torch.tensor(training_features, dtype=torch.float32).cuda()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
        X_val_tensor = torch.tensor(validation_features, dtype=torch.float32).cuda()
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).cuda()

        # if use_FIM == False:
        #     X_train_tensor = X_train_tensor[:,:8]
        #     X_val_tensor = X_val_tensor[:,:8]

        # Create DataLoader for mini-batch training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        val_losses = []

        # running_mean_mean = []
        # running_var_mean = []

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                train_loss = criterion(outputs, targets)
                train_loss.backward()
                optimizer.step()

                # Store training loss for the batch
                train_losses.append(train_loss.item())

                # **Monitoring BatchNorm Statistics**
                # Register the hook to all BatchNorm layers in the model
                # for layer in model.modules():
                #     if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
                #         layer.register_forward_hook(inspect_batchnorm_output)

                # Validate on the full validation set
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                # Store validation loss for the batch
                val_losses.append(val_loss)

            if (epoch) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Last Batch Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss:.4f}"
                )

        return model, train_losses, val_losses

    @staticmethod
    def plot_loss_curves(train_losses, val_losses, save_path):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", color="red", linewidth=2)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training and Validation Loss Curves (Batch-by-Batch)")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
