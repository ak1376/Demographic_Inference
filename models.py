from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from utils import relative_squared_error


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

    def extract_features(self, simulated_params, opt_params, train_percentage=0.8):
        """
        opt_params can come from any of the inference methods.
        """

        # Extracting parameters from the flattened lists
        Nb_opt = [d["Nb"] for d in opt_params]
        N_recover_opt = [d["N_recover"] for d in opt_params]
        t_bottleneck_start_opt = [d["t_bottleneck_start"] for d in opt_params]
        t_bottleneck_end_opt = [d["t_bottleneck_end"] for d in opt_params]

        Nb_sample = [d["Nb"] for d in simulated_params]
        N_recover_sample = [d["N_recover"] for d in simulated_params]
        t_bottleneck_start_sample = [d["t_bottleneck_start"] for d in simulated_params]
        t_bottleneck_end_sample = [d["t_bottleneck_end"] for d in simulated_params]

        # Put all these features into a single 2D array
        opt_params_array = np.column_stack(
            (Nb_opt, N_recover_opt, t_bottleneck_start_opt, t_bottleneck_end_opt)
        )

        # Combine simulated parameters into targets
        targets = np.column_stack(
            (
                Nb_sample,
                N_recover_sample,
                t_bottleneck_start_sample,
                t_bottleneck_end_sample,
            )
        )

        # Features are the optimized parameters
        features = opt_params_array

        return features, targets

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
