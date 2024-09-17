from models import ShallowNN
import os
from utils import visualizing_results, calculate_model_errors
import pickle
import torch
import numpy as np

class Trainer:
    def __init__(self, experiment_directory, model_config, color_shades, main_colors, param_names, use_FIM=True):
        self.experiment_directory = experiment_directory
        self.model_config = model_config
        self.num_epochs = self.model_config["num_epochs"]
        self.learning_rate = self.model_config["learning_rate"]
        self.color_shades = color_shades
        self.main_colors = main_colors
        self.param_names = param_names
        self.use_FIM = use_FIM

    def train(
        self,
        training_data,
        training_targets,
        validation_data,
        validation_targets,
        additional_features = None,
        visualize=True,
    ):
        snn_model, train_losses, val_losses = ShallowNN.train_and_validate(
            X_train=training_data,
            y_train=training_targets,
            X_val=validation_data,
            y_val=validation_targets,
            input_size=self.model_config["input_size"],
            hidden_size=self.model_config["hidden_size"],
            output_size=self.model_config["output_size"],
            num_layers=self.model_config["num_layers"],
            num_epochs=self.model_config["num_epochs"],
            learning_rate=self.model_config["learning_rate"],
            weight_decay=self.model_config["weight_decay"],
            dropout_rate=self.model_config["dropout_rate"],
            additional_features = additional_features,
            use_FIM=self.use_FIM

        )

        if visualize:
            ShallowNN.plot_loss_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                save_path=f"{self.experiment_directory}/loss_curves.png",
            )

        return snn_model, train_losses, val_losses

    def predict(
        self,
        snn_model,
        training_data,
        validation_data,
        training_targets,
        validation_targets,
        additional_features,
        visualize=True,
    ):

        num_sims, num_reps, num_analyses, num_params = training_data.shape[0], training_data.shape[1], training_data.shape[2], training_data.shape[3]

        training_features = training_data.reshape(num_sims * num_reps, num_analyses*num_params)
        training_targets = np.expand_dims(training_targets[:,:,0,:], axis = 2)

        if additional_features is not None:
            additional_features_training = additional_features['training'].reshape(-1, additional_features['training'].shape[3])
            training_features = np.concatenate((training_features, additional_features_training), axis = 1) 

        training_features = torch.tensor(training_features, dtype = torch.float32).cuda()

        num_sims, num_reps, num_analyses, num_params = validation_data.shape[0], validation_data.shape[1], validation_data.shape[2], validation_data.shape[3]

        validation_features = validation_data.reshape(num_sims * num_reps, num_analyses*num_params)
        validation_targets = np.expand_dims(validation_targets[:,:,0,:], axis = 2)      

        if additional_features is not None:
            additional_features_validation = additional_features['validation'].reshape(-1, additional_features['validation'].shape[3])
            validation_features = np.concatenate((validation_features, additional_features_validation), axis = 1) 

        validation_features = torch.tensor(validation_features, dtype = torch.float32).cuda()

        training_predictions = snn_model.predict(training_features).reshape(training_data.shape[0], num_reps, 1, num_params)
        validation_predictions = snn_model.predict(validation_features).reshape(validation_data.shape[0], num_reps, 1, num_params)

        snn_mdl_obj = {}
        snn_mdl_obj["training"] = {}
        snn_mdl_obj["validation"] = {}
        snn_mdl_obj["testing"] = {}

        snn_mdl_obj["training"]["predictions"] = training_predictions
        snn_mdl_obj["training"]["targets"] = training_targets

        snn_mdl_obj["validation"]["predictions"] = validation_predictions
        snn_mdl_obj["validation"]["targets"] = validation_targets

        snn_mdl_obj["param_names"] = self.param_names

        if visualize:
            visualizing_results(
                snn_mdl_obj,
                "snn_results",
                save_loc=self.experiment_directory,
                stages=["training", "validation"],
                color_shades=self.color_shades, 
                main_colors=self.main_colors
            )

        #TODO: Fix this for 3 dimensional
        model_error = calculate_model_errors(
            snn_mdl_obj, "snn", datasets=["training", "validation"]
        )

        # Print results
        with open(f"{self.experiment_directory}/MLP_model_error.txt", "w") as f:
            for model, datasets in model_error.items():
                f.write(f"\n{model.upper()} Model Errors:\n")
                for dataset, error in datasets.items():
                    f.write(f"  {dataset.capitalize()} RMSE: {error:.4f}\n")
        print(
            f"Results have been saved to {self.experiment_directory}/MLP_model_error.txt"
        )

        return snn_mdl_obj

        # Save the results object
        # with open(f"{self.experiment_directory}/snn_results.pkl", 'wb') as f:
        #     pickle.dump(snn_mdl_obj, f)
