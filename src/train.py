# train.py

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from src.utils import visualizing_results, calculate_model_errors


class MLPTrainer:
    def __init__(
        self,
        experiment_directory,
        model_config,
        color_shades,
        main_colors,
        param_names,
    ):
        """
        Handles preparing DataLoaders, configuring PyTorch Lightning Trainer,
        training a model, and running predictions/visualizations.
        """
        self.experiment_directory = experiment_directory
        self.model_config = model_config
        self.color_shades = color_shades
        self.main_colors = main_colors
        self.param_names = param_names

    def _prepare_dataloaders(self, X_train, y_train, X_val, y_val):
        """
        Converts input arrays to Torch datasets and wraps them in DataLoaders.
        """
        # Convert to float32 numpy arrays
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        # Build TensorDatasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )

        # Create DataLoaders
        batch_size = self.model_config["neural_net_hyperparameters"]["batch_size"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    def _configure_trainer(self):
        """
        Builds and returns a PyTorch Lightning Trainer, optionally with EarlyStopping.
        """
        callbacks = []
        hyperparams = self.model_config["neural_net_hyperparameters"]

        # EarlyStopping configuration (if enabled)
        if hyperparams.get("EarlyStopping", False):
            patience_value = hyperparams.get("patience", 10)
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=patience_value,
                mode="min",
            )
            callbacks.append(early_stopping)

        # TensorBoard logger
        logger = TensorBoardLogger(save_dir="tb_logs", name="my_model")

        # Accelerator and device config
        device_type = "gpu" if torch.cuda.is_available() else "cpu"
        max_epochs = hyperparams.get("num_epochs", 10)

        # Build Trainer
        trainer = Trainer(
            logger=logger,
            max_epochs=max_epochs,
            callbacks=callbacks,
            accelerator=device_type,
            devices=1,  # typically 1 device for simpler training
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        return trainer

    def train(self, model, X_train, y_train, X_val, y_val):
        """
        Orchestrates the data preparation, trainer configuration, and model fitting process.
        Returns the trained model and lists of train/validation losses per epoch.
        """
        # Prepare DataLoaders
        train_loader, val_loader = self._prepare_dataloaders(X_train, y_train, X_val, y_val)

        # Configure Trainer
        trainer = self._configure_trainer()

        # Fit the model
        trainer.fit(model, train_loader, val_loader)

        # Retrieve losses from rank 0 (main) process
        if trainer.is_global_zero:
            train_losses = model.train_losses_per_epoch
            val_losses = model.val_losses_per_epoch
        else:
            train_losses, val_losses = [], []

        return model, train_losses, val_losses

    def predict(
        self,
        model,
        training_data,
        validation_data,
        training_targets,
        validation_targets,
        visualize=True,
    ):
        """
        Runs inference on training/validation data, optionally visualizes results,
        and saves MSE errors to file.
        """
        # Convert arrays to torch.Tensor
        training_features = torch.tensor(training_data.to_numpy(), dtype=torch.float32)
        validation_features = torch.tensor(validation_data.to_numpy(), dtype=torch.float32)

        # Inference (train mode for training set, eval mode for validation set)
        training_predictions = model.predict_from_trained_network(training_features, eval_mode=False)
        validation_predictions = model.predict_from_trained_network(validation_features, eval_mode=True)

        # Prepare results dictionary
        snn_mdl_obj = {
            "training": {
                "predictions": training_predictions,
                "targets": training_targets,
            },
            "validation": {
                "predictions": validation_predictions,
                "targets": validation_targets,
            },
            "param_names": self.param_names,
            "num_params": model.num_params,
        }

        # Visualization (if requested)
        if visualize:
            visualizing_results(
                snn_mdl_obj,
                "snn_results",
                save_loc=self.experiment_directory,
                stages=["training", "validation"],
                color_shades=self.color_shades,
                main_colors=self.main_colors,
            )

        # Calculate and save model errors
        model_error = calculate_model_errors(snn_mdl_obj, "snn", datasets=["training", "validation"])
        with open(f"{self.experiment_directory}/MLP_model_error.txt", "w") as f:
            for model_name, dataset_errors in model_error.items():
                f.write(f"\n{model_name.upper()} Model Errors:\n")
                for dataset, error_value in dataset_errors.items():
                    f.write(f"  {dataset.capitalize()} RMSE: {error_value:.4f}\n")

        return snn_mdl_obj
