from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
import torch
from src.utils import visualizing_results, calculate_model_errors
from pytorch_lightning.loggers import TensorBoardLogger


class MLPTrainer:
    def __init__(
        self, experiment_directory, model_config, color_shades, main_colors, param_names
    ):
        self.experiment_directory = experiment_directory
        self.model_config = model_config
        self.color_shades = color_shades
        self.main_colors = main_colors
        self.param_names = param_names

    def train(self, model, X_train, y_train, X_val, y_val):
        # Prepare datasets and dataloaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config["neural_net_hyperparameters"]["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_config["neural_net_hyperparameters"]["batch_size"],
            shuffle=False
        )

        # Add early stopping callback if enabled
        callbacks = []
        if self.model_config["neural_net_hyperparameters"]["EarlyStopping"]:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.model_config["neural_net_hyperparameters"]["patience"],
                mode="min",
            )
            callbacks.append(early_stopping)

        # Set up the logger
        logger = TensorBoardLogger("tb_logs", name="my_model")

        # Instantiate the PyTorch Lightning Trainer
        trainer = Trainer(
            logger=logger,
            max_epochs=self.model_config["neural_net_hyperparameters"]["num_epochs"],
            callbacks=callbacks,
            # log_every_n_steps=10,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            # enable_progress_bar=True,
            enable_checkpointing=True,
            devices = 3
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Store batch losses
        self.train_losses_per_epoch = model.train_losses_per_epoch
        self.val_losses_per_epoch = model.val_losses_per_epoch

        model.num_params = len(self.param_names)

        return model, self.train_losses_per_epoch, self.val_losses_per_epoch

    def predict(
        self,
        model,
        training_data,
        validation_data,
        training_targets,
        validation_targets,
        visualize=True,
    ):
        # Convert data to tensors
        training_features = torch.tensor(training_data, dtype=torch.float32)
        validation_features = torch.tensor(validation_data, dtype=torch.float32)

        # Predictions
        training_predictions = model.predict_from_trained_network(
            training_features, eval_mode=False
        )
        validation_predictions = model.predict_from_trained_network(
            validation_features, eval_mode=True
        )

        # Prepare the result object
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

        if visualize:
            visualizing_results(
                snn_mdl_obj,
                "snn_results",
                save_loc=self.experiment_directory,
                stages=["training", "validation"],
                color_shades=self.color_shades,
                main_colors=self.main_colors,
            )

        model_error = calculate_model_errors(
            snn_mdl_obj, "snn", datasets=["training", "validation"]
        )

        # Save results to a text file
        with open(f"{self.experiment_directory}/MLP_model_error.txt", "w") as f:
            for model_name, datasets in model_error.items():
                f.write(f"\n{model_name.upper()} Model Errors:\n")
                for dataset, error in datasets.items():
                    f.write(f"  {dataset.capitalize()} RMSE: {error:.4f}\n")

        return snn_mdl_obj


# from src.utils import visualizing_results, calculate_model_errors, plot_loss_curves
# import torch
# from torch.amp.autocast_mode import autocast  # Correct import for autocast
# from torch.amp.grad_scaler import GradScaler  # Correct import for GradScaler
# import torch
# import torch.nn as nn
# from torch.optim.adam import Adam
# from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard.writer import SummaryWriter

# class Trainer:
#     def __init__(self, experiment_directory, model_config, color_shades, main_colors, param_names):
#         self.experiment_directory = experiment_directory
#         self.model_config = model_config
#         self.num_epochs = self.model_config['neural_net_hyperparameters']["num_epochs"]
#         self.learning_rate = self.model_config['neural_net_hyperparameters']["learning_rate"]
#         self.weight_decay = self.model_config['neural_net_hyperparameters']["weight_decay"]
#         self.batch_size = self.model_config['neural_net_hyperparameters']["batch_size"]
#         self.EarlyStopping = self.model_config['neural_net_hyperparameters']["EarlyStopping"]
#         self.patience = self.model_config['neural_net_hyperparameters']["patience"]


#         self.color_shades = color_shades
#         self.main_colors = main_colors
#         self.param_names = param_names


#     def train(
#         self,
#         model,
#         X_train,
#         y_train,
#         X_val,
#         y_val,
#         num_epochs=1000
#     ):
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         writer = SummaryWriter(log_dir='./runs/shallow_nn_experiment')


#         scaler = GradScaler('cuda')  # Optional: change this later for mixed precision.

#         model = model.to(device)

#         criterion = nn.MSELoss()
#         optimizer = Adam(
#             model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
#         )

#         # Convert training and validation data into PyTorch tensors
#         X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#         y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
#         X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
#         y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

#         # Create DataLoader for mini-batch training
#         train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

#         validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)
#         validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

#         train_losses = []
#         val_losses = []

#         # Variables for early stopping
#         best_val_loss = float('inf')
#         patience_counter = 0

#         for epoch in range(num_epochs):
#             model.train()  # Set model to training mode
#             epoch_train_loss = 0.0
#             epoch_val_loss = 0.0

#             # Training step
#             for batch_idx, (train_inputs, train_targets) in enumerate(train_loader):
#                 optimizer.zero_grad()  # Clear gradients
#                 with autocast('cuda'):
#                     train_outputs = model(train_inputs)  # Forward pass for training
#                     train_loss = criterion(train_outputs, train_targets)  # Calculate training loss
#                 scaler.scale(train_loss).backward()  # Backward pass

#                 scaler.step(optimizer)
#                 scaler.update()

#                 # Accumulate training loss for the epoch
#                 epoch_train_loss += train_loss.item()


#                 # Log weight distributions
#                 if epoch % 10 == 0:  # Log every 10 epochs
#                     for name, param in model.named_parameters():
#                         writer.add_histogram(f'Weights/{name}', param, epoch)
#                         if param.grad is not None:
#                             writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

#             # Validation step
#             model.eval()  # Set model to evaluation mode
#             with torch.no_grad():  # Disable gradient calculation for validation
#                 for val_inputs, val_targets in validation_loader:
#                     val_outputs = model(val_inputs)  # Forward pass for validation
#                     val_loss = criterion(val_outputs, val_targets).item()  # Calculate validation loss
#                     epoch_val_loss += val_loss

#             # Calculate average losses for the epoch
#             avg_train_loss = epoch_train_loss / len(train_loader)
#             avg_val_loss = epoch_val_loss / len(validation_loader)

#             writer.add_scalar('Loss/Train', avg_train_loss, epoch)
#             writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

#             train_losses.append(avg_train_loss)
#             val_losses.append(avg_val_loss)

#             # Print train and validation loss for this epoch
#             if (epoch + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}")

#             # Early Stopping Logic
#             if self.EarlyStopping:
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     patience_counter = 0  # Reset patience counter if validation loss improves
#                 else:
#                     patience_counter += 1  # Increment patience counter if no improvement

#                 if patience_counter >= self.patience:
#                     print(f"Early stopping at epoch {epoch+1}")
#                     break

#         print("Neural Network trained LFG")
#         writer.close()


#         plot_loss_curves(train_losses, val_losses, f'{self.experiment_directory}/loss_curves.png')

#         return model, train_losses, val_losses

#     def predict_from_trained_network(self, model, X, eval_mode = False):
#         """
#         Make predictions using the trained model.

#         Args:
#         X (numpy.ndarray or torch.Tensor): Input data for prediction.

#         Returns:
#         numpy.ndarray: Predicted outputs.
#         """

#         if eval_mode:
#             model.eval()  # Set the model to evaluation mode

#         else:
#             model.train()

#         with torch.no_grad():
#             # if isinstance(X, np.ndarray):
#             #     X = torch.tensor(X, dtype=torch.float32)
#             if X.device != next(model.parameters()).device:
#                 X = X.to(next(model.parameters()).device)
#             predictions = model(X)
#             # print(predictions)

#         return predictions.cpu().numpy()

#     def predict(
#         self,
#         snn_model,
#         training_data,
#         validation_data,
#         training_targets,
#         validation_targets,
#         visualize=True,
#     ):

#         training_features = torch.tensor(training_data, dtype = torch.float32).cuda()
#         validation_features = torch.tensor(validation_data, dtype = torch.float32).cuda()

#         training_predictions = snn_model.predict(training_features, eval_mode = False)
#         validation_predictions = snn_model.predict(validation_features, eval_mode = True)

#         snn_mdl_obj = {}
#         snn_mdl_obj["training"] = {}
#         snn_mdl_obj["validation"] = {}
#         snn_mdl_obj["testing"] = {}

#         snn_mdl_obj["training"]["predictions"] = training_predictions
#         snn_mdl_obj["training"]["targets"] = training_targets

#         snn_mdl_obj["validation"]["predictions"] = validation_predictions
#         snn_mdl_obj["validation"]["targets"] = validation_targets

#         snn_mdl_obj["param_names"] = self.param_names
#         snn_mdl_obj['num_params'] = snn_model.num_params

#         if visualize:
#             visualizing_results(
#                 snn_mdl_obj,
#                 "snn_results",
#                 save_loc=self.experiment_directory,
#                 stages=["training", "validation"],
#                 color_shades=self.color_shades,
#                 main_colors=self.main_colors
#             )

#         #TODO: Fix this for 3 dimensional
#         model_error = calculate_model_errors(
#             snn_mdl_obj, "snn", datasets=["training", "validation"]
#         )

#         # Print results
#         with open(f"{self.experiment_directory}/MLP_model_error.txt", "w") as f:
#             for model, datasets in model_error.items():
#                 f.write(f"\n{model.upper()} Model Errors:\n")
#                 for dataset, error in datasets.items():
#                     f.write(f"  {dataset.capitalize()} RMSE: {error:.4f}\n")
#         print(
#             f"Results have been saved to {self.experiment_directory}/MLP_model_error.txt"
#         )

#         return snn_mdl_obj

#         # Save the results object
#         # with open(f"{self.experiment_directory}/snn_results.pkl", 'wb') as f:
#         #     pickle.dump(snn_mdl_obj, f)
