from models import ShallowNN
import os
from utils import visualizing_results, calculate_model_errors
import pickle


class Trainer: 
    def __init__(self, experiment_directory, model_config, use_FIM = True):
        self.experiment_directory = experiment_directory
        self.model_config = model_config
        self.num_epochs=self.model_config['num_epochs']
        self.learning_rate=self.model_config['learning_rate']
        self.use_FIM = use_FIM


    def train(self, training_data, training_targets, validation_data, validation_targets, visualize = True):
        snn_model, train_losses, val_losses = ShallowNN.train_and_validate(
            X_train=training_data,
            y_train=training_targets,
            X_val=validation_data,
            y_val=validation_targets,
            input_size=self.model_config['input_size'],
            hidden_size=self.model_config['hidden_size'],
            output_size=self.model_config['output_size'],
            num_layers=self.model_config['num_layers'],
            num_epochs=self.model_config['num_epochs'],
            learning_rate=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay'],
            dropout_rate=self.model_config['dropout_rate'],
            use_FIM=self.use_FIM
        )

        if visualize:
            ShallowNN.plot_loss_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=f"{self.experiment_directory}/loss_curves.png",
        )

        return snn_model, train_losses, val_losses
    
    def predict(self, snn_model, training_data, validation_data, training_targets, validation_targets, visualize = True):

        # if self.use_FIM == False:
        #     training_predictions = snn_model.predict(training_data[:,:8])
        #     validation_predictions = snn_model.predict(validation_data[:,:8])

        # else:
        #     training_predictions = snn_model.predict(training_data)
        #     validation_predictions = snn_model.predict(validation_data)
        
        training_predictions = snn_model.predict(training_data)
        validation_predictions = snn_model.predict(validation_data)

        snn_mdl_obj = {}
        snn_mdl_obj["training"] = {}
        snn_mdl_obj["validation"] = {}
        snn_mdl_obj["testing"] = {}

        snn_mdl_obj["training"]["predictions"] = training_predictions
        snn_mdl_obj["training"]["targets"] = training_targets

        snn_mdl_obj["validation"]["predictions"] = validation_predictions
        snn_mdl_obj["validation"]["targets"] = validation_targets


        if visualize:
            visualizing_results(
            snn_mdl_obj, "snn_results", save_loc=self.experiment_directory, stages=["training", "validation"]
        )
            
        model_error = calculate_model_errors(snn_mdl_obj, "snn", datasets=["training", "validation"])

        # Print results
        with open(f"{self.experiment_directory}/MLP_model_error.txt", "w") as f:
            for model, datasets in model_error.items():
                f.write(f"\n{model.upper()} Model Errors:\n")
                for dataset, error in datasets.items():
                    f.write(f"  {dataset.capitalize()} RMSE: {error:.4f}\n")
        print(
            f"Results have been saved to {self.experiment_directory}/MLP_model_error.txt"
        )

        # Save the results object
        # with open(f"{self.experiment_directory}/snn_results.pkl", 'wb') as f:
        #     pickle.dump(snn_mdl_obj, f)


            




    