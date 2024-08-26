import pickle
import argparse
from models import ShallowNN
import ray
import os
from utils import visualizing_results, calculate_model_errors
import json
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Trainer: 
    def __init__(self, experiment_directory, model_config, use_FIM = True):
        self.experiment_directory = experiment_directory
        self.model_config = model_config
        self.num_epochs=self.model_config['num_epochs']
        self.learning_rate=self.model_config['learning_rate']
        self.use_FIM = use_FIM
        ray.init(num_cpus=os.cpu_count(), num_gpus=3, local_mode=False)


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

        print(f'Training data shape: {training_data.shape}')
        print(f'Validation data shape: {validation_data.shape}')
        if self.use_FIM == False:
            training_predictions = snn_model.predict(training_data[:,:8])
            # print(f'Training predictions shape: {training_predictions.shape}')
            # print(f'Validation Data shape: {validation_data.shape}')
            validation_predictions = snn_model.predict(validation_data[:,:8])


        else:
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

        return snn_mdl_obj

def main(experiment_directory, model_config_file, features_file, use_FIM = True):
    # Load model config
    with open(model_config_file, 'r') as f:
        model_config = json.load(f)

    # Load features
    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    
    # print(features.keys())
    # print(features['training'].keys())
    
    trainer = Trainer(experiment_directory, model_config, use_FIM = use_FIM)

    # Train the model
    snn_model, train_losses, val_losses = trainer.train(
        features['training']['features'],
        features['training']['targets'],
        features['validation']['features'],
        features['validation']['targets']
    )

    # Make predictions
    snn_results = trainer.predict(
        snn_model,
        features['training']['features'],
        features['validation']['features'],
        features['training']['targets'],
        features['validation']['targets']
    )

    # Save the trained model
    torch.save(snn_model.state_dict(), f"{experiment_directory}/snn_model.pth")

    snn_results['train_losses'] = train_losses
    snn_results['val_losses'] = val_losses
    
    
    with open(f"{experiment_directory}/snn_results.pkl", 'wb') as f:
        pickle.dump(snn_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_directory', type=str, required=True)
    parser.add_argument('--model_config_file', type=str, required=True)
    parser.add_argument('--features_file', type=str, required=True)
    parser.add_argument('--use_FIM', type=str2bool, default=True)
    args = parser.parse_args()

    main(args.experiment_directory, args.model_config_file, args.features_file, args.use_FIM)