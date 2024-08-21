from experiment_manager import Experiment_Manager
import ray
import os
import multiprocessing
import warnings
# import time
import pickle
from train import Trainer

# Suppress the specific warning about delim_whitespace
warnings.filterwarnings(
    "ignore", message="The 'delim_whitespace' keyword in pd.read_csv is deprecated"
)

# os.chdir("Demographic_Inference")  # Feel like this is too hacky

# total_cores = multiprocessing.cpu_count()


# Let's define
upper_bound_params = {
    "N0": 10000,
    "Nb": 2000,
    "N_recover": 8000,
    "t_bottleneck_end": 1000,
    "t_bottleneck_start": 2000,  # In generations
}

lower_bound_params = {
    "N0": 8000,
    "Nb": 1000,
    "N_recover": 4000,
    "t_bottleneck_end": 800,
    "t_bottleneck_start": 1500,  # In generations
}


num_simulations_pretrain = 1000
num_simulations_inference = 1000
num_samples = 20

# Neural Net Hyperparameters
input_size = 8  # Number of features
hidden_size = 100  # Number of neurons in the hidden layer
output_size = 4  # Number of output classes
num_epochs = 5000
learning_rate = 3e-4
num_layers = 4

model_config = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "num_layers": num_layers,
}


# TODO: Add an option where I can specify some text to place in a readme for this experiment.
config_file = {
    "upper_bound_params": upper_bound_params,
    "lower_bound_params": lower_bound_params,
    "num_sims_pretrain": num_simulations_pretrain,
    "num_sims_inference": num_simulations_inference,
    "num_samples": num_samples,
    "experiment_name": "neural_net_linear_regression",
    "dadi_analysis": True,
    "moments_analysis": True,
    "momentsLD_analysis": False,
    "num_windows": 50,
    "window_length": 1e5,
    "maxiter": 100,
    "genome_length": 1e7,
    "mutation_rate": 1.26e-8,
    "recombination_rate": 1.007e-8,
    "seed": 295,
    "normalization": False, 
    "remove_outliers": True, 
    "neural_net_hyperparameters": model_config
}

linear_experiment = Experiment_Manager(config_file)
# linear_experiment.obtaining_features()
preprocessing_results_obj = linear_experiment.load_features("/sietch_colab/akapoor/experiments/neural_net_linear_regression_keep/processing_results_obj.pkl")

training_features = preprocessing_results_obj["training"]["predictions"]
training_targets = preprocessing_results_obj["training"]["targets"]
validation_features = preprocessing_results_obj["validation"]["predictions"]
validation_targets = preprocessing_results_obj["validation"]["targets"]

# testing_features = preprocessing_results_obj["testing"]["predictions"]
# testing_targets = preprocessing_results_obj["testing"]["targets"]

trainer = Trainer(experiment_directory=linear_experiment.experiment_directory, model_config=model_config)
snn_model, train_losses, val_losses = trainer.train(training_features, training_targets, validation_features, validation_targets, visualize = True)
trainer.predict(snn_model, training_features, validation_features, training_targets, validation_targets, visualize = True)

# linear_experiment.inference()
ray.shutdown()


