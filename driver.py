from experiment_manager import Experiment_Manager
import ray
import os
import multiprocessing
import warnings

# Suppress the specific warning about delim_whitespace
warnings.filterwarnings(
    "ignore", message="The 'delim_whitespace' keyword in pd.read_csv is deprecated"
)

# os.chdir("Demographic_Inference")  # Feel like this is too hacky

# total_cores = multiprocessing.cpu_count()


# Let's define
upper_bound_params = {
    "N0": 15000,
    "Nb": 7000,
    "N_recover": 9000,
    "t_bottleneck_end": 1000,
    "t_bottleneck_start": 3000,  # In generations
}

lower_bound_params = {
    "N0": 10000,
    "Nb": 1000,
    "N_recover": 8000,
    "t_bottleneck_end": 400,
    "t_bottleneck_start": 1100,  # In generations
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
num_layers = 10

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
linear_experiment.pretrain()
# linear_experiment.inference()
ray.shutdown()


