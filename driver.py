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


num_simulations = 1000
num_samples = 20
#TODO: Add an option where I can specify some text to place in a readme for this experiment. 
# TODO: Add an option whether I want to do feature and target standardization.
config_file = {
    "upper_bound_params": upper_bound_params,
    "lower_bound_params": lower_bound_params,
    "num_sims": num_simulations,
    "num_samples": num_samples,
    "experiment_name": "linear_model_bottleneck_bigger",
    "dadi_analysis": True, 
    "moments_analysis": True,
    "momentsLD_analysis": False,
    "num_windows": 50,
    "window_length": 1e5,
    "maxiter": 100,
    "genome_length": 1e7,
    "mutation_rate": 1.26e-8,
    "recombination_rate": 1.007e-8,
}

linear_experiment = Experiment_Manager(config_file)
linear_experiment.pretrain()
# linear_experiment.inference()
ray.shutdown()
