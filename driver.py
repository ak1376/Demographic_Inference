from experiment_manager import Experiment_Manager

import os
os.chdir("Demographic_Inference") # Feel like this is too hacky

# Let's define 
upper_bound_params = {'N0': 10000,
                      'Nb': 2000, 
                      'N_recover': 8000, 
                      't_bottleneck_end': 1000, 
                      't_bottleneck_start': 5000  # In generations            
                     }

lower_bound_params = {'N0': 8000, 
                      'Nb': 1000, 
                      'N_recover': 4000, 
                      't_bottleneck_end': 800, 
                      't_bottleneck_start': 2000 # In generations
                     }


num_simulations = 100
num_samples = 20

config_file = {
    'upper_bound_params': upper_bound_params,
    'lower_bound_params': lower_bound_params,
    'num_sims': num_simulations,
    'num_samples': num_samples,
    'experiment_name': 'xgboost_bottleneck', 
    'num_windows': 200
}

xgboost_experiment = Experiment_Manager(config_file)
xgboost_experiment.run()
