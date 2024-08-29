from experiment_manager import Experiment_Manager
import os
import warnings
from train import Trainer

# Suppress the specific warning about delim_whitespace
warnings.filterwarnings(
    "ignore", message="The 'delim_whitespace' keyword in pd.read_csv is deprecated"
)

# os.chdir("Demographic_Inference")  # Feel like this is too hacky

# total_cores = multiprocessing.cpu_count()


# Let's define
# upper_bound_params = {
#     "N0": 15000,
#     "Nb": 7000,
#     "N_recover": 9000,
#     "t_bottleneck_end": 1000,
#     "t_bottleneck_start": 3000,  # In generations
# }

# lower_bound_params = {
#     "N0": 10000,
#     "Nb": 1000,
#     "N_recover": 8000,
#     "t_bottleneck_end": 400,
#     "t_bottleneck_start": 1100,  # In generations
# }

upper_bound_params = {
"N0": 10000,
"Nb": 5000,
"N_recover": 7000,
"t_bottleneck_end": 1000,
"t_bottleneck_start": 2000
}
lower_bound_params = {
"N0": 8000,
"Nb": 4000,
"N_recover": 6000,
"t_bottleneck_end": 800,
"t_bottleneck_start": 1500
}
model_config = {
"input_size": 8,
"hidden_size": 1000,
"output_size": 4,
"num_epochs": 1000,
"learning_rate": 3e-4,
"num_layers": 3,
"dropout_rate": 0,
"weight_decay": 0
}

config = {
    "upper_bound_params": upper_bound_params,
    "lower_bound_params": lower_bound_params,
    "num_sims_pretrain": 1000,
    "num_sims_inference": 1000,
    "num_samples": 20,
    "experiment_name": "dadi_moments_analysis_new",
    "dadi_analysis": True,
    "moments_analysis": True,
    "momentsLD_analysis": False,
    "num_windows": 50,
    "window_length": 1e6,
    "maxiter": 100,
    "genome_length": 1e8,
    "mutation_rate": 1.26e-8,
    "recombination_rate": 1.007e-8,
    "seed": 42,
    "normalization": False,
    "remove_outliers": True,
    "use_FIM": False,
    "neural_net_hyperparameters": model_config
}

linear_experiment = Experiment_Manager(config)
linear_experiment.obtaining_features()
preprocessing_results_obj = linear_experiment.load_features(f"{os.getcwd()}/experiments/dadi_moments_analysis/preprocessing_results_obj.pkl")
# preprocessing_results_obj = linear_experiment.load_features("/sietch_colab/akapoor/Demographic_Inference/experiments/dadi_moments_analysis/preprocessing_results_obj.pkl")
training_features = preprocessing_results_obj["training"]["predictions"]
training_targets = preprocessing_results_obj["training"]["targets"]
validation_features = preprocessing_results_obj["validation"]["predictions"]
validation_targets = preprocessing_results_obj["validation"]["targets"]

testing_features = preprocessing_results_obj["testing"]["predictions"]
testing_targets = preprocessing_results_obj["testing"]["targets"]

trainer = Trainer(experiment_directory=linear_experiment.experiment_directory, model_config=model_config, use_FIM=config['use_FIM'])
snn_model, train_losses, val_losses = trainer.train(training_features, training_targets, validation_features, validation_targets, visualize = True)
trainer.predict(snn_model, training_features, validation_features, training_targets, validation_targets, visualize = True)

print("siema")
# linear_experiment.inference()


