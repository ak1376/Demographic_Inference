from experiment_manager import Experiment_Manager
import os
import warnings
from train import Trainer
from inference import Inference
import pickle

# Suppress the specific warning about delim_whitespace
warnings.filterwarnings(
    "ignore", message="The 'delim_whitespace' keyword in pd.read_csv is deprecated"
)

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
    "t_bottleneck_start": 2000,
}
lower_bound_params = {
    "N0": 8000,
    "Nb": 4000,
    "N_recover": 6000,
    "t_bottleneck_end": 800,
    "t_bottleneck_start": 1500,
}
model_config = {
    "input_size": 10,
    "hidden_size": 1000,
    "output_size": 5,
    "num_epochs": 1000,
    "learning_rate": 3e-4,
    "num_layers": 3,
    "dropout_rate": 0,
    "weight_decay": 0,
}

config = {
    "upper_bound_params": upper_bound_params,
    "lower_bound_params": lower_bound_params,
    "num_sims_pretrain": 5,
    "num_sims_inference": 5,
    "num_samples": 20,
    "experiment_name": "dadi_moments_analysis_new",
    "dadi_analysis": True,
    "moments_analysis": True,
    "momentsLD_analysis": False,
    "num_windows": 50,
    "window_length": 1e6,
    "maxiter": 100,
    "genome_length": 1e6,
    "mutation_rate": 1.26e-8,
    "recombination_rate": 1.007e-8,
    "seed": 42,
    "normalization": False,
    "remove_outliers": True,
    "use_FIM": False,
    "neural_net_hyperparameters": model_config,
    "demographic_model": "bottleneck_model",
    "parameter_names": ["N0", "Nb", "N_recover", "t_bottleneck_end", "t_bottleneck_start"], # these should be a list of parameters that we want to optimize 
    "optimization_initial_guess": [0.25, 0.75, 0.1, 0.05]
}

linear_experiment = Experiment_Manager(config)
linear_experiment.obtaining_features()
preprocessing_results_obj = linear_experiment.load_features(
    f"{os.getcwd()}/experiments/dadi_moments_analysis_new/preprocessing_results_obj.pkl"
)
# preprocessing_results_obj = linear_experiment.load_features("/sietch_colab/akapoor/Demographic_Inference/experiments/dadi_moments_analysis/preprocessing_results_obj.pkl")
training_features = preprocessing_results_obj["training"]["predictions"]
training_targets = preprocessing_results_obj["training"]["targets"]
validation_features = preprocessing_results_obj["validation"]["predictions"]
validation_targets = preprocessing_results_obj["validation"]["targets"]

testing_features = preprocessing_results_obj["testing"]["predictions"]
testing_targets = preprocessing_results_obj["testing"]["targets"]


trainer = Trainer(
    experiment_directory=linear_experiment.experiment_directory,
    model_config=model_config,
    use_FIM=config["use_FIM"],
    color_shades=linear_experiment.color_shades,
    main_colors=linear_experiment.main_colors,
    param_names=config["parameter_names"]
)
snn_model, train_losses, val_losses = trainer.train(
    training_features,
    training_targets,
    validation_features,
    validation_targets,
    visualize=True,
)
trainer.predict(
    snn_model,
    training_features,
    validation_features,
    training_targets,
    validation_targets,
    visualize=True,
)
inference_obj = Inference(
    vcf_filepath="GHIST-bottleneck.vcf.gz",
    txt_filepath="wisent.txt",
    popname="wisent",
    config=config,
    experiment_directory=linear_experiment.experiment_directory,
)
inference_obj.obtain_features()

with open(
    f"{os.getcwd()}/experiments/dadi_moments_analysis_new/inference_results_obj.pkl",
    "rb",
) as file:
    inference_results = pickle.load(file)

inference_obj.evaluate_model(snn_model, inference_results)
