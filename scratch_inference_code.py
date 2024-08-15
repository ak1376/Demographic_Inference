import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import allel
import dadi
import moments


# Read the VCF file into a NumPy array structure
vcf_data = allel.read_vcf('/sietch_colab/akapoor/Demographic_Inference/GHIST-bottleneck.vcf.gz')

num_samples = len(vcf_data['samples'])



dd = dadi.Misc.make_data_dict_vcf("/sietch_colab/akapoor/Demographic_Inference/GHIST-bottleneck.vcf.gz", "/sietch_colab/akapoor/Demographic_Inference/wisent.txt")


fs = dadi.Spectrum.from_data_dict(dd, ['wisent'], projections = [2*num_samples], polarized = False)

# Run dadi inference
from parameter_inference import run_inference_dadi

p0=[0.25, 0.75, 0.1, 0.05]
lower_bound=[0.001, 0.001, 0.001, 0.001]
upper_bound=[10, 10, 10, 10]
sampled_params = None

# run_inference_dadi(
#     fs,
#     p0,
#     sampled_params,
#     num_samples,
#     lower_bound=[0.001, 0.001, 0.001, 0.001],
#     upper_bound=[10, 10, 10, 10],
#     maxiter=20,
# )

# DADI Inference
model_func = dadi.Demographics1D.three_epoch

# Make the extrapolating version of our demographic model function.
# func_ex = dadi.Numerics.make_extrap_log_func(model_func)

p_guess = moments.Misc.perturb_params(
    p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
)

opt_params = dadi.Inference.optimize_log_lbfgsb(
    p_guess,
    fs,
    model_func,
    pts=2 * num_samples,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    maxiter=100,
)

model = model_func(opt_params, fs.sample_sizes, 2 * num_samples)

opt_theta = dadi.Inference.optimal_sfs_scaling(model, fs)

# MOMENTS INFERENCE
model_func = moments.Demographics1D.three_epoch

# Make the extrapolating version of our demographic model function.
# func_ex = dadi.Numerics.make_extrap_log_func(model_func)

p_guess = moments.Misc.perturb_params(
    p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
)

opt_params_moments = moments.Inference.optimize_log_lbfgsb(
    p_guess,
    fs,
    model_func,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    maxiter=100,
)

features = np.concatenate((opt_params.reshape(1, opt_params.shape[0]), opt_params_moments.reshape(1, opt_params_moments.shape[0])), axis = 1)

import torch
import torch.nn as nn

from models import ShallowNN

## SHALLOW NEURAL NETWORK
# Define model hyperparameters
input_size = features.shape[1]  # Number of features
hidden_size = 100  # Number of neurons in the hidden layer
output_size = 4  # Number of output classes
num_epochs = 1000
learning_rate = 3e-4

# Instantiate the model
model = ShallowNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    loo=None,
    experiment_directory='/sietch_colab/akapoor/Demographic_Inference',
)

# Load the saved model state dictionary
model.load_state_dict(torch.load('/sietch_colab/akapoor/experiments/linear_model_bottleneck/neural_network_model.pth'))

features_torch = torch.tensor(features, dtype=torch.float32)

model.eval()
# Make predictions
with torch.no_grad():  # Disable gradient computation for inference
    output = model(features_torch)
    print(output)

