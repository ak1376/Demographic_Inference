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


fs = dadi.Spectrum.from_data_dict(dd, ['wisent'], projections = [2*num_samples], polarized = True)

# # Sum of all entries in the observed SFS (excluding masked values)
# total_observed_sites = numpy.sum(dd['wisent'][1:])

# # Normalize the observed SFS
# normalized_fs_observed = fs / total_observed_sites

import pickle
with open('/sietch_colab/akapoor/experiments/linear_model_bottleneck/dadi_dict.pkl', 'rb') as file:
    dadi_dict = pickle.load(file)

# with open('/sietch_colab/akapoor/experiments/linear_model_bottleneck/generative_params.pkl', 'rb') as file:
#     generative_params = pickle.load(file)


# sampled_params = generative_params[0]

simulated_sfs = dadi_dict['model_sfs'][0]

dadi.Plotting.plot_1d_fs(simulated_sfs)
plt.savefig("simulated_sfs.png")


# Step 1: Get the optimal parameters from dadi inference 
# Step 2: Get the optimal theta value from dadi inference
# Step 3: Get the estimated ground truth ancestral size by doing theta / (4*mu)




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
N_est = opt_theta / (4 * 1.26e-8*1e8)

opt_params[0] *= N_est
opt_params[1] *= N_est
opt_params[3] = opt_params[3] * 2 * N_est
opt_params[2] = 2*N_est*opt_params[2] + opt_params[3]

# fs /= opt_theta

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

opt_params_moments[0] *= N_est
opt_params_moments[1] *= N_est
opt_params_moments[3] = opt_params_moments[3] * 2 * N_est
opt_params_moments[2] = 2*N_est*opt_params_moments[2] + opt_params_moments[3]


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
model.load_state_dict(torch.load('/sietch_colab/akapoor/experiments/linear_model_bottleneck/final_model.pth'))

features_torch = torch.tensor(features, dtype=torch.float32)

model.eval()
# Make predictions
with torch.no_grad():  # Disable gradient computation for inference
    output = model(features_torch)
    print(output)

# Replace 'model.pkl' with the path to your saved model
with open('/sietch_colab/akapoor/experiments/linear_model_bottleneck/linear_regression_model.pkl', 'rb') as file:
    linear_mdl = pickle.load(file)