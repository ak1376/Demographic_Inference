import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import moments

dadi_dict = '/sietch_colab/akapoor/experiments/archive/linear_model_bottleneck_5/dadi_dict.pkl'
with open(dadi_dict, 'rb') as f:
    dadi_dict = pickle.load(f)

moments_dict = '/sietch_colab/akapoor/experiments/archive/linear_model_bottleneck_5/moments_dict.pkl'
with open(moments_dict, 'rb') as f:
    moments_dict = pickle.load(f)

# Load in the generative demographic parameters 

generative_params = '/sietch_colab/akapoor/experiments/archive/linear_model_bottleneck_5/generative_params.pkl'
with open(generative_params, 'rb') as f:
    generative_params = pickle.load(f)

# Load in the indices for the outliers for both moments and dadi 
moments_outliers = pd.read_csv('/sietch_colab/akapoor/experiments/archive/linear_model_bottleneck_5/outlier_indices_moments.csv', header = None)
moments_outliers = moments_outliers[0].values.astype(int)   
generative_params_outliers_moments = [generative_params[i] for i in moments_outliers]

# Read the CSV file
dadi_outliers = pd.read_csv('/sietch_colab/akapoor/experiments/archive/linear_model_bottleneck_5/outlier_indices_dadi.csv', header=None)
dadi_outliers = dadi_outliers[0].values.astype(int)
generative_params_outliers_dadi = [generative_params[i] for i in dadi_outliers]

# Let's do moments first 

# I want to see detailed information on the optimization procedure for these outlier values . 

# Let's just do one example first: 

sampled_params = generative_params_outliers_moments[0]

# input_theta = 4*sampled_params['N0']*1.5e-8

input_theta = 10000

# Let's define the ground truth parameter values. We will try to recover these values using the moments optimization procedure.

sampled_params = {}

sampled_params['N0'] = 10000
sampled_params['Nb'] = 8000
sampled_params['N_recover'] = 9000
sampled_params['t_bottleneck_end'] = 400
sampled_params['t_bottleneck_start'] = 500


nuB = sampled_params['Nb']/sampled_params['N0']
nuF = sampled_params['N_recover']/sampled_params['N0']
TB = (sampled_params['t_bottleneck_start'] - sampled_params['t_bottleneck_end'])/(2*sampled_params['N0'])
TF = sampled_params['t_bottleneck_end']/(2*sampled_params['N0'])

params = [nuB, nuF, TB, TF]
model_func = moments.Demographics1D.three_epoch
model = model_func(params, [200])
model = input_theta * model
data = model.sample()

plt.figure()
plt.hist(np.array(data))
plt.show()
plt.savefig('bottleneck_sfs.png')

p_guess = [0.75, 0.85, 0.1, 0.01]
lower_bound = [1e-4, 1e-4, 1e-4, 1e-4]
upper_bound = [1, 1, 1, 1]

p_guess = moments.Misc.perturb_params(
    p_guess, lower_bound=lower_bound, upper_bound=upper_bound)

opt_params = moments.Inference.optimize_log_fmin(
    p_guess, data, model_func,
    lower_bound=lower_bound, upper_bound=upper_bound,
    verbose=20) # report every 20 iterations

# refit_theta = moments.Inference.optimal_sfs_scaling(
#     model_func(opt_params, data.sample_sizes), data)

p_guess = moments.Misc.perturb_params(
    opt_params, lower_bound=lower_bound, upper_bound=upper_bound, fold = 2)


opt_params = moments.Inference.optimize_log_fmin(
    p_guess, data, model_func,
    lower_bound=lower_bound, upper_bound=upper_bound,
    verbose=20) # report every 20 iterations

p_guess = moments.Misc.perturb_params(
    opt_params, lower_bound=lower_bound, upper_bound=upper_bound)

opt_params = moments.Inference.optimize_log_fmin(
    p_guess, data, model_func,
    lower_bound=lower_bound, upper_bound=upper_bound,
    verbose=20) # report every 20 iterations



# uncerts = moments.Godambe.FIM_uncert(
#     model_func, opt_params, data)

# print_params = params + [input_theta]
# print_opt = np.concatenate((opt_params, [refit_theta]))

# print("Params\tnu1\tnu2\tT_div\tm_sym\ttheta")
# print(f"Input\t" + "\t".join([str(p) for p in print_params]))
# print(f"Refit\t" + "\t".join([f"{p:.4}" for p in print_opt]))
# print(f"Std-err\t" + "\t".join([f"{u:.3}" for u in uncerts]))

# moments.Plotting.plot_1d_comp_multinom(
#     model_func(opt_params, data.sample_sizes), data)

# Define the grid of hyperparameters
# epsilon_grid = [1e-2, 1e-3, 1e-4]
# gtol_grid = [1e-4, 1e-5, 1e-6]
# maxiter_grid = [100, 500, 1000]

# # Store the results
# results = []

# # Example data and initial parameters (replace with your actual data and p0)

# # Iterate over all combinations of hyperparameters
# for epsilon in epsilon_grid:
#     for gtol in gtol_grid:
#         for maxiter in maxiter_grid:
#             print(f"Running optimization with epsilon={epsilon}, gtol={gtol}, maxiter={maxiter}")
            
#             # Run the optimization
#             result = moments.Inference.optimize_log(
#                 p0=p_guess,
#                 data=data,
#                 model_func=model_func,
#                 epsilon=epsilon,
#                 gtol=gtol,
#                 maxiter=maxiter,
#                 verbose=0
#             )
            
#             # Store the result with the hyperparameters
#             results.append({
#                 'epsilon': epsilon,
#                 'gtol': gtol,
#                 'maxiter': maxiter,
#                 'result': result
#             })

# # After running all combinations, find the best result
# best_result = min(results, key=lambda x: x['result']['final_value'])  # Assuming lower is better

# print(f"Best result found with epsilon={best_result['epsilon']}, gtol={best_result['gtol']}, maxiter={best_result['maxiter']}")
# print("Result:", best_result['result'])
