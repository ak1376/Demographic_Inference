import dadi
import moments
import os
import numpy as np
import time
from moments.Godambe import _get_godambe
import nlopt
import src.demographic_models as demographic_models
from src.optimize import opt, ld_opt
import ray

# Define your function with Ray's remote decorator
@ray.remote
def get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path):
    ray.init(ignore_reinit_error=True)
    ld_stats = moments.LD.Parsing.compute_ld_statistics( #type:ignore
        vcf_file,
        rec_map_file=flat_map_path,
        pop_file=pop_file_path,
        pops=["N1", "N2"], # TODO: Change later
        r_bins=r_bins,
        report=False,
    )

    return ld_stats

def compute_ld_stats_parallel(folderpath, num_reps, r_bins):

    #     print("parsing LD statistics in parallel")
    # # Submit tasks to Ray in parallel using .remote()
    # futures = [get_LD_stats.remote(ii, r_bins) for ii in range(num_reps)]
    # # Gather results with ray.get() to collect them once the tasks are finished
    # ld_stats = ray.get(futures)
    # # Optionally, you can convert the list of results into a dictionary with indices
    # ld_stats_dict = {ii: result for ii, result in enumerate(ld_stats)}


    flat_map_path = os.path.join(folderpath, "flat_map.txt")
    pop_file_path = os.path.join(folderpath, "samples.txt")
    vcf_files = [
        os.path.join(folderpath, f"rep.{rep_ii}.vcf.gz")
        for rep_ii in range(num_reps)
    ]

    # Launch the tasks in parallel using Ray
    futures = [
        get_LD_stats.remote(vcf_file, r_bins, flat_map_path, pop_file_path)
        for vcf_file in vcf_files
    ]

    # Wait for all the tasks to complete and retrieve results
    results = ray.get(futures)
    return results

# def compute_ld_stats_sequential(flat_map_path, samples_path, metadata_path, r_bins):
#     # Start by defining paths
    
#     # List of VCF files
#     # Read the file and store each line (filepath) into a list
#     with open(metadata_path, 'r') as f:
#         vcf_files = [line.strip() for line in f]

#     # List to store LD statistics results
#     ld_stats_list = []

#     # Sequentially compute LD statistics for each VCF file
#     for vcf_file in vcf_files:
#         ld_stats = get_LD_stats(vcf_file, r_bins, flat_map_path, samples_path)
#         ld_stats_list.append(ld_stats)
    
#     return ld_stats_list


def run_inference_dadi(
    sfs,
    p0,
    num_samples,
    demographic_model,
    k,
    lower_bound=[0.001, 0.001, 0.001, 0.001],
    upper_bound=[1, 1, 1, 1],
    mutation_rate=1.26e-8,
    length=1e8,
    top_values_k=3
):
    """
    This should do the parameter inference for dadi
    """
    if demographic_model == "bottleneck_model":
        model_func = dadi.Demographics1D.three_epoch

    elif demographic_model == "split_isolation_model":
        model_func = demographic_models.split_isolation_model_dadi

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    func_ex = dadi.Numerics.make_extrap_log_func(model_func)
    pts_ext = [num_samples + 20, num_samples + 30, num_samples + 40]

    opt_params_dict_list = []
    model_list = []
    opt_theta_list = []
    ll_list = []

    for i in np.arange(k):
        p_guess = moments.Misc.perturb_params(
            p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
        )

        start = time.time()

        # Optimization with dadi
        opt_params, ll_value = dadi.Inference.opt(
            p_guess,
            sfs,
            func_ex,
            pts=pts_ext,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            algorithm=nlopt.LN_BOBYQA,
            maxeval=10,
        )

        ll_list.append(ll_value)
        opt_params_dict_list.append(opt_params)

        print(f"OPT DADI PARAMETER: {opt_params}")

    # Find the indices of the top top_k_values (those with the highest likelihood)
    top_k_indices = np.argsort(ll_list)[-top_values_k:]
    top_k_indices = np.array(top_k_indices, dtype=int)  # Convert to numpy integer array

    opt_params_dict_list_top_values = [opt_params_dict_list[i] for i in top_k_indices]
    opt_params_final_list = []

    for j in np.arange(top_values_k):
        opt_params = opt_params_dict_list_top_values[j]
        model = func_ex(opt_params, sfs.sample_sizes, 2 * num_samples)
        opt_theta = dadi.Inference.optimal_sfs_scaling(model, sfs)
        N_ref = opt_theta / (4 * mutation_rate * length)

        # Initialize opt_params_dict properly in all cases
        opt_params_dict = {}

        if demographic_model == "bottleneck_model":
            opt_params_dict = {
                "N0": N_ref,
                "Nb": opt_params[0] * N_ref,
                "N_recover": opt_params[1] * N_ref,
                "t_bottleneck_start": (opt_params[2]+opt_params[3]) * 2 * N_ref,
                "t_bottleneck_end": opt_params[2] * 2 * N_ref,
            }

        elif demographic_model == "split_isolation_model":
            opt_params_dict = {
                "Na": N_ref,
                "N1": opt_params[0]*N_ref,
                "N2": opt_params[1]*N_ref,
                "t_split": opt_params[2]*2*N_ref,
                "m": opt_params[3]*2*N_ref
            }

        else:
            raise ValueError(f"Unsupported demographic model: {demographic_model}")

        model = model * opt_theta

        model_list.append(model)
        opt_theta_list.append(opt_theta)
        opt_params_final_list.append(opt_params_dict)

    return model_list, opt_theta_list, opt_params_final_list, ll_list

def run_inference_moments(
    sfs,
    p0,
    demographic_model,
    k,
    lower_bound=[0.001, 0.001, 0.001, 0.001],
    upper_bound=[1, 1, 1, 1],
    use_FIM=False,
    mutation_rate=1.26e-8,
    length=1e7,
    top_values_k = 3
):
    """
    This should do the parameter inference for moments
    """
    ll_list = []
    model_list = []
    opt_theta_list = []
    opt_params_dict_list = []

    for i in range(k):

        p_guess = moments.Misc.perturb_params(
            p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
        )
        if demographic_model == "bottleneck_model":
            model_func = moments.Demographics1D.three_epoch

        elif demographic_model == "split_isolation_model":
            model_func = demographic_models.split_isolation_model_moments

        else:
            raise ValueError(f"Unsupported demographic model: {demographic_model}")

        opt_params, ll =opt(
            p_guess,
            sfs,
            model_func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            log_opt=True, 
            algorithm=nlopt.LN_BOBYQA,
            maxeval=10,
            verbose = 3
        ) # I don't want the log likelihood. 

        ll_list.append(ll)
        opt_params_dict_list.append(opt_params)

        print(f"OPT MOMENTS PARAMETER: {opt_params}")
    
    # Now find the indices of the top top_k_values (those with the highest likelihood)
    top_k_indices = np.array(np.argsort(ll_list)[-top_values_k:], dtype = int)    
    
    opt_params_dict_list_top_values = [opt_params_dict_list[i] for i in top_k_indices]
    opt_params_final_list = []

    for j in np.arange(top_values_k):
        opt_params = opt_params_dict_list_top_values[j]
        model = model_func(opt_params, sfs.sample_sizes)
        opt_theta = moments.Inference.optimal_sfs_scaling(model, sfs)

        N_ref = opt_theta / (4 * mutation_rate * length)

        if use_FIM:

            # Let's extract the hessian
            H = _get_godambe(
                model_func,
                all_boot=[],
                p0=opt_params,
                data=sfs,
                eps=1e-6,
                log=False,
                just_hess=True,
            )
            FIM = -1 * H

            # Get the indices of the upper triangular part (including the diagonal)
            upper_tri_indices = np.triu_indices(FIM.shape[0])  # type: ignore

            # Extract the upper triangular elements
            upper_triangular = FIM[upper_tri_indices]  # type: ignore

        else:
            upper_triangular = None

        opt_params_dict = {}

        if demographic_model == "bottleneck_model":

            opt_params_dict = {
                "N0": N_ref,
                "Nb": opt_params[0] * N_ref,
                "N_recover": opt_params[1] * N_ref,
                "t_bottleneck_start": (opt_params[2]+opt_params[3]) * 2 * N_ref, # type: ignore
                "t_bottleneck_end": opt_params[2] * 2 * N_ref, # type: ignore
                "upper_triangular_FIM": upper_triangular,
            }

        elif demographic_model == "split_isolation_model":
            opt_params_dict = {
                "Na": N_ref,
                "N1": opt_params[0]*N_ref,
                "N2": opt_params[1]*N_ref,
                "t_split": opt_params[2]*2*N_ref,
                "m": opt_params[3]*2*N_ref,
                "upper_triangular_FIM": upper_triangular,
            }

        else:
            raise ValueError(f"Unsupported demographic model: {demographic_model}")

        if opt_params_dict["upper_triangular_FIM"] is None:
            del opt_params_dict["upper_triangular_FIM"]

        model = model * opt_theta

        model_list.append(model)
        opt_theta_list.append(opt_theta)
        opt_params_final_list.append(opt_params_dict)

    # print(f'Log Likelihood Values for Moments: {ll_list}')
    return model_list, opt_theta_list, opt_params_final_list, ll_list


def run_inference_momentsLD(folderpath, demographic_model, p_guess, num_reps):
    """
    This should do the parameter inference for momentsLD
    index: unique simulation number
    """

    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])

    print("parsing LD statistics")


    ld_stats = {}
    results = compute_ld_stats_parallel(folderpath, num_reps, r_bins)

    for i, result in enumerate(results):
        ld_stats[i] = result

    # print("computing mean and varcov matrix from LD statistics sums")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)  # type: ignore
    print("SHAPE OF THE COVARIANCE MATRIX")
    print(mv["varcovs"][-1].shape)
    # mv["varcovs"][-1].shape = (1, 1)

    if demographic_model == "bottleneck_model":
        demo_func = moments.LD.Demographics1D.three_epoch # type: ignore

    elif demographic_model == "split_isolation_model":
        demo_func = demographic_models.split_isolation_model_momentsLD

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # Set up the initial guess
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1) # type: ignore
    opt_params, LL = moments.LD.Inference.optimize_log_lbfgsb( #type:ignore
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins,
    )

    physical_units = moments.LD.Util.rescale_params( # type: ignore
        opt_params, ["nu", "nu", "T", "m", "Ne"]
)
    # p_guess = moments.LD.Util.perturb_params(p_guess, fold=1)  # type: ignore
    
    # Define necessary arguments for the new opt function
    # p0 = p_guess  # Initial parameters guess
    # data = [mv["means"], mv["varcovs"]]  # Means and varcovs
    # model_func = demo_func  # Demographic model function
    # func_args = [r_bins]  # Pass r_bins as part of additional function arguments
    # ftol_abs = 1e-6  # You can specify or adjust these based on your use case
    # xtol_abs = 1e-6
    # maxeval = maxiter  # Use maxiter as max function evaluations   
    
    # opt_params, LL = ld_opt(
    #     p0=p0,
    #     data=data,
    #     model_func=model_func,
    #     lower_bound=None,  # Add your actual lower bounds if applicable
    #     upper_bound=None,  # Add your actual upper bounds if applicable
    #     func_args=func_args,
    #     ftol_abs=ftol_abs,
    #     xtol_abs=xtol_abs,
    #     maxeval=maxeval,
    #     verbose=0
    # )

    opt_params, LL = moments.LD.Inference.optimize_log_lbfgsb( #type:ignore
    p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins, verbose = 3
    )

    opt_params_dict = {}
    if demographic_model == "bottleneck_model":

        opt_params_dict = {
            # "N0": opt_params[4],
            "Nb": opt_params[0] * opt_params[4],
            "N_recover": opt_params[1] * opt_params[4],
            "t_bottleneck_start": (opt_params[2]+opt_params[3]) * 2 * opt_params[4],
            "t_bottleneck_end": opt_params[3] * 2 * opt_params[4]
        }

    elif demographic_model == "split_isolation_model":
        physical_units = moments.LD.Util.rescale_params( #type:ignore
            opt_params, ["nu", "nu", "T", "m", "Ne"]
        )

        print(physical_units)

        opt_params_dict = {
            "N1": physical_units[0],
            "N2": physical_units[1],
            "t_split": physical_units[2],
            "m": physical_units[3], 
            'Na': physical_units[4]
        }
    
    print(f'Moments LD results: {opt_params_dict}')

    return opt_params_dict
