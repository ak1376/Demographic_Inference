import dadi
import moments
import os
from tqdm import tqdm
import numpy as np
import ray
import time


@ray.remote
def get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path):
    start_time = time.time()
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file=flat_map_path,
        pop_file=pop_file_path,
        pops=["A"],
        r_bins=r_bins,
        report=False,
    )
    end_time = time.time()

    print(f"LD stats for {vcf_file} computed in {end_time - start_time} seconds")

    return ld_stats


def compute_ld_stats_parallel(folderpath, num_reps, r_bins):
    start_time = time.time()

    flat_map_path = os.path.join(folderpath, "flat_map.txt")
    pop_file_path = os.path.join(folderpath, "samples.txt")
    vcf_files = [
        os.path.join(folderpath, f"bottleneck_window.{rep_ii}.vcf.gz")
        for rep_ii in range(num_reps)
    ]

    preparation_time = time.time()
    # print(f"Preparation took {preparation_time - start_time:.2f} seconds")

    # Create a list of remote function calls
    futures = [
        get_LD_stats.remote(vcf_file, r_bins, flat_map_path, pop_file_path)
        for vcf_file in vcf_files
    ]

    # Execute the function in parallel and collect the results
    results = ray.get(futures)

    execution_time = time.time()
    print(f"Execution took {execution_time - preparation_time:.2f} seconds")

    return results


def run_inference_dadi(
    sfs,
    p0,
    sampled_params,
    num_samples,
    lower_bound=[0.01, 0.01, 0.01, 0.01],
    upper_bound=[10, 10, 10, 10],
    maxiter=20,
):
    """
    This should do the parameter inference for dadi
    """

    model_func = dadi.Demographics1D.three_epoch

    # Make the extrapolating version of our demographic model function.
    # func_ex = dadi.Numerics.make_extrap_log_func(model_func)

    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    opt_params = dadi.Inference.optimize_log_lbfgsb(
        p_guess,
        sfs,
        model_func,
        pts=2 * num_samples,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        maxiter=maxiter,
    )

    model = model_func(opt_params, sfs.sample_sizes, 2 * num_samples)

    opt_theta = dadi.Inference.optimal_sfs_scaling(model, sfs)

    opt_params_dict = {
        "N0": sampled_params["N0"],
        "Nb": opt_params[0] * sampled_params["N0"],
        "N_recover": opt_params[1] * sampled_params["N0"],
        "t_bottleneck_end": opt_params[3] * 2 * sampled_params["N0"],
        "t_bottleneck_start": opt_params[2] * 2 * sampled_params["N0"],
    }

    model = model * opt_theta

    return model, opt_theta, opt_params_dict


def run_inference_moments(
    sfs,
    p0,
    sampled_params,
    lower_bound=[0.01, 0.01, 0.01, 0.01],
    upper_bound=[10, 10, 10, 10],
    maxiter=20,
):
    """
    This should do the parameter inference for moments
    """
    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    model_func = moments.Demographics1D.three_epoch
    opt_params = moments.Inference.optimize_log_fmin(
        p_guess,
        sfs,
        model_func,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        maxiter=maxiter,
    )

    model = model_func(opt_params, sfs.sample_sizes)
    opt_theta = moments.Inference.optimal_sfs_scaling(model, sfs)

    # opt_params_dict = {
    #     'N0': N0_opt,
    #     'Nb': opt_params[0]*N0_opt,
    #     'N_recover': opt_params[1]*N0_opt,
    #     't_bottleneck_end': opt_params[3]*2*N0_opt,
    #     't_bottleneck_start': opt_params[2]*2*N0_opt
    # }

    opt_params_dict = {
        "N0": sampled_params["N0"],
        "Nb": opt_params[0] * sampled_params["N0"],
        "N_recover": opt_params[1] * sampled_params["N0"],
        "t_bottleneck_end": opt_params[3] * 2 * sampled_params["N0"],
        "t_bottleneck_start": opt_params[2] * 2 * sampled_params["N0"],
    }

    model = model * opt_theta

    return model, opt_theta, opt_params_dict


def run_inference_momentsLD(folderpath, num_windows, param_sample, p_guess, maxiter=20):
    """
    This should do the parameter inference for momentsLD
    """

    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])

    print("parsing LD statistics")

    ld_stats = {}

    results = compute_ld_stats_parallel(folderpath, num_windows, r_bins)

    for i, result in enumerate(results):
        ld_stats[i] = result

    print("computing mean and varcov matrix from LD statistics sums")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    mv["varcovs"][-1].shape = (1, 1)

    demo_func = moments.LD.Demographics1D.three_epoch
    # Set up the initial guess
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=1)
    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess,
        [mv["means"], mv["varcovs"]],
        [demo_func],
        rs=r_bins,
        maxiter=maxiter,
    )

    opt_params_dict = {
        "N0": opt_params[4],
        "Nb": opt_params[0] * opt_params[4],
        "N_recover": opt_params[1] * opt_params[4],
        "t_bottleneck_end": opt_params[3] * 2 * opt_params[4],
        "t_bottleneck_start": opt_params[2] * 2 * opt_params[4],
    }

    return opt_params_dict
