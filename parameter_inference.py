import dadi
import moments
import os
from tqdm import tqdm
import numpy as np
import ray
import time
from dadi.Godambe import get_godambe
from moments.Godambe import _get_godambe
import nlopt
import demographic_models
from optimize import opt


def get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path):
    start_time = time.time()
    ld_stats = moments.LD.Parsing.compute_ld_statistics(  # type: ignore
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
        get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path)
        for vcf_file in vcf_files
    ]
    return futures


def run_inference_dadi(
    sfs,
    p0,
    sampled_params,
    num_samples,
    demographic_model,
    lower_bound=[0.001, 0.001, 0.001, 0.001],
    upper_bound=[1, 1, 1, 1],
    maxiter=100,
    mutation_rate=1.26e-8,
    length=1e8,
):
    """
    This should do the parameter inference for dadi
    """
    if demographic_model == "bottleneck_model":
        model_func = dadi.Demographics1D.three_epoch

    elif demographic_model == "split_isolation_model":
        model_func = demographic_models.split_isolation_model

    func_ex = dadi.Numerics.make_extrap_log_func(model_func)
    pts_ext = [num_samples + 20, num_samples + 30, num_samples + 40]

    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )
    start = time.time()

    # print(f'GUESS PARAMETER: {p_guess}')

    opt_params = dadi.Inference.opt(
        p_guess,
        sfs,
        func_ex,
        pts=pts_ext,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        algorithm=nlopt.LN_BOBYQA,
        maxeval=10,
    )

    opt_params = opt_params[0]

    print(f"OPT DADI PARAMETER: {opt_params}")

    end = time.time()
    # print(f"Dadi optimization took {end - start} seconds")

    model = func_ex(opt_params, sfs.sample_sizes, 2 * num_samples)
    opt_theta = dadi.Inference.optimal_sfs_scaling(model, sfs)

    opt_params_dict = {}

    if demographic_model == "bottleneck_model":
        N_ref = opt_theta / (4 * mutation_rate * length)
        opt_params_dict = {
            "N0": N_ref,
            "Nb": opt_params[0] * N_ref,
            "N_recover": opt_params[1] * N_ref,
            "t_bottleneck_end": opt_params[3] * 2 * N_ref,  # type: ignore
            "t_bottleneck_start": opt_params[2] * 2 * N_ref,  # type: ignore
        }
    
    elif demographic_model == "split_isolation_model":
        N_ref = opt_theta / (4 * mutation_rate * length)
        opt_params_dict = {
            "N0": N_ref,
            "N1": opt_params[0] * N_ref,
            "N2": opt_params[1] * N_ref,
            "t_split": opt_params[2] * 2 * N_ref,
            "t_isolation_start": opt_params[3] * 2 * N_ref,
            "t_isolation_end": opt_params[4] * 2 * N_ref
        }

    model = model * opt_theta
    
    return model, opt_theta, opt_params_dict


def run_inference_moments(
    sfs,
    p0,
    sampled_params,
    demographic_model,
    lower_bound=[0.001, 0.001, 0.001, 0.001],
    upper_bound=[1, 1, 1, 1],
    maxiter=100,
    use_FIM=False,
    mutation_rate=1.26e-8,
    length=1e7,
):
    """
    This should do the parameter inference for moments
    """
    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    if demographic_model == "bottleneck_model":
        model_func = moments.Demographics1D.three_epoch

    start = time.time()

    opt_params =opt(
        p_guess,
        sfs,
        model_func,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        log_opt=True, 
        algorithm=nlopt.LN_BOBYQA
    )[0] # I don't want the log likelihood. 

    print(f"OPT MOMENTS PARAMETER: {opt_params}")

    model = model_func(opt_params, sfs.sample_sizes)
    opt_theta = moments.Inference.optimal_sfs_scaling(model, sfs)

    N_ref = opt_theta / (4 * mutation_rate * length)

    end = time.time()

    if use_FIM:

        # Let's extract the hessian
        # H = moments.Godambe\.get_hess(func = model_func, p0 = opt_params, eps = 1e-6, args = (np.array(sfs.sample_sizes),))
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
            "t_bottleneck_end": opt_params[3] * 2 * N_ref,  # type: ignore
            "t_bottleneck_start": opt_params[2] * 2 * N_ref,  # type: ignore
            "upper_triangular_FIM": upper_triangular,
        }

        if opt_params_dict["upper_triangular_FIM"] is None:
            del opt_params_dict["upper_triangular_FIM"]

    model = model * opt_theta

    return model, opt_theta, opt_params_dict


def run_inference_momentsLD(
    folderpath, num_windows, param_sample, p_guess, demographic_model, maxiter=20
):
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
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)  # type: ignore
    mv["varcovs"][-1].shape = (1, 1)

    demo_func = moments.LD.Demographics1D.three_epoch  # type: ignore
    # Set up the initial guess
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=1)  # type: ignore
    opt_params, LL = moments.LD.Inference.optimize_log_fmin(  # type: ignore
        p_guess,
        [mv["means"], mv["varcovs"]],
        [demo_func],
        rs=r_bins,
        maxiter=maxiter,
    )

    opt_params_dict = {}
    if demographic_model == "bottleneck_model":

        opt_params_dict = {
            "N0": opt_params[4],
            "Nb": opt_params[0] * opt_params[4],
            "N_recover": opt_params[1] * opt_params[4],
            "t_bottleneck_end": opt_params[3] * 2 * opt_params[4],
            "t_bottleneck_start": opt_params[2] * 2 * opt_params[4],
        }

    return opt_params_dict
