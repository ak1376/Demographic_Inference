import dadi
import moments
import os
import numpy as np
import time
from moments.Godambe import _get_godambe
import nlopt
import src.demographic_models as demographic_models
from src.optimize import opt

# Define your function with Ray's remote decorator
def get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path):
    ld_stats = moments.LD.Parsing.compute_ld_statistics( #type:ignore
        vcf_file,
        rec_map_file=flat_map_path,
        pop_file=pop_file_path,
        pops=["N1", "N2"], # TODO: Change later
        r_bins=r_bins,
        report=False,
        use_genotypes = False
    )

    return ld_stats

# def compute_ld_stats_parallel(folderpath, num_reps, r_bins):

#     #     print("parsing LD statistics in parallel")
#     # # Submit tasks to Ray in parallel using .remote()
#     # futures = [get_LD_stats.remote(ii, r_bins) for ii in range(num_reps)]
#     # # Gather results with ray.get() to collect them once the tasks are finished
#     # ld_stats = ray.get(futures)
#     # # Optionally, you can convert the list of results into a dictionary with indices
#     # ld_stats_dict = {ii: result for ii, result in enumerate(ld_stats)}


#     flat_map_path = os.path.join(folderpath, "flat_map.txt")
#     pop_file_path = os.path.join(folderpath, "samples.txt")
#     vcf_files = [
#         os.path.join(folderpath, f"rep.{rep_ii}.vcf.gz")
#         for rep_ii in range(num_reps)
#     ]

#     # Launch the tasks in parallel using Ray
#     futures = [
#         get_LD_stats.remote(vcf_file, r_bins, flat_map_path, pop_file_path)
#         for vcf_file in vcf_files
#     ]

#     # Wait for all the tasks to complete and retrieve results
#     results = ray.get(futures)
#     return results

def compute_ld_stats_sequential(flat_map_path, pop_file_path, metadata_path, r_bins):
    print("=== Computing LD statistics sequentially ===")
    # Debugging: Print the path to check if it's correct
    print(f"Looking for metadata file at: {metadata_path}")

    # Check if the file exists before trying to open it
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
    else:
        print(f"Metadata file found at {metadata_path}, proceeding to open it...")

        # Try opening the file and read its contents
        try:
            with open(metadata_path, 'r') as f:
                vcf_files = [line.strip() for line in f]
            
        
        except Exception as e:
            print(f"Error while reading metadata file: {str(e)}")

    # List to store LD statistics results
    ld_stats_list = []

    # Sequentially compute LD statistics for each VCF file
    for vcf_file in vcf_files:
        ld_stats = get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path)
        ld_stats_list.append(ld_stats)
    
    return ld_stats_list

def run_inference_dadi(
    sfs,
    p0,
    num_samples,
    demographic_model,
    lower_bound=[0.001, 0.001, 0.001, 0.001],
    upper_bound=[1, 1, 1, 1],
    mutation_rate=1.26e-8,
    length=1e8,
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


    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    # Optimization with dadi
    opt_params, ll_value = dadi.Inference.opt(
        p_guess,
        sfs,
        func_ex,
        pts=pts_ext,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        algorithm=nlopt.LN_BOBYQA,
        maxeval=10
    )

    print(f"OPT DADI PARAMETER: {opt_params}")

    # # Find the indices of the top top_k_values (those with the highest likelihood)
    # top_k_indices = np.argsort(ll_list)[-top_values_k:]
    # top_k_indices = np.array(top_k_indices, dtype=int)  # Convert to numpy integer array

    # opt_params_dict_list_top_values = [opt_params_dict_list[i] for i in top_k_indices]
    # opt_params_final_list = []

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
            "ll": ll_value
        }

    elif demographic_model == "split_isolation_model":
        opt_params_dict = {
            "Na": N_ref,
            "N1": opt_params[0]*N_ref,
            "N2": opt_params[1]*N_ref,
            "t_split": opt_params[2]*2*N_ref,
            "m": opt_params[3]*2*N_ref, 
            "ll": ll_value 
        }

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    model = model * opt_theta

    return model, opt_theta, opt_params_dict

def run_inference_moments(
    sfs,
    p0,
    demographic_model,
    lower_bound=[0.001, 0.001, 0.001, 0.001],
    upper_bound=[1, 1, 1, 1],
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
        maxeval=10
    ) # I don't want the log likelihood. 

    print(f"OPT MOMENTS PARAMETER: {opt_params}")

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
                "ll": ll
            }

        elif demographic_model == "split_isolation_model":
            opt_params_dict = {
                "Na": N_ref,
                "N1": opt_params[0]*N_ref,
                "N2": opt_params[1]*N_ref,
                "t_split": opt_params[2]*2*N_ref,
                "m": opt_params[3]*2*N_ref,
                "upper_triangular_FIM": upper_triangular,
                "ll": ll
            }

        else:
            raise ValueError(f"Unsupported demographic model: {demographic_model}")

        if opt_params_dict["upper_triangular_FIM"] is None:
            del opt_params_dict["upper_triangular_FIM"]

        model = model * opt_theta

    # print(f'Log Likelihood Values for Moments: {ll_list}')
    return model, opt_theta, opt_params

def run_inference_momentsLD(ld_stats, demographic_model, p_guess):
    """
    This should do the parameter inference for momentsLD.
    """
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    ll_list = []
    opt_params_dict_list = []

    print("====================================================")
    print(ld_stats.keys())

    mv = moments.LD.Parsing.bootstrap_data(ld_stats)  # type: ignore

    if demographic_model == "bottleneck_model":
        demo_func = moments.LD.Demographics1D.three_epoch  # type: ignore
    elif demographic_model == "split_isolation_model":
        demo_func = demographic_models.split_isolation_model_momentsLD
    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # Perform optimization
    opt_params, ll = moments.LD.Inference.optimize_log_lbfgsb(  # type: ignore
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins, verbose=3, maxiter=20
    )

    # Rescale parameters to physical units
    physical_units = moments.LD.Util.rescale_params(  # type: ignore
        opt_params, ["nu", "nu", "T", "m", "Ne"]
    )
    ll_list.append(ll)

    opt_params_dict = {}
    if demographic_model == "bottleneck_model":
        opt_params_dict = {
            "Nb": opt_params[0] * opt_params[4],
            "N_recover": opt_params[1] * opt_params[4],
            "t_bottleneck_start": (opt_params[2] + opt_params[3]) * 2 * opt_params[4],
            "t_bottleneck_end": opt_params[3] * 2 * opt_params[4]
        }

    elif demographic_model == "split_isolation_model":
        physical_units = moments.LD.Util.rescale_params(  # type: ignore
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

        print("best fit parameters:")
        print(f"  N(deme0)         :  {physical_units[0]:.1f}")
        print(f"  N(deme1)         :  {physical_units[1]:.1f}")
        print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
        print(f"  Migration rate   :  {physical_units[3]:.6f}")
        print(f"  N(ancestral)     :  {physical_units[4]:.1f}")

        opt_params_dict_list.append(opt_params_dict)

    return opt_params_dict_list, ll_list