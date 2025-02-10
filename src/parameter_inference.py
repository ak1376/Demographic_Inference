import dadi
import moments
import os
import numpy as np
import time
from moments.Godambe import _get_godambe
import nlopt
import src.demographic_models as demographic_models
from src.optimize import opt
import time
import multiprocessing
from collections import OrderedDict
import nlopt

TIMEOUT_SECONDS = 20 * 60  # 20 minutes = 1200 seconds

def norm(p, mean, stddev):
    return [(x - m) / s for x, m, s in zip(p, mean, stddev)]


def unnorm(z, mean, stddev):
    return [z_i * s + m for z_i, m, s in zip(z, mean, stddev)]

def real_to_dadi_params(real_params):
    """
    Suppose real_params = (N1, N2, M12, M21, Tsplit_in_gen)
    all in real units (pop sizes, migration per gen, time in generations, etc.)
    
    We want to scale them to: (nu1, nu2, m12, m21, t_split)
    which are the parameters expected by the original dadi function:
       - nu1, nu2 = N1 / N_anc  (for example)
       - m12, m21 = (some formula for scaled migration)
       - t_split = Tsplit_in_gen / (2*N_anc)
    or whatever your chosen reference is.
    
    You define that reference once (N_anc, mu, generation_time, etc.).
    Then you do the math here.
    """

    # Example:
    # real_params = (N1, N2, M12, M21, T_gen)
    # Let’s define some reference:
    N_anc = 1000.0  # <--- you pick your reference size
    # For migration: in coalescent units, m12 (the symmetric migration parameter)
    # is typically 4*N_anc*m (if haploid or diploid factor?). This depends on how
    # your data was simulated. Let's assume the simple version:
    #   m12_dadi = M12 (per gen) * 2*N_anc
    #   T_dadi = T_gen / (2*N_anc)
    # likewise for population sizes.
    
    (N1, N2, M12, M21, T_gen) = real_params
    
    nu1 = N1 / N_anc
    nu2 = N2 / N_anc
    m12 = M12 * 2 * N_anc
    m21 = M21 * 2 * N_anc
    t_split = T_gen / (2 * N_anc)
    
    scaled_params = (nu1, nu2, m12, m21, t_split)
    return scaled_params


def diffusion_sfs_moments(parameters: list[float], sample_sizes: OrderedDict, demographic_model) -> moments.Spectrum:
    """
    Get the expected SFS under the diffusion approximation (moments).
    """
    # 1) Convert our parameter list into a dictionary

    if demographic_model == "split_migration_model":
        demo_model = demographic_models.split_migration_model_simulation
        param_dict = {
            "N0": parameters[0],
            "N1": parameters[1],
            "N2": parameters[2],
            "m12": parameters[3],
            "m21": parameters[4],
            "t_split": parameters[5],
        }
    elif demographic_model == "split_isolation_model":
        demo_model = demographic_models.split_isolation_model_simulation
        param_dict = {
            "N0": parameters[0],
            "N1": parameters[1],
            "N2": parameters[2],
            "m": parameters[3],
            "t_split": parameters[4],
        }
    elif demographic_model == "bottleneck_model":
        demo_model = demographic_models.bottleneck_model
        param_dict = {
            "N0": parameters[0],
            "Nb": parameters[1],
            "N_recover": parameters[2],
            "t_bottleneck_end": parameters[3],
        }
    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # 2) Build the demes graph
    demes_graph = demo_model(param_dict)

    # 3) Construct the Spectrum via from_demes
    sampled_demes = list(sample_sizes.keys())
    haploid_sample_sizes = [n * 2 for n in sample_sizes.values()]

    # Typically, you'd set theta = 4 * Nref * mu * L for a given reference size Nref,
    # but here we just use the first population's size as "N0".
    # Adjust to your model's convention for Nref if needed.
    Nref = parameters[0]  # or define some other reference population size
    theta = 4 * Nref * mutation_rate * sequence_length

    sfs = moments.Spectrum.from_demes(
        demes_graph,
        sample_sizes=haploid_sample_sizes,
        sampled_demes=sampled_demes,
        theta=theta,
    )
    return sfs

def diffusion_sfs_dadi(
    parameters: list[float],
    sample_sizes: OrderedDict,
    demographic_model: str,
    mutation_rate: float,
    sequence_length: float,
    pts: list[int],
) -> dadi.Spectrum:
    """
    Get the expected SFS under the diffusion approximation (using dadi)
    by building a demes.Graph and converting to a dadi model via demes_dadi.

    Parameters
    ----------
    parameters : list[float]
        Model parameters in demographic units or scaled units
        (depending on how your demes builder interprets them).
    sample_sizes : OrderedDict
        e.g. {"N1": 15, "N2": 8} for 2D data
    demographic_model : str
        One of ["split_migration_model", "split_isolation_model", "bottleneck_model"].
    mutation_rate : float
        Mutation rate per generation per base.
    sequence_length : float
        Total number of base pairs or length of region.
    pts : list of int
        Extrapolation grid sizes for dadi (e.g. [60, 70, 80]).

    Returns
    -------
    dadi.Spectrum
        The model-predicted SFS (on the largest grid in `pts`).
    """
    # 1) Parse the parameters and pick the correct demes-based function
    parameters.insert(0,10000) # Insert the ancestral pop size. 
    print(f'THE PARAMETERS PROVIDED ARE: {parameters}')
    
    if demographic_model == "split_migration_model":
        # For example, [N0, N1, N2, m12, m21, t_split]
        param_dict = {
            "N0": parameters[0],
            "N1": parameters[1],
            "N2": parameters[2],
            "m12": parameters[3],
            "m21": parameters[4],
            "t_split": parameters[5],
        }
        demo_func = demographic_models.split_migration_model_simulation

    elif demographic_model == "split_isolation_model":
        # For example, [N0, N1, N2, m, t_split]
        param_dict = {
            "N0": parameters[0],
            "N1": parameters[1],
            "N2": parameters[2],
            "m": parameters[3],
            "t_split": parameters[4],
        }
        demo_func = demographic_models.split_isolation_model_simulation

    elif demographic_model == "bottleneck_model":
        # For example, [N0, Nb, N_recover, t_bottleneck_end, ...]
        # Adjust indexing to match your code.
        param_dict = {
            "N0": parameters[0],
            "Nb": parameters[1],
            "N_recover": parameters[2],
            "t_bottleneck_end": parameters[3],
            # etc. if more parameters
        }
        demo_func = demographic_models.bottleneck_model

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # 2) Build the demes graph
    start = time.time()
    demes_graph = demo_func(param_dict)  # e.g. returns a demes.Graph
    end = time.time()

    # print(f'TIME TO BUILD DEMES GRAPH: {end - start}')

    # 3) Convert the demes Graph to a dadi function using demes_dadi
    #    This creates a python function: model_func(pts, ns, [params]) -> fs
    #    But the demes graph is already fully specified. We just pass it in.
    # dadi_model_func = demes_dadi.Demes2Dadi(demes_graph)

    # 4) Evaluate on the extrapolation grid
    #    For a 2D model, sample_sizes might be [n1, n2]. For 1D, [n]. For 3D, etc.
    ns = [2 * n for n in sample_sizes.values()]  # e.g., [30, 16] if input is [15, 8]
    # print(f'SAMPLE SIZES: {ns}')

    start = time.time()

    model_fs = dadi.Spectrum.from_demes(
        demes_graph,
        sampled_demes = list(sample_sizes.keys()),
        sample_sizes = ns,
        pts = pts
    )

    end = time.time()

    # print(f'TIME TO BUILD SFS: {end - start}')
    
    # model_fs = dadi_model_func(pts, ns)

    # 5) Scale by theta if you want an absolute SFS. Typically, we do:
    #    theta = 4 * Nref * mu * L. If you consider parameters[0] = N0, then:
    # Nref = parameters[0]
    # theta = 4.0 * Nref * mutation_rate * sequence_length

    # model_fs *= theta

    # model_fs is a dadi.Spectrum. You can return it directly.
    # print(f'Diffusion SFS Dadi: {model_fs}')
    # print(f'Diffusion SFS Dadi Shape: {model_fs.shape}')
    return model_fs

# Define your function with Ray's remote decorator
def get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path):

    # Read the file and extract unique populations
    with open(pop_file_path, "r") as file:
        # Skip the header
        lines = file.readlines()[1:]
        # Extract the population column
        populations = [line.strip().split("\t")[1] for line in lines]
        # Get unique populations
        unique_populations = list(set(populations))

    ld_stats = moments.LD.Parsing.compute_ld_statistics( #type:ignore
        vcf_file,
        rec_map_file=flat_map_path,
        pop_file=pop_file_path,
        pops=unique_populations,
        r_bins=r_bins,
        report=True,
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

def _optimize_dadi(
    queue,
    init_z,             # initial guess in z-space
    sfs,
    model_func,         # e.g. split_migration_model_dadi
    pts_ext,
    lower_bound,        # real-space bound
    upper_bound,        # real-space bound
    mean,               # used for un‐z‐scoring
    stddev              # used for un‐z‐scoring
):
    """
    Runs dadi optimization in *z-scored* space, then returns the
    best-fit *un-z-scored* parameters + LL to the parent process via 'queue'.
    """

    # 1) Convert real-space bounds to z-space
    lb_z = norm(lower_bound, mean, stddev)
    ub_z = norm(upper_bound, mean, stddev)

    # 2) Make a wrapper that un‐z‐scores inside
    def z_wrapper(z_params, ns, pts):
        scaled_params = unnorm(z_params, mean, stddev)
        return model_func(scaled_params, ns, pts)

    # 3) Extrapolation
    func_ex = dadi.Numerics.make_extrap_log_func(z_wrapper)
    print(f'Starting Dadi Optimization')

    # 4) Run the optimizer in z-space
    opt_params_z, ll_value = dadi.Inference.opt(
        init_z,
        sfs,
        func_ex,
        pts=pts_ext,
        lower_bound=lb_z,
        upper_bound=ub_z,
        algorithm=nlopt.LN_BOBYQA,  # or LN_SBPLX, etc.
        maxeval=400,
        verbose=1
    )

    # 5) Convert best-fit from z-space to real (scaled) space
    fitted_params = unnorm(opt_params_z, mean, stddev)
    print(f'The converted to real space parameters are: {fitted_params}')

    # 6) Send them back to the parent
    queue.put((fitted_params, ll_value))

def _optimize_moments(queue, p_guess, sfs, model_func, lower_bound, upper_bound, sample_sizes_fit):
    """
    This function wraps moments.Inference.optimize_log_powell
    so we can run it in a separate process. We'll push results
    (opt_params, ll) into 'queue'.
    """

    # Optimize using Powell
    print(f'Empirical SFS shape: {sfs.shape}')
    xopt = moments.Inference.optimize_powell(
        norm(start),
        sfs,
        lambda z, n: diffusion_sfs_moments(unnorm(z), sample_sizes_fit),
        lower_bound=norm(lower_bound),
        upper_bound=norm(upper_bound),
        multinom=False,
        verbose=10,
        flush_delay=0.0,
        full_output=True
    )

    fitted_params = unnorm(xopt[0])
    ll = xopt[1]

    # full_output=True => returns (best_params, best_ll, ...)
    # xopt = moments.Inference.optimize_log_lbfgsb(
    #     p_guess,
    #     sfs,
    #     model_func,
    #     lower_bound=lower_bound,
    #     upper_bound=upper_bound,
    #     verbose=20,
    #     full_output=True
    # )
    # xopt[0] = opt_params, xopt[1] = ll
    queue.put((fitted_params, ll))  # pass (opt_params, ll) back to the parent


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
    Perform dadi parameter inference on a masked, unnormalized SFS.
    We'll do the optimization in z-scored space,
    but the final best-fit parameters remain in coalescent space for dadi.
    """

    # 0) sample_sizes_fit, not strictly needed below, but for reference
    sample_sizes_fit = OrderedDict(
        (p, (n - 1)//2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )
    ns = sfs.sample_sizes

    print(f"lower bound: {lower_bound}")
    print(f"upper bound: {upper_bound}")

    # 1) Build "mean" and "stddev" from your (scaled) bounds
    mean = [(l+u)/2 for (l, u) in zip(lower_bound, upper_bound)]
    stddev = [(u - l)/np.sqrt(12) for (l, u) in zip(lower_bound, upper_bound)]

    # 2) Pick the correct dadi model in scaled space
    if demographic_model == "bottleneck_model":
        model_func = demographic_models.three_epoch_fixed
    elif demographic_model == "split_isolation_model":
        model_func = demographic_models.split_isolation_model_dadi
    elif demographic_model == "split_migration_model":
        model_func = demographic_models.split_migration_model_dadi
    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # 3) Setup grids for extrapolation
    pts_ext = [max(ns) + 10, max(ns) + 20, max(ns) + 30]

    # 4) Perturb the initial guess to avoid local minima
    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    # 5) Convert that guess to z-space
    init_z = norm(p_guess, mean, stddev)

    # 6) Launch the optimization in a separate process
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_optimize_dadi,
        args=(
            queue,          # for returning results
            init_z,         # initial guess in z-space
            sfs,
            model_func,     # the original dadi model
            pts_ext,
            lower_bound,
            upper_bound,
            mean,
            stddev
        )
    )
    process.start()
    TIMEOUT_SECONDS = 20*60  # e.g. 20 mins
    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        print("dadi optimization took too long. Terminating...")
        process.terminate()
        process.join()
        return None, None, {"N0": np.nan, "ll": np.nan}

    # 7) Retrieve best-fit from the queue
    opt_params_scaled, ll_value = queue.get()
    print(f"Best-fit scaled params: {opt_params_scaled}")
    print(f"Log-likelihood: {ll_value}")

    # 8) Reconstruct final model
    func_ex = dadi.Numerics.make_extrap_log_func(model_func)
    model = func_ex(opt_params_scaled, ns, pts_ext)

    # 9) From that model & data, compute best-fit theta
    opt_theta = dadi.Inference.optimal_sfs_scaling(model, sfs)
    print(f"Best-fit theta = {opt_theta}")

    # 10) Convert that theta -> N_ref = theta / (4*mu*L)
    N_ref = opt_theta / (4.0 * mutation_rate * length)
    print(f"Estimated N_ref = {N_ref}")

    # 11) Build dictionary. E.g. for split_migration_model:
    if demographic_model == "split_migration_model":
        # Suppose we have (nu1, nu2, m12, m21, t_split)
        nu1, nu2, m12, m21, t_split = opt_params_scaled
        opt_params_dict = {
            "Na": N_ref,
            "N1": nu1*N_ref,
            "N2": nu2*N_ref,
            "t_split": t_split * 2 * N_ref,    # in generations
            "m12": m12 * 2 * N_ref,           # migration per generation
            "m21": m21 * 2 * N_ref,
            "ll": ll_value
        }
    elif demographic_model == "split_isolation_model":
        # e.g. (nu1, nu2, T, m)
        nu1, nu2, T, m = opt_params_scaled
        opt_params_dict = {
            "Na": N_ref,
            "N1": nu1*N_ref,
            "N2": nu2*N_ref,
            "t_split": T * 2 * N_ref,
            "m": m * 2 * N_ref,
            "ll": ll_value,
        }
    elif demographic_model == "bottleneck_model":
        # e.g. (nuB, nuF, T1, T2)
        nuB, nuF, T1, T2 = opt_params_scaled
        opt_params_dict = {
            "N0": N_ref,
            "Nb": nuB * N_ref,
            "N_recover": nuF * N_ref,
            "t_bottleneck_end": T1 * 2 * N_ref,  # or however your model is set
            "ll": ll_value,
        }
    else:
        opt_params_dict = {}

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
    Run parameter inference for Moments in a separate process. 
    If the optimization takes more than 20 minutes, kill it 
    and return np.nan for all parameters (including the FIM).
    """

    # Recompute sample sizes from the SFS shape.
    sample_sizes_fit = OrderedDict((p, (n - 1) // 2)
                                   for p, n in zip(sfs.pop_ids, sfs.shape))

    
    # Extract lower and upper bounds
    lb = experiment_config["lower_bound_params"]
    ub = experiment_config["upper_bound_params"]

    # Compute mean and standard deviation
    mean = [(lb[param] + ub[param]) / 2 for param in lb]
    stddev = [(ub[param] - lb[param]) / np.sqrt(12) for param in lb]

    # 1) Perturb initial guess
    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    # 2) Pick the correct demographic model
    if demographic_model == "bottleneck_model":
        model_func = demographic_models.three_epoch_fixed_moments  # built-in example
    elif demographic_model == "split_isolation_model":
        model_func = demographic_models.split_isolation_model_moments
    elif demographic_model == "split_migration_model":
        model_func = demographic_models.split_migration_model_moments
    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # 3) Create a Queue and Process for the optimization
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_optimize_moments,
        args=(queue, p_guess, sfs, model_func, lower_bound, upper_bound)
    )

    # 4) Start the process and wait up to 20 minutes
    process.start()
    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        # 4a) If still running after 20 minutes, kill it
        print("moments optimization took longer than 20 minutes. Terminating...")
        process.terminate()
        process.join()

        # Return np.nan for everything, including FIM if needed
        if demographic_model == "bottleneck_model":
            opt_params_dict = {
                "N0": np.nan,
                "Nb": np.nan,
                "N_recover": np.nan,
                "t_bottleneck_end": np.nan,
                "ll": np.nan,
            }
            if use_FIM:
                opt_params_dict["upper_triangular_FIM"] = np.nan
            return None, None, opt_params_dict

        elif demographic_model == "split_isolation_model":
            opt_params_dict = {
                "Na": np.nan,
                "N1": np.nan,
                "N2": np.nan,
                "t_split": np.nan,
                "m": np.nan,
                "ll": np.nan,
            }
            if use_FIM:
                num_params = len(p0)
                num_diag_elements = (num_params * (num_params + 1)) // 2  # formula for upper triangle including diagonal
                opt_params_dict["upper_triangular_FIM"] = np.full((num_diag_elements,), np.nan)
            return None, None, opt_params_dict

        elif demographic_model == "split_migration_model":
            opt_params_dict = {
                "Na": np.nan,
                "N1": np.nan,
                "N2": np.nan,
                "t_split": np.nan,
                "m12": np.nan,
                "m21": np.nan,
                "ll": np.nan,
            }
            if use_FIM:
                num_params = len(p0)
                num_diag_elements = (num_params * (num_params + 1)) // 2

    # 5) Otherwise, retrieve the results from the queue
    opt_params, ll = queue.get()
    print(f"OPT MOMENTS PARAMETER: {opt_params}")
    print(f"LL: {ll}")

    # 6) Build the final model
    model = model_func(opt_params, sfs.sample_sizes)
    opt_theta = moments.Inference.optimal_sfs_scaling(model, sfs)

    # 7) Compute N_ref
    N_ref = opt_theta / (4 * mutation_rate * length)

    # 8) Optionally compute the Hessian / FIM
    upper_triangular = None
    if use_FIM:
        H = _get_godambe(
            model_func,
            all_boot=[],
            p0=opt_params,
            data=sfs,
            eps=1e-6,
            log=False,
            just_hess=True,
        )
        FIM = -1 * H  # typical sign for Hessian
        upper_tri_indices = np.triu_indices(FIM.shape[0])
        upper_triangular = FIM[upper_tri_indices]

    # 9) Build the param dictionary 
    #    (separately for each demographic_model)
    if demographic_model == "bottleneck_model":
        nuB, nuF, time_since_recovery = opt_params
        # Times are in coalescent units => multiply by 2*N_ref to get generations
        # Sizes are multiples of N_ref
        opt_params_dict = {
            "N0": N_ref,
            "Nb": nuB * N_ref,
            "N_recover": nuF * N_ref,
            "t_bottleneck_end": time_since_recovery * 2 * N_ref,           # generations
            "ll": ll,
        }
        if use_FIM:
            opt_params_dict["upper_triangular_FIM"] = upper_triangular

    elif demographic_model == "split_isolation_model":
        opt_params_dict = {
            "Na": N_ref,
            "N1": opt_params[0] * N_ref,
            "N2": opt_params[1] * N_ref,
            "t_split": opt_params[2] * 2 * N_ref,
            "m": opt_params[3] * 2 * N_ref,
            "ll": ll,
        }
        if use_FIM:
            opt_params_dict["upper_triangular_FIM"] = upper_triangular

    elif demographic_model == "split_migration_model":
        opt_params_dict = {
            "Na": N_ref,
            "N1": opt_params[0] * N_ref,
            "N2": opt_params[1] * N_ref,
            "t_split": opt_params[4] * 2 * N_ref,
            "m12": opt_params[2] * 2 * N_ref,
            "m21": opt_params[3] * 2 * N_ref,
            "ll": ll,
        }
        if use_FIM:
            opt_params_dict["upper_triangular_FIM"] = upper_triangular

    # 10) Scale the model by opt_theta
    model *= opt_theta

    # 11) Return the final outputs
    return model, opt_theta, opt_params_dict

def run_inference_momentsLD(ld_stats, demographic_model, p_guess, experiment_config):
    """
    This should do the parameter inference for momentsLD.
    """
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    ll_list = []
    opt_params_dict_list = []

    print("====================================================")
    print(ld_stats.keys())

    mv = moments.LD.Parsing.bootstrap_data(ld_stats)  # type: ignore
    print('MV CREATION COMPLETED!')

    if demographic_model == "bottleneck_model":
        demo_func = demographic_models.three_epoch_fixed_MomentsLD #TODO: CHANGE
    elif demographic_model == "split_isolation_model":
        demo_func = demographic_models.split_isolation_model_momentsLD

    elif demographic_model == "split_migration_model":
        demo_func = demographic_models.split_migration_model_momentsLD

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # Perform optimization
    opt_params, ll = moments.LD.Inference.optimize_log_fmin(  # type: ignore
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins, verbose=1, maxiter=400, 
        lower_bound = experiment_config['lower_bound_optimization'],
        upper_bound = experiment_config['upper_bound_optimization']
    )

    ll_list.append(ll)

    opt_params_dict = {}
    if demographic_model == "bottleneck_model":
        # opt_params[0]: Nb
        # opt_params[1]: N_recover 
        # opt_params[2]: t_bottleneck_end
        # opt_params[3]: N_ref
        opt_params_dict = {
            "N0": opt_params[3],
            "Nb": opt_params[0] * opt_params[3],
            "N_recover": opt_params[1] * opt_params[3],
            "t_bottleneck_end": opt_params[2] * 2 * opt_params[3]
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

    elif demographic_model == "split_migration_model":
        physical_units = moments.LD.Util.rescale_params(
            opt_params, ["nu", "nu", "T", "m", "m", "Ne"]  # Use "m" for both migration rates
        )

        print(physical_units)

        opt_params_dict = {
            "N1": physical_units[0],
            "N2": physical_units[1],
            "t_split": physical_units[2],
            "m12": physical_units[3],  # Matches first "m"
            "m21": physical_units[4],  # Matches second "m"
            'Na': physical_units[5]
        }

        print("best fit parameters:")
        print(f"  N(deme1)         :  {physical_units[0]:.1f}")
        print(f"  N(deme2)         :  {physical_units[1]:.1f}")
        print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
        print(f"  Migration rate 1 :  {physical_units[3]:.6f}")
        print(f"  Migration rate 2 :  {physical_units[4]:.6f}")
        print(f"  N(ancestral)     :  {physical_units[5]:.1f}")

    opt_params_dict_list.append(opt_params_dict)

    return opt_params_dict_list, ll_list