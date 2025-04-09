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

def real_to_dadi_params(real_params, demographic_model, parameter_names=None):
    """
    Suppose real_params = (N1, N2, M12, M21, Tsplit_in_gen) or a dict if parameter_names is None.
    If parameter_names is provided, it must be a sequence of names with the same length as real_params,
    and we’ll convert real_params into a dict mapping names→values before scaling.
    """

    # If the user passed in a list/tuple of values plus names, build a dict
    if parameter_names is not None:
        if len(parameter_names) != len(real_params):
            raise ValueError(
                f"parameter_names length ({len(parameter_names)}) does not match "
                f"real_params length ({len(real_params)})"
            )
        real_params = dict(zip(parameter_names, real_params))

    scaled_params = None

    if demographic_model == "split_migration_model":
        N_anc = real_params['N0']
        N1 = real_params['N1']
        N2 = real_params['N2']
        M12 = real_params['m12']
        M21 = real_params['m21']
        T_gen = real_params['t_split']

        scaled_params = {
            'nu1': N1 / N_anc,
            'nu2': N2 / N_anc,
            'm12': M12 * (2 * N_anc),
            'm21': M21 * (2 * N_anc),
            't_split': T_gen / (2 * N_anc),
        }

    elif demographic_model == "split_isolation_model":
        N_anc = real_params['Na']
        N1 = real_params['N1']
        N2 = real_params['N2']
        M = real_params['m']
        T_gen = real_params['t_split']

        scaled_params = {
            'nu1': N1 / N_anc,
            'nu2': N2 / N_anc,
            'm': M * (2 * N_anc),
            't_split': T_gen / (2 * N_anc),
        }

    elif demographic_model == "bottleneck_model":
        N_anc = real_params['N0']
        Nb = real_params['Nb']
        N_recover = real_params['N_recover']
        t_end = real_params['t_bottleneck_end']

        scaled_params = {
            'nuB': Nb / N_anc,
            'nuF': N_recover / N_anc,
            't_bottleneck_end': t_end / (2 * N_anc),
        }

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    return scaled_params


def diffusion_sfs_moments(parameters: list[float],
    sample_sizes: OrderedDict,
    demographic_model: str,
    mutation_rate: float,
    sequence_length: float) -> moments.Spectrum:
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
            "Na": parameters[0],
            "N1": parameters[1],
            "N2": parameters[2],
            "m": parameters[4],
            "t_split": parameters[3],
        }
    elif demographic_model == "bottleneck_model":
        demo_model = demographic_models.bottleneck_model
        param_dict = {
            "N_recover": parameters[0],
            "t_bottleneck_start": parameters[1],
            "t_bottleneck_end": parameters[2]
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
    # parameters.insert(0,10000) # Insert the ancestral pop size. 
    
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
            "Na": parameters[0],
            "N1": parameters[1],
            "N2": parameters[2],
            "m": parameters[4],
            "t_split": parameters[3],
        }
        demo_func = demographic_models.split_isolation_model_simulation

    elif demographic_model == "bottleneck_model":
        # For example, [N0, Nb, N_recover, t_bottleneck_end, ...]
        # Adjust indexing to match your code.
        param_dict = {
            "N_recover": parameters[0],
            "t_bottleneck_start": parameters[1],
            "t_bottleneck_end": parameters[2],
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
    Nref = parameters[0]
    theta = 4.0 * Nref * mutation_rate * sequence_length

    model_fs *= theta

    # model_fs is a dadi.Spectrum. You can return it directly.
    # print(f'Diffusion SFS Dadi: {model_fs}')
    # print(f'Diffusion SFS Dadi Shape: {model_fs.shape}')
    return model_fs

# Define your function with Ray's remote decorator
def get_LD_stats(vcf_file, r_bins, flat_map_path, pop_file_path):
    # Read the file and extract unique populations while preserving order
    with open(pop_file_path, "r") as file:
        lines = file.readlines()[1:]  # Skip header
        populations = [line.strip().split("\t")[1] for line in lines]

        # Preserve order while ensuring uniqueness
        unique_populations = list(dict.fromkeys(populations)) 

        print(f"Unique populations (order preserved): {unique_populations}")

    # Open the VCF file
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file=flat_map_path,
        pop_file=pop_file_path,
        pops=unique_populations,  # Now correctly ordered
        r_bins=r_bins,
        report=False,
        use_h5=False
    )

    return ld_stats

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

import nlopt

def _optimize_dadi(
    queue,
    p_guess,             # initial guess in z-space
    sfs,
    demographic_model,  # model name
    sample_sizes_fit,   # OrderedDict of sample sizes
    mutation_rate,
    sequence_length,
    pts_ext,
    lower_bound,        # real-space bound
    upper_bound        # real-space bound
):
    """
    Runs dadi optimization in *z-scored* space, then returns the
    best-fit *un-z-scored* parameters + LL to the parent process via 'queue'.
    Uses the diffusion SFS from dadi.
    """

    def raw_wrapper(scaled_params, ns, pts):
        # scaled_params = unnorm(z_params, mean, stddev)
        # print(f'unscaled Parameters are: {scaled_params}')
        return diffusion_sfs_dadi(
            scaled_params,
            sample_sizes_fit,
            demographic_model,
            mutation_rate,
            sequence_length,
            pts
        )

    # 3) Extrapolation function
    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    print(f'Lower bound: {lower_bound}')
    print(f'Upper bound: {upper_bound}')

    fitted_params, ll_value = dadi.Inference.opt(
    p_guess,
    sfs, 
    func_ex,
    pts=pts_ext,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    algorithm=nlopt.LN_BOBYQA,
    maxeval=5000,
    verbose=1
    )

    # 3) Run the optimizer in z-space
    # xopt = dadi.Inference.optimize_log_powell(
    #     p_guess,
    #     sfs,
    #     func_ex,
    #     pts=pts_ext,
    #     lower_bound=lower_bound,
    #     upper_bound=upper_bound,
    #     multinom=False,
    #     verbose=True,
    #     flush_delay=0.0,
    #     full_output=True,
    #     maxiter=1000
    # )
    
    # fitted_params = xopt[0]
    # ll_value = xopt[1]

    # fitted_params = unnorm(opt_params_z, mean, stddev)
    print(f"Best-fit dadi params (real-space): {fitted_params}")

    # # 5) Convert best-fit from z-space to real (scaled) space
    # # print(f'The initial guess in real space is: {unnorm(init_z, mean, stddev)}')
    # print(f'The optimized parameters in real space are : {fitted_params}')

    # 6) Send them back to the parent process
    queue.put((fitted_params, ll_value))

def _optimize_moments(
    queue,
    init_z,  # Initial guess in z-space
    sfs,
    demographic_model,
    sample_sizes_fit,
    lower_bound,
    upper_bound,
    mutation_rate,
    sequence_length,
    mean,
    stddev
):
    """
    Runs moments optimization in *z-scored* space, then returns the
    best-fit *un-z-scored* parameters + LL to the parent process via 'queue'.
    Uses the diffusion SFS from moments.
    """

    # 1) Convert real-space bounds to z-space
    lb_z = norm(lower_bound, mean, stddev)
    ub_z = norm(upper_bound, mean, stddev)

    # 2) Define wrapper function for moments optimization in z-space
    def z_wrapper(z_params, ns):
        scaled_params = unnorm(z_params, mean, stddev)
        return diffusion_sfs_moments(scaled_params, sample_sizes_fit, demographic_model, mutation_rate, sequence_length)

    # 3) Run the optimizer in z-space
    xopt = moments.Inference.optimize_powell(
        init_z,
        sfs,
        lambda z, n: z_wrapper(z, n),
        lower_bound=lb_z,
        upper_bound=ub_z,
        multinom=False,
        verbose=0,
        flush_delay=0.0,
        full_output=True
    )

    # 4) Convert best-fit from z-space to real (scaled) space
    fitted_params = unnorm(xopt[0], mean, stddev)
    ll_value = xopt[1]

    print(f"Best-fit moments params (real-space): {fitted_params}")
    
    # 5) Send results back to parent process
    queue.put((fitted_params, ll_value))


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

    pop_ids = None

    if demographic_model == "split_migration_model":
        pop_ids = ['N1', 'N2']
    elif demographic_model == "split_isolation_model":
        pop_ids = ['N1', 'N2']
    elif demographic_model == "bottleneck_model":
        pop_ids = ['N0']
    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    sfs.pop_ids = pop_ids

    sample_sizes_fit = OrderedDict(
        (p, (n - 1)//2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )
    ns = sfs.sample_sizes

    # 3) Setup grids for extrapolation
    pts_ext = [max(ns) + 60, max(ns) + 70, max(ns) + 80]

    # 4) Perturb the initial guess to avoid local minima
    p_guess = moments.Misc.perturb_params(
        p0, fold=0.1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    p_guess = p0.copy()
    # 6) Launch the optimization in a separate process
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_optimize_dadi,
        args=(
            queue,           # for returning results
            p_guess,          # initial guess in z-space
            sfs,             # empirical SFS
            demographic_model,  # model name as a string (e.g., "split_migration_model")
            sample_sizes_fit,  # OrderedDict of sample sizes
            mutation_rate,   # mutation rate
            length,          # sequence length
            pts_ext,         # extrapolation grid points
            lower_bound,     # real-space lower bound
            upper_bound,     # real-space upper bound
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

    # 8) Generate the model SFS using diffusion_sfs_dadi #TODO: This may fail for the bottleneck. Need to check that this works. 
    model_sfs = diffusion_sfs_dadi(
        opt_params_scaled,  # Best-fit parameters from optimization
        sample_sizes_fit,   # Sample sizes as an OrderedDict
        demographic_model,  # Model name (e.g., "split_migration_model")
        mutation_rate,      # Mutation rate
        length,             # Sequence length
        pts_ext             # Extrapolation grid points
    )

    # 9) Compute best-fit theta
    opt_theta = dadi.Inference.optimal_sfs_scaling(model_sfs, sfs)

    # 11) Build dictionary. E.g. for split_migration_model:
    if demographic_model == "split_migration_model":
        # Suppose we have (nu1, nu2, m12, m21, t_split)
        N_ref, nu1, nu2, m12, m21, t_split = opt_params_scaled
        opt_params_dict = {
            "N0": N_ref,
            "N1": nu1,
            "N2": nu2,
            "t_split": t_split,    # in generations
            "m12": m12,           # migration per generation
            "m21": m21,
            "ll": ll_value
        }
    elif demographic_model == "split_isolation_model":
        # e.g. (nu1, nu2, T, m)
        # These are all in real units
        Na, nu1, nu2, T, m = opt_params_scaled
        print("====================================")
        print(f'N_ref: {Na}')
        print(f'nu1: {nu1}')
        print(f'nu2: {nu2}')
        print(f'T: {T}')
        print(f'm: {m}')
        # print(f'The difference between N_ref and N0 is: {N_ref - N0}')
        print("====================================")

        opt_params_dict = {
            "Na": Na,
            "N1": nu1,
            "N2": nu2,
            "t_split": T,
            "m": m,
            "ll": ll_value,
        }
    elif demographic_model == "bottleneck_model":
        print(f'opt_params_scaled: {opt_params_scaled}')
        # e.g. (nuB, nuF, T1, T2)
        nuF, T1, T2 = opt_params_scaled
        opt_params_dict = {
            "N_recover": nuF,
            "t_bottleneck_start": T1,  # or however your model is set
            "t_bottleneck_end": T2,  # or however your model is set
            "ll": ll_value
        }
    else:
        opt_params_dict = {}

    return model_sfs, opt_theta, opt_params_dict

def run_inference_moments(
    sfs,
    p0,
    demographic_model,
    lower_bound=[0.001, 0.001, 0.001, 0.001],
    upper_bound=[1, 1, 1, 1],
    use_FIM=False,
    mutation_rate=1.26e-8,
    length=1e7
):
    """
    Perform moments parameter inference on a masked, unnormalized SFS.
    """

    # Compute sample sizes
    sample_sizes_fit = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    # Compute mean and stddev for z-score normalization
    mean = [(l + u) / 2 for (l, u) in zip(lower_bound, upper_bound)]
    stddev = [(u - l) / np.sqrt(12) for (l, u) in zip(lower_bound, upper_bound)]

    print(f'Mean is: {mean}')
    print(f'Stddev is: {stddev}')

    # Perturb the initial guess
    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    p_guess = p0.copy()
    print(f'Initial guess in real-space: {p_guess}')

    # Convert to z-space
    init_z = norm(p_guess, mean, stddev)

    print(f'Initial guess in z-space for moments: {init_z}')

    # Run optimization in a separate process
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_optimize_moments,
        args=(queue, init_z, sfs, demographic_model, sample_sizes_fit,
              lower_bound, upper_bound, mutation_rate, length, mean, stddev)
    )

    process.start()
    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        print("Moments optimization took too long. Terminating...")
        process.terminate()
        process.join()
        return None, None, {"N0": np.nan, "ll": np.nan}

    # Retrieve best-fit parameters from queue
    opt_params_scaled, ll_value = queue.get()
    print(f"Best-fit moments params (scaled): {opt_params_scaled}")
    print(f"Log-likelihood: {ll_value}")

    # Generate the model SFS using diffusion_sfs_moments
    model_sfs = diffusion_sfs_moments(
        opt_params_scaled, sample_sizes_fit, demographic_model, mutation_rate, length
    )

    # Compute best-fit theta
    opt_theta = moments.Inference.optimal_sfs_scaling(model_sfs, sfs)
    print(f"Best-fit theta (moments): {opt_theta}")

    # # Convert theta to N_ref
    # N_ref = opt_theta / (4.0 * mutation_rate * length)
    # print(f"Estimated N_ref (moments): {N_ref}")

    # Compute FIM if needed
    upper_triangular = None

    if demographic_model == "split_migration_model":
        model_func = demographic_models.split_migration_model_moments
    elif demographic_model == "split_isolation_model":
        model_func = demographic_models.split_isolation_model_moments
    elif demographic_model == "bottleneck_model":
        model_func = demographic_models.three_epoch_fixed_moments
    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    if use_FIM:
        H = _get_godambe(
            model_func,
            all_boot=[],
            p0=opt_params_scaled,
            data=sfs,
            eps=1e-6,
            log=False,
            just_hess=True,
        )
        FIM = -1 * H  # Typical sign convention
        upper_tri_indices = np.triu_indices(FIM.shape[0])
        upper_triangular = FIM[upper_tri_indices]

    # Construct parameter dictionary

    if demographic_model == "split_migration_model":
    
        n0, n1, n2, m12, m21, t_split = opt_params_scaled

        opt_params_dict = {
            "N0": n0,
            "N1": n1,
            "N2": n2,
            "t_split": t_split,
            "m12": m12,
            "m21": m21,
            "ll": ll_value
        }
    elif demographic_model == "split_isolation_model":
        n0, n1, n2, t_split, m = opt_params_scaled

        opt_params_dict = {
            "Na": n0,
            "N1": n1,
            "N2": n2,
            "t_split": t_split,
            "m": m,
            "ll": ll_value
        }

    elif demographic_model == "bottleneck_model":
        n_recover, t_bottleneck_start, t_bottleneck_end = opt_params_scaled

        opt_params_dict = {
            "N_recover": n_recover,
            "t_bottleneck_start": t_bottleneck_start,
            "t_bottleneck_end": t_bottleneck_end,
            "ll": ll_value
        }

    if use_FIM:
        opt_params_dict["upper_triangular_FIM"] = upper_triangular

    return model_sfs, opt_theta, opt_params_dict

def run_inference_momentsLD(ld_stats, demographic_model, p_guess, sampled_params, experiment_config):
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
        # demo_func = demographic_models.three_epoch_fixed_MomentsLD
        demo_func = moments.LD.Demographics1D.three_epoch
        demes_func = demographic_models.bottleneck_model
    elif demographic_model == "split_isolation_model":
        demo_func = demographic_models.split_isolation_model_momentsLD
        demes_func = demographic_models.split_isolation_model_simulation

    elif demographic_model == "split_migration_model":
        demo_func = demographic_models.split_migration_model_momentsLD
        demes_func = demographic_models.split_migration_model_simulation

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    print('Demes Graph Computation')
    g = demes_func(sampled_params)

    # Expected LD stats plots
    if demographic_model == "bottleneck_model":
        y = moments.Demes.LD(g, sampled_demes=["N0"], rho=4 * sampled_params["N0"] * r_bins)
    elif demographic_model == "split_migration_model":
        y = moments.Demes.LD(g, sampled_demes=["N1", "N2"], rho=4 * sampled_params["N0"] * r_bins)
    elif demographic_model == "split_isolation_model":
        y = moments.Demes.LD(g, sampled_demes=["N1", "N2"], rho=4 * sampled_params["Na"] * r_bins)
    else:
        raise ValueError(f"Unknown demographic model: {demographic_model}")

    y = moments.LD.LDstats(
        [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    # Determine which LD statistics to plot based on the demographic model
    if demographic_model == "bottleneck_model":
        stats_to_plot = [
            ["DD_0_0"],
            ["Dz_0_0_0"],
            ["pi2_0_0_0_0"],
        ]
        labels = [
            [r"$D_0^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$\pi_{2;0,0,0,0}$"],
        ]

    elif demographic_model in ["split_isolation_model", "split_migration_model"]:
        stats_to_plot = [
            ["DD_0_0"],
            ["DD_0_1"],
            ["DD_1_1"],
            ["Dz_0_0_0"],
            ["Dz_0_1_1"],
            ["Dz_1_1_1"],
            ["pi2_0_0_1_1"],
            ["pi2_0_1_0_1"],
            ["pi2_1_1_1_1"],
        ]
        labels = [
            [r"$D_0^2$"],
            [r"$D_0 D_1$"],
            [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$Dz_{0,1,1}$"],
            [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"],
            [r"$\pi_{2;0,1,0,1}$"],
            [r"$\pi_{2;1,1,1,1}$"],
        ]
    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # Plot LD curves
    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=r_bins,
        stats_to_plot=stats_to_plot,
        labels=labels,
        rows=3,
        plot_vcs=True,
        show=False,
        fig_size=(6, 4),
    )

    p_guess = [0.5, 0.5, 0.075, 2, 50000]
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)
    bottleneck_size_fixed = sampled_params['N_recover']/sampled_params['N0']
    print(f'bottleneck size fixed: {bottleneck_size_fixed}')

    opt_params, ll = moments.LD.Inference.optimize_log_lbfgsb(
        p_guess, 
        [mv["means"], mv["varcovs"]], 
        [demo_func], 
        rs=r_bins,
        fixed_params=[None, bottleneck_size_fixed, None, None, None],
        upper_bound=[1, 3, 0.2, 0.2, 100000],
        verbose=1
    )

    physical_units = moments.LD.Util.rescale_params(
        opt_params, ["nu", "nu", "T", "T", "Ne"]
    )

    print("Simulated parameters:")
    print(f"  N1(deme0)         :  {g.demes[0].epochs[1].start_size:.1f}")
    print(f"  N2(deme1)         :  {g.demes[0].epochs[2].start_size:.1f}")
    print(f"  T1 (gen)  :  {g.demes[0].epochs[1].start_time:.1f}")
    print(f"  T2 (gen)   :  {g.demes[0].epochs[2].start_time:.6f}")
    print(f"  N(ancestral)     :  {g.demes[0].epochs[0].start_size:.1f}")
    
    print("best fit parameters:")
    print(f"  N1(deme0)         :  {physical_units[0]:.1f}")
    print(f"  N2(deme1)         :  {physical_units[1]:.1f}")
    print(f"  T1 (gen)  :  {physical_units[2]:.1f}")
    print(f"  T2 (gen)   :  {physical_units[3]:.6f}")
    print(f"  N(ancestral)     :  {physical_units[4]:.1f}")

    ll_list.append(ll)

    opt_params_dict = {}
    if demographic_model == "bottleneck_model":
        # opt_params[0]: Nb
        # opt_params[1]: N_recover 
        # opt_params[2]: t_bottleneck_end
        # opt_params[3]: N_ref
        opt_params_dict = {
            "N0": opt_params[4],
            "Nb": opt_params[0] * opt_params[4],
            "N_recover": opt_params[1] * opt_params[4],
            "t_bottleneck_end": opt_params[3] * 2 * opt_params[4]
        }

        print(f'best fit parameters:')
        print(f'  N0         :  {opt_params[4]:.1f}')
        print(f'  Nb         :  {opt_params[0] * opt_params[4]:.1f}')
        print(f'  N_recover  :  {opt_params[1] * opt_params[4]:.1f}')
        print(f'  t_bottleneck_end (gen)  :  {opt_params[3] * 2 * opt_params[4]:.1f}')
        

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
            opt_params, ["nu", "nu", "m", "m", "T", "Ne"]  # Use "m" for both migration rates
        )

        print(physical_units)

        opt_params_dict = {
            "N1": physical_units[0],
            "N2": physical_units[1],
            "t_split": physical_units[4],
            "m12": physical_units[2],  # Matches first "m"
            "m21": physical_units[3],  # Matches second "m"
            'N0': physical_units[5]
        }

        print("best fit parameters:")
        print(f"  N(deme1)         :  {physical_units[0]:.1f}")
        print(f"  N(deme2)         :  {physical_units[1]:.1f}")
        print(f"  Div. time (gen)  :  {physical_units[4]:.1f}")
        print(f"  Migration rate 1 :  {physical_units[2]:.6f}")
        print(f"  Migration rate 2 :  {physical_units[3]:.6f}")
        print(f"  N(ancestral)     :  {physical_units[5]:.1f}")

    opt_params_dict_list.append(opt_params_dict)

    return opt_params_dict_list, ll_list, fig 