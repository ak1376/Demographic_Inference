import dadi
import moments
import os
import numpy as np
import time
from moments.Godambe import _get_godambe
import src.demographic_models as demographic_models
from src.optimize import opt
import time
import multiprocessing
from collections import OrderedDict
import moments.LD as LD

TIMEOUT_SECONDS = 300 * 60  # 20 minutes = 1200 seconds
REF_SIZE = 10000  # Reference population size for scaling

def diffusion_sfs_moments(parameters: list[float],
    sample_sizes: OrderedDict,
    demographic_model: str,
    mutation_rate: float,
    sequence_length: float) -> moments.Spectrum:
    """
    Get the expected SFS under the diffusion approximation (moments).
    """
    
    # 1) Convert our parameter list into a dictionary
    if demographic_model == "island_model":
        demo_model = demographic_models.island_model_simulation
        param_dict = {
            "N1": parameters[0],
            "N2": parameters[1],
            "m12": parameters[2],
            "m21": parameters[3],
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
    Nref = REF_SIZE  # Reference population size
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

    if demographic_model == "island_model":
        # For example, [N0, N1, N2, m12, m21]
        param_dict = {
            "N1": parameters[0],
            "N2": parameters[1],
            "m12": parameters[2],
            "m21": parameters[3],
        }

        demo_func = demographic_models.island_model_simulation

    else:
        raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # 2) Build the demes graph
    demes_graph = demo_func(param_dict)  # e.g. returns a demes.Graph

    ns = [2 * n for n in sample_sizes.values()]  # e.g., [30, 16] if input is [15, 8]
    # print(f'SAMPLE SIZES: {ns}')

    model_fs = dadi.Spectrum.from_demes(
        demes_graph,
        sampled_demes = list(sample_sizes.keys()),
        sample_sizes = ns,
        pts = pts
    )

    # 5) Scale by theta if you want an absolute SFS. Typically, we do:
    Nref = REF_SIZE  # Reference population size
    theta = 4.0 * Nref * mutation_rate * sequence_length

    model_fs *= theta

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

def _optimize_dadi(
    p_guess,             # initial guess *in real space*
    sfs,
    demographic_model,   # model name string
    sample_sizes_fit,    # OrderedDict of diploid sample sizes
    mutation_rate,
    sequence_length,
    pts_ext,
    lower_bound,         # real-space bounds
    upper_bound
):
    """
    Runs dadi optimisation in REAL parameter space (no z-scoring).
    Returns best-fit params + log-likelihood through `queue`.
    """

    # ----- model wrapper ----------------------------------------------------
    def raw_wrapper(real_params, ns, pts):
        """
        real_params: [N1, N2, m12, m21] or whatever your model expects,
        already in demographic units.  Returns a dadi.Spectrum.
        """
        return diffusion_sfs_dadi(
            real_params,
            sample_sizes_fit,
            demographic_model,
            mutation_rate,
            sequence_length,
            pts
        )

    # Extrapolated version for dadi
    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    print(f"Lower bounds: {lower_bound}")
    print(f"Upper bounds: {upper_bound}")
    print(f"Initial guess (real): {p_guess}")

    # ----- optimisation -----------------------------------------------------
    xopt = dadi.Inference.optimize_log_powell(
        p_guess,
        sfs,
        func_ex,
        pts=pts_ext,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        multinom=False,
        verbose=0,
        flush_delay=0.0,
        full_output=True,
        maxiter=1000
    )

    fitted_params, ll_value = xopt[0], xopt[1]

    print(f"Best-fit dadi params (real): {fitted_params}")
    print(f"Log-likelihood: {ll_value}")

    # Send results back to the parent process
    return fitted_params, ll_value
    # queue.put((fitted_params, ll_value))


def _optimize_moments(
    init_z,               # NOW TREATED AS *REAL* INITIAL GUESS
    sfs,
    demographic_model,    # e.g. "island_model"
    sample_sizes_fit,     # OrderedDict of diploid counts
    lower_bound,          # list/array, real space
    upper_bound,          # list/array, real space
    mutation_rate,
    sequence_length
):
    """
    moments optimisation in REAL space.
    Sends (best_params, logL) back via `queue`.
    """

    init_params = list(init_z)   # rename for clarity

    # ───────────────── model wrapper ───────────────────────────────
    def model_func(params, ns):
        # Reject out-of-bounds proposals with a “terrible” SFS
        for p, lb, ub in zip(params, lower_bound, upper_bound):
            if p < lb or p > ub:
                bad = sfs.copy()
                bad.data[:] = 1e-10
                return bad
        try:
            return diffusion_sfs_moments(
                params,
                sample_sizes_fit,
                demographic_model,
                mutation_rate,
                sequence_length
            )
        except Exception as e:
            print("Model evaluation error:", e)
            bad = sfs.copy()
            bad.data[:] = 1e-10
            return bad

    # ───────────────── optimisation ────────────────────────────────
    print(" moments | initial params (real):", init_params)
    print(" moments | bounds:", list(zip(lower_bound, upper_bound)))

    try:
        xopt = moments.Inference.optimize_log_powell(
            init_params,
            sfs,
            model_func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            multinom=False,
            verbose=0,
            flush_delay=0.0,
            full_output=True,
            maxiter=200,   # adjust if needed
        )
        fitted_params, ll_value = xopt[0], xopt[1]
        print(" moments | best-fit params:", fitted_params)
        print(" moments | log-likelihood:", ll_value)
    except Exception as e:
        print(" moments | optimisation failed:", e)
        fitted_params, ll_value = init_params, -np.inf

    return fitted_params, ll_value

def run_inference_dadi(
    sfs,
    p0,
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
    pop_ids = ['N1', 'N2']

    sfs.pop_ids = pop_ids

    sample_sizes_fit = OrderedDict(
        (p, (n - 1)//2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )
    ns = sfs.sample_sizes

    # 3) Setup grids for extrapolation
    pts_ext = [max(ns) + 20, max(ns) + 30, max(ns) + 40]

    # 4) Perturb the initial guess to avoid local minima
    # p_guess = moments.Misc.perturb_params(
    #     p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    # )

    p_guess = p0.copy()

    fitted_params, ll_value = _optimize_dadi(
        p_guess,          # initial guess in real space
        sfs,             # empirical SFS
        demographic_model,  # model name as a string (e.g., "split_migration_model")
        sample_sizes_fit,  # OrderedDict of sample sizes
        mutation_rate,   # mutation rate
        length,          # sequence length
        pts_ext,         # extrapolation grid points
        lower_bound,     # real-space lower bound
        upper_bound,     # real-space upper bound
    )

    # 7) Retrieve best-fit from the queue
    print(f"Best-fit scaled params: {fitted_params}")
    print(f"Log-likelihood: {ll_value}")

    # 8) Generate the model SFS using diffusion_sfs_dadi #TODO: This may fail for the bottleneck. Need to check that this works. 
    model_sfs = diffusion_sfs_dadi(
        fitted_params,  # Best-fit parameters from optimization
        sample_sizes_fit,   # Sample sizes as an OrderedDict
        demographic_model,  # Model name (e.g., "split_migration_model")
        mutation_rate,      # Mutation rate
        length,             # Sequence length
        pts_ext             # Extrapolation grid points
    )

    # 9) Compute best-fit theta
    opt_theta = dadi.Inference.optimal_sfs_scaling(model_sfs, sfs)

    if demographic_model == "island_model":
        # e.g. (nu1, nu2, m12, m21)
        nu1, nu2, m12, m21 = fitted_params
        opt_params_dict = {
            "N1": nu1,
            "N2": nu2,
            "m12": m12,
            "m21": m21,
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

    p_guess = p0.copy()
    print(f'Initial guess in real-space: {p_guess}')

    fitted_params, ll_value = _optimize_moments(
        p_guess,          # initial guess in real space
        sfs,             # empirical SFS
        demographic_model,  # model name as a string (e.g., "split_migration_model")
        sample_sizes_fit,  # OrderedDict of sample sizes
        lower_bound,     # real-space lower bound
        upper_bound,     # real-space upper bound
        mutation_rate,   # mutation rate
        length           # sequence length
    )

    # Retrieve best-fit parameters from queue
    print(f"Best-fit moments params (scaled): {fitted_params}")
    print(f"Log-likelihood: {ll_value}")

    # Generate the model SFS using diffusion_sfs_moments
    model_sfs = diffusion_sfs_moments(
        fitted_params, sample_sizes_fit, demographic_model, mutation_rate, length
    )

    # Compute best-fit theta
    opt_theta = moments.Inference.optimal_sfs_scaling(model_sfs, sfs)
    print(f"Best-fit theta (moments): {opt_theta}")

    # # Convert theta to N_ref
    # N_ref = opt_theta / (4.0 * mutation_rate * length)
    # print(f"Estimated N_ref (moments): {N_ref}")

    # Compute FIM if needed
    # upper_triangular = None

    # if demographic_model == "split_migration_model":
    #     model_func = demographic_models.split_migration_model_moments
    # elif demographic_model == "split_isolation_model":
    #     model_func = demographic_models.split_isolation_model_moments
    # elif demographic_model == "bottleneck_model":
    #     model_func = demographic_models.three_epoch_fixed_moments
    # elif demographic_model == "island_model":
    #     model_func = demographic_models.island_model_moments
    # else:
    #     raise ValueError(f"Unsupported demographic model: {demographic_model}")

    # if use_FIM:

    #     H = _get_godambe(
    #         model_func,
    #         all_boot=[],
    #         p0=fitted_params
    #         data=sfs,
    #         eps=1e-6,
    #         log=False,
    #         just_hess=True
    #     )
    #     FIM = -1 * H  # Typical sign convention
    #     upper_tri_indices = np.triu_indices(FIM.shape[0])
    #     upper_triangular = FIM[upper_tri_indices]

    # Construct parameter dictionary

    if demographic_model == "island_model":
        n1, n2, m12, m21 = fitted_params

        opt_params_dict = {
            "N1": n1,
            "N2": n2,
            "m12": m12,
            "m21": m21,
            "ll": ll_value
        }

    # if use_FIM:
    #     opt_params_dict["upper_triangular_FIM"] = upper_triangular

    return model_sfs, opt_theta, opt_params_dict

# def run_inference_momentsLD(ld_stats, demographic_model, p_guess, sampled_params, experiment_config):
#     """
#     This should do the parameter inference for momentsLD.
#     """
#     r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
#     ll_list = []
#     opt_params_dict_list = []

#     print("====================================================")
#     print(ld_stats.keys())

#     mv = moments.LD.Parsing.bootstrap_data(ld_stats)  # type: ignore
#     print('MV CREATION COMPLETED!')

#     demo_func = LD.Demographics2D.island_model
#     demes_func = demographic_models.island_model_simulation

#     print('Demes Graph Computation')
#     g = demes_func(sampled_params)

#     # Expected LD stats plots
#     y = moments.Demes.LD(g, sampled_demes=["N1", "N2"], rho=4 * 10000 * r_bins)

#     y = moments.LD.LDstats(
#         [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
#         num_pops=y.num_pops,
#         pop_ids=y.pop_ids,
#     )
#     y = moments.LD.Inference.sigmaD2(y)

#     # Determine which LD statistics to plot based on the demographic model
#     if demographic_model == "bottleneck_model":
#         stats_to_plot = [
#             ["DD_0_0"],
#             ["Dz_0_0_0"],
#             ["pi2_0_0_0_0"],
#         ]
#         labels = [
#             [r"$D_0^2$"],
#             [r"$Dz_{0,0,0}$"],
#             [r"$\pi_{2;0,0,0,0}$"],
#         ]

#     elif demographic_model in ["split_isolation_model", "split_migration_model", "island_model"]:
#         stats_to_plot = [
#             ["DD_0_0"],
#             ["DD_0_1"],
#             ["DD_1_1"],
#             ["Dz_0_0_0"],
#             ["Dz_0_1_1"],
#             ["Dz_1_1_1"],
#             ["pi2_0_0_1_1"],
#             ["pi2_0_1_0_1"],
#             ["pi2_1_1_1_1"],
#         ]
#         labels = [
#             [r"$D_0^2$"],
#             [r"$D_0 D_1$"],
#             [r"$D_1^2$"],
#             [r"$Dz_{0,0,0}$"],
#             [r"$Dz_{0,1,1}$"],
#             [r"$Dz_{1,1,1}$"],
#             [r"$\pi_{2;0,0,1,1}$"],
#             [r"$\pi_{2;0,1,0,1}$"],
#             [r"$\pi_{2;1,1,1,1}$"],
#         ]
#     else:
#         raise ValueError(f"Unsupported demographic model: {demographic_model}")

#     # Plot LD curves
#     fig = moments.LD.Plotting.plot_ld_curves_comp(
#         y,
#         mv["means"][:-1],
#         mv["varcovs"][:-1],
#         rs=r_bins,
#         stats_to_plot=stats_to_plot,
#         labels=labels,
#         rows=3,
#         plot_vcs=True,
#         show=False,
#         fig_size=(6, 4),
#     )

#     print(f'p_guess is: {p_guess}')

#     p_guess_scaled = real_to_dadi_params(p_guess, demographic_model)
#     # p_guess_scaled = real_to_dadi_params(sampled_params, demographic_model) # Let's pass in the ground truth and see if we can recover the true generative params. Sanity check. 

#     lower_bound_scaled = real_to_dadi_params(experiment_config["lower_bound_optimization"], demographic_model)
#     upper_bound_scaled = real_to_dadi_params(experiment_config["upper_bound_optimization"], demographic_model)

#     # lower_bound_scaled = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
#     # upper_bound_scaled = [10, 10, 10, 10, 10]

#     print(f'the scaled parameters are: {p_guess_scaled}')
#     # print(f'the lower bound scaled parameters are: {lower_bound_scaled}')
#     # print(f'the upper bound scaled parameters are: {upper_bound_scaled}')

#     # p_guess_scaled is a dictionary. We need to be very careful when converting it to a list for optimization
#     p_guess_scaled = [p_guess_scaled['nu1'], p_guess_scaled['nu2'], p_guess_scaled['m12'], p_guess_scaled['m21']]
#     lower_bound_scaled = [lower_bound_scaled['nu1'], lower_bound_scaled['nu2'], lower_bound_scaled['m12'], lower_bound_scaled['m21']]
#     upper_bound_scaled = [upper_bound_scaled['nu1'], upper_bound_scaled['nu2'], upper_bound_scaled['m12'], upper_bound_scaled['m21']]

#     # p_guess_scaled = moments.LD.Util.perturb_params(p_guess_scaled, fold=0.1)
#     # lower_bound_scaled = moments.LD.Util.perturb_params(lower_bound_scaled, fold=0.1) 
#     # upper_bound_scaled = moments.LD.Util.perturb_params(upper_bound_scaled, fold=0.1)

#     # Append the real ancestral size to the end of p_guess_scaled
#     p_guess_scaled = np.append(p_guess_scaled, 10000)
#     lower_bound_scaled = np.array([0,0,0,0, 100])  # For island model, we set lower bounds to zero
#     upper_bound_scaled = np.array([1, 1, 100, 100, 30000])  # For island model, we set upper bounds to one

#     # lower_bound_scaled = [5000/8000, 5000/8000, 0.001/(2*8000), 0.001/(2*8000), 1500/(2*8000)]
#     # upper_bound_scaled = [8000/10000, 8000/10000, 0.005/(2*10000), 0.005/(2*10000), 20000/(2*10000)]

#     print(f'Lower bound in scaled space: {lower_bound_scaled}')
#     print(f'Initial guess in scaled space: {p_guess_scaled}')
#     print(f'Upper bound in scaled space: {upper_bound_scaled}')


#     opt_params, ll = moments.LD.Inference.optimize_log_lbfgsb(
#         p_guess_scaled, [mv["means"], mv["varcovs"]],
#         [moments.LD.Demographics2D.island_model],
#         rs=r_bins, verbose=1, lower_bound=lower_bound_scaled,
#         upper_bound=upper_bound_scaled)
    
#     physical_units = moments.LD.Util.rescale_params(opt_params, ["nu","nu","m","m","Ne"])

#     ll_list.append(ll)

#     opt_params_dict = {
#         "N1": physical_units[0],
#         "N2": physical_units[1],
#         "m12": physical_units[2],
#         "m21": physical_units[3],
#         "N0": physical_units[4],  # This is the ancestral population size
#     }

#     opt_params_dict_list.append(opt_params_dict)

#     return opt_params_dict_list, ll_list, fig 