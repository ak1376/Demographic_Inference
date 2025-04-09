import numpy as np
import dadi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
import msprime
import demes
import time
import pickle
import nlopt
import moments
import os
import json
import tskit
import ray
import io
import contextlib  # <-- for redirecting stdout

##############################
# DEMOGRAPHIC SIMULATION CODE
##############################
def bottleneck_model(param_dict=None):
    N0 = 10000
    nuB = 4000
    nuF, t_bottleneck_start, t_bottleneck_end = (
        param_dict["N_recover"],
        param_dict["t_bottleneck_start"],
        param_dict["t_bottleneck_end"],
    )

    b = demes.Builder()
    b.add_deme(
        "N0",
        epochs=[
            dict(start_size=N0, end_time=t_bottleneck_start),
            dict(start_size=nuB, end_time=t_bottleneck_end),
            dict(start_size=nuF, end_time=0),
        ],
    )
    g = b.resolve()
    return g

def simulate_chromosome(g, length=1e7, mutation_rate=5.7e-9, recombination_rate=3.386e-9):
    samples = {"N0": 50}
    demog = msprime.Demography.from_demes(g)
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demog,
        sequence_length=length,
        recombination_rate=recombination_rate,
        random_seed=295,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate)
    return ts, g

def create_SFS(ts):
    sample_sets = [
        ts.samples(population=pop.id)
        for pop in ts.populations()
        if len(ts.samples(population=pop.id)) > 0
    ]
    sfs = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False
    )
    sfs = moments.Spectrum(sfs)
    return sfs

################################
# DADI OPTIMIZATION INFRASTRUCTURE
################################
def diffusion_sfs_dadi(parameters, sample_sizes, mutation_rate, sequence_length, pts):
    param_dict = {
        "N_recover": parameters[0],
        "t_bottleneck_start": parameters[1],
        "t_bottleneck_end": parameters[2],
    }
    demes_graph = bottleneck_model(param_dict)
    ns = [2 * n for n in sample_sizes.values()]
    model_fs = dadi.Spectrum.from_demes(
        demes_graph,
        sampled_demes=list(sample_sizes.keys()),
        sample_sizes=ns,
        pts=pts
    )
    Nref = parameters[0]
    theta = 4.0 * Nref * mutation_rate * sequence_length
    model_fs *= theta
    return model_fs

def _optimize_dadi(
    p_guess,
    sfs,
    sampled_params,
    sample_sizes_fit,
    mutation_rate,
    sequence_length,
    pts_ext,
    lower_bound,
    upper_bound,
    replicate_id=None,
    results_dir="results",
):
    """
    Runs dadi.Inference.opt, capturing stdout so we can save it to a log file.
    Returns (fitted_params, ll_value, log_text).
    """
    def raw_wrapper(scaled_params, ns, pts):
        return diffusion_sfs_dadi(
            scaled_params,
            sample_sizes_fit,
            mutation_rate,
            sequence_length,
            pts
        )

    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)


    fitted_params, ll_value = dadi.Inference.opt(
        p_guess,
        sfs,
        func_ex,
        pts=pts_ext,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        algorithm=nlopt.LN_BOBYQA,
        maxeval=5000,
        verbose=10,  # set verbose>0 so we actually get some text output
    )

    return fitted_params, ll_value

def run_inference_dadi(
    sfs,
    p0,
    num_samples,
    sampled_params,
    lower_bound,
    upper_bound,
    mutation_rate,
    length,
    replicate_id=None,
    results_dir="results",
):
    """
    Runs a single replicate of the optimization.
    Saves figure, params, and log to 'results_dir'.
    """
    # ensure results_dir exists
    os.makedirs(results_dir, exist_ok=True)

    pop_ids = ["N0"]
    sfs.pop_ids = pop_ids
    sample_sizes_fit = OrderedDict(
        (p, (n - 1)//2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )
    ns = sfs.sample_sizes
    pts_ext = [max(ns) + 60, max(ns) + 70, max(ns) + 80]

    # randomize the guess
    p_guess = moments.Misc.perturb_params(
        p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound
    )

    # Capture all text output from the optimization
    opt_params, ll_value = _optimize_dadi(
        p_guess,
        sfs,
        sampled_params,
        sample_sizes_fit,
        mutation_rate,
        length,
        pts_ext,
        lower_bound,
        upper_bound,
        replicate_id=replicate_id,
        results_dir=results_dir,
    )

    # Evaluate best-fit model
    model_sfs = diffusion_sfs_dadi(
        opt_params,
        sample_sizes_fit,
        mutation_rate,
        length,
        pts_ext
    )
    opt_theta = dadi.Inference.optimal_sfs_scaling(model_sfs, sfs)

    # N0, nuB, nuF, T1, T2 = opt_params
    nuF, T1, T2 = opt_params
    opt_params_dict = {
        "N0": 10000,
        "Nb": 4000,
        "N_recover": nuF,
        "t_bottleneck_start": T1,
        "t_bottleneck_end": T2,
        "ll": ll_value,
        "theta": opt_theta,
    }

    # Save replicate-specific figure & results if replicate_id is given
    if replicate_id is not None:
        # 1) Figure comparing model & data SFS
        fig = plt.figure()
        dadi.Plotting.plot_1d_comp_multinom(model_sfs, sfs)
        fig_filename = os.path.join(results_dir, f"replicate_{replicate_id:03d}_sfs.png")
        plt.savefig(fig_filename)
        plt.close(fig)

        # 2) Save parameters to .pkl
        params_filename = os.path.join(results_dir, f"replicate_{replicate_id:03d}_params.pkl")
        with open(params_filename, "wb") as fh:
            pickle.dump(opt_params_dict, fh)

        # If you wanted to also store the log text in a single file for all replicates,
        # you already do it in _optimize_dadi, but you could do it again here if needed.

    return opt_params_dict

##############################
# RAY PARALLELIZATION
##############################
@ray.remote
def replicate_inference_worker(
    replicate_id,
    sfs_data,
    p0,
    num_samples,
    sampled_params,
    lower_bound,
    upper_bound,
    mutation_rate,
    length,
    results_dir="results"
):
    """
    Single replicate with dadi, in parallel via Ray.
    We pass results_dir so each replicate's logs, figs, etc. get saved in 'results'.
    """
    return run_inference_dadi(
        sfs=sfs_data,
        p0=p0,
        num_samples=num_samples,
        sampled_params=sampled_params,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        mutation_rate=mutation_rate,
        length=length,
        replicate_id=replicate_id,
        results_dir=results_dir,
    )

def main():
    ##############################
    # STEP 1: Load or Simulate Data
    ##############################
    length = 1e7
    mutation_rate = 1.5e-8
    recombination_rate = 1.5e-8

    # True parameters
    param_dict = {
        "N_recover": 8000,
        "t_bottleneck_start": 1800,
        "t_bottleneck_end": 1500,
    }

    ts_filename = "simulated_tree_sequence.trees"
    sfs_filename = "simulated_sfs.pkl"

    # 1) TreeSequence
    if os.path.exists(ts_filename):
        print(f"Found existing tree sequence file: {ts_filename}. Loading...")
        ts = tskit.load(ts_filename)
    else:
        print("No existing tree sequence found. Simulating new tree sequence...")
        g = bottleneck_model(param_dict)
        ts, g = simulate_chromosome(
            g,
            length=length,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate
        )
        ts.dump(ts_filename)
        print(f"Tree sequence saved to {ts_filename}")

    # 2) SFS
    if os.path.exists(sfs_filename):
        print(f"Found existing SFS file: {sfs_filename}. Loading...")
        with open(sfs_filename, "rb") as f:
            sfs = pickle.load(f)
    else:
        print("No existing SFS found. Creating from TreeSequence.")
        sfs = create_SFS(ts)
        with open(sfs_filename, "wb") as f:
            pickle.dump(sfs, f)
        print(f"SFS saved to {sfs_filename}")

    ##############################
    # STEP 2: Ray Initialization
    ##############################
    ray.init(num_cpus=24, ignore_reinit_error=True)

    ##############################
    # STEP 3: Launch Replicates
    ##############################
    n_replicates = 100  # or 100
    print(f"Starting {n_replicates} replicate optimizations with Ray...")

    # Dadi bounds and initial guess
    lower_bound = [2000, 1701, 100]     # [N0, Nb, Nrec, Tstart, Tdur]
    upper_bound = [20000, 3000, 1700]
    p0 = [8000, 1800, 1500]         # Tstart=1800, Tdur=400 => Tend=1400

    num_samples = 50
    sampled_params = {}
    results_dir = "results"

    # Make sure the 'results' folder exists
    os.makedirs(results_dir, exist_ok=True)

    # Ray tasks
    replicate_tasks = []
    for replicate_id in range(n_replicates):
        task = replicate_inference_worker.remote(
            replicate_id,
            sfs,
            p0,
            num_samples,
            sampled_params,
            lower_bound,
            upper_bound,
            mutation_rate,
            length,
            results_dir=results_dir
        )
        replicate_tasks.append(task)

    # Gather results from all replicate tasks
    all_results = ray.get(replicate_tasks)
    print("All replicates finished.")

    ##############################
    # STEP 4: Plot Histograms of All Parameter Estimates
    ##############################
    true_vals = {
        # "N0": param_dict["N0"],
        # "Nb": param_dict["Nb"],
        "N_recover": param_dict["N_recover"],
        "t_bottleneck_start": param_dict["t_bottleneck_start"],
        "t_bottleneck_end": param_dict["t_bottleneck_end"],
    }
    param_names = ["N_recover", "t_bottleneck_start", "t_bottleneck_end"]
    param_estimates = {p: [] for p in param_names}
    for result_dict in all_results:
        for param in param_names:
            param_estimates[param].append(result_dict[param])

    # print("Parameter estimates distribution:")
    # for p in param_names:
    #     print(p, param_estimates[p])

    # Sort all_results by ll_value
    all_results.sort(key=lambda x: x["ll"], reverse=True)
    # Print the top 3 results 
    print("Top 3 results:")
    for i, result in enumerate(all_results[:3]):
        print(f"Result {i+1}: {result}")

    fig, axes = plt.subplots(nrows=len(param_names), ncols=1, figsize=(6, 3*len(param_names)))
    for i, param in enumerate(param_names):
        ax = axes[i]
        ax.hist(param_estimates[param], bins=20, alpha=0.7)
        ax.axvline(true_vals[param], color="red", linestyle="--", label="True value")
        ax.set_xlabel(param)
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    hist_filename = os.path.join(results_dir, "parameter_histograms_ray.png")
    plt.savefig(hist_filename)
    plt.close(fig)

    print(f"Saved parameter histograms to {hist_filename}")
    print("Done!")

if __name__ == "__main__":
    main()
