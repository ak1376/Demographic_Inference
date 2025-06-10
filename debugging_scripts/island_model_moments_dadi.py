#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
island_model_inference.py
--------------------------
Simulates a single replicate under the island model and performs inference
using dadi only. Uses the exact demographic model from island_ld_debug.py
without changing the optimization logic.
"""

from pathlib import Path
import numpy as np
import msprime
import demes
import dadi
import pickle
import argparse
from collections import OrderedDict
import nlopt
import moments
import time

def norm(p, mean, stddev):
    return [(x - m) / s for x, m, s in zip(p, mean, stddev)]


def unnorm(z, mean, stddev):
    return [z_i * s + m for z_i, m, s in zip(z, mean, stddev)]

def demographic_model(params):
    N1, N2, m12, m21 = params
    b = demes.Builder()
    b.add_deme("ancestral", epochs=[{"start_size": 10_000, "end_time": 100}])
    b.add_deme("pop0", ancestors=["ancestral"], epochs=[{"start_size": N1}])
    b.add_deme("pop1", ancestors=["ancestral"], epochs=[{"start_size": N2}])
    b.add_migration(source="pop0", dest="pop1", rate=m12)
    b.add_migration(source="pop1", dest="pop0", rate=m21)
    return b.resolve()


def diffusion_sfs_moments(
    parameters: list[float],
    sample_sizes: OrderedDict,
    mutation_rate: float,
    sequence_length: float) -> moments.Spectrum:
    """
    Get the expected SFS under the diffusion approximation (moments).
    """
    # 1) Convert our parameter list into a dictionary
    param_dict = [parameters[0], parameters[1], parameters[2], parameters[3]]

    # 2) Build the demes graph
    demes_graph = demographic_model(param_dict)

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
    mutation_rate: float,
    sequence_length: float,
    pts: list[int],
) -> dadi.Spectrum:
    param_dict = [parameters[0], parameters[1], parameters[2], parameters[3]]
    demo_func = demographic_model
    demes_graph = demo_func(param_dict)
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

def _optimize_moments(
    init_z,  # Initial guess in z-space
    sfs,
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
        return diffusion_sfs_moments(scaled_params, sample_sizes_fit, mutation_rate, sequence_length)

    # 3) Run the optimizer in z-space
    xopt = moments.Inference.optimize_powell(
        init_z,
        sfs,
        lambda z, n: z_wrapper(z, n),
        lower_bound=lb_z,
        upper_bound=ub_z,
        multinom=False,
        verbose=1,
        flush_delay=0.0,
        full_output=True
    )

    # 4) Convert best-fit from z-space to real (scaled) space
    fitted_params = unnorm(xopt[0], mean, stddev)
    ll_value = xopt[1]

    print(f"Best-fit moments params (real-space): {fitted_params}")
    
    # 5) Send results back to parent process
    return fitted_params, ll_value

def _optimize_dadi(
    p_guess,
    sfs,
    sample_sizes_fit,
    mutation_rate,
    sequence_length,
    pts_ext,
    lower_bound,
    upper_bound,
    mean,
    stddev
):
    def raw_wrapper(scaled_params, ns, pts):
        return diffusion_sfs_dadi(
            scaled_params,
            sample_sizes_fit,
            mutation_rate,
            sequence_length,
            pts
        )
    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)
    print(f'Lower bound: {lower_bound}')
    print(f'Upper bound: {upper_bound}')
    xopt = dadi.Inference.optimize_log_powell(
        p_guess,
        sfs,
        func_ex,
        pts=pts_ext,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        multinom=False,
        verbose=1,
        flush_delay=0.0,
        full_output=True,
        maxiter=1000
    )
    fitted_params = xopt[0]
    ll_value = xopt[1]
    print(f"The optimized parameters in real space are : {fitted_params}")
    return fitted_params, ll_value


def run_moments_inference(sfs, mutation_rate, seq_length):
    lower_bound = [100, 100, 1e-6, 1e-6]
    upper_bound = [20000, 20000, 1e-1, 1e-1]
    p_guess = [2000, 5000, 1E-2, 1E-4]
    sample_sizes_fit = OrderedDict((p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape))
    ns = sfs.sample_sizes
    mean = [(l + u) / 2 for (l, u) in zip(lower_bound, upper_bound)]
    stddev = [(u - l) / np.sqrt(12) for (l, u) in zip(lower_bound, upper_bound)]

    # Convert initial guess to z-space
    p_guess_scaled = norm(p_guess, mean, stddev)

    opt_params_scaled, ll_value = _optimize_moments(
        p_guess_scaled,
        sfs,
        sample_sizes_fit,
        lower_bound,
        upper_bound,
        mutation_rate,
        seq_length,
        mean,
        stddev
    )
    model_sfs = diffusion_sfs_moments(
        opt_params_scaled,
        sample_sizes_fit,
        mutation_rate,
        seq_length
    )
    opt_theta = moments.Inference.optimal_sfs_scaling(model_sfs, sfs)
    return opt_params_scaled

def run_dadi_inference(sfs, mutation_rate, seq_length):
    lower_bound = [100, 100, 1e-6, 1e-6]
    upper_bound = [20000, 20000, 1e-1, 1e-1]
    p_guess = [2000, 5000, 1E-2, 1E-4]
    sample_sizes_fit = OrderedDict((p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape))
    ns = sfs.sample_sizes
    pts_ext = [max(ns) + 40, max(ns) + 50, max(ns) + 60]
    mean = [(l + u) / 2 for (l, u) in zip(lower_bound, upper_bound)]
    stddev = [(u - l) / np.sqrt(12) for (l, u) in zip(lower_bound, upper_bound)]
    opt_params_scaled, ll_value = _optimize_dadi(
        p_guess,
        sfs,
        sample_sizes_fit,
        mutation_rate,
        seq_length,
        pts_ext,
        lower_bound,
        upper_bound,
        mean,
        stddev
    )
    model_sfs = diffusion_sfs_dadi(
        opt_params_scaled,
        sample_sizes_fit,
        mutation_rate,
        seq_length,
        pts_ext
    )
    opt_theta = dadi.Inference.optimal_sfs_scaling(model_sfs, sfs)
    return opt_params_scaled

def simulate_and_infer(seq_length, mutation_rate, recombination_rate, n_samples, output_path):
    TRUE_PARAMS = [10000, 500, 1E-6, 1E-5]
    OUTPUT_DIR = Path(output_path)
    OUTPUT_DIR.mkdir(exist_ok=True)
    GRAPH = demographic_model(TRUE_PARAMS)
    print("Simulating data...")
    demog = msprime.Demography.from_demes(GRAPH)
    ts = msprime.sim_ancestry(
        samples={"pop0": n_samples, "pop1": n_samples},
        demography=demog,
        sequence_length=seq_length,
        recombination_rate=recombination_rate,
        random_seed=42
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=43)
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
    sfs.pop_ids = ["pop0", "pop1"]
    with open(OUTPUT_DIR / "simulated_sfs.pkl", "wb") as f:
        pickle.dump(sfs, f)
    print("Running dadi inference...")
    opt_params_scaled_dadi = run_dadi_inference(sfs, mutation_rate, seq_length)
    N1_dadi, N2_dadi, m12_dadi, m21_dadi = opt_params_scaled_dadi

    opt_params_scaled_moments = run_moments_inference(sfs, mutation_rate, seq_length)
    N1_moments, N2_moments, m12_moments, m21_moments = opt_params_scaled_moments

    print(f"Estimated dadi parameters: N1={N1_dadi}, N2={N2_dadi}, m12={m12_dadi}, m21={m21_dadi}")
    print(f"Estimated moments parameters: N1={N1_moments}, N2={N2_moments}, m12={m12_moments}, m21={m21_moments}")
    print(f"True parameters: {TRUE_PARAMS}")

def main():
    parser = argparse.ArgumentParser(description="Run island model dadi inference.")
    parser.add_argument("--seq_length", type=float, default=1_000_000, help="Sequence length")
    parser.add_argument("--mu", type=float, default=1.5e-8, help="Mutation rate")
    parser.add_argument("--recombination_rate", type=float, default=1.5e-8, help="Recombination rate")
    parser.add_argument("--n_samples", type=int, default=30, help="Samples per population")
    parser.add_argument("--output_path", type=str, default="./single_inference", help="Output directory")
    args = parser.parse_args()
    simulate_and_infer(
        seq_length=args.seq_length,
        mutation_rate=args.mu,
        recombination_rate=args.recombination_rate,
        n_samples=args.n_samples,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
