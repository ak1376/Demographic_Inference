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
import matplotlib.pyplot as plt

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
    init_params,  # Initial guess in real space (changed from z-space)
    sfs,
    sample_sizes_fit,
    lower_bound,
    upper_bound,
    mutation_rate,
    sequence_length,
    mean=None,  # Not used anymore
    stddev=None  # Not used anymore
):
    """
    Runs moments optimization in real space (no z-scoring).
    Uses the diffusion SFS from moments.
    """
    
    # Define wrapper function for moments optimization
    def model_func(params, ns):
        # Ensure parameters stay within reasonable bounds during optimization
        for i, (p, lb, ub) in enumerate(zip(params, lower_bound, upper_bound)):
            if p < lb or p > ub:
                # Return a very bad likelihood for out-of-bounds parameters
                bad_sfs = sfs.copy()
                bad_sfs.data[:] = 1e-10
                return bad_sfs
        
        try:
            return diffusion_sfs_moments(params, sample_sizes_fit, mutation_rate, sequence_length)
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            # Return a bad SFS on error
            bad_sfs = sfs.copy()
            bad_sfs.data[:] = 1e-10
            return bad_sfs

    # Run the optimizer in real space
    print(f"Starting optimization with initial params: {init_params}")
    print(f"Lower bounds: {lower_bound}")
    print(f"Upper bounds: {upper_bound}")
    
    try:
        xopt = moments.Inference.optimize_log_powell(
            init_params,
            sfs,
            model_func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            multinom=False,
            verbose=1,
            flush_delay=0.0,
            full_output=True,
            maxiter=100
        )
        
        fitted_params = xopt[0]
        ll_value = xopt[1]
        
        print(f"Optimization completed successfully")
        print(f"Best-fit moments params: {fitted_params}")
        print(f"Log-likelihood: {ll_value}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return initial guess on failure
        fitted_params = init_params
        ll_value = -np.inf
    
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
    lower_bound = [100, 100, 1e-8, 1e-8]
    upper_bound = [20000, 20000, 1e-2, 1e-2]
    p_guess = [2000, 5000, 0.5E-2, 1E-4]
    
    # Fix sample sizes: for a (2n+1, 2n+1) SFS, we have 2n haploid samples
    # So we need to extract n (diploid count) from the SFS shape
    sample_sizes_fit = OrderedDict()
    for pop_id, dim in zip(sfs.pop_ids, sfs.shape):
        n_diploid = (dim - 1) // 2  # Convert from SFS dimension to diploid count
        sample_sizes_fit[pop_id] = n_diploid
    
    print(f"SFS shape: {sfs.shape}")
    print(f"Sample sizes for fitting: {sample_sizes_fit}")
    
    # Try multiple initial guesses
    initial_guesses = [
        p_guess,
        [5000, 1000, 1E-3, 5E-3],
        [8000, 800, 5E-4, 2E-3],
        [1500, 300, 2E-3, 8E-3]
    ]
    
    best_ll = -np.inf
    best_params = None
    
    for i, guess in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {guess}")
        
        # Check if guess is within bounds
        if all(l <= g <= u for g, l, u in zip(guess, lower_bound, upper_bound)):
            try:
                opt_params_scaled, ll_value = _optimize_moments(
                    guess,  # Pass real-space parameters directly
                    sfs,
                    sample_sizes_fit,
                    lower_bound,
                    upper_bound,
                    mutation_rate,
                    seq_length
                )
                
                print(f"Log-likelihood: {ll_value}")
                
                if ll_value > best_ll:
                    best_ll = ll_value
                    best_params = opt_params_scaled
                    
            except Exception as e:
                print(f"Optimization failed with error: {e}")
                continue
        else:
            print("Initial guess outside bounds, skipping...")
    
    if best_params is None:
        print("All optimizations failed!")
        return p_guess
    
    print(f"\nBest parameters found: {best_params}")
    print(f"Best log-likelihood: {best_ll}")
    
    return best_params

def run_dadi_inference(sfs, mutation_rate, seq_length):
    lower_bound = [100, 100, 1e-6, 1e-6]
    upper_bound = [20000, 20000, 1e-1, 1e-1]
    p_guess = [2000, 5000, 1E-2, 1E-4]
    sample_sizes_fit = OrderedDict((p, (n) // 2) for p, n in zip(sfs.pop_ids, sfs.shape))
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

def compare_sfs_simulations(seq_length, mutation_rate, recombination_rate, n_samples, output_path):
    """Compare SFS simulated by dadi vs msprime using the same parameters."""
    TRUE_PARAMS = [10000, 500, 1E-6, 1E-5]
    OUTPUT_DIR = Path(output_path)
    OUTPUT_DIR.mkdir(exist_ok=True)
    GRAPH = demographic_model(TRUE_PARAMS)
    
    # Simulate with msprime
    print("Simulating SFS using msprime...")
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
    sfs_msprime = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False
    )
    sfs_msprime = moments.Spectrum(sfs_msprime)
    sfs_msprime.pop_ids = ["pop0", "pop1"]
    
    # Simulate with dadi
    print("Simulating SFS using dadi...")
    sample_sizes_dict = OrderedDict([("pop0", n_samples), ("pop1", n_samples)])
    ns = [2 * n for n in sample_sizes_dict.values()]
    pts_sim = [max(ns) + 20, max(ns) + 30, max(ns) + 40]
    
    # Get expected SFS from dadi
    sfs_dadi_expected = diffusion_sfs_dadi(
        TRUE_PARAMS,
        sample_sizes_dict,
        mutation_rate,
        seq_length,
        pts_sim
    )
    
    # Sample from the expected SFS
    sfs_dadi_sampled = sfs_dadi_expected.sample()
    sfs_dadi_sampled = moments.Spectrum(sfs_dadi_sampled.data, pop_ids=["pop0", "pop1"])
    
    # Simulate with moments
    print("Simulating SFS using moments...")
    sfs_moments_expected = diffusion_sfs_moments(
        TRUE_PARAMS,
        sample_sizes_dict,
        mutation_rate,
        seq_length
    )
    
    # Sample from the moments expected SFS
    sfs_moments_sampled = sfs_moments_expected.sample()
    
    # Compare the SFS
    print("\n=== SFS Comparison ===")
    print(f"Total SNPs (msprime): {sfs_msprime.S():.0f}")
    print(f"Total SNPs (dadi expected): {sfs_dadi_expected.S():.2f}")
    print(f"Total SNPs (dadi sampled): {sfs_dadi_sampled.S():.0f}")
    print(f"Total SNPs (moments expected): {sfs_moments_expected.S():.2f}")
    print(f"Total SNPs (moments sampled): {sfs_moments_sampled.S():.0f}")
    
    # Save all SFS for further analysis
    with open(OUTPUT_DIR / "sfs_comparison.pkl", "wb") as f:
        pickle.dump({
            "msprime": sfs_msprime,
            "dadi_expected": sfs_dadi_expected,
            "dadi_sampled": sfs_dadi_sampled,
            "moments_expected": sfs_moments_expected,
            "moments_sampled": sfs_moments_sampled,
            "true_params": TRUE_PARAMS
        }, f)
    
    # Calculate some basic statistics
    print(f"\nSFS shapes:")
    print(f"msprime: {sfs_msprime.shape}")
    print(f"dadi: {sfs_dadi_sampled.shape}")
    print(f"moments: {sfs_moments_sampled.shape}")
    
    # Compare marginal spectra
    marg_msprime_0 = sfs_msprime.marginalize([1])
    marg_msprime_1 = sfs_msprime.marginalize([0])
    marg_dadi_0 = sfs_dadi_sampled.marginalize([1])
    marg_dadi_1 = sfs_dadi_sampled.marginalize([0])
    marg_moments_0 = sfs_moments_sampled.marginalize([1])
    marg_moments_1 = sfs_moments_sampled.marginalize([0])
    
    print(f"\nMarginal spectrum comparison (pop0):")
    print(f"Total SNPs - msprime: {marg_msprime_0.S():.0f}, dadi: {marg_dadi_0.S():.0f}, moments: {marg_moments_0.S():.0f}")
    
    print(f"\nMarginal spectrum comparison (pop1):")
    print(f"Total SNPs - msprime: {marg_msprime_1.S():.0f}, dadi: {marg_dadi_1.S():.0f}, moments: {marg_moments_1.S():.0f}")
    
    # Create comparison plots
    plot_sfs_comparison(sfs_msprime, sfs_dadi_sampled, sfs_moments_sampled, OUTPUT_DIR)
    
    return sfs_msprime, sfs_dadi_sampled, sfs_moments_sampled


def plot_sfs_comparison(sfs_msprime, sfs_dadi, sfs_moments, output_dir):
    """Plot comparison of msprime vs dadi vs moments SFS."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: 2D heatmap of msprime SFS
    ax = axes[0, 0]
    sfs_msprime_masked = sfs_msprime.copy()
    sfs_msprime_masked.mask_corners()
    im1 = ax.imshow(np.log10(sfs_msprime_masked.data + 1), cmap='viridis', aspect='auto')
    ax.set_title('msprime SFS (log10 scale)')
    ax.set_xlabel('Pop1 frequency')
    ax.set_ylabel('Pop0 frequency')
    plt.colorbar(im1, ax=ax, label='log10(count + 1)')
    
    # Plot 2: 2D heatmap of dadi SFS
    ax = axes[0, 1]
    sfs_dadi_masked = sfs_dadi.copy()
    sfs_dadi_masked.mask_corners()
    im2 = ax.imshow(np.log10(sfs_dadi_masked.data + 1), cmap='viridis', aspect='auto')
    ax.set_title('dadi SFS (log10 scale)')
    ax.set_xlabel('Pop1 frequency')
    ax.set_ylabel('Pop0 frequency')
    plt.colorbar(im2, ax=ax, label='log10(count + 1)')
    
    # Plot 3: 2D heatmap of moments SFS
    ax = axes[0, 2]
    sfs_moments_masked = sfs_moments.copy()
    sfs_moments_masked.mask_corners()
    im3 = ax.imshow(np.log10(sfs_moments_masked.data + 1), cmap='viridis', aspect='auto')
    ax.set_title('moments SFS (log10 scale)')
    ax.set_xlabel('Pop1 frequency')
    ax.set_ylabel('Pop0 frequency')
    plt.colorbar(im3, ax=ax, label='log10(count + 1)')
    
    # Plot 4: Marginal spectra comparison for pop0
    ax = axes[1, 0]
    marg_msprime_0 = sfs_msprime.marginalize([1])
    marg_dadi_0 = sfs_dadi.marginalize([1])
    marg_moments_0 = sfs_moments.marginalize([1])
    
    # Create frequency bins (excluding fixed bins)
    n_bins_0 = len(marg_msprime_0) - 2
    freq_bins_0 = np.arange(1, n_bins_0 + 1)
    
    ax.plot(freq_bins_0, marg_msprime_0[1:-1], 'o-', label='msprime', markersize=6, alpha=0.7)
    ax.plot(freq_bins_0, marg_dadi_0[1:-1], 's-', label='dadi', markersize=6, alpha=0.7)
    ax.plot(freq_bins_0, marg_moments_0[1:-1], '^-', label='moments', markersize=6, alpha=0.7)
    ax.set_xlabel('Frequency bin')
    ax.set_ylabel('Count')
    ax.set_title('Marginal SFS - Pop0')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Marginal spectra comparison for pop1
    ax = axes[1, 1]
    marg_msprime_1 = sfs_msprime.marginalize([0])
    marg_dadi_1 = sfs_dadi.marginalize([0])
    marg_moments_1 = sfs_moments.marginalize([0])
    
    n_bins_1 = len(marg_msprime_1) - 2
    freq_bins_1 = np.arange(1, n_bins_1 + 1)
    
    ax.plot(freq_bins_1, marg_msprime_1[1:-1], 'o-', label='msprime', markersize=6, alpha=0.7)
    ax.plot(freq_bins_1, marg_dadi_1[1:-1], 's-', label='dadi', markersize=6, alpha=0.7)
    ax.plot(freq_bins_1, marg_moments_1[1:-1], '^-', label='moments', markersize=6, alpha=0.7)
    ax.set_xlabel('Frequency bin')
    ax.set_ylabel('Count')
    ax.set_title('Marginal SFS - Pop1')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Difference plot
    ax = axes[1, 2]
    # Calculate relative differences from msprime
    diff_dadi = (sfs_dadi_masked.data - sfs_msprime_masked.data) / (sfs_msprime_masked.data + 1)
    diff_moments = (sfs_moments_masked.data - sfs_msprime_masked.data) / (sfs_msprime_masked.data + 1)
    
    # Plot absolute mean differences
    ax.bar([0], [np.mean(np.abs(diff_dadi))], width=0.4, label='dadi vs msprime', alpha=0.7)
    ax.bar([1], [np.mean(np.abs(diff_moments))], width=0.4, label='moments vs msprime', alpha=0.7)
    ax.set_ylabel('Mean absolute relative difference')
    ax.set_title('SFS Differences from msprime')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['dadi', 'moments'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "sfs_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def simulate_and_infer(seq_length, mutation_rate, recombination_rate, n_samples, output_path, use_dadi_sim=False, skip_dadi=True):
    TRUE_PARAMS = [10000, 500, 1E-6, 1E-5]
    OUTPUT_DIR = Path(output_path)
    OUTPUT_DIR.mkdir(exist_ok=True)
    GRAPH = demographic_model(TRUE_PARAMS)
    
    if use_dadi_sim:
        print("Simulating data using dadi...")
        # Set up sample sizes for dadi simulation
        sample_sizes_dict = OrderedDict([("pop0", n_samples), ("pop1", n_samples)])
        ns = [2 * n for n in sample_sizes_dict.values()]  # Convert to haploid counts
        
        # Use grid points for simulation
        pts_sim = [max(ns) + 20, max(ns) + 30, max(ns) + 40]
        
        # Generate SFS using dadi
        sfs = diffusion_sfs_dadi(
            TRUE_PARAMS,
            sample_sizes_dict,
            mutation_rate,
            seq_length,
            pts_sim
        )
        
        # Sample from the expected SFS to get a realized SFS
        sfs = sfs.sample()
        
        # Convert to moments.Spectrum for consistency
        sfs = moments.Spectrum(sfs.data, pop_ids=["pop0", "pop1"])
    else:
        print("Simulating data using msprime...")
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
    
    if not skip_dadi:
        print("Running dadi inference...")
        opt_params_scaled_dadi = run_dadi_inference(sfs, mutation_rate, seq_length)
        N1_dadi, N2_dadi, m12_dadi, m21_dadi = opt_params_scaled_dadi
        print(f"Estimated dadi parameters: N1={N1_dadi}, N2={N2_dadi}, m12={m12_dadi}, m21={m21_dadi}")
    else:
        print("Skipping dadi inference (--skip_dadi=True)")
    
    print("\nRunning moments inference...")
    opt_params_scaled_moments = run_moments_inference(sfs, mutation_rate, seq_length)
    N1_moments, N2_moments, m12_moments, m21_moments = opt_params_scaled_moments
    
    print(f"\nEstimated moments parameters: N1={N1_moments}, N2={N2_moments}, m12={m12_moments}, m21={m21_moments}")
    print(f"True parameters: {TRUE_PARAMS}")

def main():
    parser = argparse.ArgumentParser(description="Run island model dadi inference.")
    parser.add_argument("--seq_length", type=float, default=1_000_000, help="Sequence length")
    parser.add_argument("--mu", type=float, default=1.5e-8, help="Mutation rate")
    parser.add_argument("--recombination_rate", type=float, default=1.5e-8, help="Recombination rate")
    parser.add_argument("--n_samples", type=int, default=30, help="Samples per population")
    parser.add_argument("--output_path", type=str, default="./single_inference", help="Output directory")
    parser.add_argument("--use_dadi_sim", action="store_true", help="Use dadi to simulate SFS instead of msprime")
    parser.add_argument("--skip_dadi", action="store_true", default=False, help="Skip dadi inference (default: True)")
    parser.add_argument("--mode", type=str, default="infer", choices=["infer", "compare"], 
                        help="Mode: 'infer' runs inference, 'compare' compares dadi vs msprime SFS")
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_sfs_simulations(
            seq_length=args.seq_length,
            mutation_rate=args.mu,
            recombination_rate=args.recombination_rate,
            n_samples=args.n_samples,
            output_path=args.output_path
        )
    else:
        simulate_and_infer(
            seq_length=args.seq_length,
            mutation_rate=args.mu,
            recombination_rate=args.recombination_rate,
            n_samples=args.n_samples,
            output_path=args.output_path,
            use_dadi_sim=args.use_dadi_sim,
            skip_dadi=args.skip_dadi
        )

if __name__ == "__main__":
    main()
