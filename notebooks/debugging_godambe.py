"""
This script uses msprime to simulate under an isolation with migration model,
writing the outputs to VCF. We'll simulate a small dataset: 100 x 1Mb regions,
each with recombination and mutation rates of 1.5e-8. We'll then use moments
to compute LD statistics from each of the 100 replicates to compute statistic
means and variances/covariances. These are then used to refit the simulated
model using moments.LD, and then we use bootstrapped datasets to estimate
confidence intervals.

The demographic model is a population of size 10,000 that splits into a
population of size 2,000 and a population of size 20,000. The split occurs
1,500 generations ago followed by symmetric migration at rate 1e-4.
"""

import os
import time
import gzip
import numpy as np
import pickle
import msprime
import moments
import demes
import ray
import json

assert msprime.__version__ >= "1"

if not os.path.isdir("./data/"):
    os.makedirs("./data/")
os.system("rm ./data/*.vcf.gz")
os.system("rm ./data/*.h5")

upper_bound_params = {
    "t_split": 5000, 
    "m": 1e-4,
    "N1": 10000,
    "N2": 10000,
    "Na": 20000
}

lower_bound_params =  {
    "t_split": 100, 
    "m": 1e-8,
    "N1": 100,
    "N2": 100,
    "Na": 10000

}

def sample_params():
    sampled_params = {}
    for key in lower_bound_params:
        lower_bound = lower_bound_params[key]
        upper_bound = upper_bound_params[key]
        sampled_value = np.random.uniform(lower_bound, upper_bound)

        # Initialize adjusted_value with sampled_value by default
        adjusted_value = sampled_value

        # Check if the sampled parameter is equal to the mean of the uniform distribution
        mean_value = (lower_bound + upper_bound) / 2
        if sampled_value == mean_value:
            # Add a small random value to avoid exact mean, while keeping within bounds
            adjustment = np.random.uniform(-0.1 * (upper_bound - lower_bound), 0.1 * (upper_bound - lower_bound))
            adjusted_value = sampled_value + adjustment
            # Ensure the adjusted value is still within the bounds
            adjusted_value = max(min(adjusted_value, upper_bound), lower_bound)

        # Assign adjusted_value to sampled_params
        if key == "m":
            sampled_params[key] = adjusted_value
        else:
            sampled_params[key] = int(adjusted_value)

    return sampled_params

def demographic_model(sampled_params):

    # Unpack the sampled parameters
    Na, N1, N2, m, t_split = (
        sampled_params["Na"],  # Effective population size of the ancestral population
        sampled_params["N1"],  # Size of population 1 after split
        sampled_params["N2"],  # Size of population 2 after split
        sampled_params["m"],   # Migration rate between populations
        sampled_params["t_split"],  # Time of the population split (in generations)
    )

    b = demes.Builder()
    b.add_deme("Na", epochs=[dict(start_size=Na, end_time=t_split)])
    b.add_deme("N1", ancestors=["Na"], epochs=[dict(start_size=N1)])
    b.add_deme("N2", ancestors=["Na"], epochs=[dict(start_size=N2)])
    b.add_migration(demes=["N1", "N2"], rate=m)
    g = b.resolve()
    return g

# def demographic_model():
#     b = demes.Builder()
#     b.add_deme("Na", epochs=[dict(start_size=18575, end_time=1408)])
#     b.add_deme("N1", ancestors=["Na"], epochs=[dict(start_size=617)])
#     b.add_deme("N2", ancestors=["Na"], epochs=[dict(start_size=4559)])
#     b.add_migration(demes=["N1", "N2"], rate=7.701758925243914e-05)
#     g = b.resolve()
#     return g

# def run_msprime_replicates(experiment_config, num_reps=100):
#     # Set up the demography from demes
#     g = demographic_model()
#     demog = msprime.Demography.from_demes(g)

#     # Dynamically define the samples using msprime.SampleSet, based on the sample_sizes dictionary
#     samples = [
#         msprime.SampleSet(sample_size, population=pop_name, ploidy=1)
#         for pop_name, sample_size in experiment_config['num_samples'].items()
#     ]

#     tree_sequences = msprime.sim_ancestry(
#         samples,
#         demography=demog,
#         sequence_length=experiment_config['genome_length'],
#         recombination_rate=experiment_config['recombination_rate'],
#         num_replicates=num_reps,
#         random_seed=42,
#     )
#     for ii, ts in enumerate(tree_sequences):
#         ts = msprime.sim_mutations(ts, rate=experiment_config['mutation_rate'], random_seed=ii + 1)
#         vcf_name = "./data/split_mig.{0}.vcf".format(ii)
#         with open(vcf_name, "w+") as fout:
#             ts.write_vcf(fout, allow_position_zero=True)
#         os.system(f"gzip {vcf_name}")


def run_msprime_replicates(sampled_params, experiment_config):

    g = demographic_model(sampled_params)
    demog = msprime.Demography.from_demes(g)
    tree_sequences = msprime.sim_ancestry(
        {"N1": experiment_config['num_samples']['N1'], "N2": experiment_config['num_samples']['N2']},
        demography=demog,
        sequence_length=experiment_config['genome_length'],
        recombination_rate=experiment_config['recombination_rate'],
        num_replicates=experiment_config['num_reps'],
        random_seed=experiment_config['seed'],
    )
    for ii, ts in enumerate(tree_sequences):
        ts = msprime.sim_mutations(ts, rate=experiment_config['mutation_rate'], random_seed=ii + 1)
        vcf_name = "./data/split_mig.{0}.vcf".format(ii)
        with open(vcf_name, "w+") as fout:
            ts.write_vcf(fout, allow_position_zero=True)
        os.system(f"gzip {vcf_name}")

def write_samples_and_rec_map(experiment_config):

    # Define the file paths
    samples_file = "./data/samples.txt"
    flat_map_file ="./data/flat_map.txt"

    # Open and write the sample file
    with open(samples_file, "w+") as fout:
        fout.write("sample\tpop\n")

        # Dynamically define samples based on the num_samples dictionary
        sample_idx = 0  # Initialize sample index
        for pop_name, sample_size in experiment_config['num_samples'].items():
            for _ in range(sample_size):
                fout.write(f"tsk_{sample_idx}\t{pop_name}\n")
                sample_idx += 1

    # Write the recombination map file
    with open(flat_map_file, "w+") as fout:
        fout.write("pos\tMap(cM)\n")
        fout.write("0\t0\n")
        fout.write(f"{experiment_config['genome_length']}\t{experiment_config['recombination_rate'] * experiment_config['genome_length'] * 100}\n")

# def write_samples_and_rec_map(L=1000000, r=1.5e-8, n=18):
#     # samples file
#     with open("./data/samples.txt", "w+") as fout:
#         fout.write("sample\tpop\n")
#         for jj in range(2):
#             for ii in range(n):
#                 fout.write(f"tsk_{jj * n + ii}\tdeme{jj}\n")
#     # recombination map
#     with open("./data/flat_map.txt", "w+") as fout:
#         fout.write("pos\tMap(cM)\n")
#         fout.write("0\t0\n")
#         fout.write(f"{L}\t{r * L * 100}\n")

# Define your function with Ray's remote decorator
@ray.remote
def get_LD_stats(rep_ii, r_bins):
    vcf_file = f"./data/split_mig.{rep_ii}.vcf.gz"
    time1 = time.time()
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file="./data/flat_map.txt",
        pop_file="./data/samples.txt",
        pops=["N1", "N2"],
        r_bins=r_bins,
        report=False,
    )
    time2 = time.time()
    print("  finished rep", rep_ii, "in", int(time2 - time1), "seconds")
    return ld_stats


if __name__ == "__main__":
    num_reps = 100
    # define the bin edges
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])

    with open("/sietch_colab/akapoor/Demographic_Inference/experiment_config.json") as f:
        experiment_config = json.load(f)


    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Sample parameters
    sampled_params = sample_params()

    print("running msprime and writing vcfs")
    run_msprime_replicates(sampled_params, experiment_config)
    # run_msprime_replicates(experiment_config=experiment_config, num_reps=num_reps)

    print("writing samples and recombination map")
    write_samples_and_rec_map(experiment_config=experiment_config)

    print("parsing LD statistics in parallel")
    # Submit tasks to Ray in parallel using .remote()
    futures = [get_LD_stats.remote(ii, r_bins) for ii in range(num_reps)]
    # Gather results with ray.get() to collect them once the tasks are finished
    ld_stats = ray.get(futures)
    # Optionally, you can convert the list of results into a dictionary with indices
    ld_stats_dict = {ii: result for ii, result in enumerate(ld_stats)}

    print("computing mean and varcov matrix from LD statistics sums")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats_dict)
    with open(f"./data/means.varcovs.split_mig.{num_reps}_reps.bp", "wb+") as fout:
        pickle.dump(mv, fout)
    print(
        "computing bootstrap replicates of mean statistics (for confidence intervals"
    )
    all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats_dict)
    with open(f"./data/bootstrap_sets.split_mig.{num_reps}_reps.bp", "wb+") as fout:
        pickle.dump(all_boot, fout)
    os.system("rm ./data/*.vcf.gz")
    os.system("rm ./data/*.h5")

# print("computing expectations under the model")
g = demographic_model(sampled_params)
# y = moments.Demes.LD(g, sampled_demes=["deme0", "deme1"], rho=4 * 10000 * r_bins)
# y = moments.LD.LDstats(
#     [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
#     num_pops=y.num_pops,
#     pop_ids=y.pop_ids,
# )
# y = moments.LD.Inference.sigmaD2(y)

# plot simulated data vs expectations under the model
# fig = moments.LD.Plotting.plot_ld_curves_comp(
#     y,
#     mv["means"][:-1],
#     mv["varcovs"][:-1],
#     rs=r_bins,
#     stats_to_plot=[
#         ["DD_0_0"],
#         ["DD_0_1"],
#         ["DD_1_1"],
#         ["Dz_0_0_0"],
#         ["Dz_0_1_1"],
#         ["Dz_1_1_1"],
#         ["pi2_0_0_1_1"],
#         ["pi2_0_1_0_1"],
#         ["pi2_1_1_1_1"],
#     ],
#     labels=[
#         [r"$D_0^2$"],
#         [r"$D_0 D_1$"],
#         [r"$D_1^2$"],
#         [r"$Dz_{0,0,0}$"],
#         [r"$Dz_{0,1,1}$"],
#         [r"$Dz_{1,1,1}$"],
#         [r"$\pi_{2;0,0,1,1}$"],
#         [r"$\pi_{2;0,1,0,1}$"],
#         [r"$\pi_{2;1,1,1,1}$"],
#     ],
#     rows=3,
#     plot_vcs=True,
#     show=False,
#     fig_size=(6, 4),
#     output="split_mig_comparison.pdf",
# )

print("running inference")
# Run inference using the parsed data
demo_func = moments.LD.Demographics2D.split_mig
# Set up the initial guess
# The split_mig function takes four parameters (nu0, nu1, T, m), and we append
# the last parameter to fit Ne, which doesn't get passed to the function but
# scales recombination rates so can be simultaneously fit
p_guess = [0.1, 2, 0.075, 2, 10000]
p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)
opt_params, LL = moments.LD.Inference.optimize_log_lbfgsb(
    p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins,
)

physical_units = moments.LD.Util.rescale_params(
    opt_params, ["nu", "nu", "T", "m", "Ne"]
)

print("Simulated parameters:")
print(f"  N(deme0)         :  {g.demes[1].epochs[0].start_size:.1f}")
print(f"  N(deme1)         :  {g.demes[2].epochs[0].start_size:.1f}")
print(f"  Div. time (gen)  :  {g.demes[1].epochs[0].start_time:.1f}")
print(f"  Migration rate   :  {g.migrations[0].rate:.6f}")
print(f"  N(ancestral)     :  {g.demes[0].epochs[0].start_size:.1f}")

print("best fit parameters:")
print(f"  N(deme0)         :  {physical_units[0]:.1f}")
print(f"  N(deme1)         :  {physical_units[1]:.1f}")
print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
print(f"  Migration rate   :  {physical_units[3]:.6f}")
print(f"  N(ancestral)     :  {physical_units[4]:.1f}")

print("computing confidence intervals for parameters")
uncerts = moments.LD.Godambe.GIM_uncert(
    demo_func, all_boot, opt_params, mv["means"], mv["varcovs"], r_edges=r_bins,
)

lower = opt_params - 1.96 * uncerts
upper = opt_params + 1.96 * uncerts

lower_pu = moments.LD.Util.rescale_params(lower, ["nu", "nu", "T", "m", "Ne"])
upper_pu = moments.LD.Util.rescale_params(upper, ["nu", "nu", "T", "m", "Ne"])

print("95% CIs:")
print(f"  N(deme0)         :  {lower_pu[0]:.1f} - {upper_pu[0]:.1f}")
print(f"  N(deme1)         :  {lower_pu[1]:.1f} - {upper_pu[1]:.1f}")
print(f"  Div. time (gen)  :  {lower_pu[2]:.1f} - {upper_pu[2]:.1f}")
print(f"  Migration rate   :  {lower_pu[3]:.6f} - {upper_pu[3]:.6f}")
print(f"  N(ancestral)     :  {lower_pu[4]:.1f} - {upper_pu[4]:.1f}")