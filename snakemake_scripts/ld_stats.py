from src.parameter_inference import get_LD_stats
import argparse
import numpy as np
import pickle
import json
import os
import msprime
import src.demographic_models  # Ensure this module is accessible
from src.preprocess import Processor  # To generate sample and flat map files

def ld_stat_creation(sampled_params_path, sim_directory, sim_number, window_number, experiment_config_filepath):
    # Define recombination bins
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    
    print(f"Calculating LD stats for replicate {window_number}, sim {sim_number}")

    # Load the sampled parameters and experiment configuration
    with open(sampled_params_path, "rb") as f:
        sampled_params = pickle.load(f)
    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    # Build the demography using your bottleneck model
    g = src.demographic_models.bottleneck_model(sampled_params)
    demog = msprime.Demography.from_demes(g)
    
    # Simulate the tree sequence (ancestry)
    ts = msprime.sim_ancestry(
        {"N0": 20},  # Adjust key as needed
        demography=demog,
        sequence_length=experiment_config['genome_length'],
        recombination_rate=experiment_config['recombination_rate'],
        random_seed=experiment_config['seed'] + window_number
    )
    # Simulate mutations so that genotype data is available
    ts = msprime.sim_mutations(ts, rate=experiment_config['mutation_rate'], random_seed=window_number + 1)
    if ts.num_sites == 0:
        raise ValueError("No mutations were simulated. Please check that your mutation rate and genome length are high enough.")
    
    # Write the mutated tree sequence to VCF on disk
    window_dir = f"{sim_directory}/sim_{sim_number}/window_{window_number}"
    os.makedirs(window_dir, exist_ok=True)
    vcf_filepath = f"{window_dir}/vcf_window.{window_number}.vcf"
    with open(vcf_filepath, "w+") as fout:
        ts.write_vcf(fout, allow_position_zero=True)
    os.system(f"gzip {vcf_filepath}")
    vcf_filepath += ".gz"
    print(f"VCF file written to: {vcf_filepath}")

    # Generate the samples and flat map files internally
    # Processor.write_samples_and_rec_map() is assumed to create these files in window_dir.
    Processor.write_samples_and_rec_map(experiment_config, window_number=window_number, folderpath=window_dir)
    # Now assume the samples file and flat map file are:
    pop_file_path = f"{window_dir}/samples.txt"
    flat_map_path = f"{window_dir}/flat_map.txt"

    # Calculate LD stats
    ld_stats = get_LD_stats(vcf_filepath, r_bins, flat_map_path, pop_file_path)
    
    # Save LD stats to a pickle file
    output_file = f"{window_dir}/ld_stats_window.{window_number}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(ld_stats, f)
    print(f"LD stats successfully created for window {window_number}, sim {sim_number}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LD statistics for specified simulation windows.")
    parser.add_argument("--sampled_params_path", type=str, required=True, help="Path to the sampled parameters pickle file")
    parser.add_argument("--experiment_config_filepath", type=str, required=True, help="Path to the experiment configuration JSON file")
    parser.add_argument("--sim_directory", type=str, required=True, help="Path to the simulation directory (for LD inferences)")
    parser.add_argument("--sim_number", type=int, required=True, help="Simulation number")
    parser.add_argument("--window_number", type=int, required=True, help="Window number")
    args = parser.parse_args()

    ld_stat_creation(
        sampled_params_path=args.sampled_params_path,
        sim_directory=args.sim_directory,
        sim_number=args.sim_number,
        window_number=args.window_number,
        experiment_config_filepath=args.experiment_config_filepath
    )
