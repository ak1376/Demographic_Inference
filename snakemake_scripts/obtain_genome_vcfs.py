'''
I want this file to create the VCF file windows. These should be wildcards in the snakemake workflow I think. 
'''

import argparse
from src.preprocess import Processor
import tskit
import json
import os 
def main(tree_sequence_file, experiment_config_filepath, genome_sim_directory, window_number, sim_number):

    ts = tskit.load(tree_sequence_file)

    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    # Simulate process and save windows as VCF files

    # Ensure the base directory is absolute
    base_directory = os.getcwd()

    # Construct the directory for saving VCF windows dynamically
    directory_for_windows = os.path.join(base_directory, genome_sim_directory, f"sim_{sim_number}")
    os.makedirs(directory_for_windows, exist_ok=True)  # Ensure the directory exists

    Processor.run_msprime_replicates(ts, experiment_config, window_number, directory_for_windows)
    Processor.write_samples_and_rec_map(experiment_config, window_number = window_number, folderpath=directory_for_windows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_sequence_file", type=str, required=True)
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--genome_sim_directory", type=str, required=True)
    parser.add_argument("--window_number", type=int, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    main(args.tree_sequence_file, args.experiment_config_filepath, args.genome_sim_directory, args.window_number, args.sim_number)



