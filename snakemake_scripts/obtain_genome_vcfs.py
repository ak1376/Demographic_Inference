'''
I want this file to create the VCF file windows. These should be wildcards in the snakemake workflow I think. 
'''

import argparse
from src.preprocess import Processor
import tskit
import json
import os 

def main(tree_sequence_file, experiment_config_filepath, genome_sim_directory, window_number, sim_number):
    # print(f"Debug: Starting main function")
    # print(f"Debug: tree_sequence_file = {tree_sequence_file}")
    # print(f"Debug: genome_sim_directory = {genome_sim_directory}")
    
    ts = tskit.load(tree_sequence_file)
    # print(f"Debug: Successfully loaded tree sequence")

    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)
    # print(f"Debug: Successfully loaded config")

    # Use the sim_X directory as the parent directory
    # This is where we were going wrong before
    parent_dir = genome_sim_directory  # This is already /projects/.../sim_0
    # print(f"Debug: Parent directory = {parent_dir}")
    
    # print(f"Debug: About to run msprime_replicates")
    Processor.run_msprime_replicates(ts, experiment_config, window_number, parent_dir)
    # print(f"Debug: Completed msprime_replicates")

    # print(f"Debug: About to write samples and rec map")
    Processor.write_samples_and_rec_map(experiment_config, window_number=window_number, folderpath=parent_dir)
    # print(f"Debug: Completed writing samples and rec map")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_sequence_file", type=str, required=True)
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--genome_sim_directory", type=str, required=True)
    parser.add_argument("--window_number", type=int, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    main(args.tree_sequence_file, args.experiment_config_filepath, args.genome_sim_directory, args.window_number, args.sim_number)



