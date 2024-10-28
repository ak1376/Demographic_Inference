'''
I want this file to create the VCF file windows. These should be wildcards in the snakemake workflow I think. 
'''


import pickle
import json
import src.demographic_models as demographic_models
import argparse
from src.preprocess import Processor

def main(experiment_config_file, sampled_params_path, genome_sim_directory, sim_number):

    # Open the pickled file with the sampled parameters
    with open(sampled_params_path, 'rb') as f:
        sampled_params = pickle.load(f)

    # Open the json file with the experiment configuration
    with open(experiment_config_file, 'r') as f:
        experiment_config = json.load(f)

    # Simulate process and save windows as VCF files
    directory_for_windows = f"{genome_sim_directory}/sim_{sim_number}"

    Processor.run_msprime_replicates(sampled_params=sampled_params, experiment_config = experiment_config, folderpath=directory_for_windows)
    print("MSPRIME REPLICATES DONE!!!!!!")
    Processor.write_samples_and_rec_map(experiment_config, directory_for_windows)
    print("SAMPLES AND REC MAP WRITTEN!!!!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--sampled_params_path", type=str, required=True)
    parser.add_argument("--genome_sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    main(args.experiment_config_filepath, args.sampled_params_path, args.genome_sim_directory, args.sim_number)



