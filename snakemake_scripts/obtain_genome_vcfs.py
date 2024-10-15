'''
I want this file to create the VCF file windows. These should be wildcards in the snakemake workflow I think. 
'''


import pickle
import json
import src.demographic_models as demographic_models
import argparse
from src.preprocess import Processor

def main(experiment_config_file, sampled_params_path, sim_directory, sim_number):

    # Open the pickled file with the sampled parameters
    with open(sampled_params_path, 'rb') as f:
        sampled_params = pickle.load(f)

    # Open the json file with the experiment configuration
    with open(experiment_config_file, 'r') as f:
        experiment_config = json.load(f)


    if experiment_config["demographic_model"] == "bottleneck_model":
        demographic_model = demographic_models.bottleneck_model

    elif experiment_config["demographic_model"] == "split_isolation_model":
        demographic_model = demographic_models.split_isolation_model_simulation

    else:
        raise ValueError(f"Unsupported demographic model: {experiment_config['demographic_model']}")

    # Simulate process and save windows as VCF files
    g = demographic_model(sampled_params)
    directory_for_windows = f"{sim_directory}/sampled_genome_windows/sim_{sim_number}"
    Processor.run_msprime_replicates(experiment_config, g, directory_for_windows)
    print("MSPRIME REPLICATES DONE!!!!!!")
    samples_file, flat_map_file = Processor.write_samples_and_rec_map(experiment_config, directory_for_windows)
    print("SAMPLES AND REC MAP WRITTEN!!!!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--sampled_params_path", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    main(args.experiment_config_filepath, args.sampled_params_path, args.sim_directory, args.sim_number)



