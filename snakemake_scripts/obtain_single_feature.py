# obtain_features.py

import numpy as np
import pickle
from src.preprocess import Processor
import json
import src.demographic_models as demographic_models
from src.parameter_inference import run_inference_dadi, run_inference_moments, run_inference_momentsLD
import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# I want to load in the 

def obtain_feature(SFS, sampled_params, experiment_config, sim_directory, sim_number):

    # Load in the experiment config 
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    upper_bound = [b if b is not None else None for b in experiment_config['upper_bound_optimization']]
    lower_bound = [b if b is not None else None for b in experiment_config['lower_bound_optimization']]

    # Load in the SFS file
    with open(SFS, "rb") as f:
        SFS = pickle.load(f)

    # Load in the sampled params
    with open(sampled_params, "rb") as f:
        sampled_params = pickle.load(f)

    # if experiment_config["demographic_model"] == "bottleneck_model":
    #     demographic_model = demographic_models.bottleneck_model

    # elif experiment_config["demographic_model"] == "split_isolation_model":
    #     demographic_model = demographic_models.split_isolation_model_simulation

    mega_result_dict = (
            {}
        )  # This will store all the results (downstream postprocessing) later
    
    mega_result_dict = {"simulated_params": sampled_params, "sfs": SFS}

    # Load the experiment config and run the simulation (as before)
    # processor = Processor(
    #     experiment_config,
    #     experiment_directory=sim_directory,
    #     recombination_rate=experiment_config["recombination_rate"],
    #     mutation_rate=experiment_config["mutation_rate"],
    # )


    # Conditional analysis based on provided functions
    if experiment_config["dadi_analysis"]:
        model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi, ll_list_dadi = (
            run_inference_dadi(
                sfs = SFS,
                p0= experiment_config['optimization_initial_guess'],
                lower_bound= lower_bound,
                upper_bound= upper_bound,
                num_samples=100,
                demographic_model=experiment_config['demographic_model'],
                mutation_rate=experiment_config['mutation_rate'],
                length=experiment_config['genome_length'],
                k  = experiment_config['k'], 
                top_values_k = experiment_config['top_values_k']
            )
        )

        dadi_results = {
            "model_sfs_dadi": model_sfs_dadi,
            "opt_theta_dadi": opt_theta_dadi,
            "opt_params_dadi": opt_params_dict_dadi,
            "ll_all_replicates_dadi": ll_list_dadi,
        }
        
        mega_result_dict.update(dadi_results)


    if experiment_config["moments_analysis"]:
        model_sfs_moments, opt_theta_moments, opt_params_dict_moments, ll_list_moments = (
            run_inference_moments(
                sfs = SFS,
                p0=experiment_config['optimization_initial_guess'],
                lower_bound= lower_bound,
                upper_bound= upper_bound,
                demographic_model=experiment_config['demographic_model'],
                use_FIM=experiment_config["use_FIM"],
                mutation_rate=experiment_config['mutation_rate'],
                length=experiment_config['genome_length'],
                k = experiment_config['k'],
                top_values_k=experiment_config['top_values_k']
            )
        )


        moments_results = {
            "model_sfs_moments": model_sfs_moments,
            "opt_theta_moments": opt_theta_moments,
            "opt_params_moments": opt_params_dict_moments,
            "ll_all_replicates_moments": ll_list_moments
        }
        mega_result_dict.update(moments_results)

    if experiment_config["momentsLD_analysis"]:

        p_guess = experiment_config['optimization_initial_guess'].copy()
        
        p_guess.extend([10000])
        
        folderpath = f'{sim_directory}/sampled_genome_windows/sim_{sim_number}'

        opt_params_momentsLD = run_inference_momentsLD(folderpath=folderpath,
          demographic_model=experiment_config["demographic_model"],
          num_reps = experiment_config["num_reps"], 
          p_guess = p_guess
        )

        momentsLD_results = {"opt_params_momentsLD": opt_params_momentsLD}
        mega_result_dict.update(momentsLD_results)

    # Save the results in a pickle file
    with open(f"{sim_directory}/simulation_results/software_inferences_sim_{sim_number}.pkl", "wb") as f:
        pickle.dump(mega_result_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sfs_file", type=str, required=True)
    parser.add_argument("--sampled_params_pkl", type=str, required=True)
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    obtain_feature(
        SFS=args.sfs_file,
        sampled_params=args.sampled_params_pkl,
        experiment_config=args.experiment_config_filepath,
        sim_directory=args.sim_directory,
        sim_number=args.sim_number
    )