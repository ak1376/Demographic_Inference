# obtain_features.py

import pickle
import json
from src.parameter_inference import run_inference_dadi, run_inference_moments
from src.demographic_models import set_TB_fixed
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


# I want to do dadi and moments inference but separately for each replicate. Note to self: need to define another function that will aggregate the results of each replicate and then choose the top k ones. 

def obtain_feature(SFS, sampled_params, experiment_config, sim_directory, sim_number, replicate_number):

    print(f'The simulation directory is: {sim_directory}')

    # Ensure required directories exist
    dadi_dir = f"{sim_directory}/sim_{sim_number}/dadi/replicate_{replicate_number}"
    moments_dir = f"{sim_directory}/sim_{sim_number}/moments/replicate_{replicate_number}"

    print(f'The dadi dir is {dadi_dir}')
    print(f'The moments dir is {moments_dir}')

    os.makedirs(dadi_dir, exist_ok=True)
    os.makedirs(moments_dir, exist_ok=True)

    # Load in the experiment config 
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    # It's strange because we also want to optimize the ancestral size but indirectly through theta. Therefore, the ancestral population size will not be an element in the upper or lower bounds
    param_order = experiment_config["parameter_names"]
    p0 = [experiment_config["optimization_initial_guess"][param] for param in param_order]
    lower_bound = [experiment_config["lower_bound_optimization"][param] for param in param_order]
    upper_bound = [experiment_config["upper_bound_optimization"][param] for param in param_order]

    print(f'The guess is: {p0}')
    print(f'The upper bound is: {upper_bound}')
    print(f'The lower bound is: {lower_bound}')
    
    upper_bound = list(experiment_config['upper_bound_optimization'].values())
    lower_bound = list(experiment_config['lower_bound_optimization'].values())

    # Load in the SFS file
    with open(SFS, "rb") as f:
        SFS = pickle.load(f)

    # Load in the sampled params
    with open(sampled_params, "rb") as f:
        sampled_params = pickle.load(f)

    # p0 = list(experiment_config['optimization_initial_guess'].values())
    # lower_bound = lower_bound[:]
    # upper_bound = upper_bound[:]

    if experiment_config['demographic_model'] == "bottleneck_model":
        # Extract the true TB value from sampled parameters
        set_TB_fixed((sampled_params['t_bottleneck_start'] - sampled_params['t_bottleneck_end']) / (2 * sampled_params['N0']))

    # Conditional analysis based on provided functions
    if experiment_config["dadi_analysis"]:
        model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = (
            run_inference_dadi(
                sfs = SFS,
                p0= p0,
                lower_bound= lower_bound,
                upper_bound= upper_bound,
                num_samples=30,
                demographic_model=experiment_config['demographic_model'],
                mutation_rate=experiment_config['mutation_rate'],
                length=experiment_config['genome_length']
            )
        )


        dadi_results = {
            "model_sfs_dadi": model_sfs_dadi,
            "opt_theta_dadi": opt_theta_dadi,
            "opt_params_dadi": opt_params_dict_dadi,
            "ll_dadi": opt_params_dict_dadi['ll'],
        }
        
    if experiment_config["moments_analysis"]:
        model_sfs_moments, opt_theta_moments, opt_params_dict_moments = (
            run_inference_moments(
                sfs = SFS,
                p0=p0,
                lower_bound= lower_bound,
                upper_bound= upper_bound,
                demographic_model=experiment_config['demographic_model'],
                use_FIM=experiment_config["use_FIM"],
                mutation_rate=experiment_config['mutation_rate'],
                length=experiment_config['genome_length']
            )
        )

        moments_results = {
            "model_sfs_moments": model_sfs_moments,
            "opt_theta_moments": opt_theta_moments,
            "opt_params_moments": opt_params_dict_moments,
            "ll_moments": opt_params_dict_moments['ll']
        }

    
    # save the results in a pickle file
    with open(f"{dadi_dir}/replicate_{replicate_number}.pkl", "wb") as f:
        pickle.dump(dadi_results, f)

    with open(f"{moments_dir}/replicate_{replicate_number}.pkl", "wb") as f:
        pickle.dump(moments_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfs_file", type=str, required=True)
    parser.add_argument("--sampled_params_pkl", type=str, required=True)
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    parser.add_argument("--replicate_number", type=int, required=True)  # Add this line
    args = parser.parse_args()

    obtain_feature(
        SFS=args.sfs_file,
        sampled_params=args.sampled_params_pkl,
        experiment_config=args.experiment_config_filepath,
        sim_directory = args.sim_directory,
        sim_number=args.sim_number, 
        replicate_number=args.replicate_number  # Pass replicate number here
    )