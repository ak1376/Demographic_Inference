import pickle
import json
import argparse
import ray
import numpy as np
import moments
import subprocess
from src.parameter_inference import run_inference_momentsLD


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Function to load in the combined moments LD pickle files for each simulation
def obtain_feature(combined_ld_stats_path, sim_directory, sampled_params, experiment_config_filepath, sim_number):
    # Load the experiment config
    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    # Load the sampled params
    with open(sampled_params, "rb") as f:
        sampled_params = pickle.load(f)

    # Load the combined moments LD stats
    with open(combined_ld_stats_path, "rb") as f:
        combined_ld_stats = pickle.load(f)

    # Dictionary to store results for downstream postprocessing
    mega_result_dict = {"simulated_params": sampled_params}

    # Set up the initial guess for the optimization
    p_guess = experiment_config["optimization_initial_guess"].copy()
    p_guess.extend([10000])  # Example of hardcoded value; replace as necessary
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)  # type: ignore

    try:
        # Attempt inference
        opt_params_momentsLD, ll_list_momentsLD = run_inference_momentsLD(
            ld_stats=combined_ld_stats,
            demographic_model=experiment_config["demographic_model"],
            p_guess=p_guess
        )

    except (np.linalg.LinAlgError, KeyError) as e:
        print(f"Error encountered: {e}. Attempting to rerun simulation for sim_number={sim_number}.")

        # Rerun simulation via subprocess call
        rerun_command = [
            "python",
            "/path/to/rerun_simulation_script.py",  # Replace with actual path
            "--experiment_config", experiment_config_filepath,
            "--sim_directory", sim_directory,
            "--sim_number", str(sim_number),
            # Add any other necessary arguments here
        ]
        subprocess.run(rerun_command, check=True)

        # Re-attempt inference after rerunning simulation
        p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)
        opt_params_momentsLD, ll_list_momentsLD = run_inference_momentsLD(
            ld_stats=combined_ld_stats,
            demographic_model=experiment_config["demographic_model"],
            p_guess=p_guess
        )

    # Store results in dictionary and save to a pickle file
    momentsLD_results = {
        "opt_params_momentsLD": opt_params_momentsLD,
        "ll_all_replicates_momentsLD": ll_list_momentsLD,
    }
    mega_result_dict.update(momentsLD_results)

    with open(f"{sim_directory}/momentsLD_inferences_sim_{sim_number}.pkl", "wb") as f:
        pickle.dump(mega_result_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined_ld_stats_path", type=str, required=True)
    parser.add_argument("--sampled_params_pkl", type=str, required=True)
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    obtain_feature(
        combined_ld_stats_path=args.combined_ld_stats_path,
        sampled_params=args.sampled_params_pkl,
        experiment_config=args.experiment_config_filepath,
        sim_directory=args.sim_directory,
        sim_number=args.sim_number
    )
