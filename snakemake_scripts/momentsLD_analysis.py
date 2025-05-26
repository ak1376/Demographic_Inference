#!/usr/bin/env python

import argparse
import pickle
import json
import os
import numpy as np

# Adjust imports to match your project layout
from src.parameter_inference import run_inference_momentsLD
from src.demographic_models import set_TB_fixed


def main(
    combined_ld_stats_path: str,
    sampled_params_pkl: str,
    experiment_config_filepath: str,
    sim_directory: str,
    sim_number: int
):
    """
    Minimal script to run momentsLD inference on a precomputed LD stats file.
    Saves the inference results and figure, without any re-simulation or cleanup.
    """

    # 1. Load experiment config
    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    # 2. Load sampled parameters
    with open(sampled_params_pkl, "rb") as f:
        sampled_params = pickle.load(f)

    # 3. Load combined LD stats (observed data for inference)
    with open(combined_ld_stats_path, "rb") as f:
        combined_ld_stats = pickle.load(f)

    # 5. Prepare an initial parameter guess from config (dictionary or list)
    p_guess = experiment_config["optimization_initial_guess"].copy()  
    print(f"Initial guess for optimization: {p_guess}")

    # 6. Run the actual inference
    print("Running momentsLD inference...")
    opt_params_momentsLD, ll_list_momentsLD, ld_stats_fig = run_inference_momentsLD(
        ld_stats=combined_ld_stats,
        demographic_model=experiment_config["demographic_model"],
        p_guess=p_guess,
        sampled_params=sampled_params,
        experiment_config=experiment_config
    )

    # 7. Save the results as a pickle
    results_dict = {
        "simulated_params": sampled_params,
        "opt_params_momentsLD": opt_params_momentsLD,
        "ll_all_replicates_momentsLD": ll_list_momentsLD
    }
    output_pickle_path = os.path.join(sim_directory, f"momentsLD_inferences_sim_{sim_number}.pkl")
    with open(output_pickle_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Inference results saved to {output_pickle_path}")

    # 8. Save the LD stats figure (no re-simulation or cleanup)
    figure_path = os.path.join(sim_directory, f"LD_stats_sim_{sim_number}.png")
    ld_stats_fig.savefig(figure_path)
    print(f"LD stats figure saved to {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal script to run momentsLD inference.")
    parser.add_argument("--combined_ld_stats_path", type=str, required=True, help="Path to combined LD stats (pickle).")
    parser.add_argument("--sampled_params_pkl", type=str, required=True, help="Path to sampled params pickle.")
    parser.add_argument("--experiment_config_filepath", type=str, required=True, help="Path to experiment config JSON.")
    parser.add_argument("--sim_directory", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--sim_number", type=int, required=True, help="Simulation number or identifier.")

    args = parser.parse_args()

    main(
        combined_ld_stats_path=args.combined_ld_stats_path,
        sampled_params_pkl=args.sampled_params_pkl,
        experiment_config_filepath=args.experiment_config_filepath,
        sim_directory=args.sim_directory,
        sim_number=args.sim_number
    )
