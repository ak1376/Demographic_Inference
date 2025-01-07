#!/usr/bin/env python3

import pickle
import json
import argparse
import os
import numpy as np
from tqdm import tqdm

# If "run_inference_momentsLD" is in a local module (src.parameter_inference), import it:
from src.parameter_inference import run_inference_momentsLD
import moments

def obtain_feature(combined_ld_stats_path, experiment_config_filepath, output_dir):
    """
    Perform MomentsLD parameter inference given combined LD stats and an experiment configuration.
    Results are saved to output_dir (no retries, cleanup, or sampled parameters).
    """

    # 1. Load the experiment config
    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    # 2. Load the combined LD stats
    with open(combined_ld_stats_path, "rb") as f:
        combined_ld_stats = pickle.load(f)

    # Prepare a result dictionary
    mega_result_dict = {}

    # 3. Build an initial guess and slightly perturb it
    p_guess = experiment_config["optimization_initial_guess"].copy()
    # If your model needs additional parameters, add them here:
    p_guess.extend([10000])  
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)  # type: ignore

    # 4. Run a single inference attempt (no retries)
    print("Starting MomentsLD optimization...")
    opt_params_momentsLD, ll_list_momentsLD = run_inference_momentsLD(
        ld_stats=combined_ld_stats,
        demographic_model=experiment_config["demographic_model"],
        p_guess=p_guess,
    )
    print("Optimization completed successfully.")

    # 5. Store inference results in the dictionary
    momentsLD_results = {
        "opt_params_momentsLD": opt_params_momentsLD,
        "ll_all_replicates_momentsLD": ll_list_momentsLD,
    }
    mega_result_dict.update(momentsLD_results)

    # 6. Save everything to a pickle file in the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "momentsLD_inferences.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(mega_result_dict, f)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MomentsLD inference without retries, cleanup, or sampled parameters.")
    parser.add_argument("--combined_ld_stats_path", type=str, required=True,
                        help="Path to the combined LD stats pickle file.")
    parser.add_argument("--experiment_config_filepath", type=str, required=True,
                        help="Path to the JSON experiment config.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the inference results.")

    args = parser.parse_args()

    obtain_feature(
        combined_ld_stats_path=args.combined_ld_stats_path,
        experiment_config_filepath=args.experiment_config_filepath,
        output_dir=args.output_dir,
    )
