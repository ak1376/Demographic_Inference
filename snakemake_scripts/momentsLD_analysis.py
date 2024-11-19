import pickle
import json
import argparse
import ray
import numpy as np
import moments
import subprocess
from src.parameter_inference import run_inference_momentsLD
import os
import shutil

def cleanup_files(sim_directory, sim_number):
    simulation_results_directory = '/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results'
    genome_windows_directory = f'/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}'

    # Define files to delete
    files_to_delete = [
        f"{simulation_results_directory}/SFS_sim_{sim_number}.pkl",
        f"{simulation_results_directory}/ts_sim_{sim_number}.trees",
        f"{simulation_results_directory}/sampled_params_{sim_number}.pkl",
        f"{simulation_results_directory}/sampled_params_metadata_{sim_number}.txt"
    ]

    # Delete individual files
    for filepath in files_to_delete:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted {filepath}")
        else:
            print(f"File not found, skipping deletion: {filepath}")

    # Delete genome windows directory
    if os.path.exists(genome_windows_directory):
        shutil.rmtree(genome_windows_directory)
        print(f"Deleted genome windows directory: {genome_windows_directory}")
    else:
        print(f"Genome windows directory not found: {genome_windows_directory}")

def obtain_feature(combined_ld_stats_path, sim_directory, sampled_params, experiment_config_filepath, sim_number):
    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    with open(sampled_params, "rb") as f:
        sampled_params = pickle.load(f)

    with open(combined_ld_stats_path, "rb") as f:
        combined_ld_stats = pickle.load(f)

    mega_result_dict = {"simulated_params": sampled_params}

    p_guess = experiment_config["optimization_initial_guess"].copy()
    p_guess.extend([10000])
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)

    def reoptimize():
        print("Attempting optimization...")
        opt_params_momentsLD, ll_list_momentsLD = run_inference_momentsLD(
            ld_stats=combined_ld_stats,
            demographic_model=experiment_config["demographic_model"],
            p_guess=p_guess
        )
        print("Optimization completed successfully.")
        return opt_params_momentsLD, ll_list_momentsLD

    try:
        opt_params_momentsLD, ll_list_momentsLD = reoptimize()
    except (np.linalg.LinAlgError, KeyError) as e:
        print("======================================================================================")
        print(f"Error encountered: {e}. Attempting to resimulate for sim_number={sim_number}.")
        print("======================================================================================")

        # Clean up files before rerun
        cleanup_files(sim_directory, sim_number)

        # Resimulate
        rerun_command = [
            "python",
            "/projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/single_simulation.py",
            "--experiment_config", experiment_config_filepath,
            "--sim_directory", '/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences',
            "--sim_number", str(sim_number)
        ]
        subprocess.run(rerun_command, check=True)
        print("Resimulation completed.")

        # Rerun genome window creation
        for window_number in range(experiment_config['num_windows']):
            rerun_window_command = [
                "python",
                "/projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/obtain_genome_vcfs.py",
                "--tree_sequence_file", f"/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/ts_sim_{sim_number}.trees",
                "--experiment_config_filepath", experiment_config_filepath,
                "--genome_sim_directory", '/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows',
                "--window_number", str(window_number),
                "--sim_number", str(sim_number)
            ]
            subprocess.run(rerun_window_command, check=True)
            print(f"Genome window creation for window {window_number} rerun successfully.")

        # Rerun the creation of LD stats 
        # ld_stat_creation(vcf_filepath, flat_map_path, pop_file_path, sim_directory, sim_number, window_number)
        from tqdm import tqdm

        for window_number in tqdm(range(experiment_config['num_windows'])):
            rerun_window_command = [
                "python",
                "/projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/ld_stats.py",
                "--vcf_filepath", f"/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/window.{window_number}.vcf.gz",
                "--flat_map_path", f"/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/flat_map.txt",
                "--pop_file_path", f"/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/samples.txt", 
                "--sim_directory", '/projects/kernlab/akapoor/Demographic_Inference/LD_inferences',
                "--sim_number", str(sim_number),
                "--window_number", str(window_number)
            ]

        # Retry optimization
        print("Retrying optimization after resimulation...")
        p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)
        opt_params_momentsLD, ll_list_momentsLD = reoptimize()

    # Store results in dictionary and save to a pickle file
    momentsLD_results = {
        "opt_params_momentsLD": opt_params_momentsLD,
        "ll_all_replicates_momentsLD": ll_list_momentsLD,
    }
    mega_result_dict.update(momentsLD_results)

    output_path = f"{sim_directory}/momentsLD_inferences_sim_{sim_number}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(mega_result_dict, f)
    print(f"Results saved to {output_path}")

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
        experiment_config_filepath=args.experiment_config_filepath,
        sim_directory=args.sim_directory,
        sim_number=args.sim_number
    )
