import os
import pickle
import json
import argparse
import subprocess
import shutil
import numpy as np
from tqdm import tqdm
from src.parameter_inference import run_inference_momentsLD
import moments

# Get the current working directory dynamically
# BASE_DIR = os.getcwd()
BASE_DIR = "/sietch_colab/akapoor/"

def cleanup_files(sim_number):
    """Clean up simulation-related files for a given simulation number."""
    simulation_results_directory = os.path.join(BASE_DIR, "Demographic_Inference", "simulated_parameters_and_inferences", "simulation_results")
    genome_windows_directory = os.path.join(BASE_DIR, "Demographic_Inference", "sampled_genome_windows", f"sim_{sim_number}")

    # Define files to delete
    files_to_delete = [
        os.path.join(simulation_results_directory, f"SFS_sim_{sim_number}.pkl"),
        os.path.join(simulation_results_directory, f"ts_sim_{sim_number}.trees"),
        os.path.join(simulation_results_directory, f"sampled_params_{sim_number}.pkl"),
        os.path.join(simulation_results_directory, f"sampled_params_metadata_{sim_number}.txt"),
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

    # Delete the LD stats directory 
    ld_inferences_dir = os.path.join(BASE_DIR, "Demographic_Inference", "LD_inferences", f"sim_{sim_number}")
    if os.path.exists(ld_inferences_dir):
        shutil.rmtree(ld_inferences_dir)
        print(f"Deleted LD inferences directory: {ld_inferences_dir}")
    else:
        print(f"LD inferences directory not found: {ld_inferences_dir}")

def resimulate(sim_number, experiment_config_filepath):
    """Rerun simulation and regenerate genome windows."""
    print(f"Resimulating for simulation {sim_number}...")

    # Run the simulation script
    rerun_command = [
        "python",
        os.path.join(BASE_DIR, "Demographic_Inference", "snakemake_scripts", "single_simulation.py"),
        "--experiment_config",
        experiment_config_filepath,
        "--sim_directory",
        os.path.join(BASE_DIR, "Demographic_Inference", "simulated_parameters_and_inferences"),
        "--sim_number",
        str(sim_number),
    ]
    subprocess.run(rerun_command, check=True)
    print("Resimulation completed.")

    # Regenerate genome windows
    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    print(f"Regenerating genome windows for simulation {sim_number}...")

    for window_number in range(experiment_config["num_windows"]):
        regenerate_window_command = [
            "python",
            os.path.join(BASE_DIR, "Demographic_Inference", "snakemake_scripts", "obtain_genome_vcfs.py"),
            "--tree_sequence_file",
            os.path.join(BASE_DIR, "Demographic_Inference", "simulated_parameters_and_inferences", "simulation_results", f"ts_sim_{sim_number}.trees"),
            "--experiment_config_filepath",
            experiment_config_filepath,
            "--genome_sim_directory",
            'sampled_genome_windows',
            "--window_number",
            str(window_number),
            "--sim_number",
            str(sim_number),
        ]
        subprocess.run(regenerate_window_command, check=True)
        print(f"Regenerated genome window {window_number}.")

    # Recompute the LD stats
    for window_number in range(experiment_config['num_windows']):
        regenerate_window_command = [
            "python", 
            os.path.join(BASE_DIR, "Demographic_Inference", "snakemake_scripts", "ld_stats.py"),
            "--vcf_filepath",
            os.path.join(BASE_DIR, "Demographic_Inference", "sampled_genome_windows", f"sim_{sim_number}", f"window_{window_number}", f"window.{window_number}.vcf.gz"),
            "--flat_map_path",
            os.path.join(BASE_DIR, "Demographic_Inference", "sampled_genome_windows", f"sim_{sim_number}", f"window_{window_number}", "flat_map.txt"),
            "--pop_file_path",
            os.path.join(BASE_DIR, "Demographic_Inference", "sampled_genome_windows", f"sim_{sim_number}", f"window_{window_number}", "samples.txt"),
            "--sim_directory",
            os.path.join(BASE_DIR, "Demographic_Inference", "LD_inferences"),
            "--sim_number",
            str(sim_number),
            "--window_number",
            str(window_number),
        ]
        subprocess.run(regenerate_window_command, check=True)
        print(f"Recomputed LD stats for simulation {sim_number} and window {window_number}.")

def reoptimize_with_retries(combined_ld_stats, p_guess, experiment_config, sim_number):
    """Attempt optimization with retries, handling exceptions."""
    def reoptimize():
        print("Attempting optimization...")
        opt_params_momentsLD, ll_list_momentsLD = run_inference_momentsLD(
            ld_stats=combined_ld_stats,
            demographic_model=experiment_config["demographic_model"],
            p_guess=p_guess,
            experiment_config=experiment_config
        )
        print("Optimization completed successfully.")
        print(f'The optimal parameters are: {opt_params_momentsLD}')
        return opt_params_momentsLD, ll_list_momentsLD

    try:
        return reoptimize()
    except (np.linalg.LinAlgError, KeyError) as e:
        print("================================================================================")
        print(f"Error encountered during optimization: {e}. Resimulating for sim_number={sim_number}.")
        print("================================================================================")
        raise  # Signal that the caller should handle resimulation

def obtain_feature(combined_ld_stats_path, sim_directory, sampled_params, experiment_config_filepath, sim_number):
    """Main function to infer momentsLD features and handle errors."""
    with open(experiment_config_filepath, "r") as f:
        experiment_config = json.load(f)

    with open(sampled_params, "rb") as f:
        sampled_params = pickle.load(f)

    with open(combined_ld_stats_path, "rb") as f:
        combined_ld_stats = pickle.load(f)

    mega_result_dict = {"simulated_params": sampled_params}

    # Initial guess for optimization
    p_guess = experiment_config["optimization_initial_guess"].copy()
    p_guess.extend([10000])  # Extend with additional parameters if needed
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1) #type:ignore

    # Attempt optimization with a retry mechanism
    try:
        opt_params_momentsLD, ll_list_momentsLD = reoptimize_with_retries(
            combined_ld_stats, p_guess, experiment_config, sim_number
        )
    except Exception as e:
        print(f"Exception occurred: {e}")
        cleanup_files(sim_number)
        resimulate(sim_number, experiment_config_filepath)

        # Retry optimization after resimulation
        opt_params_momentsLD, ll_list_momentsLD = reoptimize_with_retries(
            combined_ld_stats, p_guess, experiment_config, sim_number
        )

    # Store results in dictionary and save to a pickle file
    momentsLD_results = {
        "opt_params_momentsLD": opt_params_momentsLD,
        "ll_all_replicates_momentsLD": ll_list_momentsLD,
    }
    mega_result_dict.update(momentsLD_results)

    output_path = os.path.join(sim_directory, f"momentsLD_inferences_sim_{sim_number}.pkl")
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

    try:
        obtain_feature(
            combined_ld_stats_path=args.combined_ld_stats_path,
            sampled_params=args.sampled_params_pkl,
            experiment_config_filepath=args.experiment_config_filepath,
            sim_directory=args.sim_directory,
            sim_number=args.sim_number,
        )
    except Exception as e:
        print(f"Fatal error during feature generation: {e}")
        exit(1)
