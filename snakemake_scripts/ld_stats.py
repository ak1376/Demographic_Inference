from src.parameter_inference import get_LD_stats
import argparse
import numpy as np
import pickle
import json
import tskit
import os
import shutil

# Function to create LD statistics
def ld_stat_creation(vcf_filepath, flat_map_path, pop_file_path, sim_directory, sim_number, window_number):
    # Define recombination bins
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    errors_to_retry = (IndexError, ValueError)  # Specific errors for retry logic

    try:
        print(f"Calculating LD stats for window {window_number}, sim {sim_number}")

        # Calculate LD stats
        ld_stats = get_LD_stats(vcf_filepath, r_bins, flat_map_path, pop_file_path)

        # Save LD stats to a file
        os.makedirs(f"{sim_directory}/sim_{sim_number}/window_{window_number}/", exist_ok = True)
        output_file = f"{sim_directory}/sim_{sim_number}/window_{window_number}/ld_stats_window.{window_number}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(ld_stats, f)

        print(f"LD stats successfully created for window {window_number}, sim {sim_number}")
    except errors_to_retry as e:
        print(f"Error encountered ({e}) for window {window_number}, sim {sim_number}. Regenerating the window...")

        # First delete the windows
        dir_path = f"/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}"
        print("========================================================")
        print(f'WINDOWS PATH TO REMOVE: {dir_path}')
        print("========================================================")

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # Recursively delete the directory and its contents
            print(f"Deleted directory and all contents: {dir_path}")
        else:
            print(f"Directory not found: {dir_path}")
                    
        # Delete the LD stats that correspond to each window
        [os.remove(os.path.join(f"/projects/kernlab/akapoor/Demographic_Inference/LD_inferences/sim_{sim_number}", f)) for f in os.listdir(f"/projects/kernlab/akapoor/Demographic_Inference/LD_inferences/sim_{sim_number}") if os.path.isfile(os.path.join(f"/projects/kernlab/akapoor/Demographic_Inference/LD_inferences/sim_{sim_number}", f))]

        from src.preprocess import Processor

        # Reload experiment configuration
        with open("/projects/kernlab/akapoor/Demographic_Inference/experiment_config.json", "r") as f:
            experiment_config = json.load(f)

        # Load the tree sequence
        ts_path = f"/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/ts_sim_{sim_number}.trees"
        ts = tskit.load(ts_path)

        genome_window_dir = f"/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}"

        # Regenerate the window
        Processor.run_msprime_replicates(ts, experiment_config, window_number, genome_window_dir)
        Processor.write_samples_and_rec_map(experiment_config, window_number, genome_window_dir)

        # Update file paths for the regenerated window
        new_vcf_filepath = f"{genome_window_dir}/window_{window_number}/window.{window_number}.vcf.gz"
        new_flat_map_path = f"{genome_window_dir}/window_{window_number}/flat_map.txt"

        print(f"Retrying LD stats calculation for regenerated window {window_number}, sim {sim_number}")
        os.makedirs(f"{sim_directory}/sim_{sim_number}/window_{window_number}/", exist_ok = True)
        ld_stat_creation(vcf_filepath= new_vcf_filepath, 
        flat_map_path = new_flat_map_path, 
        pop_file_path = pop_file_path, 
        sim_directory = sim_directory, 
        sim_number = sim_number, 
        window_number = window_number)

    except Exception as e:
        print(f"Unexpected error: {e} for window {window_number}, sim {sim_number}. Type: {type(e)}")
        print(f"Failed to create LD stats for window {window_number}, sim {sim_number}.")

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LD statistics for specified simulation windows.")
    parser.add_argument("--vcf_filepath", type=str, required=True, help="Path to the VCF file containing simulated data")
    parser.add_argument("--flat_map_path", type=str, required=True, help="Path to the flat map file")
    parser.add_argument("--pop_file_path", type=str, required=True, help="Path to the population file")
    parser.add_argument("--sim_directory", type=str, required=True, help="Path to the simulation directory")
    parser.add_argument("--sim_number", type=int, required=True, help="Simulation number")
    parser.add_argument("--window_number", type=int, required=True, help="Window number")
    args = parser.parse_args()

    # Run the LD statistics creation function
    ld_stat_creation(
        vcf_filepath=args.vcf_filepath,
        flat_map_path=args.flat_map_path,
        pop_file_path=args.pop_file_path,
        sim_directory=args.sim_directory,
        sim_number=args.sim_number,
        window_number=args.window_number
    )
