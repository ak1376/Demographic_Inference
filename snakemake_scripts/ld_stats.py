from src.parameter_inference import get_LD_stats
import argparse
import numpy as np
import pickle
import cProfile
import pstats
import io
import json

def ld_stat_creation(vcf_filepath, flat_map_path, pop_file_path, sim_directory, sim_number, window_number):
    try:
        r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
        ld_stats = get_LD_stats(vcf_filepath, r_bins, flat_map_path, pop_file_path)
        
        with open(f"{sim_directory}/sim_{sim_number}/ld_stats_window.{window_number}.pkl", "wb") as f:
            pickle.dump(ld_stats, f)
            
    except IndexError as e:
        print(f"Binning error in window {window_number}, sim number {sim_number} -- regenerating window...")
        # Code to regenerate the window
        from src.preprocess import Processor
        import json
        import tskit
        
        # Get your experiment config somehow - might need to pass it as an argument
        with open("/projects/kernlab/akapoor/Demographic_Inference/experiment_config.json", "r") as f:
            experiment_config = json.load(f)
        
        # Load in the tree sequence
        ts = tskit.load(f'/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/ts_sim_{sim_number}.trees')
            
        # Regenerate just this window
        Processor.run_msprime_replicates(ts, experiment_config, window_number, f"{sim_directory}/sim_{sim_number}")
        Processor.write_samples_and_rec_map(experiment_config, window_number, f"{sim_directory}/sim_{sim_number}")
        
        # Try calculating LD stats again with new window
        new_vcf = f"{sim_directory}/sim_{sim_number}/window_{window_number}/window.{window_number}.vcf.gz"
        new_map = f"{sim_directory}/sim_{sim_number}/window_{window_number}/flat_map.txt"
        ld_stats = get_LD_stats(new_vcf, r_bins, new_map, pop_file_path)
        
        with open(f"{sim_directory}/sim_{sim_number}/ld_stats_window.{window_number}.pkl", "wb") as f:
            pickle.dump(ld_stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf_filepath", type=str, help="The path to the VCF file that contains the simulated data")
    parser.add_argument("--flat_map_file", type=str, help="The path to the flat map file")
    parser.add_argument("--pop_file_path", type=str, help="The path to the population file")
    parser.add_argument("--sim_directory", type=str, help="The path to the simulation directory")
    parser.add_argument("--sim_number", type=int, help="The simulation number")
    parser.add_argument("--window_number", type=int, help="The window number")
    args = parser.parse_args()

    ld_stat_creation(args.vcf_filepath, args.flat_map_file, args.pop_file_path, args.sim_directory, args.sim_number, args.window_number)