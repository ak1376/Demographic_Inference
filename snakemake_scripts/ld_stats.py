from src.parameter_inference import get_LD_stats
import argparse
import numpy as np
import pickle
import cProfile
import pstats
import io

def ld_stat_creation(vcf_filepath, flat_map_path, pop_file_path, sim_directory, sim_number, window_number):
    # Start profiling
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # Your existing code
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    ld_stats = get_LD_stats(vcf_filepath, r_bins, flat_map_path, pop_file_path)
    
    with open(f"{sim_directory}/sim_{sim_number}/ld_stats_window.{window_number}.pkl", "wb") as f:
        pickle.dump(ld_stats, f)
    
    # # Stop profiling
    # profiler.disable()

    # # Create a string buffer to capture the stats output
    # s = io.StringIO()
    # stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    
    # # Print stats to the buffer, limiting to top 20 functions
    # stats.print_stats(20)
    
    # # Save both the raw stats and human-readable format
    # stats.dump_stats(f'{sim_directory}/sim_{sim_number}/profile_window_{window_number}.stats')
    
    # # Save the human-readable output to a text file
    # with open(f'{sim_directory}/sim_{sim_number}/profile_window_{window_number}.txt', 'w') as f:
    #     f.write(s.getvalue())
    #     # Add some additional information
    #     f.write("\n\nTotal time in minutes: {:.2f}".format(stats.total_tt / 60))
    
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