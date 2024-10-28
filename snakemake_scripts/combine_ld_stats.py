import argparse
import pickle
import os

def combine_ld_stats(LD_inferences_path, sim_number):

    # Print all the files in the directory to see what's there
    all_files = os.listdir(LD_inferences_path)
    
    # Use os.path.join to get the absolute paths for the files and filter for 'ld_stats' files
    ld_stats_files = [os.path.join(LD_inferences_path, f) for f in all_files if f.startswith('ld_stats_window') and f.endswith('.pkl')]

    ld_stats = {}
    
    # Load and combine all LD stats pickle files
    for ii, ld_stats_file in enumerate(ld_stats_files):
        print(f"Processing file: {ld_stats_file}")
        with open(ld_stats_file, 'rb') as f:
            file_stats = pickle.load(f)
            ld_stats[ii] = file_stats
    
    # Save the combined results
    combined_file_path = os.path.join('combined_LD_inferences', f'sim_{sim_number}', f'combined_LD_stats_sim_{sim_number}.pkl')
    with open(combined_file_path, 'wb') as f:
        pickle.dump(ld_stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--LD_inferences_path", type=str, help="Contains the simulation directory for this particular simulation")
    parser.add_argument("--sim_number", type=int, help="The simulation number")
    args = parser.parse_args()

    combine_ld_stats(args.LD_inferences_path, args.sim_number)