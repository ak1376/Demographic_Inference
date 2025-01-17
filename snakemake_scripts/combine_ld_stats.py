import argparse
import pickle
import os

def combine_ld_stats(ld_stats_files, sim_number):

    ld_stats = {}

    # Read the file paths from the provided file-like object
    with open(ld_stats_files, 'r') as f:
        ld_stats_files = [line.strip() for line in f if line.strip()]

    print(f"Processing simulation {args.sim_number}")
    print(f"LD Stats Files: {ld_stats_files}")

    # Load and combine all LD stats pickle files
    for ii, ld_stats_file in enumerate(ld_stats_files):
        print(f"Processing file: {ld_stats_file}")
        with open(ld_stats_file, 'rb') as f:
            file_stats = pickle.load(f)
            ld_stats[ii] = file_stats
    
    # Save the combined results
    base_directory = os.getcwd()
    combined_file_path = os.path.join(base_directory, f'sim_{sim_number}', f'combined_LD_stats_sim_{sim_number}.pkl')
    with open(combined_file_path, 'wb') as f:
        pickle.dump(ld_stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ld_stats_files", type=str, help="List of the pkl window LD stat files")
    parser.add_argument("--sim_number", type=int, help="The simulation number")
    args = parser.parse_args()

    combine_ld_stats(args.ld_stats_files, args.sim_number)