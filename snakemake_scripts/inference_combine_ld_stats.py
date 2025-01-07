import argparse
import pickle
import os

def combine_ld_stats(ld_stats_dir, output_dir):
    """
    Combines LD stats from multiple pickle files into a single dictionary and saves the result.
    
    Parameters:
    - ld_stats_dir (str): Directory containing window folders with LD stats pickle files.
    - output_dir (str): Directory to save the combined LD stats pickle file.
    """
    ld_stats = {}

    # Get all the window directories in the specified folder
    window_dirs = sorted([d for d in os.listdir(ld_stats_dir) if d.startswith("window_")])
    
    print(f"Found {len(window_dirs)} window directories.")
    
    # Process each window directory
    for window_dir in window_dirs:
        window_path = os.path.join(ld_stats_dir, window_dir)
        pkl_file = os.path.join(window_path, f"ld_stats_window.{window_dir.split('_')[1]}.pkl")

        if not os.path.exists(pkl_file):
            print(f"Warning: File {pkl_file} not found. Skipping.")
            continue

        print(f"Processing file: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            file_stats = pickle.load(f)
            # Use the window number as the key
            window_number = int(window_dir.split('_')[1])
            ld_stats[window_number] = file_stats

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    combined_file_path = os.path.join(output_dir, 'combined_LD_stats.pkl')

    # Save the combined results
    with open(combined_file_path, 'wb') as f:
        pickle.dump(ld_stats, f)
    
    print(f"Combined LD stats saved to {combined_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine LD stats from multiple pickle files into one.")
    parser.add_argument("--ld_stats_dir", type=str, required=True, help="Directory containing the LD stats window folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the combined LD stats pickle file.")
    args = parser.parse_args()

    combine_ld_stats(args.ld_stats_dir, args.output_dir)
