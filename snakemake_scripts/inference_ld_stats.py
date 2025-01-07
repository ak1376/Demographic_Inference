from src.parameter_inference import get_LD_stats
import argparse
import numpy as np
import pickle
import json
import tskit
import os
import shutil

# Function to create LD statistics
def ld_stat_creation(vcf_filepath, flat_map_path, pop_file_path, output_dir, window_number):
    # Define recombination bins
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    errors_to_retry = (IndexError, ValueError)  # Specific errors for retry logic

    os.makedirs(output_dir, exist_ok = True)

    try:
        print(f"Calculating LD stats for window {window_number}")

        # Calculate LD stats
        ld_stats = get_LD_stats(vcf_filepath, r_bins, flat_map_path, pop_file_path)

        # Save LD stats to a file
        os.makedirs(f"{output_dir}/window_{window_number}/", exist_ok = True)
        output_file = f"{output_dir}/window_{window_number}/ld_stats_window.{window_number}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(ld_stats, f)

        print(f"LD stats successfully created for window {window_number}")

    except Exception as e:
        print(f"Unexpected error: {e} for window {window_number}. Type: {type(e)}")
        print(f"Failed to create LD stats for window {window_number}.")

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LD statistics for specified simulation windows.")
    parser.add_argument("--vcf_filepath", type=str, required=True, help="Path to the VCF file containing simulated data")
    parser.add_argument("--flat_map_path", type=str, required=True, help="Path to the flat map file")
    parser.add_argument("--pop_file_path", type=str, required=True, help="Path to the population file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--window_number", type=int, required=True, help="Window number")
    args = parser.parse_args()

    # Run the LD statistics creation function
    ld_stat_creation(
        vcf_filepath=args.vcf_filepath,
        flat_map_path=args.flat_map_path,
        pop_file_path=args.pop_file_path,
        output_dir=args.output_dir,
        window_number=args.window_number
    )
