#!/usr/bin/env python3

import pandas as pd
import pickle
import os
import argparse

def load_results(pickle_file, prefix=None):
    """
    Load results from a pickle file and optionally prefix the keys.
    
    Parameters:
        pickle_file (str): Path to the pickle file.
        prefix (str): Optional prefix for column names.
    
    Returns:
        pd.DataFrame: DataFrame with the loaded results.
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in {pickle_file}, but got {type(data)}.")

    # Convert to DataFrame and optionally add prefixes
    df = pd.DataFrame([data])  # Create a single-row DataFrame
    if prefix:
        df = df.add_prefix(f"{prefix}_")
    
    return df

def combine_results(dadi_path, moments_path, momentsld_path, output_file):
    """
    Combine results from dadi, moments, and momentsLD into a single DataFrame.
    
    Parameters:
        dadi_path (str): Path to the dadi results pickle file.
        moments_path (str): Path to the moments results pickle file.
        momentsld_path (str): Path to the momentsLD results pickle file.
        output_file (str): Path to save the combined DataFrame as a CSV file.
    """
    # Load dadi results
    print(f"Loading dadi results from {dadi_path}...")
    dadi_df = load_results(dadi_path, prefix="dadi")

    # Load moments results
    print(f"Loading moments results from {moments_path}...")
    moments_df = load_results(moments_path, prefix="moments")

    # Load momentsLD results
    print(f"Loading momentsLD results from {momentsld_path}...")
    momentsld_df = load_results(momentsld_path, prefix="momentsLD")

    # Concatenate all DataFrames columnwise
    print("Concatenating results...")
    combined_df = pd.concat([dadi_df, moments_df, momentsld_df], axis=1)

    # Save the combined DataFrame to a CSV file
    print(f"Saving combined results to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    print("Combined results saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine dadi, moments, and momentsLD inference results into a single DataFrame.")
    parser.add_argument("--dadi_path", type=str, required=True, help="Path to the dadi results pickle file.")
    parser.add_argument("--moments_path", type=str, required=True, help="Path to the moments results pickle file.")
    parser.add_argument("--momentsld_path", type=str, required=True, help="Path to the momentsLD results pickle file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the combined results as a CSV file.")
    
    args = parser.parse_args()

    # Combine the results
    combine_results(
        dadi_path=args.dadi_path,
        moments_path=args.moments_path,
        momentsld_path=args.momentsld_path,
        output_file=args.output_file,
    )
