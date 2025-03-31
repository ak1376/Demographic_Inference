#!/usr/bin/env python3

import argparse
import pickle
import dadi
import json
import os
from src.parameter_inference import run_inference_dadi, run_inference_moments

def create_joint_sfs(vcf_file, pop_file, popnames, num_samples, polarized, output_file):
    """
    Create a Joint Site Frequency Spectrum (SFS) from a VCF file for two populations.

    Parameters:
        vcf_file (str): Path to the VCF file.
        pop_file (str): Path to the population file.
        popnames (list): List of two population names to use from the population file.
        num_samples (list): List of sample counts for the two populations (diploid individuals).
        polarized (bool): Whether to create a polarized SFS.
        output_file (str): Path to save the SFS pickle file.
    """
    print(f"Creating data dictionary from VCF: {vcf_file} and population file: {pop_file}")
    dd = dadi.Misc.make_data_dict_vcf(vcf_file, pop_file)

    print(f"Generating Joint SFS for populations: {popnames}")
    sfs = dadi.Spectrum.from_data_dict(
        dd,
        popnames,  # Two populations
        projections=[2 * num_samples[0], 2 * num_samples[1]],  # Projection sizes
        polarized=polarized,  # Polarization setting
    )

    print(f"Saving Joint SFS to file: {output_file}")
    # Use pickle.dump to save the SFS as a binary pickle file
    with open(output_file, "wb") as f:
        pickle.dump(sfs, f)

    print("Joint SFS creation completed successfully.")


def obtain_feature(SFS, experiment_config, output_dir):
    """
    Perform dadi and moments demographic inference on the provided SFS.

    Parameters:
        SFS (str): Path to the SFS file.
        experiment_config (str): Path to the experiment configuration JSON file.
        output_dir (str): Directory to save inference results.
    """
    # Ensure output directories exist
    dadi_dir = os.path.join(output_dir, "dadi")
    moments_dir = os.path.join(output_dir, "moments")
    os.makedirs(dadi_dir, exist_ok=True)
    os.makedirs(moments_dir, exist_ok=True)

    # Load experiment configuration
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    # Load the SFS
    with open(SFS, "rb") as f:
        SFS = pickle.load(f)

    # Extract bounds
    upper_bound = [b if b is not None else None for b in experiment_config['upper_bound_optimization']]
    lower_bound = [b if b is not None else None for b in experiment_config['lower_bound_optimization']]

    # Perform dadi analysis
    if experiment_config["dadi_analysis"]:
        model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = run_inference_dadi(
            sfs=SFS,
            p0=experiment_config['optimization_initial_guess'],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_samples=30,
            demographic_model=experiment_config['demographic_model'],
            mutation_rate=experiment_config['mutation_rate'],
            length=experiment_config['genome_length']
        )
        dadi_results = {
            "model_sfs_dadi": model_sfs_dadi,
            "opt_theta_dadi": opt_theta_dadi,
            "opt_params_dadi": opt_params_dict_dadi,
            "ll_dadi": opt_params_dict_dadi['ll'],
        }
        with open(os.path.join(dadi_dir, "dadi_results.pkl"), "wb") as f:
            pickle.dump(dadi_results, f)
        print("Dadi inference completed and saved.")

    # Perform moments analysis
    if experiment_config["moments_analysis"]:
        model_sfs_moments, opt_theta_moments, opt_params_dict_moments = run_inference_moments(
            sfs=SFS,
            p0=experiment_config['optimization_initial_guess'],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            demographic_model=experiment_config['demographic_model'],
            use_FIM=experiment_config["use_FIM"],
            mutation_rate=experiment_config['mutation_rate'],
            length=experiment_config['genome_length']
        )
        moments_results = {
            "model_sfs_moments": model_sfs_moments,
            "opt_theta_moments": opt_theta_moments,
            "opt_params_moments": opt_params_dict_moments,
            "ll_moments": opt_params_dict_moments['ll']
        }
        with open(os.path.join(moments_dir, "moments_results.pkl"), "wb") as f:
            pickle.dump(moments_results, f)
        print("Moments inference completed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Joint SFS and perform dadi/moments inference.")
    parser.add_argument("--vcf_file", type=str, required=True, help="Path to the VCF file.")
    parser.add_argument("--pop_file", type=str, required=True, help="Path to the population file.")
    parser.add_argument("--popnames", type=str, nargs=2, required=True,
                        help="Two population names to use from the population file (e.g., pop1 pop2).")
    parser.add_argument("--num_samples", type=int, nargs=2, required=True,
                        help="Number of diploid individuals in each population (e.g., 10 8).")
    parser.add_argument("--polarized", action="store_true", help="Create a polarized SFS (default: unpolarized).")
    parser.add_argument("--sfs_file", type=str, required=True, help="Path to save the Joint SFS pickle file.")
    parser.add_argument("--experiment_config_filepath", type=str, required=True, help="Path to the experiment configuration JSON.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results.")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Create the Joint SFS
    create_joint_sfs(
        vcf_file=args.vcf_file,
        pop_file=args.pop_file,
        popnames=args.popnames,
        num_samples=args.num_samples,
        polarized=args.polarized,
        output_file=args.sfs_file,
    )

    # Step 2: Perform dadi/moments inference
    obtain_feature(
        SFS=args.sfs_file,
        experiment_config=args.experiment_config_filepath,
        output_dir=args.output_dir,
    )