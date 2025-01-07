import random
import gzip
import os
import json
import argparse
from src.preprocess import Processor

def extract_window(vcf_file, output_dir, experiment_config, window_number):
    """
    Extracts a random genomic window from a VCF file and saves it to a new file.

    Args:
        vcf_file (str): Path to the input VCF file (gzip-compressed).
        output_dir (str): Directory to save the output VCF file.
        experiment_config (str): Path to the JSON file containing the window size.
        window_number (int): Identifier for the window (used in the output file name).

    Returns:
        str: Path to the output VCF file.
    """
    # Ensure output directory exists
    output_dir_window = os.path.join(output_dir, f'window_{window_number}')
    print(f'The output dir for this window is : {output_dir}')
    os.makedirs(output_dir_window, exist_ok=True)

    # Step 1: Identify the contigs and lengths
    contigs = {}
    with gzip.open(vcf_file, "rt") as f:
        for line in f:
            if line.startswith("##contig"):
                parts = line.strip().split("<")[1].strip(">").split(",")
                contig_name = parts[0].split("=")[1]
                contig_length = int(parts[1].split("=")[1])
                contigs[contig_name] = contig_length
            elif line.startswith("#CHROM"):
                break

    # Step 2: Load the window size from the experiment config
    with open(experiment_config, "r") as f:
        config = json.load(f)

    window_size = int(config["window_length"])  # Ensure window_size is an integer
    random_contig = random.choice(list(contigs.keys()))
    contig_length = int(contigs[random_contig])  # Ensure contig_length is an integer

    # Prevent errors if the contig length is smaller than the window size
    if contig_length <= window_size:
        raise ValueError(f"Contig {random_contig} is too short ({contig_length} bp) for window size {window_size}.")

    start = random.randint(1, contig_length - window_size)  # This will now work
    end = start + window_size

    print(f"Extracting window: {random_contig}:{start}-{end}")

    # Step 3: Extract the variants in the window
    output_vcf = os.path.join(output_dir_window, f"window.{window_number}.vcf.gz")
    with gzip.open(vcf_file, "rt") as f_in, gzip.open(output_vcf, "wt") as f_out:
        for line in f_in:
            if line.startswith("#"):
                # Write header lines to the output VCF
                f_out.write(line)
            else:
                # Parse the variant line
                parts = line.strip().split("\t")
                chrom = parts[0]
                pos = int(parts[1])

                # Check if the variant falls within the selected window
                if chrom == random_contig and start <= pos <= end:
                    f_out.write(line)

    print(f"Random window saved to: {output_vcf}")

    Processor.write_samples_and_rec_map(config, window_number, output_dir)

    return output_vcf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a random genomic window from a VCF file.")
    parser.add_argument("--vcf_file", type=str, required=True, help="Path to the input VCF file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output VCF file.")
    parser.add_argument("--experiment_config", type=str, required=True, help="Path to the experiment config JSON file.")
    parser.add_argument("--window_number", type=int, required=True, help="Window number for the output file name.")
    args = parser.parse_args()

    extract_window(args.vcf_file, args.output_dir, args.experiment_config, args.window_number)
