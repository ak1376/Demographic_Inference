import random
import gzip
import os
import json
import argparse

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
    os.makedirs(output_dir, exist_ok=True)

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

    window_size = config["window_length"]
    random_contig = random.choice(list(contigs.keys()))
    contig_length = contigs[random_contig]
    start = random.randint(1, contig_length - window_size)
    end = start + window_size

    # Step 3: Extract the variants in the window
    output_vcf = os.path.join(output_dir, f"window.{window_number}.vcf.gz")
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
    return output_vcf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a random genomic window from a VCF file.")
    parser.add_argument("--vcf_file", type=str, required=True, help="Path to the input VCF file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output VCF file.")
    parser.add_argument("--experiment_config", type=str, required=True, help="Path to the experiment config JSON file.")
    parser.add_argument("--window_number", type=int, required=True, help="Window number for the output file name.")
    args = parser.parse_args()

    extract_window(args.vcf_file, args.output_dir, args.experiment_config, args.window_number)
