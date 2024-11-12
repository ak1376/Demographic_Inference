import argparse
import pickle
import heapq

def load_results(input_files):
    results = {"moments": [], "dadi": []}
    for file in input_files:
        with open(file, "rb") as f:
            result = pickle.load(f)
            # Sort files into moments or dadi based on their directory path
            if "moments" in file:
                results["moments"].append(result)
            elif "dadi" in file:
                results["dadi"].append(result)
    return results

def find_top_k_parameters(results, top_k, analysis):
    loglikelihood_key = "ll_" + analysis
    # Extract log-likelihood values
    loglikelihoods = [res[loglikelihood_key] for res in results if loglikelihood_key in res]
    # Get top k indices
    top_k_indices = heapq.nlargest(top_k, range(len(loglikelihoods)), key=loglikelihoods.__getitem__)
    # Retrieve parameters for the top k indices
    top_k_parameters = [{"loglikelihood": loglikelihoods[idx], "parameters": results[idx]["opt_params_" + analysis]} for idx in top_k_indices]
    return top_k_parameters

def main(input_files, top_k, output_file):
    results = load_results(input_files)
    aggregated_data = {}

    for analysis in ["moments", "dadi"]:
        top_k_parameters = find_top_k_parameters(results[analysis], top_k, analysis)
        aggregated_data[analysis] = top_k_parameters

    # Save combined results for moments and dadi in one output file
    with open(output_file, "wb") as f:
        pickle.dump(aggregated_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs='+', type=str, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    main(args.input_files, args.top_k, args.output_file)
