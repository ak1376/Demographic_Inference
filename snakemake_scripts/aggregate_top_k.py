import argparse
import pickle
import heapq

def load_results(input_files):
    results = {"moments": [], "dadi": []}
    for file in input_files:
        with open(file, "rb") as f:
            result = pickle.load(f)
            # Classify results by analysis type based on file path
            if "moments" in file:
                results["moments"].append(result)
            elif "dadi" in file:
                results["dadi"].append(result)
    return results

def find_top_k_parameters(results, top_k, analysis):
    loglikelihood_key = "ll_" + analysis
    # Extract log-likelihood values and get top k indices
    loglikelihoods = [res[loglikelihood_key] for res in results if loglikelihood_key in res]
    top_k_indices = heapq.nlargest(top_k, range(len(loglikelihoods)), key=loglikelihoods.__getitem__)
    # Retrieve parameters and model SFS for top k entries
    top_k_data = {
        "loglikelihoods": [loglikelihoods[idx] for idx in top_k_indices],
        "opt_params": [results[idx]["opt_params_" + analysis] for idx in top_k_indices],
        "model_sfs": [results[idx]["model_sfs_" + analysis] for idx in top_k_indices],
        "opt_theta": [results[idx]["opt_theta_" + analysis] for idx in top_k_indices],
    }
    return top_k_data

def main(input_files, top_k, output_file):
    results = load_results(input_files)
    
    # Initialize dictionary to store the required keys
    aggregated_data = {
        "simulated_params": None,
        "sfs": None,
        "model_sfs_dadi": [],
        "opt_theta_dadi": [],
        "opt_params_dadi": [],
        "ll_dadi": [],
        "model_sfs_moments": [],
        "opt_theta_moments": [],
        "opt_params_moments": [],
        "ll_moments": []
    }

    # Assuming simulated parameters and SFS data are the same across replicates, extract from the first entry of each analysis type
    for analysis in ["moments", "dadi"]:
        top_k_data = find_top_k_parameters(results[analysis], top_k, analysis)

        # Populate the corresponding fields in aggregated_data
        if analysis == "dadi":
            aggregated_data["model_sfs_dadi"] = top_k_data["model_sfs"]
            aggregated_data["opt_theta_dadi"] = top_k_data["opt_theta"]
            aggregated_data["opt_params_dadi"] = top_k_data["opt_params"]
            aggregated_data["ll_dadi"] = top_k_data["loglikelihoods"]
        elif analysis == "moments":
            aggregated_data["model_sfs_moments"] = top_k_data["model_sfs"]
            aggregated_data["opt_theta_moments"] = top_k_data["opt_theta"]
            aggregated_data["opt_params_moments"] = top_k_data["opt_params"]
            aggregated_data["ll_moments"] = top_k_data["loglikelihoods"]

    # Extract common fields from one of the replicates (assuming these are consistent across replicates)
    sample_result = results["dadi"][0] if results["dadi"] else results["moments"][0]
    aggregated_data["simulated_params"] = sample_result.get("simulated_params", None)
    aggregated_data["sfs"] = sample_result.get("sfs", None)

    # Save the combined data as a single pickle file
    with open(output_file, "wb") as f:
        pickle.dump(aggregated_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs='+', type=str, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    main(args.input_files, args.top_k, args.output_file)
