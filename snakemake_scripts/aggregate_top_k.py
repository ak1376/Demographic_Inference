import argparse
import pickle
import heapq
import os

def load_results_from_files(dadi_files, moments_files, sfs_file, params_file):
    # Initialize the results dictionary
    results = {"moments": [], "dadi": []}
    
    # Load dadi results
    for file in dadi_files:
        with open(file, "rb") as f:
            results["dadi"].append(pickle.load(f))
            
    # Load moments results
    for file in moments_files:
        with open(file, "rb") as f:
            results["moments"].append(pickle.load(f))
    
    # Load simulated parameters and SFS
    with open(params_file, "rb") as f:
        simulated_params = pickle.load(f)
    with open(sfs_file, "rb") as f:
        sfs = pickle.load(f)
        
    return results, simulated_params, sfs

def find_top_k_results(results, top_k, method):
    # Get likelihoods
    if method == "dadi":
        lls = [res["ll_dadi"] for res in results]
    else:  # moments
        lls = [res["ll_moments"] for res in results]
        
    # Get top k indices
    top_k_indices = heapq.nlargest(top_k, range(len(lls)), key=lls.__getitem__)
    
    return {
        "lls": [lls[i] for i in top_k_indices],
        "model_sfs": [results[i][f"model_sfs_{method}"] for i in top_k_indices],
        "opt_theta": [results[i][f"opt_theta_{method}"] for i in top_k_indices],
        "opt_params": [results[i][f"opt_params_{method}"] for i in top_k_indices]
    }

def main(dadi_files, moments_files, sfs_file, params_file, top_k, sim_number):
    # Load all results
    results, simulated_params, sfs = load_results_from_files(dadi_files, moments_files, sfs_file, params_file)
    
    # Initialize aggregated data dictionary
    aggregated_data = {
        "simulated_params": simulated_params,
        "sfs": sfs,
        "model_sfs_dadi": [],
        "opt_theta_dadi": [],
        "opt_params_dadi": [],
        "ll_all_replicates_dadi": [],
        "model_sfs_moments": [],
        "opt_theta_moments": [],
        "opt_params_moments": [],
        "ll_all_replicates_moments": []
    }
    
    # Process dadi results
    dadi_top_k = find_top_k_results(results["dadi"], top_k, "dadi")
    aggregated_data["model_sfs_dadi"] = dadi_top_k["model_sfs"]
    aggregated_data["opt_theta_dadi"] = dadi_top_k["opt_theta"]
    aggregated_data["opt_params_dadi"] = dadi_top_k["opt_params"]
    aggregated_data["ll_all_replicates_dadi"] = dadi_top_k["lls"]
    
    # Process moments results
    moments_top_k = find_top_k_results(results["moments"], top_k, "moments")
    aggregated_data["model_sfs_moments"] = moments_top_k["model_sfs"]
    aggregated_data["opt_theta_moments"] = moments_top_k["opt_theta"]
    aggregated_data["opt_params_moments"] = moments_top_k["opt_params"]
    aggregated_data["ll_all_replicates_moments"] = moments_top_k["lls"]
    
    # Save aggregated results
    base_directory = os.getcwd()
    output_path = os.path.join(base_directory, f'software_inferences_sim_{sim_number}.pkl')
    with open(output_path, "wb") as f:
        pickle.dump(aggregated_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dadi_files", nargs='+', type=str, required=True)
    parser.add_argument("--moments_files", nargs='+', type=str, required=True)
    parser.add_argument("--sfs_file", type=str, required=True)
    parser.add_argument("--params_file", type=str, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    
    args = parser.parse_args()
    
    main(args.dadi_files, 
         args.moments_files, 
         args.sfs_file, 
         args.params_file, 
         args.top_k, 
         args.sim_number)