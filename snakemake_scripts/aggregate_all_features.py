import argparse
import pickle
import numpy as np
import json
import pandas as pd

def main(experiment_config_file, sim_directory, inferences_file_list):
    '''
    Aggregates the software and momentsLD inferences for each simulation, ensuring training and validation data are kept separate.
    '''
    # Load the experiment configuration
    with open(experiment_config_file, "r") as f:
        experiment_config = json.load(f)

    # Split the data into training and validation sets
    all_indices = np.arange(experiment_config["num_sims_pretrain"])
    np.random.shuffle(all_indices)
    training_indices = all_indices[:int(experiment_config["training_percentage"] * experiment_config["num_sims_pretrain"])]
    validation_indices = all_indices[int(experiment_config["training_percentage"] * experiment_config["num_sims_pretrain"]):]

    preprocessing_results_obj = {
        stage: {} for stage in ["training", "validation"]
    }

    # Separate loop to process each stage (training/validation)
    for stage, indices in [
        ("training", training_indices),
        ("validation", validation_indices)
    ]:
        
        # Step 1: Initialize lists to hold simulation data
        all_simulations_data = []   # Inferred parameters
        all_targets_data = []       # Simulated parameters (targets)

        # Step 2: Dynamically extract and append data for each analysis type
        for idx in indices:
            sim_data = {}  # Dictionary to hold inferred parameters for each simulation
            target_data = {}  # Dictionary to hold target parameters for each simulation

            inference_file = inferences_file_list[idx]
            print(f"Processing inference file: {inference_file}")

            # Handle loading of inferences
            with open(inference_file, "rb") as f:
                result = pickle.load(f)

            print(result.keys())  # Debugging step to check the keys in the result

            # Collect moments_analysis data
            if experiment_config['moments_analysis']:
                for replicate, params in enumerate(result.get('opt_params_moments', [])):
                    for key, value in params.items():
                        if key != 'upper_triangular_FIM':
                            sim_data[f'Moments_rep{replicate+1}_{key}'] = value
                    
                    # Handle upper triangular matrix for FIM
                    if experiment_config['use_FIM'] and 'upper_triangular_FIM' in params:
                        upper_triangular = params['upper_triangular_FIM']
                        flat_fim = upper_triangular.flatten()
                        for idx, fim_value in enumerate(flat_fim):
                            sim_data[f'Moments_rep{replicate+1}_FIM_element_{idx}'] = fim_value

            # Collect momentsLD_analysis data
            if experiment_config['momentsLD_analysis']:
                if 'opt_params_momentsLD' in result and result['opt_params_momentsLD']:
                    print(f"Found momentsLD params for sim {idx}")  # Debugging output
                    for key, value in result['opt_params_momentsLD'][0].items():
                        sim_data[f'MomentsLD_{key}'] = value
                else:
                    print(f"No momentsLD params found for sim {idx}")  # Debugging output

            # Collect dadi_analysis data
            if experiment_config['dadi_analysis']:
                for replicate, params in enumerate(result.get('opt_params_dadi', [])):
                    for key, value in params.items():
                        sim_data[f'Dadi_rep{replicate+1}_{key}'] = value

            # Collect simulated_params (targets)
            for key, value in result.get('simulated_params', {}).items():
                target_data[f'simulated_params_{key}'] = value

            # Append the inferred parameters and targets to the respective lists
            all_simulations_data.append(sim_data)
            all_targets_data.append(target_data)

        # Step 3: Create DataFrames from the simulation data
        features_df = pd.DataFrame(all_simulations_data, index=[f'Sim_{i}' for i in range(len(indices))])
        targets_df = pd.DataFrame(all_targets_data, index=[f'Sim_{i}' for i in range(len(indices))])

        # Store the DataFrames in the preprocessing object for later use
        preprocessing_results_obj[stage]["predictions"] = features_df
        preprocessing_results_obj[stage]["targets"] = targets_df
        preprocessing_results_obj['parameter_names'] = list(targets_df.columns)  # type:ignore

        # Save DataFrames for each stage (training or validation)
        features_df.to_csv(f"{sim_directory}/{stage}_features.csv", index=True)
        targets_df.to_csv(f"{sim_directory}/{stage}_targets.csv", index=True)

        # Save the DataFrames as .npy files for further processing
        np.save(f"{sim_directory}/{stage}_features.npy", features_df.values)
        np.save(f"{sim_directory}/{stage}_targets.npy", targets_df.values)

    # Save the preprocessing results object (optional)
    with open(f"{sim_directory}/preprocessing_results_obj.pkl", "wb") as file:
        pickle.dump(preprocessing_results_obj, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config_file", type=str, help="Path to the experiment config file")
    parser.add_argument("sim_directory", type=str, help="Path to the simulation directory")
    parser.add_argument("inferences_file_list", type=str, nargs='+', help="List of filepaths to the inferences results")
    args = parser.parse_args()
    main(args.experiment_config_file, args.sim_directory, args.inferences_file_list)
