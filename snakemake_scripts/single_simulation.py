from src.preprocess import Processor
import json
import os
import pickle
import argparse
import src.demographic_models as demographic_models
import tskit  # Add this import

def safe_save_tree_sequence(ts, final_filename):
    """Ensure atomic write of tree sequence"""
    temp_filename = final_filename + '.tmp'
    try:
        # Write to temporary file first
        ts.dump(temp_filename)
        # Verify the file
        test_ts = tskit.load(temp_filename)
        if test_ts.num_trees > 0:
            # Move temp file to final location atomically
            os.rename(temp_filename, final_filename)
            return True
    except Exception as e:
        print(f"Error saving tree sequence: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    return False


def main(experiment_config, sim_directory, sim_number):

    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    # Create the subdirectory to store the results
    simulation_results_directory = os.path.join(sim_directory, "simulation_results")
    os.makedirs(simulation_results_directory, exist_ok=True)

    # Load the experiment config and run the simulation (as before)
    processor = Processor(
        experiment_config,
        experiment_directory=sim_directory,
        recombination_rate=experiment_config["recombination_rate"],
        mutation_rate=experiment_config["mutation_rate"],
    )
    sampled_params = processor.sample_params()

    if experiment_config["demographic_model"] == "bottleneck_model":
        demographic_model = demographic_models.bottleneck_model

    elif experiment_config["demographic_model"] == "split_isolation_model":
        demographic_model = demographic_models.split_isolation_model_simulation

    else:
        raise ValueError(
            f"Unsupported demographic model: {experiment_config['demographic_model']}"
        )
    

    print("BEGINNING THE PROCESS OF SIMULATING THE CHROMOSOME")

    # Now simulate the chromosome
    ts = processor.simulate_chromosome(
        experiment_config, # TODO: temporary 
        sampled_params,
        demographic_model=demographic_model,
        length=experiment_config["genome_length"],
        mutation_rate=experiment_config["mutation_rate"],
        recombination_rate=experiment_config["recombination_rate"],
    )

    # Now create the SFS

    SFS = processor.create_SFS(
        ts, mode="pretrain", num_samples=experiment_config["num_samples"], length = experiment_config["genome_length"]
    )

    # Save the SFS
    SFS_filename = f"{simulation_results_directory}/SFS_sim_{sim_number}.pkl"
    with open(SFS_filename, "wb") as f:
        pickle.dump(SFS, f)

    # Save the tree sequence using the new safe method
    ts_filename = f"{simulation_results_directory}/ts_sim_{sim_number}.trees"
    if not safe_save_tree_sequence(ts, ts_filename):
        raise RuntimeError(f"Failed to save tree sequence for simulation {sim_number}")

    # Save the unique identifier in a .pkl file

    if demographic_model == demographic_models.bottleneck_model:
        sampled_params = {
            key: sampled_params[key]
            for key in experiment_config["parameters_to_estimate"]
            if key in sampled_params
        }

    pkl_filename = f"{simulation_results_directory}/sampled_params_{sim_number}.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(sampled_params, f)

    # Also save the name of the .pkl file to metadata_{sim_number}.txt
    metadata_filename = f"{simulation_results_directory}/sampled_params_metadata_{sim_number}.txt"
    with open(metadata_filename, "w") as meta_f:
        meta_f.write(pkl_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    main(args.experiment_config, args.sim_directory, args.sim_number)
