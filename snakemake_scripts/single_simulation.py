from src.preprocess import Processor
import json
import os
import pickle
import argparse
import src.demographic_models as demographic_models


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
        raise ValueError(f"Unsupported demographic model: {experiment_config['demographic_model']}")

    # Now simulate the chromosome
    ts = processor.simulate_chromosome(sampled_params, num_samples, demographic_model, length=experiment_config['genome_length'], mutation_rate=experiment_config['mutation_rate'], recombination_rate = experiment_config['recombination_rate'])

    # Now create the SFS
    
    SFS = processor.create_SFS(
        ts, mode = 'pretrain', num_samples = experiment_config["num_samples"]
    )

    # Save the SFS in a .pkl file
    SFS_filename = f'{simulation_results_directory}/SFS_sim_{sim_number}.pkl'

    print("=================================================================")
    print(f'SFS filename path: {SFS_filename}')
    print("=================================================================")
    
    with open(SFS_filename, "wb") as f:
        pickle.dump(SFS, f)

    # Save the unique identifier in a .pkl file

    if demographic_model == demographic_models.bottleneck_model:
        sampled_params = {key: sampled_params[key] for key in experiment_config['parameters_to_estimate'] if key in sampled_params}

    pkl_filename = f"{simulation_results_directory}/sampled_params_{sim_number}.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(sampled_params, f)

    # Also save the name of the .pkl file to metadata_{sim_number}.txt
    metadata_filename = f"{simulation_results_directory}/metadata_{sim_number}.txt"
    with open(metadata_filename, "w") as meta_f:
        meta_f.write(pkl_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    main(args.experiment_config, args.sim_directory, args.sim_number)
