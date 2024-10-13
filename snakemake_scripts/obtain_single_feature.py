# obtain_features.py

import numpy as np
import pickle
from src.preprocess import Processor
import json
import src.demographic_models as demographic_models
from src.parameter_inference import run_inference_dadi, run_inference_moments, run_inference_momentsLD
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# I want to load in the 

def obtain_feature(SFS, sampled_params, experiment_config, sim_directory, sim_number):

    # Load in the experiment config 
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)

    upper_bound = [b if b is not None else None for b in experiment_config['upper_bound_optimization']]
    lower_bound = [b if b is not None else None for b in experiment_config['lower_bound_optimization']]

    # Load in the SFS file
    with open(SFS, "rb") as f:
        SFS = pickle.load(f)

    # Load in the sampled params
    with open(sampled_params, "rb") as f:
        sampled_params = pickle.load(f)

    # if experiment_config["demographic_model"] == "bottleneck_model":
    #     demographic_model = demographic_models.bottleneck_model

    # elif experiment_config["demographic_model"] == "split_isolation_model":
    #     demographic_model = demographic_models.split_isolation_model_simulation

    mega_result_dict = (
            {}
        )  # This will store all the results (downstream postprocessing) later
    
    mega_result_dict = {"simulated_params": sampled_params, "sfs": SFS}

    # Load the experiment config and run the simulation (as before)
    # processor = Processor(
    #     experiment_config,
    #     experiment_directory=sim_directory,
    #     recombination_rate=experiment_config["recombination_rate"],
    #     mutation_rate=experiment_config["mutation_rate"],
    # )


    # Conditional analysis based on provided functions
    if experiment_config["dadi_analysis"]:
        model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi, ll_list_dadi = (
            run_inference_dadi(
                sfs = SFS,
                p0= experiment_config['optimization_initial_guess'],
                lower_bound= lower_bound,
                upper_bound= upper_bound,
                num_samples=100,
                demographic_model=experiment_config['demographic_model'],
                mutation_rate=experiment_config['mutation_rate'],
                length=experiment_config['genome_length'],
                k  = experiment_config['k'], 
                top_values_k = experiment_config['top_values_k']
            )
        )

        dadi_results = {
            "model_sfs_dadi": model_sfs_dadi,
            "opt_theta_dadi": opt_theta_dadi,
            "opt_params_dadi": opt_params_dict_dadi,
            "ll_all_replicates_dadi": ll_list_dadi,
        }
        
        mega_result_dict.update(dadi_results)


    if experiment_config["moments_analysis"]:
        model_sfs_moments, opt_theta_moments, opt_params_dict_moments, ll_list_moments = (
            run_inference_moments(
                sfs = SFS,
                p0=experiment_config['optimization_initial_guess'],
                lower_bound= lower_bound,
                upper_bound= upper_bound,
                demographic_model=experiment_config['demographic_model'],
                use_FIM=experiment_config["use_FIM"],
                mutation_rate=experiment_config['mutation_rate'],
                length=experiment_config['genome_length'],
                k = experiment_config['k'],
                top_values_k=experiment_config['top_values_k']
            )
        )


        moments_results = {
            "model_sfs_moments": model_sfs_moments,
            "opt_theta_moments": opt_theta_moments,
            "opt_params_moments": opt_params_dict_moments,
            "ll_all_replicates_moments": ll_list_moments
        }
        mega_result_dict.update(moments_results)

    if experiment_config["momentsLD_analysis"]:

        p_guess = experiment_config['optimization_initial_guess'].copy()
        
        p_guess.extend([20000])

        flat_map_path = f'{sim_directory}/sampled_genome_windows/sim_{sim_number}/flat_map.txt'
        metadata_path = f'{sim_directory}/sampled_genome_windows/sim_{sim_number}/metadata.txt'
        samples_path = f'{sim_directory}/sampled_genome_windows/sim_{sim_number}/samples.txt'

        opt_params_momentsLD = run_inference_momentsLD(
            flat_map_path = flat_map_path,
            samples_path = samples_path,
            metadata_path = metadata_path,
            p_guess=p_guess, #TODO: Need to change this to not rely on a hardcoded value
            demographic_model=experiment_config['demographic_model'],
            maxiter=experiment_config['maxiter'],
        )

        momentsLD_results = {"opt_params_momentsLD": opt_params_momentsLD}
        mega_result_dict.update(momentsLD_results)

    # Save the results in a pickle file
    with open(f"{sim_directory}/simulation_results/software_inferences_sim_{sim_number}.pkl", "wb") as f:
        pickle.dump(mega_result_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sfs_file", type=str, required=True)
    parser.add_argument("--sampled_params_pkl", type=str, required=True)
    parser.add_argument("--experiment_config_filepath", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    parser.add_argument("--sim_number", type=int, required=True)
    args = parser.parse_args()

    obtain_feature(
        SFS=args.sfs_file,
        sampled_params=args.sampled_params_pkl,
        experiment_config=args.experiment_config_filepath,
        sim_directory=args.sim_directory,
        sim_number=args.sim_number
    )









# def obtain_features(
#     experiment_config,
#     sim_directory
# ): 

#     # Load the experiment config
#     with open(experiment_config, "r") as f:
#         experiment_config = json.load(f)

#     processor = Processor(
#         experiment_config,
#         experiment_directory = sim_directory,
#         recombination_rate=experiment_config["recombination_rate"],
#         mutation_rate=experiment_config["mutation_rate"]
        
#     )

#     # Now I want to define training, validation, and testing indices:

#     # Generate all indices and shuffle them
#     all_indices = np.arange(experiment_config["num_sims_pretrain"])
#     np.random.shuffle(all_indices)

#     # Split into training and validation indices
#     n_train = int(
#         experiment_config["training_percentage"]
#         * experiment_config["num_sims_pretrain"]
#     )

#     training_indices = all_indices[:n_train]
#     validation_indices = all_indices[n_train:]
#     testing_indices = np.arange(experiment_config["num_sims_inference"])

#     preprocessing_results_obj = {
#         stage: {} for stage in ["training", "validation", "testing"]
#     }

#     for stage, indices in [
#         ("training", training_indices),
#         ("validation", validation_indices),
#         ("testing", testing_indices),
#     ]:

#         # Your existing process_and_save_data function

#         # Call the remote function and get the ObjectRef
#         print(f"Processing {stage} data")
#         features, targets, upper_triangle_features, ll_values_data = processor.pretrain_processing(
#             indices
#         )

#         preprocessing_results_obj[stage]["predictions"] = features
#         preprocessing_results_obj[stage]["targets"] = targets
#         preprocessing_results_obj[stage]["upper_triangular_FIM"] = upper_triangle_features
#         preprocessing_results_obj[stage]["ll_values"] = ll_values_data

#         features = features.reshape(features.shape[0], -1)
#         # print(f"Features shape: {features.shape}")

#         # Concatenate the features and upper triangle features column-wise

#         if experiment_config['use_FIM'] == True:
#             upper_triangle_features = upper_triangle_features.reshape(upper_triangle_features.shape[0], -1) # type:ignore
#             preprocessing_results_obj[stage]["upper_triangular_FIM_reshape"] = upper_triangle_features
#             all_features = np.concatenate((features, upper_triangle_features), axis=1) #type:ignore

#         else:
#             all_features = features
    
#         # Save all_features and targets to a file
#         np.save(f"{sim_directory}/{stage}_features.npy", all_features)

#         #TODO: FIX 
#         targets = targets[:,0,0,:] # This extracts the ground truth values. Later we can always tile and get it to match the the features shape. 
#         np.save(f"{sim_directory}/{stage}_targets.npy", targets)

#     #TODO: Need to add fields to the below object
#     # # Open a file to save the object
#     with open(
#         f"{sim_directory}/preprocessing_results_obj.pkl", "wb"
#     ) as file:  # "wb" mode opens the file in binary write mode
#         pickle.dump(preprocessing_results_obj, file)

#     #     if experiment_config["normalization"] == True:
#     #         feature_values = normalized_features
#     #         targets_values = normalized_targets
#     #     else:
#     #         feature_values = features
#     #         targets_values = targets

#     #     preprocessing_results_obj[stage][
#     #         "predictions"
#     #     ] = feature_values  # This is for input to the ML model
#     #     preprocessing_results_obj[stage]["targets"] = targets_values
#     #     preprocessing_results_obj[stage][
#     #         "upper_triangular_FIM"
#     #     ] = upper_triangle_features

        
#     #     # Object for plotting

#     #     if experiment_config["normalization"] == True:

#     #         preprocessing_results_obj_plotting[stage][
#     #             "predictions"
#     #         ] = normalized_features  # This is for plotting
#     #         preprocessing_results_obj_plotting[stage]["targets"] = normalized_targets
#     #         preprocessing_results_obj_plotting[stage][
#     #             "upper_triangular_FIM"
#     #         ] = upper_triangle_features
#     #     else:


#     # preprocessing_results_obj["param_names"] = experiment_config["parameter_names"]
#     # preprocessing_results_obj_normalized["param_names"] = experiment_config["parameter_names"]

#     # TODO: ALL THE CODE BELOW SHOULD BE MOVED TO THE EXTRACT_FEATURES RULE
#     # rrmse_dict = calculate_and_save_rrmse(
#     #     preprocessing_results_obj, save_path=f"{sim_directory}/rrmse_dict.json"
#     # )


#     # visualizing_results(
#     #     preprocessing_results_obj_normalized,
#     #     save_loc=sim_directory,
#     #     analysis=f"preprocessing_results",
#     #     stages=["training", "validation"],
#     #     color_shades=color_shades,
#     #     main_colors=main_colors,
#     # )

#     # visualizing_results(
#     #     preprocessing_results_obj_normalized,
#     #     save_loc=sim_directory,
#     #     analysis=f"preprocessing_results_testing",
#     #     stages=["testing"],
#     #     color_shades=color_shades,
#     #     main_colors=main_colors,
#     # )

#     # print("Training complete!")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--experiment_config", type=str, required=True)
#     parser.add_argument("--sim_directory", type=str, required=True)
#     args = parser.parse_args()

#     obtain_features(
#         experiment_config=args.experiment_config,
#         sim_directory=args.sim_directory
#     )
