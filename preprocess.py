import os
import moments
from tqdm import tqdm
import numpy as np
import msprime
import dadi
import glob
import time
import demographic_models
from parameter_inference import (
    run_inference_dadi,
    run_inference_moments,
    run_inference_momentsLD,
)
import pandas as pd
from scipy.stats import zscore

def generate_window(ts, window_length, n_samples):
    start = np.random.randint(0, n_samples - window_length)
    end = start + window_length
    return ts.keep_intervals([[start, end]])


def process_window(ts_window, folderpath, ii):
    vcf_name = os.path.join(folderpath, f"bottleneck_window.{ii}.vcf")

    with open(vcf_name, "w+") as fout:
        ts_window.write_vcf(fout, allow_position_zero=True)

    os.system(f"gzip -f {vcf_name}")

    return vcf_name  # Optionally return the filename or any other relevant information


def delete_vcf_files(directory):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory, "*"))

    # Delete all files
    [os.remove(file) for file in files]

    print(f"Deleted {len(files)} files from {directory}")

class Processor:
    def __init__(
        self,
        experiment_config,
        experiment_directory,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        window_length=1e6,
    ):

        self.experiment_config = experiment_config
        self.experiment_directory = experiment_directory

        self.upper_bound_params = self.experiment_config["upper_bound_params"]
        self.lower_bound_params = self.experiment_config["lower_bound_params"]

        self.param_storage = []
        self.ts_storage = []
        self.sfs = []

        self.num_sims_pretrain = self.experiment_config["num_sims_pretrain"]
        self.num_sims_inference = self.experiment_config["num_sims_inference"]

        self.num_samples = self.experiment_config["num_samples"]
        self.L = self.experiment_config["genome_length"]
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate

        self.num_windows = self.experiment_config[
            "num_windows"
        ]  # This is just for momentsLD. Number of windows to split the genome into.
        self.window_length = self.experiment_config["window_length"]
        self.maxiter = self.experiment_config["maxiter"]

        self.mutation_rate = self.experiment_config["mutation_rate"]
        self.recombination_rate = self.experiment_config["recombination_rate"]

        self.folderpath = f"{self.experiment_directory}/sampled_genome_windows"

        self.demographic_model = self.experiment_config["demographic_model"]

        self.optimization_initial_guess = self.experiment_config['optimization_initial_guess']


    def run_msprime_replicates(self, g):
        delete_vcf_files(self.folderpath)
        demog = msprime.Demography.from_demes(g)
        ts = msprime.sim_ancestry(
            {"A": self.num_samples},
            demography=demog,
            sequence_length=self.L,
            recombination_rate=self.recombination_rate,
            random_seed=295,
        )
        ts = msprime.sim_mutations(ts, rate=self.mutation_rate)

        self.folderpath = f"{self.experiment_directory}/sampled_genome_windows"
        os.makedirs(self.folderpath, exist_ok=True)

        # Generate random windows in parallel

        windows = [
            generate_window(ts, self.window_length, self.L)
            for _ in range(self.num_windows)
        ]

        for ii, ts_window in tqdm(enumerate(windows), total=len(windows)):
            vcf_name = os.path.join(self.folderpath, f"bottleneck_window.{ii}.vcf")
            with open(vcf_name, "w+") as fout:
                ts_window.write_vcf(fout, allow_position_zero=True)
            os.system(f"gzip {vcf_name}")

    def write_samples_and_rec_map(self):

        samples_file = os.path.join(self.folderpath, f"samples.txt")
        flat_map_file = os.path.join(self.folderpath, f"flat_map.txt")

        with open(samples_file, "w+") as fout:
            fout.write("sample\tpop\n")
            for ii in range(self.num_samples):
                fout.write(f"tsk_{ii}\tA\n")

        with open(flat_map_file, "w+") as fout:
            fout.write("pos\tMap(cM)\n")
            fout.write("0\t0\n")
            fout.write(f"{self.L}\t{self.recombination_rate * self.L * 100}\n")

        return samples_file, flat_map_file

    def sample_params(self):
        sampled_params = {}
        for key in self.lower_bound_params:
            lower_bound = self.lower_bound_params[key]
            upper_bound = self.upper_bound_params[key]
            sampled_value = np.random.uniform(lower_bound, upper_bound)
            sampled_params[key] = int(sampled_value)


            # Check if the sampled parameter is equal to the mean of the uniform distribution
            mean_value = (lower_bound + upper_bound) / 2
            if sampled_value == mean_value:
                # Add a small random value to avoid exact mean, while keeping within bounds
                adjustment = np.random.uniform(-0.1 * (upper_bound - lower_bound), 0.1 * (upper_bound - lower_bound))
                adjusted_value = sampled_value + adjustment
                
                # Ensure the adjusted value is still within the bounds
                adjusted_value = max(min(adjusted_value, upper_bound), lower_bound)
                sampled_params[key] = int(adjusted_value)

        return sampled_params

    def create_SFS(self, 
        sampled_params, mode, num_samples, demographic_model, length=1e7, mutation_rate=1.26e-8, **kwargs
    ):
        """
        If we are in pretraining mode we will use a simulated SFS. If we are in inference mode we will use a real SFS.

        """

        if mode == "pretrain":
        
            g = demographic_model(sampled_params)
                
            demog = msprime.Demography.from_demes(g)

            ts = msprime.sim_ancestry(
                self.num_samples,
                demography=demog,
                sequence_length=self.L,
                recombination_rate=self.recombination_rate,
                random_seed=295,
            )
            ts = msprime.sim_mutations(ts, rate=mutation_rate)
            
            # Now create the SFS
            sfs = ts.allele_frequency_spectrum(mode="site", polarised=True)
                        
            # multiply sfs by L
            sfs *= length

            sfs = moments.Spectrum(sfs)

            
            
        elif mode == "inference":
            vcf_file = kwargs.get("vcf_file", None)
            pop_file = kwargs.get("pop_file", None)
            popname = kwargs.get("popname", None)

            if vcf_file is None or pop_file is None:
                raise ValueError(
                    "vcf_file and pop_file must be provided in inference mode."
                )

            dd = dadi.Misc.make_data_dict_vcf(vcf_file, pop_file)
            sfs = dadi.Spectrum.from_data_dict(
                dd, [popname], projections=[2 * num_samples], polarized=True
            )

        return sfs

    def pretrain_processing(self, indices_of_interest):
        """
        This really should be subdivided into more functions so that when we do inference I can separately call the helper functions.
        """

        # Placeholder if statements 
        if self.experiment_config["demographic_model"] == "bottleneck_model":
            demographic_model_simulation = demographic_models.bottleneck_model
        
        elif self.experiment_config["demographic_model"] == "split_isolation_model":
            demographic_model_simulation = demographic_models.split_isolation_model_simulation

        list_of_mega_result_dicts = []

        for i in tqdm(range(len(indices_of_interest))):
            mega_result_dict = (
                {}
            )  # This will store all the results (downstream postprocessing) later

            sampled_params = self.sample_params()

            start = time.time()
            sfs = self.create_SFS(
                sampled_params,
                mode="pretrain",
                num_samples=self.num_samples,
                demographic_model=demographic_model_simulation,
                length=self.L,
                mutation_rate=self.mutation_rate,
            )

            end = time.time()

            mega_result_dict = {"simulated_params": sampled_params, "sfs": sfs}

            print(f'Simulation Time: {end - start}')

            # Simulate process and save windows as VCF files
            if self.experiment_config["momentsLD_analysis"]:
                g = demographic_model_simulation(sampled_params)
                self.run_msprime_replicates(g)
                samples_file, flat_map_file = self.write_samples_and_rec_map()

            # Conditional analysis based on provided functions
            if self.experiment_config["dadi_analysis"]:
                model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = (
                    run_inference_dadi(
                        sfs,
                        p0= self.experiment_config['optimization_initial_guess'],
                        lower_bound=[1e-4]*(len(self.experiment_config['parameter_names']) - 1),
                        upper_bound=[None] * (len(self.experiment_config['parameter_names']) - 1),
                        num_samples=100, #TODO: Need to change this to not rely on a hardcoded value
                        demographic_model=self.experiment_config['demographic_model'],
                        mutation_rate=self.mutation_rate,
                        length=self.L,
                        k  = self.experiment_config['k']
                    )
                )


                dadi_results = {
                    "model_sfs_dadi": model_sfs_dadi,
                    "opt_theta_dadi": opt_theta_dadi,
                    "opt_params_dadi": opt_params_dict_dadi,
                }

                mega_result_dict.update(dadi_results)

            if self.experiment_config["moments_analysis"]:
                model_sfs_moments, opt_theta_moments, opt_params_dict_moments = (
                    run_inference_moments(
                        sfs,
                        p0=self.experiment_config['optimization_initial_guess'],
                        lower_bound=[1e-4]*(len(self.experiment_config['parameter_names']) -1),
                        upper_bound=[None] * (len(self.experiment_config['parameter_names']) - 1),
                        demographic_model=self.experiment_config['demographic_model'],
                        use_FIM=self.experiment_config["use_FIM"],
                        mutation_rate=self.mutation_rate,
                        length=self.L,
                        k = self.experiment_config['k']
                    )
                )

                moments_results = {
                    "model_sfs_moments": model_sfs_moments,
                    "opt_theta_moments": opt_theta_moments,
                    "opt_params_moments": opt_params_dict_moments,
                }

                mega_result_dict.update(moments_results)

                # results.update(
                #     {
                #         "opt_params_dict_moments": opt_params_dict_moments,
                #         "model_sfs_moments": model_sfs_moments,
                #         "opt_theta_moments": opt_theta_moments,
                #     }
                # )

            if self.experiment_config["momentsLD_analysis"]:

                p_guess = self.experiment_config['optimization_initial_guess'].copy()
                
                p_guess.extend([20000])
                print(f"p_guess: {p_guess}")

                opt_params_momentsLD = run_inference_momentsLD(
                    folderpath=self.folderpath,
                    num_windows=self.num_windows,
                    param_sample=sampled_params,
                    p_guess=p_guess, #TODO: Need to change this to not rely on a hardcoded value
                    demographic_model=self.experiment_config['demographic_model'],
                    maxiter=self.maxiter
                )

                momentsLD_results = {"opt_params_momentsLD": opt_params_momentsLD}

                mega_result_dict.update(momentsLD_results)

                # results.update({"opt_params_momentsLD": opt_params_momentsLD})

            list_of_mega_result_dicts.append(mega_result_dict)

        # Step 1: Initialize an empty list to collect analysis data arrays
        analysis_data = []
        upper_triangular_data = []
        targets_data = []

        # Step 2: Dynamically extract and append data based on configuration
        for analysis_type in ['dadi_analysis', 'moments_analysis', 'momentsLD_analysis']:
            if self.experiment_config.get(analysis_type):
                # Extract the appropriate data dynamically based on the analysis type
                analysis_key = 'opt_params_' + analysis_type.split('_')[0]  # This maps to 'opt_params_dadi', 'opt_params_moments', etc.

                analysis_type_data = []
                targets_type_data = []
                for result in list_of_mega_result_dicts:
                    for index in np.arange(len(result[analysis_key])):
                        param_values = list(result[analysis_key][index].values())
                        target_values = list(result['simulated_params'].values())

                        if analysis_type == 'moments_analysis' and self.experiment_config.get('use_FIM', True):
                            # Extract and store the upper triangular FIM separately
                            upper_triangular = result['opt_params_moments'][index].get('upper_triangular_FIM', None)
                            if upper_triangular is not None:
                                upper_triangular_data.append(upper_triangular)  # Store upper triangular FIM separately
                                
                                # Remove 'upper_triangular_FIM' from param_values if it was included
                                # Assuming 'upper_triangular_FIM' is the last key in the dictionary
                                param_values = [value for value in param_values if not isinstance(value, np.ndarray)]


                        # Append the processed param values to analysis_type_data
                        analysis_type_data.append(param_values)
                        targets_type_data.append(target_values)

                # Add the collected parameter data (excluding FIM if stored separately) to analysis_data
                analysis_data.append(analysis_type_data)
                targets_data.append(targets_type_data)

        # Step 3: Convert the collected data into NumPy arrays
        analysis_arrays = np.array(analysis_data)
        targets_arrays = np.array(targets_data)

        # Now we need to transpose it to get the shape (num_sims*k, num_analyses, num_params)

        num_analyses = self.experiment_config['dadi_analysis'] + self.experiment_config['moments_analysis'] + self.experiment_config['momentsLD_analysis']
        num_sims = len(list_of_mega_result_dicts)
        num_reps = len(analysis_data[0]) // num_sims
        num_params = len(analysis_data[0][0])

        # Reshape to desired format
        analysis_arrays = analysis_arrays.reshape((num_analyses, num_sims, num_reps, num_params))
        targets_arrays = targets_arrays.reshape((num_analyses, num_sims, num_reps, num_params))

        # Transpose to match the desired output shape (num_sims, num_reps, num_analyses, num_params)
        features = np.transpose(analysis_arrays, (1, 2, 0, 3))
        targets = np.transpose(targets_arrays, (1, 2, 0, 3))

        # If upper triangular data exists, convert it to a NumPy array for further analysis
        if upper_triangular_data:
            upper_triangular_array = np.array(upper_triangular_data).reshape(((1, num_sims, num_reps, upper_triangular_data[0].shape[0])))  # Array of upper triangular matrices
            upper_triangular_array = np.transpose(upper_triangular_array, (1, 2, 0, 3))
        else:
            upper_triangular_array = None  # Handle case where FIM data does not exist


        if self.experiment_config['remove_outliers'] == True:
            print("===> Removing outliers and imputing with median values.")

            # NOW LET'S DO OUTLIER REMOVAL AND MEDIAN IMPUTATION. 
            
            
            # Step 1: Reshape to (num_sims*num_reps, num_analyses*num_params)
            reshaped = features.reshape(num_sims*num_reps, num_analyses*num_params)

            # Step 2: Calculate Z-scores for the entire array
            z_scores = np.abs(zscore(reshaped, axis=0))

            # Define the threshold for outliers (Grubbs test Z-score = 3)
            threshold = 3
            outliers = z_scores > threshold

            # Step 3: Replace outliers with the median of the non-outlier values
            # Compute the median of the values that are not outliers
            median_value = np.median(reshaped[~outliers])

            # Replace outliers with the median
            reshaped[outliers] = median_value

            # Step 4: Reshape the data back to the original format
            features = reshaped.reshape((num_sims, num_reps, num_analyses, num_params))
                

        if self.experiment_config['normalization'] == True:
            print("===> Normalizing the data.")
            
            # Convert dict values to NumPy arrays for element-wise operations
            upper_bound_values = np.array(list(self.experiment_config['upper_bound_params'].values()))
            lower_bound_values = np.array(list(self.experiment_config['lower_bound_params'].values()))

            # Calculate mean and standard deviation vectors
            mean_vector = 0.5 * (upper_bound_values + lower_bound_values)
            std_vector = (upper_bound_values - lower_bound_values) / np.sqrt(12)  # Correct std deviation for uniform distribution

            # Normalize the targets
            normalized_targets = (targets - mean_vector) / (std_vector)

            # NORMALIZE THE FEATURES TOO (FOR THE PREPROCESSING PLOTTING)
            normalized_features = (features - mean_vector) / (std_vector)

            # Check for zero values in the normalized targets
            zero_target_indices = np.where(normalized_targets == 0)
            if zero_target_indices[0].size > 0:  # If any zero values are found
                print("Warning: Zero values found in the normalized targets!")
                # Extract raw target values where normalized target values are 0
                raw_target_values = targets[zero_target_indices]
                print("Raw target values corresponding to zero normalized targets:", raw_target_values)

                # Add 1 to the normalized targets that are zero
                normalized_targets[zero_target_indices] += 1
                print("Added 1 to zero normalized target values.")
            else:
                print("No zero values found in the normalized targets.")

            # Check for zero values in the normalized features
            zero_feature_indices = np.where(normalized_features == 0)
            if zero_feature_indices[0].size > 0:  # If any zero values are found
                print("Warning: Zero values found in the normalized features!")
                # Extract raw feature values where normalized feature values are 0
                raw_feature_values = features[zero_feature_indices]
                print("Raw feature values corresponding to zero normalized features:", raw_feature_values)
            else:
                print("No zero values found in the normalized features.")                        
       
        # Return features, targets, and upper triangular array (if exists)
        #TODO: Fix code in the case where experiment_config['normalization'] == False
        return features, normalized_features, normalized_targets, upper_triangular_array