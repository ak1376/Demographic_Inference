import numpy as np
import allel
import dadi
from parameter_inference import (
    run_inference_dadi,
    run_inference_moments,
    run_inference_momentsLD,
)
from preprocess import Processor
import pickle
import torch


class Inference(Processor):
    def __init__(
        self, vcf_filepath, txt_filepath, popname, config, experiment_directory, polarized = True
    ):
        self.vcf_filepath = vcf_filepath
        self.txt_filepath = txt_filepath
        self.popname = popname
        self.experiment_directory = experiment_directory
        self.polarized = polarized

        self.config = config

    def read_data(self):
        vcf_data = allel.read_vcf(self.vcf_filepath)
        self.num_samples = len(vcf_data["samples"])  # type: ignore
        data_dict = dadi.Misc.make_data_dict_vcf(self.vcf_filepath, self.txt_filepath)
        return vcf_data, data_dict

    def create_SFS(self, data_dict):
        fs = dadi.Spectrum.from_data_dict(
            data_dict,
            [self.popname],
            projections=[2 * self.num_samples],
            polarized=self.polarized,
        )
        return fs

    def dadi_inference(
        self,
        fs,
        p0,
        demographic_model,
        sampled_params,
        lower_bound,
        upper_bound,
        maxiter,
        mutation_rate,
        genome_length,
        num_samples,
        k
    ):
        model, opt_theta, opt_params_dict = run_inference_dadi(
        sfs = fs,
        p0 = p0,
        num_samples = num_samples,
        demographic_model = demographic_model,
        lower_bound=[0.001, 0.001, 0.001, 0.001],
        upper_bound=[1, 1, 1, 1],
        mutation_rate=1.26e-8,
        length=1e8,
        k = 1
        )

        return model, opt_theta, opt_params_dict

    def moments_inference(
        self,
        fs,
        p0,
        demographic_model,
        sampled_params,
        lower_bound,
        upper_bound,
        maxiter,
        mutation_rate,
        genome_length
    ):
        model, opt_theta, opt_params_dict = run_inference_moments(
            fs,
            p0,
            demographic_model,
            sampled_params,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            use_FIM=False,
            mutation_rate=mutation_rate,
            length=genome_length,

        )

        return model, opt_theta, opt_params_dict

    def obtain_features(self):

        # First read in the data
        vcf_data, data_dict = self.read_data()

        sfs = self.create_SFS(data_dict)

        mega_result_dict = (
            {}
        )  # This will store all the results (downstream postprocessing) later

        if self.config["dadi_analysis"]:

            model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = self.dadi_inference(
                sfs,
                p0 = self.config['optimization_initial_guess'],
                demographic_model = self.config["demographic_model"],
                sampled_params=None,
                lower_bound=[1e-4, 1e-4, 1e-4, 1e-4],
                upper_bound=[None, None, None, None],
                maxiter=self.config["maxiter"],
                mutation_rate=self.config["mutation_rate"],
                genome_length=self.config["genome_length"],
                num_samples=self.config['num_samples'],
                k = 1
            )

            dadi_results = {
                "model_sfs_dadi": model_sfs_dadi,
                "opt_theta_dadi": opt_theta_dadi,
                "opt_params_dadi": opt_params_dict_dadi,
            }

            mega_result_dict.update(dadi_results)

        if self.config["moments_analysis"]:
            model_sfs_moments, opt_theta_moments, opt_params_dict_moments =run_inference_moments(sfs,
                    p0 = self.config['optimization_initial_guess'],
                    demographic_model=self.config["demographic_model"],
                    lower_bound=[1e-4, 1e-4, 1e-4, 1e-4],
                    upper_bound=[None, None, None, None],
                    use_FIM=self.config["use_FIM"],
                    mutation_rate=self.config["mutation_rate"],
                    length=self.config["genome_length"], 
                    k = 1
                )

            moments_results = {
                "model_sfs_moments": model_sfs_moments,
                "opt_theta_moments": opt_theta_moments,
                "opt_params_moments": opt_params_dict_moments,
            }

            mega_result_dict.update(moments_results)

        if self.config["momentsLD_analysis"]:
            opt_params_momentsLD = run_inference_momentsLD(
                folderpath=self.folderpath,
                num_windows=self.num_windows,
                param_sample=None,
                demographic_model=self.config["demographic_model"],
                p_guess=self.config['optimization_initial_guess'].append(20000), #TODO: Change this later
                maxiter=self.maxiter,
            )

            momentsLD_results = {"opt_params_momentsLD": opt_params_momentsLD}

            mega_result_dict.update(momentsLD_results)

        list_of_mega_result_dicts = [mega_result_dict]


        # Step 1: Initialize an empty list to collect analysis data arrays
        analysis_data = []
        upper_triangular_data = []

        # Step 2: Dynamically extract and append data based on configuration
        for analysis_type in ['dadi_analysis', 'moments_analysis', 'momentsLD_analysis']:
            if self.config.get(analysis_type):
                # Extract the appropriate data dynamically based on the analysis type
                analysis_key = 'opt_params_' + analysis_type.split('_')[0]  # This maps to 'opt_params_dadi', 'opt_params_moments', etc.

                analysis_type_data = []
                for result in list_of_mega_result_dicts:
                    for index in np.arange(len(result[analysis_key])):
                        param_values = list(result[analysis_key][index].values())

                        if analysis_type == 'moments_analysis' and self.config.get('use_FIM', True):
                            # Extract and store the upper triangular FIM separately
                            upper_triangular = result['opt_params_moments'][index].get('upper_triangular_FIM', None)
                            if upper_triangular is not None:
                                upper_triangular_data.append(upper_triangular)  # Store upper triangular FIM separately
                                
                                # Remove 'upper_triangular_FIM' from param_values if it was included
                                # Assuming 'upper_triangular_FIM' is the last key in the dictionary
                                param_values = [value for value in param_values if not isinstance(value, np.ndarray)]


                        # Append the processed param values to analysis_type_data
                        analysis_type_data.append(param_values)

                # Add the collected parameter data (excluding FIM if stored separately) to analysis_data
                analysis_data.append(analysis_type_data)

        # Step 3: Convert the collected data into NumPy arrays
        analysis_arrays = np.array(analysis_data)

        # Now we need to transpose it to get the shape (num_sims*k, num_analyses, num_params)

        num_analyses = self.config['dadi_analysis'] + self.config['moments_analysis'] + self.config['momentsLD_analysis']
        num_sims = len(list_of_mega_result_dicts)
        num_reps = len(analysis_data[0]) // num_sims
        num_params = len(analysis_data[0][0])

        # Reshape to desired format
        analysis_arrays = analysis_arrays.reshape((num_analyses, num_sims, num_reps, num_params))

        # Transpose to match the desired output shape (num_sims, num_reps, num_analyses, num_params)
        features = np.transpose(analysis_arrays, (1, 2, 0, 3))

        # If upper triangular data exists, convert it to a NumPy array for further analysis
        if upper_triangular_data:
            upper_triangular_array = np.array(upper_triangular_data).reshape(((1, num_sims, num_reps, upper_triangular_data[0].shape[0])))  # Array of upper triangular matrices
            upper_triangular_array = np.transpose(upper_triangular_array, (1, 2, 0, 3))
        else:
            upper_triangular_array = None  # Handle case where FIM data does not exist

        inference_obj = {}
        inference_obj["features"] = features

        if upper_triangular_array is not None:
            inference_obj["upper_triangular_FIM"] = upper_triangular_array

        # Open a file to save the object
        with open(
            f"{self.experiment_directory}/inference_results_obj.pkl", "wb"
        ) as file:  # "wb" mode opens the file in binary write mode
            pickle.dump(inference_obj, file)

        print("Inference complete!")


    def evaluate_model(self, snn_model, inference_results, additional_features):
        """
        This should run the GHIST data through the trained model and return the inferred parameters
        """

        training_features = inference_results['features']

        num_sims, num_reps, num_analyses, num_params = training_features.shape[0], training_features.shape[1], training_features.shape[2], training_features.shape[3]

        training_features = training_features.reshape(num_sims * num_reps, num_analyses*num_params)

        if additional_features is not None:
            additional_features_training = np.expand_dims(np.squeeze(additional_features['upper_triangular_FIM']), axis = 0)
            training_features = np.concatenate((training_features, additional_features_training), axis = 1) 


        inference_features = torch.tensor(training_features, dtype = torch.float32).cuda() 
        
        inferred_params = snn_model.predict(inference_features)

        # Save the array as a text file
        np.savetxt(
            f"{self.experiment_directory}/inferred_params_GHIST_bottleneck.txt",
            inferred_params,
            delimiter=" ",
            fmt="%.5f",
        )
