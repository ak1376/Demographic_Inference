import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import allel
import dadi
from parameter_inference import run_inference_dadi, run_inference_moments, run_inference_momentsLD
from preprocess import Processor, FeatureExtractor, get_dicts
from utils import process_and_save_data, save_dict_to_pickle
from sklearn.preprocessing import StandardScaler
import pickle

class Inference(Processor):
    def __init__(self, vcf_filepath, txt_filepath, popname, config, experiment_directory):
        self.vcf_filepath = vcf_filepath
        self.txt_filepath = txt_filepath
        self.popname = popname
        self.experiment_directory = experiment_directory

        self.config = config
        self.extractor = FeatureExtractor(
            self.experiment_directory,
            dadi_analysis=self.config["dadi_analysis"], #Not applicable for inference
            moments_analysis=self.config["moments_analysis"], # Not applicable for inference
            momentsLD_analysis=self.config["momentsLD_analysis"], # Not applicable for inference
        )

    def read_data(self):
        vcf_data = allel.read_vcf(self.vcf_filepath)
        self.num_samples = len(vcf_data['samples']) # type: ignore
        data_dict = dadi.Misc.make_data_dict_vcf(self.vcf_filepath, self.txt_filepath)
        return vcf_data, data_dict
    
    def create_SFS(self, data_dict):
        fs = dadi.Spectrum.from_data_dict(data_dict, [self.popname], projections = [2*self.num_samples], polarized = False)
        return fs
    
    def dadi_inference(self, fs, p0, sampled_params, lower_bound, upper_bound, maxiter, mutation_rate, genome_length):
        model, opt_theta, opt_params_dict = run_inference_dadi(
            fs,
            p0,
            sampled_params,
            self.num_samples,
            lower_bound,
            upper_bound,
            maxiter,
            mutation_rate = mutation_rate,
            length = genome_length
        )

        return model, opt_theta, opt_params_dict
    
    def moments_inference(self, fs, p0, sampled_params, lower_bound, upper_bound, maxiter, mutation_rate, genome_length):
        model, opt_theta, opt_params_dict = run_inference_moments(
        fs,
        p0,
        sampled_params,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        maxiter=maxiter,
        use_FIM=False,
        mutation_rate =mutation_rate,
        length = genome_length
        )

        return model, opt_theta, opt_params_dict


    def obtain_features(self):

        # First read in the data
        vcf_data, data_dict = self.read_data()

        sfs = self.create_SFS(data_dict)

        mega_result_dict = {} # This will store all the results (downstream postprocessing) later

        p0 = [0.25, 0.75, 0.1, 0.05]

        if self.config["dadi_analysis"]:
       
            model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = self.dadi_inference(
                sfs,
                p0,
                sampled_params = None,
                lower_bound = [0.001, 0.001, 0.001, 0.001],
                upper_bound = [10, 10, 10, 10],
                maxiter = self.config['maxiter'], 
                mutation_rate = self.config['mutation_rate'],
                genome_length = self.config['genome_length']
            )

            dadi_results = {
                    'model_sfs_dadi': model_sfs_dadi, 
                    'opt_theta_dadi': opt_theta_dadi,
                    'opt_params_dadi': opt_params_dict_dadi
                }
            


            mega_result_dict.update(dadi_results)

        if self.config['moments_analysis']:
            model_sfs_moments, opt_theta_moments, opt_params_dict_moments = (
                run_inference_moments(
                    sfs,
                    p0=[0.25, 0.75, 0.1, 0.05],
                    lower_bound = [0.001, 0.001, 0.001, 0.001],
                    upper_bound = [10, 10, 10, 10],
                    sampled_params=None,
                    maxiter=self.config['maxiter'],
                    use_FIM = self.config["use_FIM"],
                    mutation_rate=self.config['mutation_rate'],
                    length = self.config['genome_length']
                )
            )

            moments_results = {
                'model_sfs_moments': model_sfs_moments,
                'opt_theta_moments': opt_theta_moments,
                'opt_params_moments': opt_params_dict_moments
            }

            mega_result_dict.update(moments_results)


        if self.config['momentsLD_analysis']:
            opt_params_momentsLD = run_inference_momentsLD(
                folderpath=self.folderpath,
                num_windows=self.num_windows,
                param_sample=None,
                p_guess=[0.25, 0.75, 0.1, 0.05, 20000],
                maxiter=self.maxiter
            )

            momentsLD_results = {
                'opt_params_momentsLD': opt_params_momentsLD
            }

            mega_result_dict.update(momentsLD_results)

        list_of_mega_result_dicts = [mega_result_dict]

    
        merged_dict = get_dicts(list_of_mega_result_dicts)

        #TODO: Rewrite everything below: 
        if self.config["dadi_analysis"]:
            dadi_dict = {
                'actual_sfs': sfs,
                'model_sfs': merged_dict['model_sfs_dadi'],
                'opt_theta': merged_dict['opt_theta_dadi'],
                'opt_params': merged_dict['opt_params_dadi']
            }
        else:
            dadi_dict = {}

        
        if self.config['moments_analysis']:
            moments_dict = {
                'actual_sfs': sfs,
                'model_sfs': merged_dict['model_sfs_moments'],
                'opt_theta': merged_dict['opt_theta_moments'],
                'opt_params': merged_dict['opt_params_moments']
            }
        else:
            moments_dict = {}

        if self.config['momentsLD_analysis']:
            momentsLD_dict = {
                'opt_params': merged_dict['opt_params_momentsLD']
            }
        else:
            momentsLD_dict = {}

        for name, data in [
            ("dadi", dadi_dict),
            ("moments", moments_dict),
            ("momentsLD", momentsLD_dict),
        ]:
            filename = f"{name}_dict_inference.pkl"
            save_dict_to_pickle(data, filename, self.experiment_directory)

        features_dict = {stage: {} for stage in ["inference"]}
        targets_dict = {stage: {} for stage in ["inference"]}
        feature_names = {}
        
        # Process each dictionary
        if self.extractor.dadi_analysis:
            features_dict, feature_names = self.process_batch(
                dadi_dict,
                "dadi",
                "inference",
                features_dict=features_dict,
                feature_names=feature_names,
                normalization=self.config['normalization']
            )  # type: ignore
        if self.extractor.moments_analysis:
            features_dict, feature_names = self.process_batch(
                moments_dict,
                "moments",
                "inference",
                features_dict=features_dict,
                feature_names=feature_names,
                normalization=self.config['normalization'],
            )  # type:ignore
        if self.extractor.momentsLD_analysis:
            features_dict, feature_names = self.process_batch(
                momentsLD_dict,
                "momentsLD",
                "inference",
                features_dict=features_dict,
                feature_names=feature_names,
                normalization=self.config['normalization'],
            )  # type:ignore

        # After processing all stages, finalize the processing
        # TODO: Need to rewrite this

        features, feature_names = self.finalize_processing(features_dict=features_dict, feature_names=feature_names,
        )

        inference_obj = {}
        inference_obj["features"] = features['inference']

        # Open a file to save the object
        with open(
            f"{self.experiment_directory}/inference_results_obj.pkl", "wb"
        ) as file:  # "wb" mode opens the file in binary write mode
            pickle.dump(inference_obj, file)

        print("Inference complete!")

    def extract_features(self, opt_params, normalization=True):
        """
        opt_params can come from any of the inference methods.
        """

        #TODO: Rewrite this code s.t. I am not assuming a priori the demographic model . 

        # Extracting parameters from the flattened lists
        Nb_opt = opt_params['Nb']
        N_recover_opt = opt_params['N_recover']
        t_bottleneck_start_opt = opt_params['t_bottleneck_start']
        t_bottleneck_end_opt = opt_params['t_bottleneck_end']

        #TODO: Make this a bit more elegant and streamlined.
        if "upper_triangular_FIM" in opt_params.keys():
            upper_triangular_FIM = [d["upper_triangular_FIM"] for d in opt_params]

            # Add the FIM values to the features
            opt_params_array = np.column_stack(
                (
                    Nb_opt,
                    N_recover_opt,
                    t_bottleneck_start_opt,
                    t_bottleneck_end_opt,
                    upper_triangular_FIM,
                )
            )

        # Put all these features into a single 2D array
        if "upper_triangular_FIM" in opt_params.keys():
            opt_params_array = np.column_stack(
                (Nb_opt, N_recover_opt, t_bottleneck_start_opt, t_bottleneck_end_opt, upper_triangular_FIM)
            )
        else:
            opt_params_array = np.column_stack(
                (Nb_opt, N_recover_opt, t_bottleneck_start_opt, t_bottleneck_end_opt)
            )

        if normalization:
            # Feature scaling
            feature_scaler = StandardScaler()
            features = feature_scaler.fit_transform(opt_params_array)

        else:
            # Features are the optimized parameters
            features = opt_params_array

        return features
    
    def process_batch(self, batch_data, analysis_type, stage, features_dict = {}, feature_names = [], normalization=False):

        
        if stage != "inference":
            # Original condition that applies when stage is not 'inference'
            if (
                not batch_data
                or "simulated_params" not in batch_data
                or f"opt_params_{analysis_type}" not in batch_data
            ):
                print(f"Skipping {analysis_type} {stage} due to empty or invalid data")
                return
        else:
            # Special condition that applies when stage is 'inference'
            if "opt_params" not in batch_data:
                print(f"Skipping {analysis_type} {stage} due to empty or invalid data")
                return
        
        features = self.extract_features(
            batch_data[f"opt_params"],
            normalization=normalization,
        )

        if analysis_type not in features_dict[stage]:
            features_dict[stage][analysis_type] = []

        features_dict[stage][analysis_type].extend(features)

        if analysis_type not in feature_names:
            feature_names[analysis_type] = [
                f"Nb_opt_{analysis_type}",
                f"N_recover_opt_{analysis_type}",
                f"t_bottleneck_start_opt_{analysis_type}",
                f"t_bottleneck_end_opt_{analysis_type}"
            ]

        return features_dict, feature_names
    
    def finalize_processing(self, features_dict, feature_names):
        """
        This function will create the numpy array of features and targets across all analysis types and stages. If the user specifies to remove outliers, it will remove them from the data and then resample the rows for concatenation
        """
        for stage in features_dict:
            for analysis_type in features_dict[stage]:
                features = np.array(features_dict[stage][analysis_type])

                features_dict[stage][analysis_type] = features

                # error_value = root_mean_squared_error(targets, features)
                # print(f"Final error value for {analysis_type} {stage}: {error_value}")

        concatenated_features = (
            self.concatenate_features(feature_names=feature_names, features_dict=features_dict)
        )
        
        return concatenated_features, feature_names

    def concatenate_features(self, feature_names, features_dict):
        concatenated_features = {}
        concatenated_feature_names = []

        # Concatenate feature names only once
        for analysis_type in sorted(feature_names.keys()): #type: ignore
            concatenated_feature_names.extend(feature_names[analysis_type])

        for stage in features_dict:
            all_features = []
            all_targets = []
            for analysis_type in sorted(
                features_dict[stage].keys()
            ):  # Sort to ensure consistent order
                all_features.append(features_dict[stage][analysis_type])

        concatenated_features[stage] = np.hstack(all_features)
        return concatenated_features



    
    

    
















def inference():
    """
    This should do the parameter inference for dadi and moments
    """
    pass