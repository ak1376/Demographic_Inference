import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import allel
import dadi
from parameter_inference import run_inference_dadi, run_inference_moments, run_inference_momentsLD
from preprocess import Processor, FeatureExtractor, get_dicts
from utils import process_and_save_data, save_dict_to_pickle

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
    
    def dadi_inference(self, fs, p0, sampled_params, lower_bound, upper_bound, maxiter):
        model, opt_theta, opt_params_dict = run_inference_dadi(
            fs,
            p0,
            sampled_params,
            self.num_samples,
            lower_bound,
            upper_bound,
            maxiter,
            mutation_rate = 1.26e-8,
            length = 1e7
        )

        return model, opt_theta, opt_params_dict
    
    def moments_inference(self, fs, p0, sampled_params, lower_bound, upper_bound, maxiter):
        model, opt_theta, opt_params_dict = run_inference_moments(
        fs,
        p0,
        sampled_params,
        lower_bound=[0.001, 0.001, 0.001, 0.001],
        upper_bound=[10, 10, 10, 10],
        maxiter=20,
        use_FIM=False,
        mutation_rate = 1.26e-8,
        length = 1e7
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
                maxiter = 20
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
                    lower_bound=[1e-4, 1e-4, 1e-4, 1e-4],
                    upper_bound=[None, None, None, None],
                    sampled_params=None,
                    maxiter=self.maxiter,
                    use_FIM = self.config["use_FIM"],
                    mutation_rate=self.mutation_rate,
                    length = self.L
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
                'simulated_params': merged_dict['sampled_params'], 
                'sfs': merged_dict['sfs'],
                'model_sfs': merged_dict['model_sfs_dadi'],
                'opt_theta': merged_dict['opt_theta_dadi'],
                'opt_params': merged_dict['opt_params_dadi']
            }
        else:
            dadi_dict = {}

        
        if self.config['moments_analysis']:
            moments_dict = {
                'simulated_params': merged_dict['sampled_params'], 
                'sfs': merged_dict['sfs'],
                'model_sfs': merged_dict['model_sfs_moments'],
                'opt_theta': merged_dict['opt_theta_moments'],
                'opt_params': merged_dict['opt_params_moments']
            }
        else:
            moments_dict = {}

        if self.config['momentsLD_analysis']:
            momentsLD_dict = {
                'simulated_params': merged_dict['sampled_params'], 
                'sfs': merged_dict['sfs'],
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
            save_dict_to_pickle(data, filename, self.config["experiment_directory"])

        features_dict = {stage: {} for stage in ["inference"]}
        targets_dict = {stage: {} for stage in ["inference"]}
        feature_names = {}
        
        # Process each dictionary
        if self.extractor.dadi_analysis:
            features_dict, targets_dict, feature_names = self.extractor.process_batch(
                dadi_dict,
                "dadi",
                "inference",
                features_dict=features_dict,
                targets_dict=targets_dict,
                feature_names=feature_names,
                normalization=self.config['normalization']
            )  # type: ignore
        if self.extractor.moments_analysis:
            features_dict, targets_dict, feature_names = self.extractor.process_batch(
                moments_dict,
                "moments",
                "inference",
                features_dict=features_dict,
                targets_dict=targets_dict,
                feature_names=feature_names,
                normalization=self.config['normalization'],
            )  # type:ignore
        if self.extractor.momentsLD_analysis:
            features_dict, targets_dict, feature_names = self.extractor.process_batch(
                momentsLD_dict,
                "momentsLD",
                "inference",
                features_dict=features_dict,
                targets_dict=targets_dict,
                feature_names=feature_names,
                normalization=self.config['normalization'],
            )  # type:ignore

        # After processing all stages, finalize the processing
        # TODO: Need to rewrite this

        features, targets, feature_names = self.extractor.finalize_processing(features_dict=features_dict, targets_dict=targets_dict, feature_names=feature_names,
            remove_outliers=False
        )

        print("siema")

    
    

    
















def inference():
    """
    This should do the parameter inference for dadi and moments
    """
    pass