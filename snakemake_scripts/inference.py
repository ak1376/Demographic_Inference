# run_inference.py

import numpy as np
import allel
import dadi
from parameter_inference import run_inference_dadi, run_inference_moments, run_inference_momentsLD
from preprocess import Processor, FeatureExtractor, get_dicts
from utils import save_dict_to_pickle
import pickle
import torch
import sys
import json

class Inference(Processor):
    def __init__(self, vcf_filepath, txt_filepath, popname, config, experiment_directory):
        self.vcf_filepath = vcf_filepath
        self.txt_filepath = txt_filepath
        self.popname = popname
        self.experiment_directory = experiment_directory

        self.config = config
        self.extractor = FeatureExtractor(
            self.experiment_directory,
            dadi_analysis=self.config["dadi_analysis"], # Not applicable for inference
            moments_analysis=self.config["moments_analysis"], # Not applicable for inference
            momentsLD_analysis=self.config["momentsLD_analysis"], # Not applicable for inference
        )

    def read_data(self):
        print(f'VCF FILEPATH: {self.vcf_filepath}')
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
        vcf_data, data_dict = self.read_data()
        sfs = self.create_SFS(data_dict)
        mega_result_dict = {} 

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
                folderpath=self.experiment_directory,
                num_windows=self.config['num_windows'],
                param_sample=None,
                p_guess=[0.25, 0.75, 0.1, 0.05, 20000],
                maxiter=self.config['maxiter']
            )
            momentsLD_results = {
                'opt_params_momentsLD': opt_params_momentsLD
            }
            mega_result_dict.update(momentsLD_results)

        list_of_mega_result_dicts = [mega_result_dict]
        merged_dict = get_dicts(list_of_mega_result_dicts)

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
        concatenated_features = {}
        concatenated_targets = {}
        stage = 'inference'

        if self.config['dadi_analysis']:
            concatenated_array = np.column_stack([dadi_dict['opt_params'][key] for key in dadi_dict['opt_params']])
            features_dict[stage]["dadi"] = concatenated_array
            if 'simulated_params' in dadi_dict:
                concatenated_array = np.column_stack([dadi_dict['simulated_params'][key] for key in dadi_dict['simulated_params']])
                targets_dict[stage]["dadi"] = concatenated_array

        if self.config['moments_analysis']:
            concatenated_array = np.column_stack([moments_dict['opt_params'][key] for key in moments_dict['opt_params']])
            features_dict[stage]["moments"] = concatenated_array
            if 'simulated_params' in moments_dict:
                concatenated_array = np.column_stack([moments_dict['simulated_params'][key] for key in moments_dict['simulated_params']])
                targets_dict[stage]["moments"] = concatenated_array

        if self.config['momentsLD_analysis']:
            concatenated_array = np.column_stack([momentsLD_dict['opt_params'][key] for key in momentsLD_dict['opt_params']])
            features_dict[stage]["momentsLD"] = concatenated_array
            if 'simulated_params' in momentsLD_dict:
                concatenated_array = np.column_stack([momentsLD_dict['simulated_params'][key] for key in momentsLD_dict['simulated_params']])
                targets_dict[stage]["momentsLD"] = concatenated_array

        concat_feats = np.column_stack([features_dict[stage][subkey] for subkey in features_dict[stage]])
        if targets_dict[stage]:
            concat_targets = np.column_stack([targets_dict[stage]['dadi']])
            concatenated_targets[stage] = concat_targets
        concatenated_features[stage] = concat_feats
        inference_obj = {}
        inference_obj["features"] = concatenated_features['inference']

        with open(f"{self.experiment_directory}/inference_results_obj.pkl", "wb") as file:
            pickle.dump(inference_obj, file)

        print("Inference complete!")

    def evaluate_model(self, snn_model, inference_results):
        inference_features = torch.tensor(inference_results["features"], dtype=torch.float32)
        inferred_params = snn_model.predict(inference_features)
        np.savetxt(f'{self.experiment_directory}/inferred_params_GHIST_bottleneck.txt', inferred_params, delimiter=' ', fmt='%.5f')

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--experiment_directory', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize Inference object
    inference = Inference(
        vcf_filepath=f'{config["vcf_filepath"]}',
        txt_filepath=f'{config["txt_filepath"]}',
        popname=config['popname'],
        config=config,
        experiment_directory=config['experiment_directory']
    )

    # Run feature extraction and inference
    inference.obtain_features()