import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import allel
import dadi
from parameter_inference import run_inference_dadi, run_inference_moments, run_inference_momentsLD

class Inference:
    def __init__(self, vcf_filepath, txt_filepath, popname):
        self.vcf_filepath = vcf_filepath
        self.txt_filepath = txt_filepath
        self.popname = popname


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


    def obtain_features(self, dadi_features, moments_features, momentsLD_features):

        pass

    
    

    
















def inference():
    """
    This should do the parameter inference for dadi and moments
    """
    pass