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
            sampled_value = np.random.uniform(
                self.lower_bound_params[key], self.upper_bound_params[key]
            )

            if self.demographic_model == "split_isolation_model":
                if key == "m_pre_isolation" or key == "m_post_isolation":
                    sampled_params[key] = sampled_value
                else:
                    sampled_params[key] = int(sampled_value)
            else:
                sampled_params[key] = int(sampled_value)


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
                        sampled_params=sampled_params,
                        num_samples=100, #TODO: Need to change this to not rely on a hardcoded value
                        demographic_model=self.experiment_config['demographic_model'],
                        maxiter=self.maxiter,
                        mutation_rate=self.mutation_rate,
                        length=self.L,
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
                        sampled_params=sampled_params,
                        demographic_model=self.experiment_config['demographic_model'],
                        maxiter=self.maxiter,
                        use_FIM=self.experiment_config["use_FIM"],
                        mutation_rate=self.mutation_rate,
                        length=self.L,
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

        # Ground truth params (each row is a simulation)
        df = pd.DataFrame([result['simulated_params'] for result in list_of_mega_result_dicts])
        targets = df.values

        # Step 1: Initialize an empty list to collect analysis data arrays
        analysis_data = []

        # Step 2: Dynamically extract and append data based on configuration
        for analysis_type in ['dadi_analysis', 'moments_analysis', 'momentsLD_analysis']:
            if self.experiment_config.get(analysis_type):
                # Extract the appropriate data dynamically based on the analysis type
                analysis_key = 'opt_params_' + analysis_type.split('_')[0]  # This maps to 'opt_params_dadi', 'opt_params_moments', etc.
                analysis_data.append([list(result[analysis_key].values()) for result in list_of_mega_result_dicts])

        # Step 3: Convert the collected data into NumPy arrays
        analysis_arrays = [np.array(data) for data in analysis_data]  # List of arrays, one for each analysis type

        # Step 4: Stack the arrays along a new axis if there are multiple analyses
        if len(analysis_arrays) > 1:
            features = np.stack(analysis_arrays, axis=1)  # Stack along a new axis (num_sims, num_analyses, num_params)
        else:
            features = analysis_arrays[0]  # If there's only one analysis, just use that array

        return features, targets