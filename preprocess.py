import os
import moments
from tqdm import tqdm
import numpy as np
import msprime
import dadi
import moments.Demes as demes_obj
import glob
import demes
from utils import (
    extract_features,
    visualizing_results,
    root_mean_squared_error,
    find_outlier_indices,
)
import time

import demographic_models

from sklearn.utils import resample

from parameter_inference import (
    run_inference_dadi,
    run_inference_moments,
    run_inference_momentsLD,
)

import allel


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

class FeatureExtractor:
    def __init__(
        self,
        experiment_directory,
        dadi_analysis=True,
        moments_analysis=True,
        momentsLD_analysis=True,
    ):
        self.experiment_directory = experiment_directory
        self.dadi_analysis = dadi_analysis
        self.moments_analysis = moments_analysis
        self.momentsLD_analysis = momentsLD_analysis

    def process_batch(
        self,
        batch_data,
        analysis_type,
        stage,
        features_dict={},
        targets_dict={},
        feature_names=[],
        normalization=False,
    ):

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

        if stage != "inference":
            features, targets = extract_features(
                simulated_params=batch_data["simulated_params"],
                opt_params=batch_data[f"opt_params_{analysis_type}"],
                normalization=normalization,
            )
        else:
            features = extract_features(
                simulated_params=None,
                opt_params=batch_data[f"opt_params"],
                normalization=normalization,
            )

        if analysis_type not in features_dict[stage]:
            features_dict[stage][analysis_type] = []
            if stage != "inference":
                targets_dict[stage][analysis_type] = []

        features_dict[stage][analysis_type].extend(features)

        if stage != "inference":
            targets_dict[stage][analysis_type].extend(targets)

        if analysis_type not in feature_names:
            feature_names[analysis_type] = [
                f"Nb_opt_{analysis_type}",
                f"N_recover_opt_{analysis_type}",
                f"t_bottleneck_start_opt_{analysis_type}",
                f"t_bottleneck_end_opt_{analysis_type}",
            ]

        if stage != "inference":

            return features_dict, targets_dict, feature_names
        else:
            return features_dict, feature_names

        # error_value = root_mean_squared_error(targets, features)
        # print(f"Batch error value for {analysis_type} {stage}: {error_value}")

        # visualizing_results(batch_data, save_loc = self.experiment_directory, analysis=f"{analysis_type}_{stage}")

    def finalize_processing(
        self, features_dict, targets_dict, feature_names, remove_outliers=True
    ):
        # TODO: I really need to rewrite this whole function. So many errors.
        # TODO: Maybe I can just eliminate this entire function altogether.
        """
        This function will create the numpy array of features and targets across all analysis types and stages. If the user specifies to remove outliers, it will remove them from the data and then resample the rows for concatenation
        """

        for stage in features_dict:
            for analysis_type in features_dict[stage]:
                features = np.array(features_dict[stage][analysis_type])

                if targets_dict:  # i.e. if we are not in inference mode
                    targets = np.array(targets_dict[stage][analysis_type])

                    outlier_indices = find_outlier_indices(features)
                    np.savetxt(
                        os.path.join(
                            self.experiment_directory,
                            f"outlier_indices_{analysis_type}_{stage}.csv",
                        ),
                        outlier_indices,
                        delimiter=",",
                    )

                    # Remove outliers
                    if remove_outliers:
                        features = np.delete(features, outlier_indices, axis=0)
                        targets = np.delete(targets, outlier_indices, axis=0)
                        # self.resample_features_and_targets(stage)

                    targets_dict[stage][analysis_type] = targets

                features_dict[stage][analysis_type] = features

                # error_value = root_mean_squared_error(targets, features)
                # print(f"Final error value for {analysis_type} {stage}: {error_value}")

        if targets_dict:
            concatenated_features, concatenated_targets = (
                self.concatenate_features_and_targets(
                    stage=stage,
                    feature_names=feature_names,
                    features_dict=features_dict,
                    targets_dict=targets_dict,
                )
            )  # type:ignore
        else:
            concatenated_features = self.concatenate_features_and_targets(
                stage=stage,
                feature_names=feature_names,
                features_dict=features_dict,
                targets_dict=None,
            )  # type:ignore

        if targets_dict:
            return concatenated_features, concatenated_targets, feature_names
        else:
            return concatenated_features, feature_names

    # TODO: Need to modify this s.t. I am not relying on the class instances for any of this.
    @staticmethod
    def concatenate_features_and_targets(
        feature_names, features_dict, targets_dict, stage
    ):
        concatenated_features = {}
        concatenated_targets = {}
        concatenated_feature_names = []

        # Concatenate feature names only once
        for analysis_type in sorted(feature_names.keys()):  # type: ignore
            concatenated_feature_names.extend(feature_names[analysis_type])

        for stage_key in features_dict:
            all_features = []
            all_targets = [] if stage != "inference" else None

            for analysis_type in sorted(
                features_dict[stage_key].keys()
            ):  # Sort to ensure consistent order
                all_features.append(features_dict[stage_key][analysis_type])
                if stage != "inference":
                    all_targets.append( # type:ignore
                        targets_dict[stage_key][analysis_type]
                    )  # type:ignore

            # Find the minimum number of samples across all analysis types
            min_samples = min(len(features) for features in all_features)

            # Resample features (and targets, if not in inference mode) to have the same number of samples
            resampled_features = []
            resampled_targets = [] if stage != "inference" else None

            for features, targets in zip(all_features, all_targets or []):
                if len(features) > min_samples:
                    resampled_features.append(
                        resample(features, n_samples=min_samples, random_state=42)
                    )
                    if stage != "inference":
                        resampled_targets.append(  # type:ignore
                            resample(targets, n_samples=min_samples, random_state=42)
                        )
                else:
                    resampled_features.append(features)
                    if stage != "inference":
                        resampled_targets.append(targets)  # type:ignore

            # Concatenate the resampled features (and targets, if not in inference mode)
            if targets_dict == {}:
                resampled_features = all_features.copy()

            concatenated_features[stage_key] = np.hstack(resampled_features)

            if stage != "inference":
                concatenated_targets[stage_key] = resampled_targets[ # type:ignore
                    0
                ]  # Assuming targets are the same for all analysis types #type:ignore

            # Return concatenated_features if stage is 'inference'
            if stage == "inference":
                return concatenated_features

            # Otherwise, return both concatenated_features and concatenated_targets
            return concatenated_features, concatenated_targets

    # TODO: Again, need to modify this s.t. I am not relying on the class instances for any of this.
    def resample_features_and_targets(self, features_dict, targets_dict, stage):

        analysis_types = list(features_dict[stage].keys())
        if len(analysis_types) < 2:
            return  # No need to resample if there's only one or no analysis type

        # Find the minimum number of samples across all analysis types
        min_samples = min(len(features_dict[stage][at]) for at in analysis_types)

        for analysis_type in analysis_types:
            features = np.array(features_dict[stage][analysis_type])
            targets = np.array(targets_dict[stage][analysis_type])

            if len(features) > min_samples:
                # Randomly sample without replacement
                indices = np.random.choice(len(features), min_samples, replace=False)
                features_dict[stage][analysis_type] = features[indices]
                targets_dict[stage][analysis_type] = targets[indices]

def get_dicts(list_of_mega_result_dicts):
    """
    Concatenate the values for each subdict and each main key across list elements
    """

    merged_dict = {}

    for dictionary in list_of_mega_result_dicts:
        for key, value in dictionary.items():
            if key in merged_dict:
                if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                    # Merge nested dictionaries by combining each subkey
                    for subkey, subvalue in value.items():
                        if subkey in merged_dict[key]:
                            # Append conflicting values to a list
                            if not isinstance(merged_dict[key][subkey], list):
                                merged_dict[key][subkey] = [merged_dict[key][subkey]]
                            merged_dict[key][subkey].append(subvalue)
                        else:
                            merged_dict[key][subkey] = subvalue
                else:
                    # If the key exists but is not a dictionary, overwrite the value
                    if not isinstance(merged_dict[key], list):
                        merged_dict[key] = [merged_dict[key]]
                    merged_dict[key].append(value)
            else:
                # If the key does not exist in the merged dictionary, add it
                merged_dict[key] = value

    return merged_dict


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
                {"A": self.num_samples},
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
                        lower_bound=[1e-4, 1e-4, 1e-4, 1e-4],
                        upper_bound=[None, None, None, None],
                        sampled_params=sampled_params,
                        num_samples=self.num_samples,
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
                        lower_bound=[1e-4, 1e-4, 1e-4, 1e-4],
                        upper_bound=[None, None, None, None],
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

        return get_dicts(list_of_mega_result_dicts)