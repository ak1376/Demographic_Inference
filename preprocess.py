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
from sklearn.utils import resample

from parameter_inference import (
    run_inference_dadi,
    run_inference_moments,
    run_inference_momentsLD,
)

def generate_window(ts, window_length, n_samples):
    start = np.random.randint(0, n_samples- window_length)
    end = start + window_length
    return ts.keep_intervals([[start, end]])


# def get_random_windows_parallel(ts, window_length, num_windows):
#     """
#     Get random windows from the tree sequence in parallel.

#     Parameters:
#     - ts: tskit.TreeSequence object
#     - window_length: Length of each window (in base pairs)
#     - num_windows: Number of random windows to extract

#     Returns:
#     - windows: List of tskit.TreeSequence objects containing the random windows
#     """
#     n_samples = int(ts.sequence_length - window_length)

#     # Distribute the window creation tasks across multiple workers
#     futures = [
#         generate_window.remote(ts, window_length, n_samples) for _ in range(num_windows)
#     ]

#     # Collect the results
#     windows = ray.get(futures)

#     return windows


def process_window(ts_window, folderpath, ii):
    vcf_name = os.path.join(folderpath, f"bottleneck_window.{ii}.vcf")

    with open(vcf_name, "w+") as fout:
        ts_window.write_vcf(fout, allow_position_zero=True)

    os.system(f"gzip -f {vcf_name}")

    return vcf_name  # Optionally return the filename or any other relevant information


# def parallel_process_windows(windows, folderpath):
#     # Initialize Ray if not already initialized
#     if not ray.is_initialized():
#         ray.init(num_cpus=os.cpu_count())

#     # Create a list to store futures
#     futures = []

#     # Launch tasks in parallel
#     for ii, ts_window in tqdm(enumerate(windows), total=len(windows)):
#         future = process_window.remote(ts_window, folderpath, ii)
#         futures.append(future)

#     # Collect results (this will block until all tasks are done)
#     results = ray.get(futures)

#     # Optionally, print or return results
#     print("All windows have been processed.")
#     return results


def delete_vcf_files(directory):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory, "*"))

    # Delete all files
    [os.remove(file) for file in files]

    print(f"Deleted {len(files)} files from {directory}")


def process_single_simulation(
    i,
    sample_params_func,
    create_SFS_func,
    bottleneck_model_func,
    run_msprime_replicates_func,
    write_samples_and_rec_map_func,
    run_inference_dadi_func=None,
    run_inference_moments_func=None,
    run_inference_momentsLD_func=None,
    folderpath=None,
    num_windows=None,
    num_samples=None,
    maxiter=None,
    mutation_rate = 1.26e-8
):
    sampled_params = sample_params_func()
    sfs = create_SFS_func(sampled_params, num_samples)

    # Simulate process and save windows as VCF files
    if run_inference_momentsLD_func:
        g = bottleneck_model_func(sampled_params)
        run_msprime_replicates_func(g)
        samples_file, flat_map_file = write_samples_and_rec_map_func()

    # Initialize result dictionary
    results = {
        "sampled_params": sampled_params,
        "sfs": sfs,
    }

    # Conditional analysis based on provided functions
    if run_inference_dadi_func:
        model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = run_inference_dadi_func(
            sfs,
            p0=[0.25, 0.75, 0.1, 0.05],
            lower_bound=[0.001, 0.001, 0.001, 0.001],
            upper_bound=[10, 10, 10, 10],
            sampled_params=sampled_params,
            num_samples=num_samples,
            maxiter=maxiter,
            mutation_rate = mutation_rate
        )
        results.update(
            {
                "opt_params_dict_dadi": opt_params_dict_dadi,
                "model_sfs_dadi": model_sfs_dadi,
                "opt_theta_dadi": opt_theta_dadi,
            }
        )

    if run_inference_moments_func:
        model_sfs_moments, opt_theta_moments, opt_params_dict_moments = (
            run_inference_moments_func(
                sfs,
                p0=[0.25, 0.75, 0.1, 0.05],
                lower_bound=[0.001, 0.001, 0.001, 0.001],
                upper_bound=[10, 10, 10, 10],
                sampled_params=sampled_params,
                maxiter=maxiter,
                mutation_rate = mutation_rate
            )
        )
        results.update(
            {
                "opt_params_dict_moments": opt_params_dict_moments,
                "model_sfs_moments": model_sfs_moments,
                "opt_theta_moments": opt_theta_moments,
            }
        )

    if run_inference_momentsLD_func:
        opt_params_momentsLD = run_inference_momentsLD_func(
            folderpath=folderpath,
            num_windows=num_windows,
            param_sample=sampled_params,
            p_guess=[0.25, 0.75, 0.1, 0.05, 20000],
            maxiter=maxiter
        )
        results["opt_params_momentsLD"] = opt_params_momentsLD

    return results


# def extract_and_process_features(simulated_params, opt_params, analysis_type, stage, experiment_directory, remove_outliers=True):
#     if not simulated_params or not opt_params:
#         print(f"Skipping {analysis_type} {stage} due to empty data")
#         return None, None, None

#     features, targets = extract_features(simulated_params, opt_params, normalization=False)
#     # error_value = root_mean_squared_error(y_true=targets, y_pred=features)
#     # print(f"The error value for {analysis_type} {stage} is: {error_value}")

#     outlier_indices = find_outlier_indices(features)
#     np.savetxt(os.path.join(experiment_directory, f"outlier_indices_{analysis_type}_{stage}.csv"), outlier_indices, delimiter=",")

#     if remove_outliers:
#         features = np.delete(features, np.unique(outlier_indices), axis=0)
#         targets = np.delete(targets, outlier_indices, axis=0)

#     visualizing_results({
#         "simulated_params": simulated_params,
#         "opt_params": opt_params
#     }, f"{analysis_type}_{stage}", save_loc=experiment_directory, outlier_indices=outlier_indices)

#     feature_names = [
#         f"Nb_opt_{analysis_type}",
#         f"N_recover_opt_{analysis_type}",
#         f"t_bottleneck_start_opt_{analysis_type}",
#         f"t_bottleneck_end_opt_{analysis_type}"
#     ]

#     return features, targets, feature_names


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
        self.features = {stage: {} for stage in ["training", "validation", "testing"]}
        self.targets = {stage: {} for stage in ["training", "validation", "testing"]}
        self.feature_names = {}

    def process_batch(self, batch_data, analysis_type, stage, normalization=False):
        if (
            not batch_data
            or "simulated_params" not in batch_data
            or "opt_params" not in batch_data
        ):
            print(f"Skipping {analysis_type} {stage} due to empty or invalid data")
            return

        features, targets = extract_features(
            batch_data["simulated_params"],
            batch_data["opt_params"],
            normalization=normalization,
        )

        if analysis_type not in self.features[stage]:
            self.features[stage][analysis_type] = []
            self.targets[stage][analysis_type] = []

        self.features[stage][analysis_type].extend(features)
        self.targets[stage][analysis_type].extend(targets)

        if analysis_type not in self.feature_names:
            self.feature_names[analysis_type] = [
                f"Nb_opt_{analysis_type}",
                f"N_recover_opt_{analysis_type}",
                f"t_bottleneck_start_opt_{analysis_type}",
                f"t_bottleneck_end_opt_{analysis_type}"
            ]
            

        # error_value = root_mean_squared_error(targets, features)
        # print(f"Batch error value for {analysis_type} {stage}: {error_value}")

        # visualizing_results(batch_data, save_loc = self.experiment_directory, analysis=f"{analysis_type}_{stage}")

    def finalize_processing(self, remove_outliers=True):
        """
        This function will create the numpy array of features and targets across all analysis types and stages. If the user specifies to remove outliers, it will remove them from the data and then resample the rows for concatenation
        """
        for stage in self.features:
            for analysis_type in self.features[stage]:
                features = np.array(self.features[stage][analysis_type])
                targets = np.array(self.targets[stage][analysis_type])

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

                self.features[stage][analysis_type] = features
                self.targets[stage][analysis_type] = targets

                # error_value = root_mean_squared_error(targets, features)
                # print(f"Final error value for {analysis_type} {stage}: {error_value}")

        concatenated_features, concatenated_targets = (
            self.concatenate_features_and_targets()
        )
        return concatenated_features, concatenated_targets, self.feature_names

    def concatenate_features_and_targets(self):
        concatenated_features = {}
        concatenated_targets = {}
        concatenated_feature_names = []

        # Concatenate feature names only once
        for analysis_type in sorted(self.feature_names.keys()): #type: ignore
            concatenated_feature_names.extend(self.feature_names[analysis_type])

        for stage in self.features:
            all_features = []
            all_targets = []
            for analysis_type in sorted(
                self.features[stage].keys()
            ):  # Sort to ensure consistent order
                all_features.append(self.features[stage][analysis_type])
                all_targets.append(self.targets[stage][analysis_type])

            # Find the minimum number of samples across all analysis types
            min_samples = min(len(features) for features in all_features)

            # Resample features and targets to have the same number of samples
            resampled_features = []
            resampled_targets = []
            for features, targets in zip(all_features, all_targets):
                if len(features) > min_samples:
                    resampled_features.append(
                        resample(features, n_samples=min_samples, random_state=42)
                    )
                    resampled_targets.append(
                        resample(targets, n_samples=min_samples, random_state=42)
                    )
                else:
                    resampled_features.append(features)
                    resampled_targets.append(targets)

            # Concatenate the resampled features and targets
            concatenated_features[stage] = np.hstack(resampled_features)
            concatenated_targets[stage] = resampled_targets[
                0
            ]  # Assuming targets are the same for all analysis types

            # print(f"Stage: {stage}")
            # print(f"  Concatenated features shape: {concatenated_features[stage].shape}")
            # print(f"  Concatenated targets shape: {concatenated_targets[stage].shape}")

        self.feature_names = concatenated_feature_names
        return concatenated_features, concatenated_targets

    def resample_features_and_targets(self, stage):

        analysis_types = list(self.features[stage].keys())
        if len(analysis_types) < 2:
            return  # No need to resample if there's only one or no analysis type

        # Find the minimum number of samples across all analysis types
        min_samples = min(len(self.features[stage][at]) for at in analysis_types)

        for analysis_type in analysis_types:
            features = np.array(self.features[stage][analysis_type])
            targets = np.array(self.targets[stage][analysis_type])

            if len(features) > min_samples:
                # Randomly sample without replacement
                indices = np.random.choice(len(features), min_samples, replace=False)
                self.features[stage][analysis_type] = features[indices]
                self.targets[stage][analysis_type] = targets[indices]

        # Print the number of samples after resampling
        # for analysis_type in analysis_types:
        #     print(f"Samples for {stage}:{analysis_type} after resampling: {len(self.features[stage][analysis_type])}")


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

        # Not sure if the below code is necessary yet.
        # self.temporary_realizations_location = os.path.join(os.getcwd(), "temporary_folder_realizations")
        # os.makedirs(self.temporary_realizations_location, exist_ok=True)

    def bottleneck_model(self, sampled_params):

        N0, nuB, nuF, t_bottleneck_start, t_bottleneck_end = (
            sampled_params["N0"],
            sampled_params["Nb"],
            sampled_params["N_recover"],
            sampled_params["t_bottleneck_start"],
            sampled_params["t_bottleneck_end"],
        )
        b = demes.Builder()
        b.add_deme(
            "A",
            epochs=[
                dict(start_size=N0, end_time=t_bottleneck_start),
                dict(start_size=nuB, end_time=t_bottleneck_end),
                dict(start_size=nuF, end_time=0),
            ],
        )
        g = b.resolve()

        return g

    # @ray.remote
    # def get_random_windows(self, ts, window_length, num_windows):
    #     """
    #     Get random windows from the tree sequence.

    #     Parameters:
    #     - ts: tskit.TreeSequence object
    #     - window_length: Length of each window (in base pairs)
    #     - num_windows: Number of random windows to extract

    #     Returns:
    #     - windows: List of tskit.TreeSequence objects containing the random windows
    #     """
    #     windows = []
    #     n_samples = int(ts.sequence_length - window_length)
    #     for _ in range(num_windows):
    #         start = np.random.randint(0, n_samples)
    #         end = start + window_length
    #         windows.append(ts.keep_intervals([[start, end]])) #TODO: Use genetic distance instead of base pairs

    #     return windows

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

        # delete_vcf_files(self.folderpath)

        # Generate random windows in parallel

        windows = [generate_window(ts, self.window_length, self.L) for _ in range(self.num_windows)]

        # windows = self.get_random_windows(ts, self.window_length, self.num_windows)

        # parallel_process_windows(windows, self.folderpath)

        for ii, ts_window in tqdm(enumerate(windows), total = len(windows)):
            vcf_name = os.path.join(self.folderpath,f"bottleneck_window.{ii}.vcf")
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

    @staticmethod
    def create_SFS(sampled_params, length = 1e7, mutation_rate = 1.26e-8, num_samples = 100):
        """
        If we are in pretraining mode we will use a simulated SFS. If we are in inference mode we will use a real SFS.

        """

        demography = msprime.Demography()
        demography.add_population(
            name="A", initial_size=sampled_params["N_recover"]
        )
        demography.add_population_parameters_change(
            sampled_params["t_bottleneck_end"], initial_size=sampled_params["Nb"]
        )
        demography.add_population_parameters_change(
            sampled_params["t_bottleneck_start"], initial_size=sampled_params["N0"]
        )

        demes_model = demography.to_demes()

        sfs = demes_obj.SFS(
            demes_model,
            sampled_demes=["A"],
            sample_sizes=[2 * num_samples],
            # Ne = sampled_params["N0"]
            u = mutation_rate
        )

        #multiply sfs by L
        sfs*=length

        return sfs

    def run(self, indices_of_interest):

        # Initialize lists to store results
        sample_params_storage = []
        model_sfs = []

        opt_params_dadi_list = []
        model_sfs_dadi_list = []
        opt_theta_dadi_list = []

        opt_params_moments_list = []
        model_sfs_moments_list = []
        opt_theta_moments_list = []

        opt_params_momentsLD_list = []

        # Create a list of futures to run simulations in parallel

        for i in tqdm(range(len(indices_of_interest))):
            sampled_params = self.sample_params()
            sfs = self.create_SFS(sampled_params, length = self.L, mutation_rate=self.mutation_rate, num_samples = self.num_samples)

            # Initialize result dictionary
            results = {
                "sampled_params": sampled_params,
                "sfs": sfs
            }

            # Simulate process and save windows as VCF files
            if self.experiment_config["momentsLD_analysis"]:
                g = self.bottleneck_model(sampled_params)
                self.run_msprime_replicates(g)
                samples_file, flat_map_file = self.write_samples_and_rec_map()

            # Conditional analysis based on provided functions
            if self.experiment_config['dadi_analysis']:
                model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = run_inference_dadi(
                    sfs,
                    p0=[0.25, 0.75, 0.1, 0.05],
                    lower_bound=[0.001, 0.001, 0.001, 0.001],
                    upper_bound=[10, 10, 10, 10],
                    sampled_params=sampled_params,
                    num_samples=self.num_samples,
                    maxiter=self.maxiter,
                    mutation_rate=self.mutation_rate,
                    length = self.L
                )
                results.update(
                    {
                        "opt_params_dict_dadi": opt_params_dict_dadi,
                        "model_sfs_dadi": model_sfs_dadi,
                        "opt_theta_dadi": opt_theta_dadi,
                    }
                )

            if self.experiment_config['moments_analysis']:
                model_sfs_moments, opt_theta_moments, opt_params_dict_moments = (
                    run_inference_moments(
                        sfs,
                        p0=[0.25, 0.75, 0.1, 0.05],
                        lower_bound=[0.001, 0.001, 0.001, 0.001],
                        upper_bound=[10, 10, 10, 10],
                        sampled_params=sampled_params,
                        maxiter=self.maxiter,
                        use_FIM = self.experiment_config["use_FIM"]
                    )
                )
                results.update(
                    {
                        "opt_params_dict_moments": opt_params_dict_moments,
                        "model_sfs_moments": model_sfs_moments,
                        "opt_theta_moments": opt_theta_moments,
                    }
                )

            if self.experiment_config['momentsLD_analysis']:
                opt_params_momentsLD = run_inference_momentsLD(
                    folderpath=self.folderpath,
                    num_windows=self.num_windows,
                    param_sample=sampled_params,
                    p_guess=[0.25, 0.75, 0.1, 0.05, 20000],
                    maxiter=self.maxiter
                )

                results.update({"opt_params_momentsLD": opt_params_momentsLD})

        sample_params_storage.append(results["sampled_params"])
        model_sfs.append(results["sfs"])

        if self.experiment_config["dadi_analysis"]:
            opt_params_dadi_list.append(results["opt_params_dict_dadi"])
            model_sfs_dadi_list.append(results["model_sfs_dadi"])
            opt_theta_dadi_list.append(results["opt_theta_dadi"])

        if self.experiment_config["moments_analysis"]:
            opt_params_moments_list.append(results["opt_params_dict_moments"])
            model_sfs_moments_list.append(results["model_sfs_moments"])
            opt_theta_moments_list.append(results["opt_theta_moments"])

        if self.experiment_config["momentsLD_analysis"]:
            opt_params_momentsLD_list.append(results["opt_params_momentsLD"])

        # Create the output dictionaries
        dadi_dict = (
            {
                "model_sfs": model_sfs,
                "simulated_params": sample_params_storage,
                "opt_params": opt_params_dadi_list,
                "model_sfs": model_sfs_dadi_list,
                "opt_theta": opt_theta_dadi_list,
            }
            if self.experiment_config["dadi_analysis"]
            else {}
        )

        moments_dict = (
            {
                "model_sfs": model_sfs,
                "simulated_params": sample_params_storage,
                "opt_params": opt_params_moments_list,
                "model_sfs": model_sfs_moments_list,
                "opt_theta": opt_theta_moments_list,
            }
            if self.experiment_config["moments_analysis"]
            else {}
        )

        momentsLD_dict = (
            {
                "simulated_params": sample_params_storage,
                "opt_params": opt_params_momentsLD_list,
            }
            if self.experiment_config["momentsLD_analysis"]
            else {}
        )

        print("Length of dadi preprocessing (i.e. number of simulations): ", len(dadi_dict["model_sfs"]))
        print("Length of moments preprocessing (i.e. number of simulations): ", len(moments_dict["model_sfs"]))
        print("Length of momentsLD preprocessing (i.e. number of simulations): ", len(momentsLD_dict["opt_params"]))

        return dadi_dict, moments_dict, momentsLD_dict
