import os
import moments
from tqdm import tqdm
import numpy as np
import msprime
import dadi
import dadi.Demes
import glob
import demes
from utils import save_windows_to_vcf
import ray

from parameter_inference import (
    run_inference_dadi,
    run_inference_moments,
    run_inference_momentsLD,
)


@ray.remote
def generate_window(ts, window_length, n_samples):
    start = np.random.randint(0, n_samples)
    end = start + window_length
    return ts.keep_intervals([[start, end]])


@ray.remote
def get_random_windows_parallel(ts, window_length, num_windows):
    """
    Get random windows from the tree sequence in parallel.

    Parameters:
    - ts: tskit.TreeSequence object
    - window_length: Length of each window (in base pairs)
    - num_windows: Number of random windows to extract

    Returns:
    - windows: List of tskit.TreeSequence objects containing the random windows
    """
    n_samples = int(ts.sequence_length - window_length)

    # Distribute the window creation tasks across multiple workers
    futures = [
        generate_window.remote(ts, window_length, n_samples) for _ in range(num_windows)
    ]

    # Collect the results
    windows = ray.get(futures)

    return windows


@ray.remote
def process_window(ts_window, folderpath, ii):
    vcf_name = os.path.join(folderpath, f"bottleneck_window.{ii}.vcf")

    with open(vcf_name, "w+") as fout:
        ts_window.write_vcf(fout, allow_position_zero=True)

    os.system(f"gzip -f {vcf_name}")

    return vcf_name  # Optionally return the filename or any other relevant information


def parallel_process_windows(windows, folderpath):
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(num_cpus=os.cpu_count())

    # Create a list to store futures
    futures = []

    # Launch tasks in parallel
    for ii, ts_window in tqdm(enumerate(windows), total=len(windows)):
        future = process_window.remote(ts_window, folderpath, ii)
        futures.append(future)

    # Collect results (this will block until all tasks are done)
    results = ray.get(futures)

    # Optionally, print or return results
    print("All windows have been processed.")
    return results


def delete_vcf_files(directory):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory, "*"))

    # Delete all files
    [os.remove(file) for file in files]

    print(f"Deleted {len(files)} files from {directory}")

@ray.remote
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
):
    sampled_params = sample_params_func()
    sfs = create_SFS_func(sampled_params, mode="pretrain")
    
    # Simulate process and save windows as VCF files
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
            lower_bound=[0.01, 0.01, 0.01, 0.01],
            upper_bound=[10, 10, 10, 10],
            sampled_params=sampled_params,
            num_samples=num_samples,
            maxiter=maxiter,
        )
        results.update({
            "opt_params_dict_dadi": opt_params_dict_dadi,
            "model_sfs_dadi": model_sfs_dadi,
            "opt_theta_dadi": opt_theta_dadi,
        })

    if run_inference_moments_func:
        model_sfs_moments, opt_theta_moments, opt_params_dict_moments = (
            run_inference_moments_func(
                sfs,
                p0=[0.25, 0.75, 0.1, 0.05],
                lower_bound=[0.01, 0.01, 0.01, 0.01],
                upper_bound=[10, 10, 10, 10],
                sampled_params=sampled_params,
                maxiter=maxiter,
            )
        )
        results.update({
            "opt_params_dict_moments": opt_params_dict_moments,
            "model_sfs_moments": model_sfs_moments,
            "opt_theta_moments": opt_theta_moments,
        })

    if run_inference_momentsLD_func:
        opt_params_momentsLD = run_inference_momentsLD_func(
            folderpath=folderpath,
            num_windows=num_windows,
            param_sample=sampled_params,
            p_guess=[0.25, 0.75, 0.1, 0.05, 20000],
            maxiter=maxiter,
        )
        results["opt_params_momentsLD"] = opt_params_momentsLD

    return results

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

        self.num_sims = self.experiment_config["num_sims"]

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
        windows = get_random_windows_parallel.remote(
            ts, self.window_length, self.num_windows
        )

        # Retrieve and print the windows
        windows = ray.get(windows)

        # windows = self.get_random_windows(ts, self.window_length, self.num_windows)

        parallel_process_windows(windows, self.folderpath)

        # for ii, ts_window in tqdm(enumerate(windows), total = len(windows)):
        #     vcf_name = os.path.join(self.folderpath,f"bottleneck_window.{ii}.vcf")
        #     with open(vcf_name, "w+") as fout:
        #         ts_window.write_vcf(fout, allow_position_zero=True)
        #     os.system(f"gzip {vcf_name}")

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

    def create_SFS(self, sampled_params, mode="pretrain"):
        """
        If we are in pretraining mode we will use a simulated SFS. If we are in inference mode we will use a real SFS.

        """

        if mode == "pretrain":
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

            sfs = dadi.Demes.SFS(
                demes_model,
                sampled_demes=["A"],
                sample_sizes=[2 * self.num_samples],
                pts=4 * self.num_samples,
            )

        else:
            """
            Fill this in later. This is for inference mode on the GHIST data.
            """
            pass

        return sfs

    def run(self):

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

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create a list of futures to run simulations in parallel

        # i,
        # sample_params_func,
        # create_SFS_func,
        # bottleneck_model_func,
        # run_msprime_replicates_func,
        # write_samples_and_rec_map_func,
        # run_inference_dadi_func=None,
        # run_inference_moments_func=None,
        # run_inference_momentsLD_func=None,
        # folderpath=None,
        # num_windows=None,
        # num_samples=None,
        # maxiter=None,
        futures = [
            process_single_simulation.remote(
                i,
                self.sample_params,
                self.create_SFS,
                self.bottleneck_model,
                self.run_msprime_replicates,
                self.write_samples_and_rec_map,
                run_inference_dadi_func=run_inference_dadi if self.experiment_config['dadi_analysis'] else None,
                run_inference_moments_func=run_inference_moments if self.experiment_config['moments_analysis'] else None,
                run_inference_momentsLD_func = run_inference_momentsLD if self.experiment_config['momentsLD_analysis'] else None,
                folderpath=self.folderpath,
                num_windows=self.num_windows,
                num_samples=self.num_samples,
                maxiter=self.maxiter,
            )
            for i in range(self.num_sims)
        ]

        # Collect the results
        results = ray.get(futures)

        # Unpack the results
        for result in results:
            sample_params_storage.append(result["sampled_params"])
            model_sfs.append(result["sfs"])

            if self.dadi_analysis:
                opt_params_dadi_list.append(result["opt_params_dict_dadi"])
                model_sfs_dadi_list.append(result["model_sfs_dadi"])
                opt_theta_dadi_list.append(result["opt_theta_dadi"])

            if self.moments_analysis:
                opt_params_moments_list.append(result["opt_params_dict_moments"])
                model_sfs_moments_list.append(result["model_sfs_moments"])
                opt_theta_moments_list.append(result["opt_theta_moments"])

            if run_momentsLD:
                opt_params_momentsLD_list.append(result["opt_params_momentsLD"])

        # Create the output dictionaries
        dadi_dict = {
            "model_sfs": model_sfs,
            "simulated_params": sample_params_storage,
            "opt_params": opt_params_dadi_list,
            "model_sfs": model_sfs_dadi_list,
            "opt_theta": opt_theta_dadi_list,
        } if run_dadi else {}

        moments_dict = {
            "model_sfs": model_sfs,
            "simulated_params": sample_params_storage,
            "opt_params": opt_params_moments_list,
            "model_sfs": model_sfs_moments_list,
            "opt_theta": opt_theta_moments_list,
        } if run_moments else {}

        momentsLD_dict = {
            "simulated_params": sample_params_storage,
            "opt_params": opt_params_momentsLD_list,
        } if run_momentsLD else {}

        return dadi_dict, moments_dict, momentsLD_dict
