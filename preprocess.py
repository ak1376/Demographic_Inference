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

from parameter_inference import run_inference_dadi, run_inference_moments, run_inference_momentsLD

def delete_vcf_files(directory):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    
    # Delete all files
    [os.remove(file) for file in files]
    
    print(f"Deleted {len(files)} files from {directory}")

class Processor: 
    def __init__(self, experiment_config, experiment_directory, L = 1e8, recombination_rate=1e-8, mutation_rate=1e-7, window_length = 1e6): # CHANGE WINDOW LENGTH SIZE LATER       
       
        self.experiment_config = experiment_config
        self.experiment_directory = experiment_directory
        
        self.upper_bound_params = self.experiment_config['upper_bound_params']
        self.lower_bound_params = self.experiment_config['lower_bound_params']
        
        self.param_storage = []
        self.ts_storage = []
        self.sfs = []

        self.num_sims = self.experiment_config['num_sims']

        self.num_samples = self.experiment_config['num_samples']
        self.L = L
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate

        self.window_length = window_length
        self.num_windows =self.experiment_config['num_windows'] # This is just for momentsLD. Number of windows to split the genome into.

        # Not sure if the below code is necessary yet. 
        # self.temporary_realizations_location = os.path.join(os.getcwd(), "temporary_folder_realizations")
        # os.makedirs(self.temporary_realizations_location, exist_ok=True)

    def bottleneck_model(self, sampled_params):
        
        N0, nuB, nuF, t_bottleneck_start, t_bottleneck_end = sampled_params['N0'], sampled_params['Nb'], sampled_params['N_recover'], sampled_params['t_bottleneck_start'], sampled_params['t_bottleneck_end']
        b = demes.Builder()
        b.add_deme(
            "A",
            epochs=[
                dict(start_size=N0, end_time=t_bottleneck_start),
                dict(start_size=nuB, end_time=t_bottleneck_end),
                dict(start_size=nuF, end_time=0)
            ]
        )
        g = b.resolve()

        return g
    
    def get_random_windows(self, ts, window_length, num_windows):
        """
        Get random windows from the tree sequence.

        Parameters:
        - ts: tskit.TreeSequence object
        - window_length: Length of each window (in base pairs)
        - num_windows: Number of random windows to extract

        Returns:
        - windows: List of tskit.TreeSequence objects containing the random windows
        """
        windows = []
        n_samples = int(ts.sequence_length - window_length)
        for _ in range(num_windows):
            start = np.random.randint(0, n_samples)
            end = start + window_length
            windows.append(ts.keep_intervals([[start, end]]))

        return windows

    def run_msprime_replicates(self, g):
        demog = msprime.Demography.from_demes(g)
        ts = msprime.sim_ancestry(
            {"A": self.num_samples},
            demography=demog,
            sequence_length=self.L,
            recombination_rate=self.recombination_rate,
            random_seed=295
        )
        ts = msprime.sim_mutations(ts, rate=self.mutation_rate)
        
        self.folderpath = f'experiments/{self.experiment_directory}/sampled_genome_windows'
        os.makedirs(self.folderpath, exist_ok=True)

        delete_vcf_files(self.folderpath)

        windows = self.get_random_windows(ts, self.window_length, self.num_windows)
        
        for ii, ts_window in tqdm(enumerate(windows), total = len(windows)):
            vcf_name = os.path.join(self.folderpath,f"bottleneck_window.{ii}.vcf")
            with open(vcf_name, "w+") as fout:
                ts_window.write_vcf(fout, allow_position_zero=True)
            os.system(f"gzip {vcf_name}")

    def write_samples_and_rec_map(self):
        folderpath = self.folderpath
        with open(os.path.join(folderpath, "samples.txt"), "w+") as fout:
            fout.write("sample\tpop\n")
            for ii in range(self.num_samples):
                fout.write(f"tsk_{ii}\tA\n")

        with open(os.path.join(folderpath, "flat_map.txt"), "w+") as fout:
            fout.write("pos\tMap(cM)\n")
            fout.write("0\t0\n")
            fout.write(f"{self.L}\t{self.recombination_rate * self.L * 100}\n")
    
    def sample_params(self):
        sampled_params = {}
        for key in self.lower_bound_params:
            sampled_value = np.random.uniform(self.lower_bound_params[key], self.upper_bound_params[key])
            sampled_params[key] = int(sampled_value)

        return sampled_params

    def create_SFS(self, sampled_params, mode = 'pretrain'):
        ''' 
        If we are in pretraining mode we will use a simulated SFS. If we are in inference mode we will use a real SFS.

        '''

        if mode == 'pretrain':
            demography = msprime.Demography()
            demography.add_population(name="A", initial_size=sampled_params['N_recover'])
            demography.add_population_parameters_change(sampled_params['t_bottleneck_end'], initial_size=sampled_params['Nb'])
            demography.add_population_parameters_change(sampled_params['t_bottleneck_start'], initial_size=sampled_params['N0'])

            demes_model = demography.to_demes()

            sfs = dadi.Demes.SFS(demes_model, sampled_demes= ["A"], sample_sizes = [2*self.num_samples], pts = 4*self.num_samples)
        
        else:
            '''
            Fill this in later. This is for inference mode on the GHIST data. 
            '''
            pass

        return sfs
    
    def run(self):

        sample_params_storage = [] 
        model_sfs = []

        opt_params_dadi_list = []
        model_sfs_dadi_list = []
        opt_theta_dadi_list = []

        opt_params_moments_list = []
        model_sfs_moments_list = []
        opt_theta_moments_list = []

        opt_params_momentsLD_list = []

        for i in tqdm(np.arange(self.num_sims)):
            sampled_params = self.sample_params()
            sample_params_storage.append(sampled_params)

            # Need to create an SFS for dadi and moments
            sfs = self.create_SFS(sampled_params, mode = 'pretrain')
            model_sfs.append(sfs)

            # Now need to simulate the process with these generative parameters, window, and then save the windows as VCF files
            # g = self.bottleneck_model(sampled_params)
            # self.run_msprime_replicates(g)
            # self.write_samples_and_rec_map()

            model_sfs_dadi, opt_theta_dadi, opt_params_dict_dadi = run_inference_dadi(sfs, p0 = [0.25, 0.75, 0.1, 0.05], lower_bound = [0.01, 0.01, 0.01, 0.01], upper_bound = [10, 10, 10, 10], sampled_params = sampled_params, num_samples = self.num_samples, maxiter = 100)
            model_sfs_moments, opt_theta_moments, opt_params_dict_moments = run_inference_moments(sfs, p0 = [0.25, 0.75, 0.1, 0.05], lower_bound = [0.01, 0.01, 0.01, 0.01], upper_bound = [10, 10, 10, 10], sampled_params = sampled_params, maxiter = 100)
            # opt_params_momentsLD = run_inference_momentsLD(folderpath = self.folderpath, num_windows = self.num_windows, param_sample = sampled_params, p_guess = [0.25, 0.75, 0.1, 0.05, 20000], maxiter = 100)

            opt_params_dadi_list.append(opt_params_dict_dadi)
            model_sfs_dadi_list.append(model_sfs_dadi)
            opt_theta_dadi_list.append(opt_theta_dadi)

            opt_params_moments_list.append(opt_params_dict_moments)
            model_sfs_moments_list.append(model_sfs_moments)
            opt_theta_moments_list.append(opt_theta_moments)

            # opt_params_momentsLD_list.append(opt_params_momentsLD)
        
        # Let's return dictionaries for now.
        # simulation_dict = {
        #     'sampled_params': sample_params_storage,
        #     'model_sfs': model_sfs
        # }

        dadi_dict = {
            'model_sfs': model_sfs,
            'simulated_params': sample_params_storage,
            'opt_params': opt_params_dadi_list,
            'model_sfs': model_sfs_dadi_list,
            'opt_theta': opt_theta_dadi_list
        }

        moments_dict = {
            'model_sfs': model_sfs,
            'simulated_params': sample_params_storage,
            'opt_params': opt_params_moments_list,
            'model_sfs': model_sfs_moments_list,
            'opt_theta': opt_theta_moments_list
        }

        # momentsLD_dict = {
        #     'opt_params': opt_params_momentsLD_list
        # }

        # return simulation_dict, dadi_dict, moments_dict, momentsLD_dict

        return dadi_dict, moments_dict