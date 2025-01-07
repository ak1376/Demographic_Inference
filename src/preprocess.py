import os
import moments
from tqdm import tqdm
import numpy as np
import msprime
import dadi
import glob
import src.demographic_models as demographic_models

def generate_window(ts, window_length, n_samples):
    start = np.random.randint(0, n_samples - window_length)
    end = start + window_length
    return ts.keep_intervals([[start, end]]).trim()

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


    @staticmethod
    def run_msprime_replicates(ts, experiment_config, window_number, folderpath):

        folderpath = os.path.join(folderpath, f"window_{window_number}")

        # Create directory for storing VCFs
        os.makedirs(folderpath, exist_ok=True)

        # Generate random windows
        window = generate_window(ts, experiment_config['window_length'], experiment_config['genome_length'])

        # List to store file paths of the generated VCFs
        vcf_filepath = []

        # Iterate over windows and write VCFs
        vcf_name = os.path.join(folderpath, f'window.{window_number}.vcf')
        with open(vcf_name, "w+") as fout:
            window.write_vcf(fout, allow_position_zero=True)
            
        # Compress the VCF file
        os.system(f"gzip {vcf_name}")
        
        # # Store the compressed VCF file path
        vcf_filepath.append(f"{vcf_name}.gz")
        
        # Write the metadata file with all VCF file paths
        metadata_file = os.path.join(folderpath, "individual_file_metadata.txt")
        with open(metadata_file, "w+") as metafile:
            metafile.write(vcf_name)

    @staticmethod
    def write_samples_and_rec_map(experiment_config, window_number, folderpath):

        folderpath = os.path.join(folderpath, f"window_{window_number}")

        # Define the file paths
        samples_file = os.path.join(folderpath, f"samples.txt")
        flat_map_file = os.path.join(folderpath, f"flat_map.txt")

        # Open and write the sample file
        with open(samples_file, "w+") as fout:
            fout.write("sample\tpop\n")  # Write the header

            # Iterate over populations and sample sizes
            for pop_name, sample_size in experiment_config["num_samples"].items():
                for i in range(1, sample_size + 1):
                    fout.write(f"{pop_name}_{i}\t{pop_name}\n")

        # Write the recombination map file
        with open(flat_map_file, "w+") as fout:
            fout.write("pos\tMap(cM)\n")
            fout.write("0\t0\n")
            fout.write(f"{experiment_config['genome_length']}\t{experiment_config['recombination_rate'] * experiment_config['genome_length'] * 100}\n")


    def sample_params(self):
        sampled_params = {}
        for key in self.lower_bound_params:
            lower_bound = self.lower_bound_params[key]
            upper_bound = self.upper_bound_params[key]
            sampled_value = np.random.uniform(lower_bound, upper_bound)
            sampled_params[key] = int(sampled_value)


            # Check if the sampled parameter is equal to the mean of the uniform distribution
            mean_value = (lower_bound + upper_bound) / 2
            if sampled_value == mean_value:
                # Add a small random value to avoid exact mean, while keeping within bounds
                adjustment = np.random.uniform(-0.1 * (upper_bound - lower_bound), 0.1 * (upper_bound - lower_bound))
                adjusted_value = sampled_value + adjustment
                
                # Ensure the adjusted value is still within the bounds
                adjusted_value = max(min(adjusted_value, upper_bound), lower_bound)
                sampled_params[key] = int(adjusted_value)

        return sampled_params

    def simulate_chromosome(self, experiment_config, sampled_params, demographic_model, length=1e7, mutation_rate=5.7e-9, recombination_rate = 3.386e-9, **kwargs):
        g = demographic_model(sampled_params)

        demog = msprime.Demography.from_demes(g)

        # Dynamically define the samples using msprime.SampleSet, based on the sample_sizes dictionary
        # samples = [
        #     msprime.SampleSet(sample_size, population=pop_name, ploidy=1)
        #     for pop_name, sample_size in num_samples.items()
        # ]

        samples = {"N1": experiment_config['num_samples']['N1'], "N2": experiment_config['num_samples']['N2']}

        # Simulate ancestry for two populations (joint simulation)
        ts = msprime.sim_ancestry(
            samples=samples,  # Two populations
            demography=demog,
            sequence_length=length,
            recombination_rate=recombination_rate,
            random_seed=experiment_config['seed'],
        )
        
        # Simulate mutations over the ancestry tree sequence
        ts = msprime.sim_mutations(ts, rate=mutation_rate)

        return ts

    def create_SFS(self, ts, mode, num_samples, length=1e7, **kwargs
    ):
        """
        If we are in pretraining mode we will use a simulated SFS. If we are in inference mode we will use a real SFS.

        """

        if mode == "pretrain":
            # Define sample sets dynamically for the SFS
            sample_sets = [
                ts.samples(population=pop.id) 
                for pop in ts.populations() 
                if len(ts.samples(population=pop.id)) > 0  # Exclude populations with no samples
            ]
                        
            # Create the joint allele frequency spectrum
            sfs = ts.allele_frequency_spectrum(sample_sets=sample_sets, mode="site", polarised=True)
            
            # Multiply SFS by the sequence length to adjust scale
            sfs *= length

            # Convert to moments Spectrum for further use
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