import os
import moments
from tqdm import tqdm
import numpy as np
import msprime
import dadi
import allel
import matplotlib.pylab as plt
import dadi.Demes
import glob
import demes

def delete_vcf_files(directory):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    
    # Delete all files
    [os.remove(file) for file in files]
    
    print(f"Deleted {len(files)} files from {directory}")

class Simulator: 
    def __init__(self, upper_bound_params, lower_bound_params, num_sims, num_samples, L = 1e7, recombination_rate=1e-8, mutation_rate=1e-7, num_reps = 50):
        self.upper_bound_params = upper_bound_params
        self.lower_bound_params = lower_bound_params
        
        self.param_storage = []
        self.ts_storage = []
        self.tajima_d_list = []
        self.sfs = []

        self.num_sims = num_sims

        self.num_samples = num_samples
        self.L = L
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.num_reps = num_reps # only for momentsLD
        
        self.opt_params_list = []

        self.temporary_realizations_location = os.path.join(os.getcwd(), "temporary_folder_realizations")
        os.makedirs(self.temporary_realizations_location, exist_ok=True)

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

    def run_msprime_replicates(self, g):
        demog = msprime.Demography.from_demes(g)
        tree_sequences = msprime.sim_ancestry(
            {"A": self.num_samples},
            demography=demog,
            sequence_length=self.L,
            recombination_rate=self.recombination_rate,
            num_replicates=self.num_reps,
            random_seed=295,
        )
        
        folderpath = self.temporary_realizations_location

        delete_vcf_files(folderpath)
        
        for ii, ts in tqdm(enumerate(tree_sequences)):
            ts = msprime.sim_mutations(ts, rate=self.mutation_rate, random_seed=ii + 1)
            vcf_name = os.path.join(folderpath,f"bottleneck.{ii}.vcf")
            with open(vcf_name, "w+") as fout:
                ts.write_vcf(fout, allow_position_zero=True)
            os.system(f"gzip {vcf_name}")


    def write_samples_and_rec_map(self):
        folderpath = self.temporary_realizations_location
        with open(os.path.join(folderpath, "samples.txt"), "w+") as fout:
            fout.write("sample\tpop\n")
            for ii in range(self.num_samples):
                fout.write(f"tsk_{ii}\tA\n")

        with open(os.path.join(folderpath, "flat_map.txt"), "w+") as fout:
            fout.write("pos\tMap(cM)\n")
            fout.write("0\t0\n")
            fout.write(f"{self.L}\t{self.recombination_rate * self.L * 100}\n")

    
    def get_LD_stats(self, rep_ii, r_bins):
        folderpath = self.temporary_realizations_location
        vcf_file = os.path.join(self.temporary_realizations_location, f"bottleneck.{rep_ii}.vcf.gz")
        ld_stats = moments.LD.Parsing.compute_ld_statistics(
            vcf_file,
            rec_map_file= os.path.join(folderpath, "flat_map.txt"),
            pop_file=os.path.join(folderpath, "samples.txt"),
            pops=["A"],
            r_bins=r_bins,
            report=False,
        )

        return ld_stats
    
    def forward(self, analysis, param_sample, sfs, p0 = [0.25, 0.75, 0.1, 0.05], maxiter = 100, lower_bound = [0.01, 0.01, 0.01, 0.01], upper_bound = [10, 10, 10, 10]):
        ''' 
        Should modify this code so that no for loop is being performed. 
        '''

        opt_params = []
        opt_theta_list = []
        model_sfs = []

        # sfs = self.create_SFS(param_sample) # I think the problem is that multiple SFS can represent the same generative process ? 
        # self.sfs.append(sfs)
            
        if analysis == "dadi":
            model, opt_theta, opt_params_dict = self.do_inference_dadi(sfs, p0, lower_bound, upper_bound, param_sample, maxiter)
        if analysis == "moments": 
            model, opt_theta, opt_params_dict = self.do_inference_moments(sfs, p0, lower_bound, upper_bound, param_sample, maxiter)
        if analysis == "momentsLD":
            opt_params_dict = self.do_inference_momentsLD(param_sample, p_guess = p0, maxiter = maxiter)

        if analysis == "dadi" or analysis == "moments":
            model_sfs.append(model)
            opt_theta_list.append(opt_theta)

        opt_params.append(opt_params_dict)

        return opt_params, model_sfs, opt_theta

        # return_dict = {}
        # return_dict['simulated_params'] = param_storage
        # return_dict['sfs'] = sfs
        # return_dict['opt_params'] = opt_params
        # return_dict['opt_theta'] = opt_theta
        # return_dict['model_sfs'] = model_sfs
        
        # return return_dict

    def sample_params(self):
        sampled_params = {}
        for key in self.lower_bound_params:
            sampled_value = np.random.uniform(self.lower_bound_params[key], self.upper_bound_params[key])
            sampled_params[key] = int(sampled_value)

        return sampled_params

    def create_SFS(self, sampled_params):

        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=sampled_params['N_recover'])
        demography.add_population_parameters_change(sampled_params['t_bottleneck_end'], initial_size=sampled_params['Nb'])
        demography.add_population_parameters_change(sampled_params['t_bottleneck_start'], initial_size=sampled_params['N0'])

        demes_model = demography.to_demes()

        sfs_demes = dadi.Demes.SFS(demes_model, sampled_demes= ["A"], sample_sizes = [2*self.num_samples], pts = 4*self.num_samples)

        return sfs_demes

    def do_inference_momentsLD(self, param_sample, p_guess, maxiter):
        '''
        Unfortunately this function requires a lot more overhead than dadi or regular moments:
        1. Create VCF 
        2. Create samples.txt
        3. Create recombination map 
        4. "Calculate LD stats" function
        '''
        r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
        # Do a bunch of pre-steps first
        g = self.bottleneck_model(param_sample)
        self.run_msprime_replicates(g)
        self.write_samples_and_rec_map()

        print("parsing LD statistics")
        ld_stats = {}
        for ii in tqdm(range(self.num_reps)):
            ld_stats[ii] = self.get_LD_stats(ii, r_bins)

        print("computing mean and varcov matrix from LD statistics sums")
        mv = moments.LD.Parsing.bootstrap_data(ld_stats)
        mv['varcovs'][-1].shape = (1,1)

        # all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats)

        print("computing expectations under the model")
        y = moments.Demes.LD(g, sampled_demes=["A"], rho=4 * param_sample['N0'] * r_bins)
        y = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
            num_pops=y.num_pops,
            pop_ids=y.pop_ids,
        )
        y = moments.LD.Inference.sigmaD2(y)

        demo_func = moments.LD.Demographics1D.three_epoch
        # Set up the initial guess
        # The split_mig function takes four parameters (nu0, nu1, T, m), and we append
        # the last parameter to fit Ne, which doesn't get passed to the function but
        # scales recombination rates so can be simultaneously fit
        p_guess = moments.LD.Util.perturb_params(p_guess, fold=1)
        opt_params, LL = moments.LD.Inference.optimize_log_fmin(
            p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins, verbose = 3, maxiter = maxiter)

        opt_params_dict = {
        'Nb': opt_params[0]*param_sample['N0'],
        'N_recover': opt_params[1]*param_sample['N0'], 
        't_bottleneck_end': opt_params[3]*2*param_sample['N0'],
        't_bottleneck_start': opt_params[2]*2*param_sample['N0']
        }

        return opt_params_dict
    
    def do_inference_moments(self, sfs, p0, lower_bound, upper_bound, sampled_params, maxiter):
        p_guess = moments.Misc.perturb_params(p0, fold=1,
        lower_bound=lower_bound, upper_bound=upper_bound)

        model_func = moments.Demographics1D.three_epoch
        # optimize_log_lbfgsb
        opt_params = moments.Inference.optimize_log_fmin(
            p_guess, sfs, model_func,
            lower_bound=lower_bound,
            upper_bound=upper_bound, 
            maxiter = maxiter)

        model = model_func(opt_params, sfs.sample_sizes)
        opt_theta = moments.Inference.optimal_sfs_scaling(model, sfs)

        # opt_params_dict = {
        #     'N0': N0_opt,
        #     'Nb': opt_params[0]*N0_opt,
        #     'N_recover': opt_params[1]*N0_opt, 
        #     't_bottleneck_end': opt_params[3]*2*N0_opt,
        #     't_bottleneck_start': opt_params[2]*2*N0_opt
        # }

        opt_params_dict = {
            'N0': sampled_params['N0'],
            'Nb': opt_params[0]*sampled_params['N0'],
            'N_recover': opt_params[1]*sampled_params['N0'], 
            't_bottleneck_end': opt_params[3]*2*sampled_params['N0'],
            't_bottleneck_start': opt_params[2]*2*sampled_params['N0']
        }


        model = model * opt_theta

        return model, opt_theta, opt_params_dict
    
    def do_inference_dadi(self, sfs, p0, lower_bound, upper_bound, sampled_params, maxiter):

        # pts_l = 4*self.num_samples

        model_func = dadi.Demographics1D.three_epoch
        # model_func = three_epoch_five_param

        # Make the extrapolating version of our demographic model function.
        # func_ex = dadi.Numerics.make_extrap_log_func(model_func)

        p_guess = moments.Misc.perturb_params(p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound)


        # Do the optimization. By default we assume that theta is a free parameter,
        # since it's trivial to find given the other parameters. If you want to fix
        # theta, add a multinom=False to the call.
        # The maxiter argument restricts how long the optimizer will run. For real 
        # runs, you will want to set this value higher (at least 10), to encourage
        # better convergence. You will also want to run optimization several times
        # using multiple sets of intial parameters, to be confident you've actually
        # found the true maximum likelihood parameters.
        # print('Beginning optimization ************************************************')

        opt_params = dadi.Inference.optimize_log_lbfgsb(
        p_guess, sfs, model_func, pts = 2*self.num_samples,
        lower_bound=lower_bound,
        upper_bound=upper_bound, maxiter = maxiter)

        model = model_func(opt_params, sfs.sample_sizes, 2*self.num_samples)

        opt_theta = dadi.Inference.optimal_sfs_scaling(model, sfs)
        
        # opt_params_dict = {
        #     'N0': opt_params[4],
        #     'Nb': opt_params[0]*opt_params[4],
        #     'N_recover': opt_params[1]*opt_params[4], 
        #     't_bottleneck_end': opt_params[3]*2*opt_params[4],
        #     't_bottleneck_start': opt_params[2]*2*opt_params[4]
        # }

        opt_params_dict = {
            'N0': sampled_params['N0'],
            'Nb': opt_params[0]*sampled_params['N0'],
            'N_recover': opt_params[1]*sampled_params['N0'], 
            't_bottleneck_end': opt_params[3]*2*sampled_params['N0'],
            't_bottleneck_start': opt_params[2]*2*sampled_params['N0']
        }

        model = model * opt_theta

        return model, opt_theta, opt_params_dict


    def tajimas_d(self, ts):
        # Convert the tree sequence to a genotype array
        haplotypes = ts.genotype_matrix()

        # Convert haplotypes to allel.GenotypeArray
        genotypes = allel.HaplotypeArray(haplotypes).to_genotypes(ploidy=2)

        # Calculate allele counts
        allele_counts = genotypes.count_alleles()

        # Calculate Tajima's D
        tajima_d = allel.tajima_d(allele_counts)

        return tajima_d
