import dadi
import moments
import os
from tqdm import tqdm
import numpy as np

def get_LD_stats(folderpath, rep_ii, r_bins):
    vcf_file = os.path.join(folderpath, f"bottleneck_window.{rep_ii}.vcf.gz")
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file= os.path.join(folderpath, "flat_map.txt"),
        pop_file=os.path.join(folderpath, "samples.txt"),
        pops=["A"],
        r_bins=r_bins,
        report=False,
    )

    return ld_stats

def run_inference_dadi(sfs, p0, sampled_params, num_samples, lower_bound = [0.01, 0.01, 0.01, 0.01], upper_bound = [10, 10, 10, 10], maxiter = 100):
    '''
    This should do the parameter inference for dadi
    '''

    model_func = dadi.Demographics1D.three_epoch

    # Make the extrapolating version of our demographic model function.
    # func_ex = dadi.Numerics.make_extrap_log_func(model_func)

    p_guess = moments.Misc.perturb_params(p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound)

    opt_params = dadi.Inference.optimize_log_lbfgsb(
    p_guess, sfs, model_func, pts = 2*num_samples,
    lower_bound=lower_bound,
    upper_bound=upper_bound, maxiter = maxiter)

    model = model_func(opt_params, sfs.sample_sizes, 2*num_samples)

    opt_theta = dadi.Inference.optimal_sfs_scaling(model, sfs)

    opt_params_dict = {
        'N0': sampled_params['N0'],
        'Nb': opt_params[0]*sampled_params['N0'],
        'N_recover': opt_params[1]*sampled_params['N0'], 
        't_bottleneck_end': opt_params[3]*2*sampled_params['N0'],
        't_bottleneck_start': opt_params[2]*2*sampled_params['N0']
    }

    model = model * opt_theta

    return model, opt_theta, opt_params_dict

def run_inference_moments(sfs, p0, sampled_params, lower_bound = [0.01, 0.01, 0.01, 0.01], upper_bound = [10, 10, 10, 10], maxiter = 100):
    '''
    This should do the parameter inference for moments
    '''
    p_guess = moments.Misc.perturb_params(p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound)

    model_func = moments.Demographics1D.three_epoch
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


def run_inference_momentsLD(folderpath, num_windows, param_sample, p_guess, maxiter = 100):
    '''
    This should do the parameter inference for momentsLD
    '''

    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])

    print("parsing LD statistics")
    ld_stats = {}
    for ii in tqdm(range(num_windows)):
        ld_stats[ii] = get_LD_stats(folderpath, ii, r_bins)

    print("computing mean and varcov matrix from LD statistics sums")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    mv['varcovs'][-1].shape = (1,1)


    demo_func = moments.LD.Demographics1D.three_epoch
    # Set up the initial guess
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