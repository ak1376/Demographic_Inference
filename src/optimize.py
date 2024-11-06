"""
Comparison and optimization of model spectra to data.
"""

import logging

logger = logging.getLogger("Inference")

import os, sys

import numpy
import numpy as np
from numpy import logical_and, logical_not
from moments import Misc, Numerics
from scipy.special import gammaln
import scipy.optimize
from moments.LD import Util
from moments.LD.LDstats_mod import LDstats
import math
from moments.LD import Numerics
from moments.LD import Util
import copy
import moments


#: Stores thetas
_theta_store = {}
#: Counts calls to object_func
_counter = 0

#: Returned when object_func is passed out-of-bounds params or gets a NaN ll.
_out_of_bounds_val = -1e8

import nlopt
def opt(p0, data, model_func, multinom=True,
        lower_bound=None, upper_bound=None, fixed_params=None,
        ineq_constraints=[], eq_constraints=[], 
        algorithm=nlopt.LN_BOBYQA,
        ftol_abs=1e-6, xtol_abs=1e-6,
        maxeval=int(1e9), maxtime=np.inf,
        stopval=0, log_opt = False,
        local_optimizer=nlopt.LN_BOBYQA,
        verbose=0, func_args=[], func_kwargs={},
        ):
    """
    p0: Initial parameters.
    data: Spectrum with data.
    model_func: Function to evaluate model spectrum. Should take arguments
                (params, (n1,n2...), pts)
    pts: Grid points list for evaluating likelihoods
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    lower_bound: Lower bound on parameter values. 
                 If not None, must be of same length as p0.
    upper_bound: Upper bound on parameter values.
                 If not None, must be of same length as p0.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change 
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    ineq_constraints: List of functions defining inequality constraints, specifying quantities 
                      that should be less than zero, along with tolerances.
                      Each function should take arguments func(params, grad), where params is
                      the current vector of parameter values. grad is not typically used in dadi.
                      For example, def func1(p, grad): (p[0]+p[1])-1 specifies that the total of
                      p[0]+[1] should be less than 1.
                      This would be passed into opt as ineq_constraints = [(func1, 1e-6)].
                      Here the 1e-6 is the tolerance on the constraint, which is > 0 to deal with numerical
                      rounding issues.
                      Only some algorithms support constraints. We suggest using nlopt.LN_COBYLA.
    eq_constraints: List of functions defining equality constraints, specifying quantities 
                      that should be equal to zero, along with tolerances.
                      Each function should take arguments func(params, grad), where params is
                      the current vector of parameter values. grad is not typically used in dadi.
                      For example, def func1(p, grad): 1 - (p[0]+p[1]) specifies that the total of
                      p[0]+[1] should be equal to 1.
                      This would be passed into opt as ineq_constraints = [(func1, 1e-6)].
                      Here the 1e-6 is the tolerance on the constraint, which is > 0 to deal with numerical
                      rounding issues.
                      Only some algorithms support constraints. We suggest using nlopt.LN_COBYLA.
    algorithm: Optimization algorithm to employ. See
               https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
               for possibilities.
    ftol_abs: Absolute tolerance on log-likelihood
    xtol_abs: Absolute tolerance in parameter values
              Both these tolerances should be set more stringently than your actual
              desire, because algorithms cannot generally guarantee convergence.
    maxeval: Maximum number of function evaluations
    maxtime: Maximum optimization time, in seconds
    log_opt: If True, optimization algorithm will run in terms of logs of parameters.
    stopval: Algorithm will stop when a log-likelihood of at least stopval
             is found. This is primarily useful for testing.
    local_optimizer: If using a global algorithm, this specifies the local algorithm
                     to be used for refinement.
    verbose: If > 0, print optimization status every <verbose> model evaluations.
    func_args: Additional arguments to model_func. It is assumed that 
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    (See help(dadi.Inference.optimize_log for examples of func_args and 
     fixed_params usage.)
    """
    if lower_bound is None:
            lower_bound = [-np.inf] * len(p0)
    lower_bound = _project_params_down(lower_bound, fixed_params)
    # Replace None in bounds with infinity
    if upper_bound is None:
            upper_bound = [np.inf] * len(p0)
    upper_bound = _project_params_down(upper_bound, fixed_params)
    # Replace None in bounds with infinities
    lower_bound = [_ if _ is not None else -np.inf for _ in lower_bound]
    upper_bound = [_ if _ is not None else np.inf for _ in upper_bound]

    if log_opt:
        lower_bound, upper_bound = np.log(lower_bound), np.log(upper_bound)

    p0 = _project_params_down(p0, fixed_params)

    opt = nlopt.opt(algorithm, len(p0))

    opt.set_lower_bounds(lower_bound)
    opt.set_upper_bounds(upper_bound)

    for cons, tol in ineq_constraints:
        opt.add_inequality_constraint(cons, tol)
    for cons, tol in eq_constraints:
        opt.add_equality_constraint(cons, tol)

    opt.set_stopval(stopval)
    opt.set_ftol_abs(ftol_abs)
    opt.set_xtol_abs(xtol_abs)
    opt.set_maxeval(maxeval)
    opt.set_maxtime(maxtime)

    # For some global optimizers, need to set local optimizer parameters.
    local_opt = nlopt.opt(local_optimizer, len(p0))
    local_opt.set_stopval(stopval)
    local_opt.set_ftol_abs(ftol_abs)
    local_opt.set_xtol_abs(xtol_abs)
    local_opt.set_maxeval(maxeval)
    local_opt.set_maxtime(maxtime)
    opt.set_local_optimizer(local_opt)

    def f(x, grad):
        if grad.size:
            raise ValueError("Cannot use optimization algorithms that require a derivative function.")
        if log_opt: # Convert back from log parameters
            x = np.exp(x)
        return -_object_func(x, data, model_func, 
                             verbose=verbose, multinom=multinom,
                             func_args=func_args, func_kwargs=func_kwargs, fixed_params=fixed_params)

    # NLopt can run into a roundoff error on rare occassion.
    # To account for this we put in an exception for nlopt.RoundoffLimited
    # and return -inf log-likelihood and nan parameter values.
    try:
        opt.set_max_objective(f)

        if log_opt:
            p0 = np.log(p0)
        xopt = opt.optimize(p0)
        if log_opt:
            xopt = np.exp(p0)

        opt_val = opt.last_optimum_value()
        result = opt.last_optimize_result()

        xopt = _project_params_up(xopt, fixed_params)

    except nlopt.RoundoffLimited:
        print('nlopt.RoundoffLimited occured, other jobs still running. Users might want to adjust their boundaries or starting parameters if this message occures many times.')
        opt_val = -np.inf
        xopt = [np.nan] * len(p0)

    return xopt, opt_val

def check_zero_variance(Sigmas, num_pops):
    # Get the names of the statistics
    ld_stats_names, het_stats_names = Util.moment_names(num_pops)
    for idx, Sigma in enumerate(Sigmas):
        variances = np.diag(Sigma)
        zero_var_indices = np.where(variances <= 1e-8)[0]
        if zero_var_indices.size > 0:
            # Map indices to statistic names
            stat_names = []
            for var_idx in zero_var_indices:
                if idx < len(Sigmas) - 1:
                    stat_names.append(ld_stats_names[var_idx])
                else:
                    stat_names.append(het_stats_names[var_idx])
            print(f"Covariance matrix at index {idx} has zero variance in variables: {zero_var_indices}")
            print(f"Corresponding statistics: {stat_names}")

def remove_normalized_lds(y, normalization=0):
    """
    Returns LD statistics with the normalizing statistic removed.

    :param y: An LDstats object that has been normalized to get
        :math:`\\sigma_D^2`-formatted statistics.
    :type y: :class:`LDstats` object
    :param normalization: The index of the normalizing population.
    :type normalization: int
    """
    to_delete_ld = y.names()[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))
    to_delete_h = y.names()[1].index("H_{0}_{0}".format(normalization))
    for i in range(len(y) - 1):
        if len(y[i]) != len(y.names()[0]):
            raise ValueError("Unexpected number of LD stats in data")
        y[i] = np.delete(y[i], to_delete_ld)
    if len(y[-1]) != len(y.names()[1]):
        raise ValueError("Unexpected number of H stats in data")
    y[-1] = np.delete(y[-1], to_delete_h)
    return y

def remove_nonpresent_statistics(y, statistics=[[], []]):
    """
    Removes data not found in the given set of statistics.

    :param y: LD statistics.
    :type y: :class:`LDstats` object.
    :param statistics: A list of lists for two and one locus statistics to keep.
    """
    to_delete = [[], []]
    for j in range(2):
        for i, s in enumerate(y.names()[j]):
            if s not in statistics[j]:
                to_delete[j].append(i)
    for i in range(len(y) - 1):
        y[i] = np.delete(y[i], to_delete[0])
    y[-1] = np.delete(y[-1], to_delete[1])
    return y


def _multivariate_normal_pdf(x, mu, Sigma):
    p = len(x)
    return np.sqrt(np.linalg.det(Sigma) / (2 * math.pi) ** p) * np.exp(
        -1.0 / 2 * np.dot(np.dot((x - mu).transpose(), np.linalg.inv(Sigma)), x - mu)
    )


def _ll(x, mu, Sigma_inv):
    """
    x = data
    mu = model function output
    Sigma_inv = inverse of the variance-covariance matrix
    """
    if len(x) == 0:
        return 0
    else:
        return -1.0 / 2 * np.dot(np.dot((x - mu).transpose(), Sigma_inv), x - mu)
        # - len(x)*np.pi - 1./2*np.log(np.linalg.det(Sigma))


_varcov_inv_cache = {}


def ll_over_bins(xs, mus, Sigmas, num_pops):
    """
    Compute the composite log-likelihood over LD and heterozygosity statistics, given
    data and expectations. Inputs must be in the same order, and we assume each bin
    is independent, so we sum _ll(x, mu, Sigma) over each bin.

    :param xs: A list of data arrays.
    :param mus: A list of model function output arrays, same length as ``xs``.
    :param Sigmas: A list of var-cov matrices, same length as ``xs``.
    :param num_pops: Number of populations in the data/model.
    """
    # Check for zero variance in covariance matrices
    check_zero_variance(Sigmas, num_pops)

    it = iter([xs, mus, Sigmas])
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError(
            "Lists of data, means, and varcov matrices must be the same length"
        )
    ll_vals = []
    for ii in range(len(xs)):
        # get var-cov inverse from cache dictionary, or compute it
        recompute = True
        if ii in _varcov_inv_cache and np.all(
            _varcov_inv_cache[ii]["data"] == Sigmas[ii]
        ):
            Sigma_inv = _varcov_inv_cache[ii]["inv"]
            recompute = False
        if recompute:
            _varcov_inv_cache[ii] = {}
            _varcov_inv_cache[ii]["data"] = Sigmas[ii]
            if Sigmas[ii].size == 0:
                Sigma_inv = np.array([])
            else:
                Sigma_inv = np.linalg.inv(Sigmas[ii])
            _varcov_inv_cache[ii]["inv"] = Sigma_inv
        # append log-likelihood for this bin
        ll_vals.append(_ll(xs[ii], mus[ii], Sigma_inv))
    # sum over bins to get composite log-likelihood
    ll_val = np.sum(ll_vals)
    return ll_val


_out_of_bounds_val = -1e12



def sigmaD2(y, normalization=0):
    """
    Compute the :math:`\\sigma_D^2` statistics normalizing by the heterozygosities
    in a given population.

    :param y: The input data.
    :type y: :class:`LDstats` object
    :param normalization: The index of the normalizing population
        (normalized by pi2_i_i_i_i and H_i_i), default set to 0.
    :type normalization: int, optional
    """
    if normalization >= y.num_pops or normalization < 0:
        raise ValueError("Normalization index must be for a present population")

    out = LDstats(copy.deepcopy(y[:]), num_pops=y.num_pops, pop_ids=y.pop_ids)

    for i in range(len(y))[:-1]:
        out[i] /= y[i][y.names()[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))]
    out[-1] /= y[-1][y.names()[1].index("H_{0}_{0}".format(normalization))]

    return out

def _object_func(
    params,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    multinom=True,
    flush_delay=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_stream=sys.stdout,
    store_thetas=False,
):

    """
    Objective function for optimization.
    """
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    # Check our parameter bounds
    if lower_bound is not None:
        for pval, bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val / ll_scale
    if upper_bound is not None:
        for pval, bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val / ll_scale
            
    if hasattr(data, 'sample_sizes'):
        ns = data.sample_sizes
        all_args = [params_up, ns] + list(func_args)

    else:
        ns = None  # Or handle the case without sample sizes
        # all_args = [params_up, ns] + list(func_args)
        all_args = [params_up, ns] + list(func_args)

    # ns = data.sample_sizes
    # all_args = [params_up, ns] + list(func_args)

    func_kwargs = func_kwargs.copy()
    sfs = model_func(*all_args, **func_kwargs)
    if multinom:
        result = ll_multinom(sfs, data)
    else:
        result = ll(sfs, data)

    if store_thetas:
        global _theta_store
        _theta_store[tuple(params)] = optimal_sfs_scaling(sfs, data)

    # Bad result
    if numpy.isnan(result):
        result = _out_of_bounds_val

    if (verbose > 0) and (_counter % verbose == 0):
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params_up]))
        output_stream.write(
            "%-8i, %-12g, %s%s" % (_counter, result, param_str, os.linesep)
        )
        Misc.delayed_flush(delay=flush_delay)

    return -result / ll_scale


def _object_func_log(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func(np.exp(log_params), *args, **kwargs)

def bin_stats(model_func, params, rho=[], theta=0.001, spread=None, kwargs={}):
    """
    Computes LD statist for a given model function over bins defined by ``rho``.
    Here, ``rho`` gives the bin edges, and we assume no spaces between bins. That
    is, if the length of the input recombination rates is :math:`l`, the number of
    bins is :math:`l-1`.

    :param model_func: The model function that takes parameters in the form
        ``model_func(params, rho=rho, theta=theta, **kwargs)``.
    :param params: The parameters to evaluate the model at.
    :type params: list of floats
    :param rho: The scaled recombination rate bin edges.
    :type rho: list of floats
    :param theta: The mutation rate
    :type theta: float, optional
    :param spread: A list of length rho-1 (number of bins), where each entry is an
        array of length rho+1 (number of bins plus amount outside bin range to each
        side). Each array must sum to one.
    :type spread: list of arrays
    :param kwargs: Extra keyword arguments to pass to ``model_func``.
    """
    if len(rho) < 2:
        raise ValueError(
            "number of recombination rates (bin edges) must be greater than one"
        )
    rho_mids = (np.array(rho[:-1]) + np.array(rho[1:])) / 2
    y_edges = model_func(params, rho=rho, theta=theta, **kwargs)
    y_mids = model_func(params, rho=rho_mids, theta=theta, **kwargs)
    y = [
        1.0 / 6 * (y_edges[i] + y_edges[i + 1] + 4 * y_mids[i])
        for i in range(len(rho_mids))
    ]
    if spread is None:
        y.append(y_edges[-1])
        return LDstats(y, num_pops=y_edges.num_pops, pop_ids=y_edges.pop_ids)
    else:
        if len(spread) != len(rho) - 1:
            raise ValueError("spread must be length of bins")
        y_spread = []
        for distr in spread:
            if len(distr) != len(rho) + 1:
                raise ValueError(
                    "spread distr is not the correct length (len(bins) + 2)"
                )
            if not np.isclose(np.sum(distr), 1):
                raise ValueError("spread distributions must sum to one")
            y_spread.append(
                (distr[0] * y_edges[0] + distr[1:-1].dot(y) + distr[-1] * y_edges[-2])
            )
        y_spread.append(y_edges[-1])
        return LDstats(y_spread, num_pops=y_edges.num_pops, pop_ids=y_edges.pop_ids)


def _object_func_LD(
    params,
    model_func,
    means,
    varcovs,
    fs=None,
    rs=None,
    theta=None,
    u=None,
    Ne=None,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0,
    normalization=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    use_afs=False,
    Leff=None,
    multinom=True,
    ns=None,
    statistics=None,
    pass_Ne=False,
    spread=None,
    output_stream=sys.stdout,
):
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    # Check our parameter bounds
    if lower_bound is not None:
        for pval, bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val
    if upper_bound is not None:
        for pval, bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val

    all_args = [params_up] + list(func_args)

    if theta is None:
        if Ne is None:
            Ne = params_up[-1]
            theta = 4 * Ne * u
            rhos = [4 * Ne * r for r in rs]
            if pass_Ne == False:
                all_args = [all_args[0][:-1]]
            else:
                all_args = [all_args[0][:]]
        else:
            theta = 4 * Ne * u
            rhos = [4 * Ne * r for r in rs]
    else:
        if Ne is not None:
            rhos = [4 * Ne * r for r in rs]

    ## first get ll of afs
    if use_afs == True:
        if Leff is None:
            model = theta * model_func[1](all_args[0], ns)
        else:
            model = Leff * theta * model_func[1](all_args[0], ns)
        if fs.folded:
            model = model.fold()
        if multinom == True:
            ll_afs = moments.Inference.ll_multinom(model, fs)
        else:
            ll_afs = moments.Inference.ll(model, fs)

    # Prepare model statistics
    func_kwargs = {"theta": theta, "rho": rhos, "spread": spread}
    stats = bin_stats(model_func[0], *all_args, **func_kwargs)
    stats = sigmaD2(stats, normalization=normalization)
    if statistics == None:
        stats = remove_normalized_lds(stats, normalization=normalization)
    else:
        stats = remove_nonpresent_statistics(stats, statistics=statistics)
    simp_stats = stats[:-1]
    het_stats = stats[-1]
    num_pops = stats.num_pops

    if use_afs == False:
        simp_stats.append(het_stats)

    # Check for zero variance in varcovs before computing log-likelihood
    check_zero_variance(varcovs, num_pops)

    # Compute the log-likelihood
    if use_afs == True:
        result = ll_afs + ll_over_bins(means, simp_stats, varcovs, num_pops)
    else:
        result = ll_over_bins(means, simp_stats, varcovs, num_pops)

    # Bad result
    if np.isnan(result):
        print("got bad results...")
        result = _out_of_bounds_val

    if (verbose > 0) and (_counter % verbose == 0):
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params_up]))
        output_stream.write(
            "%-8i, %-12g, %s%s" % (_counter, result, param_str, os.linesep)
        )
        Misc.delayed_flush(delay=flush_delay)

    return -result




def optimize_log(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    gtol=1e-5,
    multinom=True,
    maxiter=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using the BFGS method. This optimization
    method works well when we start reasonably close to the optimum.

    Because this works in log(params), it cannot explore values of params < 0.
    However, it should perform well when parameters range over different orders
    of magnitude.

    :param p0: Initial parameters.
    :param data: Data SFS.
    :param model_func: Function to evaluate model spectrum. Should take arguments
        ``model_func(params, (n1,n2...))``.
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0.
    :param verbose: If > 0, print optimization status every ``verbose`` steps.
    :param output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
    :param epsilon: Step-size to use for finite-difference derivatives.
    :param gtol: Convergence criterion for optimization. For more info,
        see help(scipy.optimize.fmin_bfgs)
    :param multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum iterations to run for.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
        Using func_args.
        For example, you could define your model function as
        ``def func((p1,p2), ns, f1, f2): ...``.
        If you wanted to fix f1=0.1 and f2=0.2 in the optimization, you
        would pass func_args = [0.1,0.2] (and ignore the fixed_params
        argument).
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        ll hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
        For example, suppose your model function is
        ``def func((p1,f1,p2,f2), ns): ...``
        If you wanted to fix f1=0.1 and f2=0.2 in the optimization,
        you would pass fixed_params = [None,0.1,None,0.2] (and ignore
        the func_args argument).
    :param ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
        too large. (This appears to be a flaw in the scipy
        implementation.) To overcome this, pass ll_scale > 1, which will
        simply reduce the magnitude of the log-likelihood. Once in a
        region of reasonable likelihood, you'll probably want to
        re-optimize with ll_scale=1.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_bfgs(
        _object_func_log,
        numpy.log(p0),
        epsilon=epsilon,
        args=args,
        gtol=gtol,
        full_output=True,
        disp=False,
        maxiter=maxiter,
    )
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag


def optimize_log_lbfgsb(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    pgtol=1e-5,
    multinom=True,
    maxiter=1e5,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using the L-BFGS-B method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum. This method is
    better than optimize_log if the optimum lies at one or more of the
    parameter bounds. However, if your optimum is not on the bounds, this
    method may be much slower.

    Because this works in log(params), it cannot explore values of params < 0.
    It should also perform better when parameters range over scales.

    The L-BFGS-B method was developed by Ciyou Zhu, Richard Byrd, and Jorge
    Nocedal. The algorithm is described in:

    - R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing , 16, 5, pp. 1190-1208.
    - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550-560.

    :param p0: Initial parameters.
    :param data: Spectrum with data.
    :param model_function: Function to evaluate model spectrum. Should take arguments
        (params, (n1,n2...))
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param verbose: If > 0, print optimization status every <verbose> steps.
    :param output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
    :param epsilon: Step-size to use for finite-difference derivatives.
    :param pgtol: Convergence criterion for optimization. For more info,
        see help(scipy.optimize.fmin_l_bfgs_b)
    :param multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum algorithm iterations to run.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        will hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
    :param ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
        too large. (This appears to be a flaw in the scipy
        implementation.) To overcome this, pass ll_scale > 1, which will
        simply reduce the magnitude of the log-likelihood. Once in a
        region of reasonable likelihood, you'll probably want to
        re-optimize with ll_scale=1.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        None,
        None,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    # Make bounds list. For this method it needs to be in terms of log params.
    if lower_bound is None:
        lower_bound = [None] * len(p0)
    else:
        lower_bound = [numpy.log(lb) if lb is not None else None for lb in lower_bound]
    lower_bound = _project_params_down(lower_bound, fixed_params)
    if upper_bound is None:
        upper_bound = [None] * len(p0)
    else:
        upper_bound = [numpy.log(ub) if ub is not None else None for ub in upper_bound]
    upper_bound = _project_params_down(upper_bound, fixed_params)
    bounds = list(zip(lower_bound, upper_bound))

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin_l_bfgs_b(
        _object_func_log,
        numpy.log(p0),
        bounds=bounds,
        epsilon=epsilon,
        args=args,
        iprint=-1,
        pgtol=pgtol,
        maxiter=maxiter,
        approx_grad=True,
    )
    xopt, fopt, info_dict = outputs

    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, info_dict


def minus_ll(model, data):
    """
    The negative of the log-likelihood of the data given the model sfs.
    """
    return -ll(model, data)


def ll(model, data):
    """
    The log-likelihood of the data given the model sfs.

    Evaluate the log-likelihood of the data given the model. This is based on
    Poisson statistics, where the probability of observing k entries in a cell
    given that the mean number is given by the model is
    :math:`P(k) = exp(-model) * model^k / k!`.

    Note: If either the model or the data is a masked array, the return ll will
    ignore any elements that are masked in *either* the model or the data.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    ll_arr = ll_per_bin(model, data)
    return ll_arr.sum()


def ll_per_bin(model, data, missing_model_cutoff=1e-6):
    """
    The Poisson log-likelihood of each entry in the data given the model sfs.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    :param missing_model_cutoff: Due to numerical issues, there may be entries in the
        FS that cannot be stable calculated. If these entries
        involve a fraction of the data larger than
        missing_model_cutoff, a warning is printed.
    """
    if data.folded and not model.folded:
        model = model.fold()

    # Using numpy.ma.log here ensures that any negative or nan entries in model
    # yield masked entries in result. We can then check for correctness of
    # calculation by simply comparing masks.
    # Note: Using .data attributes directly saves a little computation time. We
    # use model and data as a whole at least once, to ensure masking is done
    # properly.
    result = -model.data + data.data * model.log() - gammaln(data + 1.0)
    if numpy.all(result.mask == data.mask):
        return result

    not_data_mask = logical_not(data.mask)
    data_sum = data.sum()

    missing = logical_and(model < 0, not_data_mask)
    if numpy.any(missing) and data[missing].sum() / data.sum() > missing_model_cutoff:
        logger.warn("Model is < 0 where data is not masked.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    # If the data is 0, it's okay for the model to be 0. In that case the ll
    # contribution is 0, which is fine.
    missing = logical_and(model == 0, logical_and(data > 0, not_data_mask))
    if numpy.any(missing) and data[missing].sum() / data_sum > missing_model_cutoff:
        logger.warn("Model is 0 where data is neither masked nor 0.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    missing = numpy.logical_and(model.mask, not_data_mask)
    if numpy.any(missing) and data[missing].sum() / data_sum > missing_model_cutoff:
        print(data[missing].sum(), data_sum)
        logger.warn("Model is masked in some entries where data is not.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    missing = numpy.logical_and(numpy.isnan(model), not_data_mask)
    if numpy.any(missing) and data[missing].sum() / data_sum > missing_model_cutoff:
        logger.warn("Model is nan in some entries where data is not masked.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    return result


def ll_multinom_per_bin(model, data):
    """
    Mutlinomial log-likelihood of each entry in the data given the model.

    Scales the model sfs to have the optimal theta for comparison with the data.
    """
    theta_opt = optimal_sfs_scaling(model, data)
    return ll_per_bin(theta_opt * model, data)


def ll_multinom(model, data):
    """
    Log-likelihood of the data given the model, with optimal rescaling.

    Evaluate the log-likelihood of the data given the model. This is based on
    Poisson statistics, where the probability of observing k entries in a cell
    given that the mean number is given by the model is
    :math:`P(k) = exp(-model) * model^k / k!`.

    model is optimally scaled to maximize ll before calculation.

    Note: If either the model or the data is a masked array, the return ll will
    ignore any elements that are masked in *either* the model or the data.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    ll_arr = ll_multinom_per_bin(model, data)
    return ll_arr.sum()


def minus_ll_multinom(model, data):
    """
    The negative of the log-likelihood of the data given the model sfs.

    Return a double that is -(log-likelihood)
    """
    return -ll_multinom(model, data)


def linear_Poisson_residual(model, data, mask=None):
    """
    Return the Poisson residuals, (model - data)/sqrt(model), of model and data.

    mask sets the level in model below which the returned residual array is
    masked. The default of 0 excludes values where the residuals are not
    defined.

    In the limit that the mean of the Poisson distribution is large, these
    residuals are normally distributed. (If the mean is small, the Anscombe
    residuals are better.)

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    :param mask: Optional mask, with same size as ``model``.
    """
    if data.folded and not model.folded:
        model = model.fold()

    resid = (model - data) / numpy.ma.sqrt(model)
    if mask is not None:
        tomask = numpy.logical_and(model <= mask, data <= mask)
        resid = numpy.ma.masked_where(tomask, resid)
    return resid


def Anscombe_Poisson_residual(model, data, mask=None):
    """
    Return the Anscombe Poisson residuals between model and data.

    mask sets the level in model below which the returned residual array is
    masked. This excludes very small values where the residuals are not normal.
    1e-2 seems to be a good default for the NIEHS human data. (model = 1e-2,
    data = 0, yields a residual of ~1.5.)

    Residuals defined in this manner are more normally distributed than the
    linear residuals when the mean is small. See this reference below for
    justification: Pierce DA and Schafer DW, "Residuals in generalized linear
    models" Journal of the American Statistical Association, 81(396)977-986
    (1986).

    Note that I tried implementing the "adjusted deviance" residuals, but they
    always looked very biased for the cases where the data was 0.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    :param mask: Optional mask, with same size as ``model``.
    """
    if data.folded and not model.folded:
        model = model.fold()
    # Because my data have often been projected downward or averaged over many
    # iterations, it appears better to apply the same transformation to the data
    # and the model.
    # For some reason data**(-1./3) results in entries in data that are zero
    # becoming masked. Not just the result, but the data array itself. We use
    # the power call to get around that.
    # This seems to be a common problem, that we want to use numpy.ma functions
    # on masked arrays, because otherwise the mask on the input itself can be
    # changed. Subtle and annoying. If we need to create our own functions, we
    # can use numpy.ma.core._MaskedUnaryOperation.
    datatrans = data ** (2.0 / 3) - numpy.ma.power(data, -1.0 / 3) / 9
    modeltrans = model ** (2.0 / 3) - numpy.ma.power(model, -1.0 / 3) / 9
    resid = 1.5 * (datatrans - modeltrans) / model ** (1.0 / 6)
    if mask is not None:
        tomask = numpy.logical_and(model <= mask, data <= mask)
        tomask = numpy.logical_or(tomask, data == 0)
        resid = numpy.ma.masked_where(tomask, resid)
    # It makes more sense to me to have a minus sign here... So when the
    # model is high, the residual is positive. This is opposite of the
    # Pierce and Schafner convention.
    return -resid


def optimally_scaled_sfs(model, data):
    """
    Optimially scale model sfs to data sfs.

    Returns a new scaled model sfs.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    return optimal_sfs_scaling(model, data) * model


def optimal_sfs_scaling(model, data):
    """
    Optimal multiplicative scaling factor between model and data.

    This scaling is based on only those entries that are masked in neither
    model nor data.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    if data.folded and not model.folded:
        model = model.fold()

    model, data = Numerics.intersect_masks(model, data)
    return data.sum() / model.sum()


def optimize_log_fmin(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    multinom=True,
    maxiter=None,
    maxfun=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using Nelder-Mead.
    This optimization method may work better than BFGS when far from a
    minimum. It is much slower, but more robust, because it doesn't use
    gradient information.

    Because this works in log(params), it cannot explore values of params < 0.
    It should also perform better when parameters range over large scales.

    :param p0: Initial parameters.
    :param data: Spectrum with data.
    :param model_function: Function to evaluate model spectrum. Should take arguments
        (params, (n1,n2...))
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param verbose: If True, print optimization status every <verbose> steps.
    :param output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
    :param multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum number of iterations to run optimization.
    :param maxfun: Maximum number of objective function calls to perform.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        will hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        1.0,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin(
        _object_func_log,
        numpy.log(p0),
        args=args,
        disp=False,
        maxiter=maxiter,
        maxfun=maxfun,
        full_output=True,
    )
    xopt, fopt, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, iter, funcalls, warnflag


def optimize_powell(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    xtol=1e-4,
    ftol=1e-4,
    multinom=True,
    maxiter=None,
    maxfunc=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
    retall=False,
):
    """
    Optimize parameters using Powell's conjugate direction method.

    This method works without calculating any derivatives, and optimizes along
    one direction at a time. May be useful as an initial search for an approximate
    solution, followed by further optimization using a gradient optimizer.

    p0: Initial parameters.
    data: Spectrum with data.
    model_func: Function to evaluate model spectrum. Should take arguments
                (params, (n1,n2...)).
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0.
    verbose: If > 0, print optimization status every <verbose> steps.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    xtol: Error tolerance for line search.
    ftol: Relative error acceptable for convergence.
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    maxiter: Maximum iterations to run for.
    maxfunc: Maximum number of function evalutions.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_powell).
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, and its second argument is an array of sample sizes
               for the sfs.
               For example, you could define your model function as
               def func((p1,p2), ns, f1, f2):
                   ....
               If you wanted to fix f1=0.1 and f2=0.2 in the optimization, you
               would pass func_args = [0.1,0.2].
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
                  For example, suppose your model function is
                  def func((p1,f1,p2,f2), ns):
                      ....
                  If you wanted to fix f1=0.1 and f2=0.2 in the optimization,
                  you would pass fixed_params = [None,0.1,None,0.2].
    ll_scale: The algorithm may fail if your initial log-likelihood is
              too large. To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    retall: If True, return a list of solutions at each iteration.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_powell(
        _object_func,
        p0,
        args=args,
        xtol=xtol,
        ftol=ftol,
        maxiter=maxiter,
        maxfun=maxfunc,
        disp=False,
        full_output=True,
        retall=retall,
    )
    if retall:
        xopt, fopt, direc, iters, funcalls, warnflag, allvecs = outputs
    else:
        xopt, fopt, direc, iters, funcalls, warnflag = outputs
    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    elif retall:
        return xopt, fopt, direc, iters, funcalls, warnflag, allvecs
    else:
        return xopt, fopt, direc, iters, funcalls, warnflag


def optimize_log_powell(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    multinom=True,
    maxiter=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using Powell's conjugate direction method.

    This method works without calculating any derivatives, and optimizes along
    one direction at a time. May be useful as an initial search for an approximate
    solution, followed by further optimization using a gradient optimizer.

    Because this works in log(params), it cannot explore values of params < 0.

    :param p0: Initial parameters.
    :param data: Spectrum with data.
    :param model_function: Function to evaluate model spectrum. Should take arguments
        (params, (n1,n2...))
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param verbose: If True, print optimization status every <verbose> steps.
        output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
        multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum iterations to run for.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        will hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
        (See help(moments.Inference.optimize_log for examples of func_args and
        fixed_params usage.)
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        1.0,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_powell(
        _object_func_log,
        numpy.log(p0),
        args=args,
        disp=False,
        maxiter=maxiter,
        full_output=True,
    )
    xopt, fopt, direc, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, direc, iter, funcalls, warnflag


def optimize(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    gtol=1e-5,
    multinom=True,
    maxiter=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize params to fit model to data using the BFGS method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, (n1,n2...))
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0.
    verbose: If > 0, print optimization status every <verbose> steps.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    gtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_bfgs)
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    maxiter: Maximum iterations to run for.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    (See help(moments.Inference.optimize_log for examples of func_args and
     fixed_params usage.)
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_bfgs(
        _object_func,
        p0,
        epsilon=epsilon,
        args=args,
        gtol=gtol,
        full_output=True,
        disp=False,
        maxiter=maxiter,
    )
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag


def optimize_lbfgsb(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    pgtol=1e-5,
    multinom=True,
    maxiter=1e5,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize params to fit model to data using the L-BFGS-B method.

    Note: this optimization method can explore negative values. You must therefore
    specify lower bounds for values that cannot take negative numbers (such
    as event times, population sizes, and migration rates).

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum. This method is
    better than optimize_log if the optimum lies at one or more of the
    parameter bounds. However, if your optimum is not on the bounds, this
    method may be much slower.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, (n1,n2...))
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0. A parameter can be declared unbound by assigning
                 a bound of None.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0. A parameter can be declared unbound by assigning
                 a bound of None.
    verbose: If > 0, print optimization status every <verbose> steps.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    pgtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_l_bfgs_b)
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    maxiter: Maximum algorithm iterations evaluations to run.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    (See help(moments.Inference.optimize_log for examples of func_args and
     fixed_params usage.)
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.

    The L-BFGS-B method was developed by Ciyou Zhu, Richard Byrd, and Jorge
    Nocedal. The algorithm is described in:
      * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
        Constrained Optimization, (1995), SIAM Journal on Scientific and
        Statistical Computing , 16, 5, pp. 1190-1208.
      * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
        FORTRAN routines for large scale bound constrained optimization (1997),
        ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550-560.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        None,
        None,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    # Make bounds list. For this method it needs to be in terms of log params.
    if lower_bound is None:
        lower_bound = [None] * len(p0)
    lower_bound = _project_params_down(lower_bound, fixed_params)
    if upper_bound is None:
        upper_bound = [None] * len(p0)
    upper_bound = _project_params_down(upper_bound, fixed_params)
    bounds = list(zip(lower_bound, upper_bound))

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin_l_bfgs_b(
        _object_func,
        p0,
        bounds=bounds,
        epsilon=epsilon,
        args=args,
        iprint=-1,
        pgtol=pgtol,
        maxiter=maxiter,
        approx_grad=True,
    )
    xopt, fopt, info_dict = outputs

    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, info_dict


def _project_params_down(pin, fixed_params):
    """
    Eliminate fixed parameters from pin.
    """
    if fixed_params is None:
        return pin

    if len(pin) != len(fixed_params):
        raise ValueError(
            "fixed_params list must have same length as input " "parameter array."
        )

    pout = []
    for ii, (curr_val, fixed_val) in enumerate(zip(pin, fixed_params)):
        if fixed_val is None:
            pout.append(curr_val)

    return numpy.array(pout)


def _project_params_up(pin, fixed_params):
    """
    Fold fixed parameters into pin.
    """
    if fixed_params is None:
        return pin

    if numpy.isscalar(pin):
        pin = [pin]

    pout = numpy.zeros(len(fixed_params))
    orig_ii = 0
    for out_ii, val in enumerate(fixed_params):
        if val is None:
            pout[out_ii] = pin[orig_ii]
            orig_ii += 1
        else:
            pout[out_ii] = fixed_params[out_ii]
    return pout


index_exp = numpy.index_exp


def optimize_grid(
    data,
    model_func,
    grid,
    verbose=0,
    flush_delay=0.5,
    multinom=True,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    output_file=None,
):
    """
    Optimize params to fit model to data using brute force search over a grid.

    data: Spectrum with data.
    model_func: Function to evaluate model spectrum. Should take arguments
                (params, (n1,n2...))
    grid: Grid of parameter values over which to evaluate likelihood. See
          below for specification instructions.
    verbose: If > 0, print optimization status every <verbose> steps.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    full_output: If True, return popt, llopt, grid, llout, thetas. Here popt is
                 the best parameter set found and llopt is the corresponding
                 (composite) log-likelihood. grid is the array of parameter
                 values tried, llout is the corresponding log-likelihoods, and
                 thetas is the corresponding thetas. Note that the grid includes
                 only the parameters optimized over, and that the order of
                 indices is such that grid[:,0,2] would be a set of parameters
                 if two parameters were optimized over. (Note the : in the
                 first index.)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    (See help(moments.Inference.optimize_log for examples of func_args and
     fixed_params usage.)

    Search grids are specified using a moments.Inference.index_exp object (which
    is an alias for numpy.index_exp). The grid is specified by passing a range
    of values for each parameter. For example, index_exp[0:1.1:0.3,
    0.7:0.9:11j] will search over parameter 1 with values 0,0.3,0.6,0.9 and
    over parameter 2 with 11 points between 0.7 and 0.9 (inclusive). (Notice
    the 11j in the second parameter range specification.) Note that the grid
    list should include only parameters that are optimized over, not fixed
    parameter values.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        None,
        None,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        1.0,
        output_stream,
        full_output,
    )

    if full_output:
        global _theta_store
        _theta_store = {}

    outputs = scipy.optimize.brute(
        _object_func, ranges=grid, args=args, full_output=full_output, finish=False
    )
    if full_output:
        xopt, fopt, grid, fout = outputs
        # Thetas are stored as a dictionary, because we can't guarantee
        # iteration order in brute(). So we have to iterate back over them
        # to produce the proper order to return.
        thetas = numpy.zeros(fout.shape)
        for indices, temp in numpy.ndenumerate(fout):
            # This is awkward, because we need to access grid[:,indices]
            grid_indices = tuple([slice(None, None, None)] + list(indices))
            thetas[indices] = _theta_store[tuple(grid[grid_indices])]
    else:
        xopt = outputs
    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, grid, fout, thetas


def add_misid_param(func):
    def misid_func(params, *args, **kwargs):
        misid = params[-1]
        fs = func(params[:-1], *args, **kwargs)
        return (1 - misid) * fs + misid * Numerics.reverse_array(fs)

    return misid_func

import nlopt
import numpy as np
import sys
import copy

def remove_normalized_data(
    means, varcovs, normalization=0, num_pops=1, statistics=None
):
    """
    Returns data means and covariance matrices with the normalizing
    statistics removed.

    :param means: List of means normalized statistics, where each entry is the
        full set of statistics for a given recombination distance.
    :type means: list of arrays
    :param varcovs: List of the corresponding variance covariance matrices.
    :type varcovs: list of arrays
    :param normalization: The index of the normalizing population.
    :type normalization: int
    :param num_pops: The number of populations in the data set.
    :type num_pops: int
    """
    if len(means) != len(varcovs):
        raise ValueError("Different lengths of means and covariances")
    if statistics is None:
        stats = Util.moment_names(num_pops)
    else:
        stats = copy.copy(statistics)
    to_delete_ld = stats[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))
    to_delete_h = stats[1].index("H_{0}_{0}".format(normalization))
    ms = []
    vcs = []
    for i in range(len(means) - 1):
        if (
            len(means[i]) != len(stats[0])
            or varcovs[i].shape[0] != len(stats[0])
            or varcovs[i].shape[1] != len(stats[0])
        ):
            raise ValueError(
                "Data and statistics mismatch. Some statistics are missing "
                "or the incorrect number of populations was given."
            )
        ms.append(np.delete(means[i], to_delete_ld))
        vcs.append(
            np.delete(np.delete(varcovs[i], to_delete_ld, axis=0), to_delete_ld, axis=1)
        )
    ms.append(np.delete(means[-1], to_delete_h))
    # Single population data will have 1-D array for H
    if varcovs[-1].size > 1:
        vcs.append(
            np.delete(np.delete(varcovs[-1], to_delete_h, axis=0), to_delete_h, axis=1)
        )
    else:
        vcs.append(np.delete(varcovs[-1], to_delete_h))
    if statistics is None:
        return ms, vcs
    else:
        stats[0].pop(to_delete_ld)
        stats[1].pop(to_delete_h)
        return ms, vcs, stats


def remove_nonpresent_statistics(y, statistics=[[], []]):
    """
    Removes data not found in the given set of statistics.

    :param y: LD statistics.
    :type y: :class:`LDstats` object.
    :param statistics: A list of lists for two and one locus statistics to keep.
    """
    to_delete = [[], []]
    for j in range(2):
        for i, s in enumerate(y.names()[j]):
            if s not in statistics[j]:
                to_delete[j].append(i)
    for i in range(len(y) - 1):
        y[i] = np.delete(y[i], to_delete[0])
    y[-1] = np.delete(y[-1], to_delete[1])
    return y


def nlopt_LD(
    p0,
    data,
    model_func,
    rs=None,
    theta=None,
    u=2e-8,
    Ne=None,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    normalization=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    use_afs=False,
    Leff=None,
    multinom=False,
    ns=None,
    statistics=None,
    pass_Ne=False,
    spread=None,
    local_optimizer = nlopt.LN_BOBYQA,
    ineq_constraints=[],
    eq_constraints=[],
    stopval = 0,
    ftol_abs=1e-6,
    xtol_abs=1e-6,
    maxeval=int(1e9),
    maxtime=np.inf,
    algorithm=nlopt.LN_BOBYQA,
    log_opt = False,
    maxiter=40000,
    epsilon=1e-3,
    pgtol=1e-5
    ):

    """
    Optimize (using the log of) the parameters using the modified Powell's
    method, which optimizes slices of parameter space sequentially. Initial
    parameters ``p0``, the data ``[means, varcovs]``,
    the demographic ``model_func``, and ``rs`` to specify recombination
    bin edges are required. ``Ne`` must either be specified as a keyword
    argument or is included as the *last* parameter in ``p0``.

    It is best at burrowing down a single minimum. This method is
    better than optimize_log if the optimum lies at one or more of the
    parameter bounds. However, if your optimum is not on the bounds, this
    method may be much slower.

    Because this works in log(params), it cannot explore values of params < 0.
    It should also perform better when parameters range over scales.

    The L-BFGS-B method was developed by Ciyou Zhu, Richard Byrd, and Jorge
    Nocedal. The algorithm is described in:

    - R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing , 16, 5, pp. 1190-1208.
    - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550-560.

    :param p0: The initial guess for demographic parameters,
        demography parameters plus (optionally) Ne.
    :type p0: list
    :param data: The parsed data[means, varcovs, fs]. The frequency spectrum
        fs is optional, and used only if use_afs=True.

        - Means: The list of mean statistics within each bin
          (has length ``len(rs)`` or ``len(rs) - 1`` if using AFS). If we are
          not using the AFS, which is typical, the heterozygosity statistics
          come last.
        - varcovs: The list of varcov matrices matching the data in ``means``.

    :type data: list
    :param model_func: The demographic model to compute statistics
        for a given rho. If we are using AFS, it's a list of the two models
        [LD func, AFS func]. If we're using LD stats alone, we pass a single LD
        model  as a list: [LD func].
    :type model_func: list
    :param rs: The list of raw recombination rates defining bin edges.
    :type rs: list
    :param theta: The population scaled per base mutation rate
        (4*Ne*mu, not 4*Ne*mu*L).
    :type theta: float, optional
    :param u: The raw per base mutation rate.
        Cannot be used with ``theta``.
    :type u: float, optional
    :param Ne: The fixed effective population size to scale
        u and r. If ``Ne`` is a parameter to fit, it should be the last parameter
        in ``p0``.
    :type Ne: float, optional
    :param lower_bound: Defaults to ``None``. Constraints on the
        lower bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type lower_bound: list, optional
    :param upper_bound: Defaults to ``None``. Constraints on the
        upper bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type upper_bound: list, optional
    :param verbose: If an integer greater than 0, prints updates
        of the optimization procedure at intervals given by that spacing.
    :type verbose: int, optional
    :param func_args: Additional arguments to be passed
        to ``model_func``.
    :type func_args: list, optional
    :param func_kwargs: Additional keyword arguments to be
        passed to ``model_func``.
    :type func_kwargs: dict, optional
    :param fixed_params: Defaults to ``None``. To fix some
        parameters, this should be a list of equal length as ``p0``, with
        ``None`` for parameters to be fit and fixed values at corresponding
        indexes.
    :type fixed_params: list, optional
    :param use_afs: Defaults to ``False``. We can pass a model
        to compute the frequency spectrum and use
        that instead of heterozygosity statistics for single-locus data.
    :type use_afs: bool, optional
    :param Leff: The effective length of genome from which
        the fs was generated (only used if fitting to afs).
    :type Leff: float, optional
    :param multinom: Only used if we are fitting the AFS.
        If ``True``, the likelihood is computed for an optimally rescaled FS.
        If ``False``, the likelihood is computed for a fixed scaling of the FS
        found by theta=4*Ne*u and Leff
    :type multinom: bool, optional
    :param ns: The sample size, which is only needed
        if we are using the frequency spectrum, as the sample size does not
        affect mean LD statistics.
    :type ns: list of ints, optional
    :param statistics: Defaults to ``None``, which assumes that
        all statistics are present and in the conventional default order. If
        the data is missing some statistics, we must specify which statistics
        are present using the subset of statistic names given by
        ``moments.LD.Util.moment_names(num_pops)``.
    :type statistics: list, optional
    :param pass_Ne: Defaults to ``False``. If ``True``, the
        demographic model includes ``Ne`` as a parameter (in the final position
        of input parameters).
    :type pass_Ne: bool, optional
    :param maxiter: Defaults to 40,000. Maximum number of iterations to perform.
    :type maxiter: int
    :param epsilon: Step-size to use for finite-difference derivatives.
    :type pgtol: float
    :param pgtol: Convergence criterion for optimization. For more info,
        see help(scipy.optimize.fmin_l_bfgs_b)
    :type pgtol: float
    """

    output_stream = sys.stdout

    means = data[0]
    varcovs = data[1]
    if use_afs:
        try:
            fs = data[2]
        except IndexError:
            raise ValueError(
                "if use_afs=True, need to pass frequency spectrum in data=[means,varcovs,fs]"
            )

        if ns is None:
            raise ValueError("need to set ns if we are fitting frequency spectrum")

    else:
        fs = None

    if rs is None:
        raise ValueError("need to pass rs as bin edges")

    # get num_pops
    if Ne is None:
        if not pass_Ne:
            y = model_func[0](p0[:-1])
        else:
            y = model_func[0](p0[:])
    else:
        y = model_func[0](p0)
    num_pops = y.num_pops

    # remove normalized statistics
    ms = copy.copy(means)
    vcs = copy.copy(varcovs)
    if statistics is None:
        # if statistics is not None, assume we already filtered out the data
        ms, vcs = remove_normalized_data(
            ms, vcs, normalization=normalization, num_pops=num_pops
        )

    args = (
        model_func,
        ms,
        vcs,
        fs,
        rs,
        theta,
        u,
        Ne,
        lower_bound,
        upper_bound,
        verbose,
        flush_delay,
        normalization,
        func_args,
        func_kwargs,
        fixed_params,
        use_afs,
        Leff,
        multinom,
        ns,
        statistics,
        pass_Ne,
        spread,
        output_stream,
    )

    if lower_bound is None:
        lower_bound = [-np.inf] * len(p0)
    lower_bound = _project_params_down(lower_bound, fixed_params)
    # Replace None in bounds with infinity
    if upper_bound is None:
            upper_bound = [np.inf] * len(p0)
    upper_bound = _project_params_down(upper_bound, fixed_params)
    # Replace None in bounds with infinities
    lower_bound = [_ if _ is not None else -np.inf for _ in lower_bound]
    upper_bound = [_ if _ is not None else np.inf for _ in upper_bound]

    if log_opt:
        lower_bound, upper_bound = np.log(lower_bound), np.log(upper_bound)

    p0 = _project_params_down(p0, fixed_params)

    opt = nlopt.opt(algorithm, len(p0))

    opt.set_lower_bounds(lower_bound)
    opt.set_upper_bounds(upper_bound)

    for cons, tol in ineq_constraints:
        opt.add_inequality_constraint(cons, tol)
    for cons, tol in eq_constraints:
        opt.add_equality_constraint(cons, tol)

    opt.set_stopval(stopval)
    opt.set_ftol_abs(ftol_abs)
    opt.set_xtol_abs(xtol_abs)
    opt.set_maxeval(maxeval)
    opt.set_maxtime(maxtime)

    # For some global optimizers, need to set local optimizer parameters.
    local_opt = nlopt.opt(local_optimizer, len(p0))
    local_opt.set_stopval(stopval)
    local_opt.set_ftol_abs(ftol_abs)
    local_opt.set_xtol_abs(xtol_abs)
    local_opt.set_maxeval(maxeval)
    local_opt.set_maxtime(maxtime)
    opt.set_local_optimizer(local_opt)

    def f(x, grad):
        if grad.size:
            raise ValueError("Cannot use optimization algorithms that require a derivative function.")
        if log_opt: # Convert back from log parameters
            x = np.exp(x)

        return -_object_func_LD(params = x, model_func = model_func, means = means, varcovs = varcovs, 
                             verbose=verbose, multinom=multinom,
                             func_args=func_args, func_kwargs=func_kwargs, fixed_params=fixed_params, 
                             rs = rs, theta = theta, u = u, Ne = Ne)

    # NLopt can run into a roundoff error on rare occassion.
    # To account for this we put in an exception for nlopt.RoundoffLimited
    # and return -inf log-likelihood and nan parameter values.
    try:
        opt.set_max_objective(f)

        if log_opt:
            p0 = np.log(p0)
        xopt = opt.optimize(p0)
        if log_opt:
            xopt = np.exp(p0)

        opt_val = opt.last_optimum_value()
        result = opt.last_optimize_result()

        xopt = _project_params_up(xopt, fixed_params)

    except nlopt.RoundoffLimited:
        print('nlopt.RoundoffLimited occured, other jobs still running. Users might want to adjust their boundaries or starting parameters if this message occures many times.')
        opt_val = -np.inf
        xopt = [np.nan] * len(p0)

    return xopt, opt_val