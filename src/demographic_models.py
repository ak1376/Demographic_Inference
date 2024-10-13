import demes
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum
import moments
import numpy as np

from moments.LD.LDstats_mod import LDstats
import moments.LD.Numerics as NumericsMomentsLD
def bottleneck_model(sampled_params):

    N0, nuB, nuF, t_bottleneck_start, t_bottleneck_end = (
        sampled_params["N0"],
        sampled_params["Nb"],
        sampled_params["N_recover"],
        sampled_params["t_bottleneck_start"],
        sampled_params["t_bottleneck_end"],
    )
    b = demes.Builder()
    b.add_deme(
        "N",
        epochs=[
            dict(start_size=N0, end_time=t_bottleneck_start),
            dict(start_size=nuB, end_time=t_bottleneck_end),
            dict(start_size=nuF, end_time=0),
        ],
    )
    g = b.resolve()

    return g

def split_isolation_model_simulation(sampled_params):
    # Unpack the sampled parameters
    Na, N1, N2, t_split = (
        sampled_params["Na"],  # Effective population size of the ancestral population
        sampled_params["N1"],  # Size of population 1 after split
        sampled_params["N2"],  # Size of population 2 after split
        sampled_params["t_split"],  # Time of the population split (in generations)
    )

    # Initialize the builder
    b = demes.Builder()

    # Add the ancestral population (deme A)
    b.add_deme(
        "ancestral",
        epochs=[
            dict(start_size=Na, end_time=t_split),  # Ancestor size until the split
        ],
    )

    # Add population 1 (deme A1) that splits from the ancestral population
    b.add_deme(
        "N1",
        ancestors=["ancestral"],  # Inherits from the ancestral population
        start_time=t_split,  # Time of the split
        epochs=[
            dict(start_size=N1, end_time=0),  # Population 1 size post-split
        ],
    )

    # Add population 2 (deme A2) that also splits from the ancestral population
    b.add_deme(
        "N2",
        ancestors=["ancestral"],  # Inherits from the ancestral population
        start_time=t_split,  # Time of the split
        epochs=[
            dict(start_size=N2, end_time=0),  # Population 2 size post-split
        ],
    )

    # Resolve the model
    g = b.resolve()

    return g

def split_isolation_model_dadi(params, ns, pts):
    """
    params = (nu1, nu2, t_split)
    ns = (n1, n2)

    Split into two populations with specified sizes and no migration.
    
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T_split: Time in the past of the split (in units of 2*Na generations).
    n1, n2: Sample sizes of the resulting Spectrum.
    pts: Number of grid points to use in integration.
    """
    # Unpack parameters
    nu1, nu2, t_split = params

    # Create the default grid for the integration
    xx = Numerics.default_grid(pts)

    # Start with the ancestral population
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # Split into two populations at T_split with no migration
    phi = Integration.two_pops(phi, xx, t_split, nu1, nu2, m12=0, m21=0)

    # Calculate the site frequency spectrum (SFS)
    fs = Spectrum.from_phi(phi, ns, (xx, xx))

    return fs

def split_isolation_model_moments(params, ns, pop_ids=None):
    
    """
    Split into two populations of specifed size, with migration.

    params = (nu1, nu2, T, m = 0)

    ns = [n1, n2]

    :param params: Tuple of length 4.

        - nu1: Size of population 1 after split.
        - nu2: Size of population 2 after split.
        - T: Time in the past of split (in units of 2*Na generations)
        - m: Migration rate between populations (2*Na*m) = 0
    :param ns: List of length two specifying sample sizes n1 and n2.
    :param pop_ids: List of population IDs.
    """

    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    nu1, nu2, T = params
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate([nu1, nu2], T, m=np.array([[0, 0], [0, 0]])) # setting migration rates to constant 0
    fs.pop_ids = pop_ids
    return fs

def snm(params=None, rho=None, theta=0.001, pop_ids=None):
    """
    Equilibrium neutral model. Neutral steady state followed by split in
    the immediate past.

    :param params: Unused.
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 2.
    :type pop_ids: lits of str, optional
    """
    Y = NumericsMomentsLD.steady_state([1], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1)
    Y = Y.split(0)
    Y.pop_ids = pop_ids
    return Y


def split_isolation_model_momentsLD(params, rho=None, theta=0.001, pop_ids=None):
    """
    Split into two populations of specifed size, which then have their own
    relative constant sizes and symmetric migration between populations.

    - nu1: Size of population 1 after split.
    - nu2: Size of population 2 after split.
    - T: Time in the past of split (in units of 2*Na generations)
    - m: Migration rate between populations (2*Na*m)

    :param params: The input parameters: (nu1, nu2, T, m)
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nu1, nu2, T = params

    Y = snm(rho=rho, theta=theta)
    Y.integrate([nu1, nu2], T, rho=rho, theta=theta, m=[[0, 0], [0, 0]])
    Y.pop_ids = pop_ids
    return Y