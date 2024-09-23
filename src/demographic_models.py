import demes
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

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
        "A",
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
    N0, N1, N2, T_split = (
        sampled_params["N0"],  # Effective population size of the ancestral population
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
            dict(start_size=N0, end_time=T_split),  # Ancestor size until the split
        ],
    )

    # Add population 1 (deme A1) that splits from the ancestral population
    b.add_deme(
        "A1",
        ancestors=["ancestral"],  # Inherits from the ancestral population
        start_time=T_split,  # Time of the split
        epochs=[
            dict(start_size=N1, end_time=0),  # Population 1 size post-split
        ],
    )

    # Add population 2 (deme A2) that also splits from the ancestral population
    b.add_deme(
        "A2",
        ancestors=["ancestral"],  # Inherits from the ancestral population
        start_time=T_split,  # Time of the split
        epochs=[
            dict(start_size=N2, end_time=0),  # Population 2 size post-split
        ],
    )

    # Resolve the model
    g = b.resolve()

    return g

def split_isolation_model(params, ns, pts):
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