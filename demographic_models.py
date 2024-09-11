import demes

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

def split_isolation_model(sampled_params):
    # Unpack the sampled parameters
    N0, N1, N2, t_split, t_isolation_start, t_isolation_end = (
        sampled_params["N0"],  # Ancestral population size
        sampled_params["N1"],  # Size of population 1 after split
        sampled_params["N2"],  # Size of population 2 after split
        sampled_params["t_split"],  # Time of the population split
        sampled_params["t_isolation_start"],  # Time when isolation starts
        sampled_params["t_isolation_end"]  # Time when isolation ends (modern time, typically 0)
    )

    # Initialize the builder
    b = demes.Builder()

    # Add the ancestral population (deme A)
    b.add_deme(
        "ancestral",
        epochs=[
            dict(start_size=N0, end_time=t_split),
        ],
    )

    # Add population 1 (deme A1) that splits from the ancestral population
    b.add_deme(
        "A1",
        ancestors=["ancestral"],  # Inherits from the ancestral population
        start_time=t_split,  # Time of the split
        epochs=[
            dict(start_size=N1, end_time=t_isolation_end),  # Population 1 size post-split
        ],
    )

    # Add population 2 (deme A2) that also splits from the ancestral population
    b.add_deme(
        "A2",
        ancestors=["ancestral"],  # Inherits from the ancestral population
        start_time=t_split,  # Time of the split
        epochs=[
            dict(start_size=N2, end_time=t_isolation_end),  # Population 2 size post-split
        ],
    )

    # Resolve the model
    g = b.resolve()

    return g