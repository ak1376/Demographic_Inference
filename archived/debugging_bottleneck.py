import moments
import numpy as np

# Define the demographic model function
def bottleneck_model(params, ns):
    N_pre, N_bottleneck, T_bottleneck, T_recovery = params
    # Pre-bottleneck: equilibrium population
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    
    # Bottleneck: severe population size reduction
    fs = moments.Manips.integrate(fs, [N_bottleneck], T_bottleneck)
    
    # Post-bottleneck recovery to original size
    fs = moments.Manips.integrate(fs, [N_pre], T_recovery)
    
    return fs

# Set the sample size
ns = [20]  # 20 individuals sampled

# Define the true parameters
true_params = [10_000, 100, 10, 50]

# Simulate the data
data = bottleneck_model(true_params, ns)

# Add some noise to the data to simulate real-world data
data = moments.Spectrum(data.sample()) + np.random.normal(scale=0.01, size=len(data))

# Plot the simulated data
import matplotlib.pyplot as plt
moments.Plotting.plot_1d_fs(data)
plt.show()