#!/usr/bin/env python3
"""
run_sim_inference_plot.py
-------------------------
Automates multiple island-model simulations (via your existing helper
scripts) and compares inferred parameters (dadi + moments) to the true
parameters with scatter-plots.

* Uses **snakemake_scripts.single_simulation.main** to simulate.
* Runs **run_inference_dadi** and **run_inference_moments** from
  **src.parameter_inference**.
* Saves two PNGs (`dadi_vs_true.png`, `moments_vs_true.png`) showing the
  one-to-one comparisons of each parameter.

### Quick start

```bash
python debugging_scripts/im_model_moments_dadi.py \
    --experiment_config /projects/kernlab/akapoor/Demographic_Inference/experiment_config.json \
    --sim_directory testing_island_model \
    --n_sims 20
```
"""
import argparse
import os
import pickle
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

# ---- user-supplied helpers ---------------------------------------------------
from snakemake_scripts.single_simulation import main as run_single_sim
from src.parameter_inference import run_inference_dadi, run_inference_moments

# -----------------------------------------------------------------------------
# plotting helpers
# -----------------------------------------------------------------------------

def scatter_compare(true_mat, est_mat, param_labels, outfile):
    """Save 2×2 scatter plot comparing true vs. estimated parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.scatter(true_mat[:, i], est_mat[:, i], alpha=0.6)
        lims = [true_mat[:, i].min(), true_mat[:, i].max()]
        ax.plot(lims, lims, "--", lw=1)
        ax.set_xlabel(f"True {param_labels[i]}")
        ax.set_ylabel(f"Estimated {param_labels[i]}")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)

# -----------------------------------------------------------------------------
# main driver
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Batch simulate + infer + plot")
    parser.add_argument("--experiment_config", required=True,
                        help="Path to experiment_config.json used by the sim script")
    parser.add_argument("--sim_directory", default="testing_island_model",
                        help="Directory to write simulation outputs")
    parser.add_argument("--n_sims", type=int, default=10,
                        help="Number of simulation replicates")
    parser.add_argument("--skip_dadi", action="store_true",
                        help="Skip dadi inference (moments only)")
    args = parser.parse_args()

    sim_dir = Path(args.sim_directory)
    sim_results_dir = sim_dir / "simulation_results"
    sim_results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # run simulations + inference in a simple for-loop
    # ---------------------------------------------------------------------
    true_params_list = []
    dadi_params_list = []
    moments_params_list = []

    for i in tqdm(range(args.n_sims)):
        print(f"[rep {i+1}/{args.n_sims}] running simulation …")
        run_single_sim(experiment_config=args.experiment_config,
                       sim_directory=str(sim_dir),
                       sim_number=i)

        # --- load sampled (true) params & SFS --------------------------------
        with open(sim_results_dir / f"sampled_params_{i}.pkl", "rb") as fh:
            true_params = pickle.load(fh)
        with open(sim_results_dir / f"SFS_sim_{i}.pkl", "rb") as fh:
            sfs = pickle.load(fh)

        # ensure pop_ids
        sfs.pop_ids = ["N1", "N2"]

        # build OrderedDict of diploid sample sizes for inference helpers
        sample_sizes_fit = {
            p: (n - 1)//2 for p, n in zip(sfs.pop_ids, sfs.shape)
        }

        # --- pull constants from config -----------------------------------
        with open(args.experiment_config, "r") as fh:
            cfg = json.load(fh)
        mu = cfg["mutation_rate"]
        L  = cfg["genome_length"]

        # initial guess: midpoints of bounds (quick & dirty)
        p0 = [6700, 500, 5e-5, 5e-5]
        lb = [100, 100, 0, 0]
        ub = [30000, 30000, 1e-3, 1e-3]

        # ------------------------------------------------------------------
        # dadi inference (optional)
        # ------------------------------------------------------------------
        if not args.skip_dadi:
            print("  • dadi optimisation …", end="", flush=True)
            model_fs, opt_theta, opt_dadi = run_inference_dadi(
                sfs,
                p0,
                demographic_model="island_model",
                lower_bound=lb,
                upper_bound=ub,
                mutation_rate=mu,
                length=L,
            )
            print(" done")
            dadi_params_list.append(list(opt_dadi.values()))
        else:
            dadi_params_list.append([np.nan]*4)

        # ------------------------------------------------------------------
        # moments inference
        # ------------------------------------------------------------------
        print("  • moments optimisation …", end="", flush=True)
        model_fs, opt_theta, opt_mom = run_inference_moments(
            sfs,
            p0,
            demographic_model="island_model",
            lower_bound=lb,
            upper_bound=ub,
            use_FIM=False,
            mutation_rate=mu,
            length=L,
        )
        print(" done")
        moments_params_list.append(list(opt_mom.values()))

        true_params_list.append(list(true_params.values()))

    # ---------------------------------------------------------------------
    # convert to arrays + plots
    # ---------------------------------------------------------------------
    true_arr     = np.array(true_params_list)
    moments_arr  = np.array(moments_params_list)
    dadi_arr     = np.array(dadi_params_list)

    labels = ["N1", "N2", "m12", "m21"]
    scatter_compare(true_arr, moments_arr, labels, sim_dir / "moments_vs_true.png")
    if not args.skip_dadi:
        scatter_compare(true_arr, dadi_arr, labels, sim_dir / "dadi_vs_true.png")

    print("\nAll done. Plots saved in", sim_dir)


if __name__ == "__main__":
    main()
