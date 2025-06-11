#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_island_ld.py
----------------
Simulate a two-deme island model, generate one VCF per window, and compute
LD statistics with moments.LD – all in parallel using Ray.

Typical call
------------
python run_island_ld.py \
    --config      /projects/kernlab/akapoor/Demographic_Inference/experiment_config.json \
    --param-file  /projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/sampled_params_0.pkl \
    --sim-number  0 \
    --windows     100 \
    --outdir      /projects/kernlab/akapoor/Demographic_Inference/moments_data \
    --num-cpus    20
"""

# ────────────────────────────── stdlib
import argparse
import json
import os
import pickle
from pathlib import Path
import math

# ────────────────────────────── light, picklable deps
import numpy as np
import ray

# project-local utilities
from src.preprocess import Processor
from src.parameter_inference import get_LD_stats

# ─────────────────────── CLI parsing ──────────────────────────
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Island-model LD simulation & inference with Ray."
    )
    p.add_argument("--config", required=True,
                   help="experiment_config.json (global simulation settings)")
    p.add_argument("--param-file", required=True,
                   help="Pickle with 'N1', 'N2', 'm12', 'm21'")
    p.add_argument("--sim-number", type=int, default=0,
                   help="Simulation index (default 0)")
    p.add_argument("--windows", type=int, default=100,
                   help="Number of genomic windows to simulate (default 100)")
    p.add_argument("--outdir", default="./moments_data",
                   help="Root output directory")
    p.add_argument("--num-cpus", type=int, default=None,
                   help="How many CPUs Ray may use (default = all available)")
    return p.parse_args()

# ─────────────────── demography helper (light) ────────────────

def island_model_graph(sampled_params):
    import demes
    """
    Island model with a single ancestral root (required by moments.Demes.LD).
    Two daughter demes (N1, N2) exchange migrants continuously.

    Expected keys in sampled_params
    --------------------------------
      N1   : size of population 1
      N2   : size of population 2
      m12  : migration rate N1 -> N2   (0 ≤ m12 ≤ 1)
      m21  : migration rate N2 -> N1   (0 ≤ m21 ≤ 1)
      N0   : (optional) ancestral size.  If absent, uses mean(N1, N2).

    Returns
    -------
    demes.Graph
    """
    N0  = sampled_params["N0"]
    N1  = sampled_params["N1"]
    N2  = sampled_params["N2"]
    m12 = sampled_params["m12"]
    m21 = sampled_params["m21"]

    # sanity-check migration rates
    if not (0 <= m12 <= 1 and 0 <= m21 <= 1):
        raise ValueError(f"Invalid migration rates: m12={m12}, m21={m21}")

    b = demes.Builder()

    # 1) single root deme: from ∞ down to time 1
    b.add_deme(
        name="ancestral",
        start_time=math.inf,
        epochs=[{"end_time": 1, "start_size": N0}],
    )

    # 2) daughter demes exist from time 1 → 0 (present)
    b.add_deme(
        name="N1",
        ancestors=["ancestral"],
        start_time=1,
        epochs=[{"end_time": 0, "start_size": N1}],
    )
    b.add_deme(
        name="N2",
        ancestors=["ancestral"],
        start_time=1,
        epochs=[{"end_time": 0, "start_size": N2}],
    )

    # 3) continuous (possibly asymmetric) migration
    b.add_migration(source="N1", dest="N2", rate=m12)
    b.add_migration(source="N2", dest="N1", rate=m21)

    return b.resolve()

# ───────────────────── Ray worker task ────────────────────────
@ray.remote
def simulate_and_compute_ld(
    win_idx: int,
    cfg: dict,
    params: dict,
    sim_number: int,
    base_outdir: str | Path,
) -> str:
    """
    Simulate one window + LD statistics; return the .pkl path.
    Heavy libraries are imported _inside_ to avoid Ray-pickling issues.
    """
    import msprime
    import moments
    import moments.LD
    from pathlib import Path  # local import for symmetry

    outdir = Path(base_outdir) / f"sim_{sim_number}" / f"window_{win_idx}"
    outdir.mkdir(parents=True, exist_ok=True)

    # demography & samples
    demog = msprime.Demography.from_demes(island_model_graph(params))
    samples = {pop: n for pop, n in cfg["num_samples"].items() if n > 0}

    # ancestry + mutations
    ts = msprime.sim_ancestry(
        samples,
        demography=demog,
        sequence_length=cfg["genome_length"],
        recombination_rate=cfg["recombination_rate"],
        random_seed=cfg["seed"] + win_idx,
    )
    ts = msprime.sim_mutations(
        ts, rate=cfg["mutation_rate"], random_seed=win_idx + 1
    )
    if ts.num_sites == 0:
        raise RuntimeError(f"[window {win_idx}] No mutations simulated.")

    # VCF
    vcf_path = outdir / f"vcf_window.{win_idx}.vcf"
    with vcf_path.open("w") as fout:
        ts.write_vcf(fout, allow_position_zero=True)
    os.system(f"gzip -f {vcf_path}")
    vcf_path = vcf_path.with_suffix(".vcf.gz")

    # sample sheet & flat map
    Processor.write_samples_and_rec_map(
        cfg, window_number=win_idx, folderpath=str(outdir)
    )

    # LD stats
    r_bins = np.array(
        [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5,
         1e-4, 2e-4, 5e-4, 1e-3]
    )

    ld_stats = get_LD_stats(
        vcf_file=str(vcf_path),
        r_bins=r_bins,
        flat_map_path=str(outdir / "flat_map.txt"),
        pop_file_path=str(outdir / "samples.txt"),
    )

    ld_path = outdir / f"ld_stats_window.{win_idx}.pkl"
    with ld_path.open("wb") as fh:
        pickle.dump(ld_stats, fh)

    return str(ld_path)

# ─────────────────────────── main ─────────────────────────────
def main() -> None:
    args = parse_cli()

    # load cfg & island parameters
    with open(args.config) as fh:
        cfg = json.load(fh)
    with open(args.param_file, "rb") as fh:
        full_params = pickle.load(fh)
    isl_params = {k: full_params[k] for k in ("N0", "N1", "N2", "m12", "m21")}

    # start Ray with a _short_ temp dir
    short_tmp = "/tmp/ray_island"
    os.makedirs(short_tmp, exist_ok=True)
    ray.init(
        num_cpus=args.num_cpus,
        _temp_dir=short_tmp,
        log_to_driver=False,
    )

    # launch one task per window
    futures = [
        simulate_and_compute_ld.remote(
            win_idx=w,
            cfg=cfg,
            params=isl_params,
            sim_number=args.sim_number,
            base_outdir=args.outdir,
        )
        for w in range(args.windows)
    ]
    ld_files = ray.get(futures)   # block until all done
    ray.shutdown()

    print("\n✓ All windows finished.  LD-stats files:")
    for p in sorted(ld_files):
        print(" •", p)

# ───────────────────────── entrypoint ─────────────────────────
if __name__ == "__main__":
    main()
