#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 island_ld_debug.py  – Parallel LD pipeline + Moments-LD inference (island model)

 • Simulate NUM_REPS tree-sequences with msprime  (only for replicates that
   are missing)
 • Parse LD stats → <output>/h5/ + <output>/ld_stats/
 • Bootstrap means / var-cov, cache to .bp
 • ⚠ Plot empirical vs. theoretical LD curves  **before** inference
 • Fit Moments-LD island_model (optimize_log_lbfgsb)
"""

from __future__ import annotations

import argparse, gc, inspect, os, pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple

import demes, moments, numpy as np, ray


# ───────────────────────── configuration ────────────────────────── #
@dataclass
class Config:
    num_reps: int = 100
    seq_length: int = 1_000_000
    mu: float = 1.5e-8
    recomb: float = 1.5e-8
    n_samples: int = 30                       # diploids per deme
    true_params: Tuple[int, int, float, float] = (10_000, 6_000, 8e-6, 3e-5)

    r_bins: np.ndarray = field(
        default_factory=lambda: np.array(
            [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
        )
    )

    max_parallel: int = 8
    task_mem_gb: float = 5.5
    data_dir: Path = Path("./data")


CFG = Config()  # will be overwritten by CLI

# ─────────────────────────── demography ──────────────────────────── #
def island_graph(params: Tuple[int, int, float, float]) -> demes.Graph:
    N1, N2, m12, m21 = params
    b = demes.Builder()
    b.add_deme("ancestral", epochs=[{"start_size": 10_000, "end_time": 100}])
    b.add_deme("pop0", ancestors=["ancestral"], epochs=[{"start_size": N1}])
    b.add_deme("pop1", ancestors=["ancestral"], epochs=[{"start_size": N2}])
    b.add_migration(source="pop0", dest="pop1", rate=m12)
    b.add_migration(source="pop1", dest="pop0", rate=m21)
    return b.resolve()


GRAPH = island_graph(CFG.true_params)

# ────────────────────────── Ray tasks ────────────────────────────── #
@ray.remote(num_cpus=1,
            memory=int(CFG.task_mem_gb * 1024 ** 3),
            max_retries=1,
            retry_exceptions=True)
def simulate_vcf(rep: int) -> str:
    """
    Simulate replicate *rep* if <output>/vcf/island.rep.vcf.gz doesn’t exist.
    Return the (gzipped) VCF path.
    """
    from pathlib import Path
    import msprime

    vcf_path = CFG.data_dir / "vcf" / f"island.{rep}.vcf"
    vcf_gz   = vcf_path.with_suffix(".vcf.gz")

    if vcf_gz.exists():
        return str(vcf_gz)

    demog = msprime.Demography.from_demes(GRAPH)
    ts = next(
        msprime.sim_ancestry(
            {"pop0": CFG.n_samples, "pop1": CFG.n_samples},
            demography=demog,
            sequence_length=CFG.seq_length,
            recombination_rate=CFG.recomb,
            num_replicates=1,
            random_seed=42 + rep,
        )
    )
    ts = msprime.sim_mutations(ts, rate=CFG.mu, random_seed=rep + 1)
    with vcf_path.open("w") as f:
        ts.write_vcf(f)
    os.system(f"gzip -f {vcf_path}")
    return str(vcf_gz)


@ray.remote(num_cpus=1,
            memory=int(CFG.task_mem_gb * 1024 ** 3),
            max_retries=1,
            retry_exceptions=True)
def ld_stats(rep: int) -> str:
    """
    Compute LD stats for replicate ‘rep’ *only if* its pickle is missing.
    Return the pickle path.
    """
    pkl_path = CFG.data_dir / "ld_stats" / f"island_ld.{rep}.pkl"
    if pkl_path.exists():
        return str(pkl_path)

    vcf_gz  = CFG.data_dir / "vcf" / f"island.{rep}.vcf.gz"
    h5_path = CFG.data_dir / "h5"  / f"island.{rep}.h5"

    compute = moments.LD.Parsing.compute_ld_statistics
    kwargs = dict(
        vcf_file=str(vcf_gz),
        rec_map_file=str(CFG.data_dir / "flat_map.txt"),
        pop_file=str(CFG.data_dir / "samples.txt"),
        pops=["pop0", "pop1"],
        r_bins=CFG.r_bins,
        report=False,
    )
    if "out_prefix" in inspect.signature(compute).parameters:
        kwargs["out_prefix"] = str(h5_path.with_suffix(""))
        ld = compute(**kwargs)
    else:
        ld = compute(**kwargs)
        default_h5 = vcf_gz.with_suffix(".h5")
        if default_h5.exists():
            default_h5.replace(h5_path)

    with pkl_path.open("wb") as f:
        pickle.dump(ld, f)

    return str(pkl_path)


# ────────────────────────── helpers ─────────────────────────────── #
def prepare_static_files():
    """population label file + flat map"""
    samp = CFG.data_dir / "samples.txt"
    rmap = CFG.data_dir / "flat_map.txt"
    if not samp.exists():
        with samp.open("w") as f:
            f.write("sample\tpop\n")
            for pop in range(2):
                for i in range(CFG.n_samples):
                    f.write(f"tsk_{pop*CFG.n_samples+i}\tpop{pop}\n")
    if not rmap.exists():
        with rmap.open("w") as f:
            f.write("pos\tMap(cM)\n0\t0\n")
            f.write(f"{CFG.seq_length}\t{CFG.recomb*CFG.seq_length*100}\n")


def run_batches(task, rep_list):
    """Run Ray tasks in chunks and yield results."""
    for start in range(0, len(rep_list), CFG.max_parallel):
        futs = [task.remote(r) for r in rep_list[start:start+CFG.max_parallel]]
        for res in ray.get(futs):
            yield res


# ───────────────────────────── pipeline ─────────────────────────── #
def run_pipeline():
    print("Config:", asdict(CFG))
    prepare_static_files()
    ray.init(logging_level="WARNING", ignore_reinit_error=True)

    mv_path   = CFG.data_dir / "means_varcovs.bp"
    boots_path= CFG.data_dir / "bootstrap_sets.bp"

    # ── (1) make sure all LD pickles exist ───────────────────────── #
    existing_pkls = {int(p.stem.split(".")[1])  # extract rep id
                     for p in (CFG.data_dir / "ld_stats").glob("island_ld.*.pkl")}
    missing_reps = [rep for rep in range(CFG.num_reps) if rep not in existing_pkls]

    if missing_reps:
        print(f"Need LD for {len(missing_reps)} replicates → sim+parse only those …")

        # Simulate only if VCF missing
        vcf_missing = [r for r in missing_reps
                       if not (CFG.data_dir / "vcf" / f"island.{r}.vcf.gz").exists()]
        if vcf_missing:
            list(run_batches(simulate_vcf, vcf_missing))

        # Compute LD for the still-missing replicates
        list(run_batches(ld_stats, missing_reps))
    else:
        print("All LD pickles already present ✔")

    # ── (2) load ALL LD pickles and (re)bootstrap ────────────────── #
    ld_dict: Dict[int, moments.LD.LDstats] = {}
    for pkl_file in (CFG.data_dir / "ld_stats").glob("island_ld.*.pkl"):
        rep = int(pkl_file.stem.split(".")[1])
        ld_dict[rep] = pickle.load(pkl_file.open("rb"))

    mv    = moments.LD.Parsing.bootstrap_data(ld_dict)
    boots = moments.LD.Parsing.get_bootstrap_sets(ld_dict)
    mv_path.write_bytes(pickle.dumps(mv))
    boots_path.write_bytes(pickle.dumps(boots))

    # ── (3) plot BEFORE inference ────────────────────────────────── #
    print("Plotting LD curves …")
    theo = moments.Demes.LD(
        GRAPH, sampled_demes=["pop0", "pop1"], rho=4 * 10_000 * CFG.r_bins
    )
    theo = moments.LD.LDstats(
        [(a+b)/2 for a, b in zip(theo[:-2], theo[1:-1])] + [theo[-1]],
        num_pops=theo.num_pops, pop_ids=theo.pop_ids)
    theo = moments.LD.Inference.sigmaD2(theo)

    pdf_path = CFG.data_dir / "island_comparison.pdf"
    moments.LD.Plotting.plot_ld_curves_comp(
        theo,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=CFG.r_bins,
        stats_to_plot=[
            ["DD_0_0"], ["DD_0_1"], ["DD_1_1"],
            ["Dz_0_0_0"], ["Dz_0_1_1"], ["Dz_1_1_1"],
            ["pi2_0_0_1_1"], ["pi2_0_1_0_1"], ["pi2_1_1_1_1"]],
        labels=[
            [r"$D_0^2$"], [r"$D_0D_1$"], [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"], [r"$Dz_{0,1,1}$"], [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"], [r"$\pi_{2;0,1,0,1}$"],
            [r"$\pi_{2;1,1,1,1}$"]],
        rows=3, plot_vcs=True, show=False, fig_size=(6,4),
        output=str(pdf_path))
    print("✓ Plot saved first →", pdf_path.relative_to(Path.cwd()))

    # ── (4) Moments-LD inference (island_model) ──────────────────── #
    print("Running Moments-LD inference …")
    p0 = [1.0, 0.5, 0.05, 1.0, 1e4]            # (nu1, nu2, m12, m21, Ne) # true_params: Tuple[int, int, float, float] = (10_000, 500, 1e-6, 1e-5)
    lower_bound = [0.4 , 0.2 , 0.01 , 0.05 , 10000]   # ν₁, ν₂, M₁₂, M₂₁, Nₑ
    upper_bound = [2.0 , 1.2 , 2.0  , 5.0  , 10000]
    # p0 = moments.LD.Util.perturb_params(p0, fold=0.1)

    opt, LL = moments.LD.Inference.optimize_log_lbfgsb(
        p0, [mv["means"], mv["varcovs"]],
        [moments.LD.Demographics2D.island_model],
        rs=CFG.r_bins, verbose=1, lower_bound=lower_bound,
        upper_bound=upper_bound)
    phys = moments.LD.Util.rescale_params(opt, ["nu","nu","m","m","Ne"])

    rpt = CFG.data_dir / "inference_results.txt"
    with rpt.open("w") as f:
        f.write("Moments-LD island-model inference\n")
        f.write(f"Log-likelihood: {LL:.3f}\n\n")
        f.write("True parameters:\n")
        f.write(f"  N1  : {CFG.true_params[0]:.0f}\n")
        f.write(f"  N2  : {CFG.true_params[1]:.0f}\n")
        f.write(f"  m12 : {CFG.true_params[2]:.4g}\n")
        f.write(f"  m21 : {CFG.true_params[3]:.4g}\n\n")
        f.write("Best-fit (physical units):\n")
        f.write(f"  nu1 : {phys[0]:.3f}\n")
        f.write(f"  nu2 : {phys[1]:.3f}\n")
        f.write(f"  m12 : {phys[2]:.4g}\n")
        f.write(f"  m21 : {phys[3]:.4g}\n")
    print("✓ Inference results →", rpt.relative_to(Path.cwd()))


# ───────────────────────────── CLI entry ─────────────────────────── #
def main():
    p = argparse.ArgumentParser(
        description="Parallel LD pipeline + Moments-LD island-model inference")
    p.add_argument("--seq_length", type=float, default=CFG.seq_length)
    p.add_argument("--mu", type=float, default=CFG.mu)
    p.add_argument("--recombination_rate", type=float, default=CFG.recomb)
    p.add_argument("--n_samples", type=int,   default=CFG.n_samples)
    p.add_argument("--output_path", type=str, default=str(CFG.data_dir))
    args = p.parse_args()

    CFG.seq_length = int(args.seq_length)
    CFG.mu        = args.mu
    CFG.recomb    = args.recombination_rate
    CFG.n_samples = args.n_samples
    CFG.data_dir  = Path(args.output_path).resolve()

    for sub in ("vcf", "h5", "ld_stats"):
        (CFG.data_dir / sub).mkdir(parents=True, exist_ok=True)

    global GRAPH
    GRAPH = island_graph(CFG.true_params)

    run_pipeline()


if __name__ == "__main__":
    main()
