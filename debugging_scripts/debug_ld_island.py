#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_ld_island.py
------------------
Minimal optimiser for the 2-deme island model in moments.LD.
• No simulation – you supply a combined_LD_stats pickle.
• Lets you FIX parameters with --fix '{"m12":0.002}' etc.
• Prints best-fit vector & log-likelihood.

Parameter order for the island model
------------------------------------
    0  nu1   (size deme 1 / N0)
    1  nu2   (size deme 2 / N0)
    2  m12   (2N0·m, scaled)
    3  m21   (2N0·m, scaled)
    4  Ne    (ancestral census size, real units)

Example
-------
python debug_ld_island.py \
    --ld  /path/combined_LD_stats_sim_0.pkl \
    --guess '[0.7,0.6,10,10,10000]' \
    --lower '[0.1,0.1,0,0,1e3]' \
    --upper '[1.5,1.5,200,200,3e4]' \
    --fix '{"m12":10,"m21":10}'
"""

import argparse, json, pickle, sys, numpy as np, moments, moments.LD as LD

# ───────────────────────── CLI ──────────────────────────
ap = argparse.ArgumentParser("moments-LD minimal debugger (island model)")
ap.add_argument("--ld",    required=True,
                help="combined LD stats pickle produced earlier")
ap.add_argument("--guess", required=True,
                help="JSON list e.g. '[0.8,0.8,5,5,1e4]'")
ap.add_argument("--lower", required=True,
                help="JSON list of lower bounds")
ap.add_argument("--upper", required=True,
                help="JSON list of upper bounds")
ap.add_argument("--fix",   default="{}",
                help="JSON dict of param→value to freeze, e.g. '{\"m12\":5}'")
args = ap.parse_args()

# ───────────────────────── Load data ────────────────────
with open(args.ld, "rb") as fh:
    ld_stats = pickle.load(fh)

mv = moments.LD.Parsing.bootstrap_data(ld_stats)   # mean / var-cov

r_bins = np.array([0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3])

# ───────────────────── Optimiser set-up ─────────────────
param_order = ["nu1","nu2","m12","m21","Ne"]      # index helper
p0   = np.array(json.loads(args.guess), dtype=float)
lwr  = np.array(json.loads(args.lower), dtype=float)
upr  = np.array(json.loads(args.upper), dtype=float)

# apply --fix : for any param specified, lower = upper = value
fixed_json = json.loads(args.fix)
fixed_params = [None]*len(param_order)
for name,val in fixed_json.items():
    idx = param_order.index(name)
    lwr[idx] = upr[idx] = float(val)
    fixed_params[idx]   = float(val)

print("initial guess :", p0)
print("lower bounds  :", lwr)
print("upper bounds  :", upr)
print("fixed_params  :", fixed_params)

# sanity
if not np.all(lwr <= upr):
    sys.exit("❌ lower bound > upper bound for at least one parameter")

# ───────────────────── Run optimiser ───────────────────
opt_p, ll = LD.Inference.optimize_log_lbfgsb(
    p0,
    [mv["means"], mv["varcovs"]],
    [LD.Demographics2D.island_model],
    rs=r_bins,
    lower_bound=lwr,
    upper_bound=upr,
    fixed_params=fixed_params,      # freezes any you requested
    verbose=1,
    maxiter=2000,
)

print("\n========= finished =========")
print("best-fit params :", dict(zip(param_order, opt_p)))
print("log-likelihood  :", ll)
