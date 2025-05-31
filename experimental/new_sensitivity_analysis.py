# -------------------------------------------------------------------------
# Modified code for Morris and Sobol sampling using proper parameter names
# and actual bounds instead of percentage-based perturbations
# -------------------------------------------------------------------------

from __future__ import annotations
import json, os, itertools, math, multiprocessing as mp
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous
from pulp import LpStatusOptimal, PULP_CBC_CMD, value

from SALib.sample import saltelli, morris as morris_sample
from SALib.analyze import sobol, morris as morris_analyze


# -------------------------------------------------------------------------
# 0. Model builder (unchanged)
# -------------------------------------------------------------------------
def build_and_solve(data: dict[str, list | float]) -> float:
    """Return objective value of CFLP instance described by *data*."""
    demands, capacities = data["demands"], data["capacities"]
    fixed, transp       = data["fixed_costs"], data["transportation_costs"]
    n_cust, n_fac       = len(demands), len(capacities)

    m = LpProblem("CapFacLoc", LpMinimize)

    open_f = {j: LpVariable(f"Open_{j}", cat=LpBinary)
              for j in range(n_fac)}
    serve  = {(i, j): LpVariable(f"Serve_{i}_{j}", 0, 1, LpContinuous)
              for i in range(n_cust) for j in range(n_fac)}

    m += (lpSum(fixed[j]*open_f[j]                 for j in range(n_fac)) +
          lpSum(transp[i][j]*serve[i, j]
                for i in range(n_cust) for j in range(n_fac)))

    for i in range(n_cust):
        m += lpSum(serve[i, j] for j in range(n_fac)) >= 1

    for j in range(n_fac):
        m += (lpSum(serve[i, j]*demands[i] for i in range(n_cust))
              <= capacities[j]*open_f[j])

    m += lpSum(capacities[j]*open_f[j] for j in range(n_fac)) >= sum(demands)
    for i in range(n_cust):
        for j in range(n_fac):
            m += serve[i, j] <= open_f[j]

    m.solve(PULP_CBC_CMD(msg=False))

    if m.status != LpStatusOptimal:
        warnings.warn(f"Solver did not return optimal. Status: {m.status}")
        penalty_value = 1e6
        vars_dict = None
        return penalty_value, vars_dict
    
    vars_dict = {}
    vars_dict = {v.name: v.value() for v in m.variables()}
    return value(m.objective), vars_dict


# -------------------------------------------------------------------------
# 1. Load baseline data + bounds (unchanged)
# -------------------------------------------------------------------------
BASE_PATH   = Path("data/capfacloc_data.json")
RESULT_PATH = Path("results"); RESULT_PATH.mkdir(exist_ok=True)

with BASE_PATH.open() as fh:
    baseline = json.load(fh)

BOUNDS = dict(
    demand=(5, 36),
    capacity=(80, 120),
    tcost=(10, 20),
    fcost=(150, 200),
)

# Flatten parameters with parameter names based on the actual data structure
def flatten(data: dict) -> np.ndarray:
    """→ 1-D array in fixed order."""
    return np.hstack([
        data["demands"],
        data["capacities"],
        data["fixed_costs"],
        np.array(data["transportation_costs"]).ravel()
    ])

def inflate(vector: np.ndarray) -> dict:
    """← reconstruct dict from 1-D array (inverse of *flatten*)."""
    d = dict(baseline)  # shallow copy
    n = len(baseline["demands"])
    m = len(baseline["capacities"])
    t_len = m*n
    d["demands"]              = vector[:n].tolist()
    d["capacities"]           = vector[n:n+m].tolist()
    d["fixed_costs"]          = vector[n+m:n+2*m].tolist()
    t                          = vector[n+2*m:]
    d["transportation_costs"] = t.reshape(n, m).tolist()
    return d

# Create descriptive parameter names
def create_param_names():
    n_cust = len(baseline["demands"])
    n_fac = len(baseline["capacities"])
    
    names = []
    # Demand parameters
    for i in range(n_cust):
        names.append(f"demand_{i}")
    
    # Capacity parameters
    for j in range(n_fac):
        names.append(f"capacity_{j}")
    
    # Fixed cost parameters
    for j in range(n_fac):
        names.append(f"fixed_cost_{j}")
    
    # Transportation cost parameters
    for i in range(n_cust):
        for j in range(n_fac):
            names.append(f"transp_cost_{i}_{j}")
    
    return names

# Create parameter bounds
def create_param_bounds():
    n_cust = len(baseline["demands"])
    n_fac = len(baseline["capacities"])
    
    bounds = []
    # Demand bounds
    for _ in range(n_cust):
        bounds.append(BOUNDS["demand"])
    
    # Capacity bounds
    for _ in range(n_fac):
        bounds.append(BOUNDS["capacity"])
    
    # Fixed cost bounds
    for _ in range(n_fac):
        bounds.append(BOUNDS["fcost"])
    
    # Transportation cost bounds
    for _ in range(n_cust * n_fac):
        bounds.append(BOUNDS["tcost"])
    
    return bounds

PARAM_NAMES = create_param_names()
PARAM_BOUNDS = create_param_bounds()
NUM_P = len(PARAM_NAMES)

# -------------------------------------------------------------------------
# 2. Baseline objective (unchanged)
# -------------------------------------------------------------------------
print("Solving baseline…")
BASE_Z,BASE_VAR = build_and_solve(baseline)
print(f"Baseline objective = {BASE_Z:.2f}")
print(f"Baseline variables = {BASE_VAR}")

# -------------------------------------------------------------------------
# 3. Local elasticity (OAT) (unchanged from your original code)
# -------------------------------------------------------------------------
DELTA = 0.10  # ±10 % perturbation for elasticity
rows = []
for idx in range(NUM_P):
    base   = flatten(baseline)
    theta_plus              = base.copy()
    theta_plus[idx]        *= 1 + DELTA
    theta_minus             = base.copy()
    theta_minus[idx]       *= 1 - DELTA

    # respect bounds
    lo, hi                  = None, None
    if idx < len(baseline["demands"]):
        lo, hi = BOUNDS["demand"]
    elif idx < len(baseline["demands"])+len(baseline["capacities"]):
        lo, hi = BOUNDS["capacity"]
    elif idx < len(baseline["demands"])+2*len(baseline["capacities"]):
        lo, hi = BOUNDS["fcost"]
    else:
        lo, hi = BOUNDS["tcost"]

    theta_plus[idx]  = float(np.clip(theta_plus[idx],  lo, hi))
    theta_minus[idx] = float(np.clip(theta_minus[idx], lo, hi))

    z_plus,vars           = build_and_solve(inflate(theta_plus))
    z_minus,vars          = build_and_solve(inflate(theta_minus))

    elasticity = ((z_plus - z_minus) / (2*BASE_Z)) / DELTA
    rows.append(dict(
        param_id=idx, 
        param_name=PARAM_NAMES[idx],
        z_plus=z_plus, 
        z_minus=z_minus,
        elasticity=elasticity
    ))

pd.DataFrame(rows).to_csv(RESULT_PATH/"elasticity.csv", index=False)
print(f"OAT elasticities saved to {RESULT_PATH/'elasticity.csv'}")


# -------------------------------------------------------------------------
# 4. Morris screening (Updated to use actual bounds)
# -------------------------------------------------------------------------
problem = {
    "num_vars": NUM_P,
    "names": PARAM_NAMES,
    "bounds": PARAM_BOUNDS
}

NUM_TRAJ = 10
sample_morris = morris_sample.sample(problem, N=NUM_TRAJ, num_levels=4)

def evaluate_morris(sample_row):
    # Direct use of sampled values without percentage adjustments
    theta = sample_row
    obj_value, vars = build_and_solve(inflate(theta))
    return obj_value

with mp.Pool() as pool:
    z_vals = pool.map(evaluate_morris, sample_morris)

morris_res = morris_analyze.analyze(problem, sample_morris, np.array(z_vals),
                                    conf_level=0.95, print_to_console=False)

# Save Morris results with descriptive parameter names
pd.DataFrame({
        "parameter": problem["names"],
        "mu_star": morris_res["mu_star"],
        "sigma": morris_res["sigma"]
    }).to_csv(RESULT_PATH/"morris.csv", index=False)
print("Morris indices stored.")


# -------------------------------------------------------------------------
# 5. Sobol / Saltelli (Updated to use actual bounds)
# -------------------------------------------------------------------------
problem = {
    "num_vars": NUM_P,
    "names": PARAM_NAMES,
    "bounds": PARAM_BOUNDS
}
N_BASE = 8192
sample_saltelli = saltelli.sample(problem, N_BASE, calc_second_order=False)

def evaluate_sobol(sample_row):
    # Direct use of sampled values without percentage adjustments
    obj, vars_d = build_and_solve(inflate(sample_row))
    return obj, vars_d

with mp.Pool() as pool:
    results = pool.map(evaluate_sobol, sample_saltelli)

# Separate objectives and var‐dicts
z_saltelli   = [r[0] for r in results]
vars_dicts   = [r[1] for r in results]

sobol_res = sobol.analyze(problem, np.array(z_saltelli),
                          calc_second_order=False, print_to_console=False)

# Create a properly structured DataFrame for the samples
sample_df = pd.DataFrame()
for i, name in enumerate(problem["names"]):
    sample_df[name] = sample_saltelli[:, i]
sample_df["objective_value"] = z_saltelli
sample_df["variables"] = vars_dicts

# Save the samples and objective values
sample_df.to_csv(RESULT_PATH/"sobol_sample.csv", index=False)
print("Sobol samples written to CSV.")

# Save the Sobol indices with descriptive parameter names
pd.DataFrame({
    "parameter": problem["names"],
    "S1": sobol_res["S1"],
    "ST": sobol_res["ST"],
    "S1_conf": sobol_res["S1_conf"],
    "ST_conf": sobol_res["ST_conf"]
}).to_csv(RESULT_PATH/"sobol.csv", index=False)
print("Sobol indices written to CSV.")


# -------------------------------------------------------------------------
print("Done.  Results in:", RESULT_PATH.resolve())


# -------------------------------------------------------------------------
# 6. Visualization of Sobol Results
# -------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# # Load the Sobol results if not already in memory
# sobol_results = pd.read_csv(RESULT_PATH/"sobol.csv")
# sobol_samples = pd.read_csv(RESULT_PATH/"sobol_sample.csv")

# # Get the parameters with highest sensitivity indices
# # Sort by total effect (ST)
# sorted_params = sobol_results.sort_values(by="ST", ascending=False)
# top_params = sorted_params.head(4)  # Get top 4 parameters

# print("Top 4 sensitive parameters:")
# print(top_params[["parameter", "S1", "ST"]])

# # Create the figure
# fig = plt.figure(figsize=(15, 10), constrained_layout=True)
# gs = gridspec.GridSpec(2, 3, figure=fig)

# # Create the main subplot for model output distribution
# ax_main = fig.add_subplot(gs[:, 0])

# # Create smaller subplots for sensitivity indices
# ax_s1_1 = fig.add_subplot(gs[0, 1])
# ax_s1_2 = fig.add_subplot(gs[0, 2])
# ax_st_1 = fig.add_subplot(gs[1, 1])
# ax_st_2 = fig.add_subplot(gs[1, 2])

# # Colors for visualization
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# # Main plot - Histogram of objective values
# ax_main.hist(z_saltelli, bins=30, color='gray', alpha=0.7)
# ax_main.axvline(BASE_Z, color='red', linestyle='--', label=f'Baseline: {BASE_Z:.2f}')
# ax_main.set_xlabel('Objective Value')
# ax_main.set_ylabel('Frequency')
# ax_main.set_title('Distribution of Objective Values')
# ax_main.legend()

# # Function to create parameter value vs objective value plot
# def plot_param_sensitivity(ax, param_name, sobol_type, color_idx):
#     param_values = sobol_samples[param_name].values
#     obj_values = sobol_samples["objective_value"].values
    
#     # Sort by parameter value
#     sorted_indices = np.argsort(param_values)
#     sorted_params = param_values[sorted_indices]
#     sorted_obj = obj_values[sorted_indices]
    
#     # Bin the data to show trends more clearly
#     num_bins = 20
#     bin_edges = np.linspace(min(sorted_params), max(sorted_params), num_bins + 1)
#     bin_means = np.zeros(num_bins)
#     bin_stds = np.zeros(num_bins)
    
#     for i in range(num_bins):
#         bin_mask = (sorted_params >= bin_edges[i]) & (sorted_params < bin_edges[i + 1])
#         if np.any(bin_mask):
#             bin_means[i] = np.mean(sorted_obj[bin_mask])
#             bin_stds[i] = np.std(sorted_obj[bin_mask])
    
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     # Plot the binned data
#     ax.plot(bin_centers, bin_means, '-', color=colors[color_idx], 
#             label=f"{param_name}")
#     ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, 
#                    alpha=0.3, color=colors[color_idx])
    
#     # Add Sobol index value
#     sobol_value = sobol_results[sobol_results['parameter'] == param_name][sobol_type].values[0]
#     ax.set_title(f"{sobol_type}: {sobol_value:.3f}")
#     ax.set_xlabel(param_name)
#     ax.set_ylabel('Objective Value')
#     ax.legend(loc='best')

# # Plot top 2 parameters by first-order effect (S1)
# top_params_s1 = sobol_results.sort_values(by="S1", ascending=False).head(2)
# plot_param_sensitivity(ax_s1_1, top_params_s1.iloc[0]['parameter'], 'S1', 0)
# plot_param_sensitivity(ax_s1_2, top_params_s1.iloc[1]['parameter'], 'S1', 1)

# # Plot top 2 parameters by total effect (ST)
# top_params_st = sobol_results.sort_values(by="ST", ascending=False).head(2)
# plot_param_sensitivity(ax_st_1, top_params_st.iloc[0]['parameter'], 'ST', 2)
# plot_param_sensitivity(ax_st_2, top_params_st.iloc[1]['parameter'], 'ST', 3)

# plt.suptitle('Sobol Sensitivity Analysis Results', fontsize=16)
# plt.savefig(RESULT_PATH/"sobol_visualization.png", dpi=300, bbox_inches='tight')
# plt.close()

# # Create a simple bar plot of S1 and ST for all parameters
# plt.figure(figsize=(12, 8))
# top_n = 10  # Number of parameters to show

# # Sort parameters by total effect
# sorted_indices = np.argsort(sobol_results['ST'].values)[::-1][:top_n]
# sorted_params = sobol_results.iloc[sorted_indices]

# x = np.arange(len(sorted_params))
# width = 0.35

# fig, ax = plt.subplots(figsize=(14, 8))
# rects1 = ax.bar(x - width/2, sorted_params['S1'], width, label='First-Order (S1)', color='steelblue')
# rects2 = ax.bar(x + width/2, sorted_params['ST'], width, label='Total Effect (ST)', color='darkorange')

# # Add error bars using confidence intervals
# ax.errorbar(x - width/2, sorted_params['S1'], yerr=sorted_params['S1_conf'], 
#             fmt='none', ecolor='black', capsize=5)
# ax.errorbar(x + width/2, sorted_params['ST'], yerr=sorted_params['ST_conf'], 
#             fmt='none', ecolor='black', capsize=5)

# ax.set_ylabel('Sensitivity Index')
# ax.set_title('Sobol Sensitivity Indices for Top Parameters')
# ax.set_xticks(x)
# ax.set_xticklabels(sorted_params['parameter'], rotation=45, ha='right')
# ax.legend()
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# fig.tight_layout()
# plt.savefig(RESULT_PATH/"sobol_indices_bar.png", dpi=300, bbox_inches='tight')
# plt.close()

# # Create a plot similar to the SALib example you shared
# # This focuses on a few key parameters and their relationship
# plt.figure(figsize=(15, 10))
# gs = gridspec.GridSpec(2, 2)

# # Main plot - histogram with prediction interval
# ax0 = plt.subplot(gs[:, 0])

# # Get a representative parameter to use as x-axis
# # Here we use the parameter with the highest total effect
# key_param = sorted_params.iloc[0]['parameter']
# x_values = sobol_samples[key_param].values

# # Sort everything by this parameter
# sort_idx = np.argsort(x_values)
# x_sorted = x_values[sort_idx]
# y_sorted = sobol_samples["objective_value"].values[sort_idx]

# # Calculate running mean and percentiles
# window = max(1, len(x_sorted) // 100)  # Use 1% of data points for smoothing
# y_mean = np.convolve(y_sorted, np.ones(window)/window, mode='valid')
# x_mean = x_sorted[window-1:]

# # Calculate prediction intervals (95%)
# prediction_interval = 95
# y_samples = []
# for i in range(len(x_sorted) - window + 1):
#     y_samples.append(y_sorted[i:i+window])
# y_samples = np.array(y_samples)
# y_lower = np.percentile(y_samples, 50 - prediction_interval/2., axis=1)
# y_upper = np.percentile(y_samples, 50 + prediction_interval/2., axis=1)

# # Plot mean and prediction interval
# ax0.plot(x_mean, y_mean, label="Mean", color='black')
# ax0.fill_between(x_mean, y_lower, y_upper, alpha=0.5, color='black',
#                 label=f"{prediction_interval}% prediction interval")
# ax0.set_xlabel(key_param)
# ax0.set_ylabel("Objective Value")
# ax0.legend(loc='upper center')

# # Sobol indices for top parameters
# ax1 = plt.subplot(gs[0, 1])
# ax2 = plt.subplot(gs[1, 1])

# # Create smoothed versions of the Sobol indices
# # We'll use the parameter values as our x-axis and interpolate S1/ST values
# for i, (idx, param) in enumerate([
#         (0, top_params_s1.iloc[0]['parameter']), 
#         (1, top_params_s1.iloc[1]['parameter'])
#     ]):
#     ax = [ax1, ax2][i]
    
#     # We don't have actual S1 values for every x, so we're showing the overall S1
#     param_s1 = sobol_results[sobol_results['parameter'] == param]['S1'].values[0]
    
#     # Plot the S1 value as a horizontal line
#     ax.axhline(y=param_s1, color='black', 
#               label=f"S1$_{{{param}}}$ = {param_s1:.3f}")
    
#     ax.set_xlabel(key_param)
#     ax.set_ylabel("First-order Sobol Index")
#     ax.set_ylim(0, 1.04)
#     ax.yaxis.set_label_position("right")
#     ax.yaxis.tick_right()
#     ax.legend(loc='upper right')

# plt.suptitle('Sobol Sensitivity Analysis - Similar to SALib Example', fontsize=16)
# plt.tight_layout()
# plt.savefig(RESULT_PATH/"sobol_salib_style.png", dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Visualizations saved to {RESULT_PATH.resolve()}")