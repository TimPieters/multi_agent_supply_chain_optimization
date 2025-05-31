# Notebooks Directory Structure

This directory contains Jupyter notebooks organized to provide a clear and modular approach to data exploration, benchmark analysis, and agent performance comparison for the supply chain optimization models.

## Structure:

```
notebooks/
├── analysis/
│   ├── common_analysis_functions.py
│   ├── cflp_analysis.ipynb
│   ├── vrp_analysis.ipynb
│   └── agent_benchmark_comparison.ipynb
├── data_exploration/
│   ├── cflp_data_exploration.ipynb
│   └── vrp_data_exploration.ipynb
└── README.md
```

## Purpose of Each Component:

### `notebooks/analysis/`

This subdirectory is dedicated to the analysis of benchmark results and agent performance.

*   **`common_analysis_functions.py`**:
    *   A Python module containing reusable functions for loading benchmark data, parsing JSON columns, filtering data (e.g., for optimal solutions), and generic plotting (e.g., objective value distributions, solve time distributions).
    *   These functions are designed to be imported and used by the individual analysis notebooks, promoting code reusability and consistency.

*   **`cflp_analysis.ipynb`**:
    *   This Jupyter notebook focuses exclusively on the analysis and visualization of benchmark results for the **Capacitated Facility Location Problem (CFLP)**.
    *   It loads data from `benchmark/CFLP/cflp_benchmark_log.csv`.
    *   It includes CFLP-specific analysis, such as facility selection frequency, impact of demand/capacity/fixed cost changes, and other relevant metrics for CFLP.

*   **`vrp_analysis.ipynb`**:
    *   This Jupyter notebook is dedicated to the analysis and visualization of benchmark results for the **Vehicle Routing Problem (VRP)**.
    *   It loads data from `benchmark/VRP/vrp_benchmark_10cust_2veh.csv`.
    *   It includes VRP-specific analysis, such as route characteristics, vehicle utilization, and the impact of fleet size or demand changes.

*   **`agent_benchmark_comparison.ipynb`**:
    *   This notebook is designed for **comparing the performance of the optimization agent** (from `langgraph_sensitivity_analysis.py` logs) against the established benchmarks (from both CFLP and VRP).
    *   It will import data and plotting utilities from `common_analysis_functions.py`.
    *   Its primary focus is on comparative metrics like solution quality, solve time, and success rates between the agent's solutions and the benchmark solutions. It should avoid model-specific analysis that belongs in the individual `_analysis.ipynb` notebooks.

### `notebooks/data_exploration/`

This subdirectory contains notebooks for initial exploration and understanding of the raw input data for each optimization model.

*   **`cflp_data_exploration.ipynb`**:
    *   For exploring the raw JSON data used by the CFLP model (e.g., `models/CFLP/data/capfacloc_data_10cust_10fac.json`).
    *   Helps in understanding data structure, distributions of demands, capacities, fixed costs, and transportation costs before running benchmarks.

*   **`vrp_data_exploration.ipynb`**:
    *   For exploring the raw JSON data used by the VRP model (e.g., `models/VRP/data/vrp_data_10cust_2veh.json`).
    *   Helps in understanding distance matrices, demands, vehicle capacities, and depot locations.

## How to Use:

1.  **Run Benchmarks**: Ensure you have run the benchmark scripts (`benchmark/CFLP/cflp_benchmark.py` and `benchmark/VRP/vrp_benchmark.py`) to generate the necessary CSV log files.
2.  **Explore Raw Data**: Use the notebooks in `data_exploration/` to understand the input data for your models.
3.  **Analyze Benchmarks**: Use `cflp_analysis.ipynb` and `vrp_analysis.ipynb` for detailed, model-specific analysis of the benchmark results.
4.  **Compare Agent Performance**: Use `agent_benchmark_comparison.ipynb` to compare your agent's performance against the benchmarks.

This structured approach aims to make your analysis workflow more organized, efficient, and easier to navigate.
