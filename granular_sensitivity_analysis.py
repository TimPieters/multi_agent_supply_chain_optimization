"""
Traditional Sensitivity Analysis Framework for Capacitated Facility Location Problem
This module provides traditional sensitivity analysis methods for the Capacitated Facility Location
Problem (CFLP) using SALib and other established techniques. It serves as a benchmark for
comparison with generative AI approaches.

Methods included:
1. Local sensitivity analysis (One-at-a-time)
2. Global sensitivity analysis (Sobol, Morris)
3. Scenario-based analysis with triangular distributions
4. Parameter importance ranking
5. Performance metrics for benchmarking
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import pickle
import os
from SALib.sample import morris as morris_sample
from SALib.sample import saltelli
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol
import itertools
import random
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, LpContinuous, lpSum, LpStatus
import pulp # Make sure pulp is imported

# Assume CapacitatedFacilityLocation class is defined elsewhere and provides:
# - __init__(self, n_customers, n_facilities) # Or similar initialization
# - generate_instance(self, data) -> returns a PuLP model instance
# - solve(self, model) -> returns (status, solve_time)

class GranularSensitivityAnalysis:
    """
    A class implementing sensitivity analysis methods for
    the Capacitated Facility Location Problem at the individual coefficient level.
    """

    def __init__(self, cflp_model, output_dir="granular_sensitivity_results"):
        """
        Initialize the sensitivity analysis framework.

        Args:
            cflp_model: An instance of a class that provides:
                        - n_customers attribute (int)
                        - n_facilities attribute (int)
                        - generate_instance(data) method -> PuLP model
                        - solve(model) method -> (status, solve_time)
            output_dir: Directory to save results
        """
        # Ensure the passed model object has the required attributes/methods
        required_attrs = ['n_customers', 'n_facilities']
        required_methods = ['generate_instance', 'solve']
        for attr in required_attrs:
            if not hasattr(cflp_model, attr):
                raise AttributeError(f"Provided cflp_model object must have attribute '{attr}'")
        for meth in required_methods:
             if not hasattr(cflp_model, meth) or not callable(getattr(cflp_model, meth)):
                 raise AttributeError(f"Provided cflp_model object must have method '{meth}'")

        self.cflp_model = cflp_model
        self.output_dir = output_dir
        self.results = {}
        self.execution_times = {}

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Results will be saved in: {self.output_dir}")

    def _solve_instance_and_get_metrics(self, data):
        """
        Solve a single instance of the problem and return key metrics.

        Args:
            data: Data dictionary for the CFLP instance

        Returns:
            dict: Metrics including objective value, solve time, and solution details
                  Returns None if the model fails to solve or is infeasible/non-optimal.
        """
        start_time = time.time()
        try:
            # Generate the PuLP model using the provided cflp_model object
            model = self.cflp_model.generate_instance(data)

            # Solve the model using the provided cflp_model object's solve method
            status, solve_time = self.cflp_model.solve(model)

            # Check if the solution is optimal (PuLP status 1)
            if status != 1:
                 print(f"Warning: Model status is {LpStatus[status]} ({status}). Skipping metric calculation.")
                 # Optionally save the problematic model/data
                 # model.writeLP(f"{self.output_dir}/failed_instance_{time.time()}.lp")
                 # with open(f"{self.output_dir}/failed_data_{time.time()}.pkl", "wb") as f:
                 #    pickle.dump(data, f)
                 return None

            total_time = time.time() - start_time

            # Extract solution details
            objective_value = model.objective.value()

            # Get variable dictionary from the solved PuLP model
            model_vars = model.variablesDict()

            open_facilities = []
            # Use n_facilities from the cflp_model instance stored during __init__
            for j in range(self.cflp_model.n_facilities):
                var_name = f"Open_{j}"
                if var_name in model_vars:
                    var = model_vars[var_name]
                    # Use varValue for PuLP variables
                    if var.varValue > 0.5:
                        open_facilities.append(j)
                else:
                    # This shouldn't happen if generate_instance is correct, but good to check
                    print(f"Warning: Variable {var_name} not found in solved model.")


            # Calculate utilization of open facilities
            utilization = {}
            total_capacity = 0
            # Use n_customers from the cflp_model instance stored during __init__
            n_cust = self.cflp_model.n_customers
            n_fac = self.cflp_model.n_facilities

            if not open_facilities:
                avg_utilization = 0.0
            else:
                for j in open_facilities:
                    # Ensure capacities are indexed correctly
                    capacity = data['capacities'][j]
                    total_capacity += capacity
                    used_capacity = 0.0
                    for i in range(n_cust):
                        var_name = f"Serve_{i}_{j}"
                        if var_name in model_vars:
                            var = model_vars[var_name]
                            # Use varValue for PuLP variables
                            if var.varValue > 1e-6: # Use tolerance for continuous/binary
                                # Ensure demands are indexed correctly
                                used_capacity += data['demands'][i] * var.varValue
                        else:
                             print(f"Warning: Variable {var_name} not found in solved model.")

                    # Calculate utilization, handle division by zero
                    utilization[j] = (used_capacity / capacity) if capacity > 1e-9 else 0.0

                avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0.0

            total_demand = sum(data['demands'])
            # Calculate capacity-demand ratio for *open* facilities
            capacity_demand_ratio = total_capacity / total_demand if total_demand > 1e-9 else float('inf')

            return {
                'objective_value': objective_value,
                'solve_time': solve_time,
                'total_time': total_time,
                'status': status, # Store the numeric status
                'open_facilities_count': len(open_facilities),
                'avg_utilization': avg_utilization,
                'capacity_demand_ratio': capacity_demand_ratio
            }
        except AttributeError as ae:
             print(f"Error accessing model attribute/value: {ae}. Check variable names and model structure.")
             return None
        except Exception as e:
            print(f"Error solving instance or extracting metrics: {e}")
            # Log the data that caused the error if needed
            # with open(f"{self.output_dir}/error_data_{time.time()}.pkl", "wb") as f:
            #     pickle.dump(data, f)
            return None # Indicate failure

    def _get_individual_parameter_names(self, base_data):
        """
        Generate parameter names for each individual coefficient.

        Args:
            base_data: Base data dictionary

        Returns:
            dict: Dictionary mapping parameter types to lists of individual parameter names
        """
        n_customers = len(base_data['demands'])
        n_facilities = len(base_data['capacities'])

        param_names = {
            'demands': [f'demand_{i}' for i in range(n_customers)],
            'capacities': [f'capacity_{j}' for j in range(n_facilities)],
            'fixed_costs': [f'fixed_cost_{j}' for j in range(n_facilities)]
        }

        # Transportation costs are pairs (customer i, facility j)
        param_names['transportation_costs'] = [
            f'trans_cost_{i}_{j}' for i in range(n_customers) for j in range(n_facilities)
        ]

        return param_names

    def _modify_single_coefficient(self, base_data, param_type, index, factor):
        """
        Modify a single coefficient in the data. Ensures data types are preserved.

        Args:
            base_data: Base data dictionary
            param_type: Type of parameter ('demands', 'capacities', etc.)
            index: Index or tuple index of the coefficient to modify
            factor: Scaling factor to apply

        Returns:
            dict: Modified data with the single coefficient changed, preserving types.
        """
        # Deep copy to avoid modifying the original base_data
        modified_data = {k: v.copy() for k, v in base_data.items()}

        try:
            if param_type == 'demands':
                original_value = base_data['demands'][index]
                # Preserve integer type for demands if originally integer
                modified_data['demands'][index] = max(0, round(original_value * factor))
                if isinstance(original_value, (int, np.integer)):
                     modified_data['demands'][index] = int(modified_data['demands'][index])
            elif param_type == 'capacities':
                original_value = base_data['capacities'][index]
                modified_data['capacities'][index] = max(0.0, original_value * factor) # Allow float
            elif param_type == 'fixed_costs':
                original_value = base_data['fixed_costs'][index]
                modified_data['fixed_costs'][index] = max(0.0, original_value * factor) # Allow float
            elif param_type == 'transportation_costs':
                i, j = index  # Unpack tuple for transportation costs
                original_value = base_data['transportation_costs'][i, j]
                modified_data['transportation_costs'][i, j] = max(0.0, original_value * factor) # Allow float
            else:
                 raise ValueError(f"Unknown parameter type: {param_type}")

        except IndexError:
            print(f"Error: Index {index} out of bounds for parameter type {param_type}.")
            return None # Indicate error
        except KeyError:
            print(f"Error: Parameter type {param_type} not found in base_data.")
            return None # Indicate error

        return modified_data

    def one_at_a_time_analysis_granular(self, base_data, factor_range=(0.7, 1.3), n_samples=5):
        """
        Perform one-at-a-time sensitivity analysis at the coefficient level.

        Args:
            base_data: Base data dictionary
            factor_range: Tuple (min_factor, max_factor) for scaling coefficients
            n_samples: Number of samples per coefficient

        Returns:
            pd.DataFrame: Results of the analysis
        """
        print("Running Granular One-at-a-Time Sensitivity Analysis...")
        start_time = time.time()

        # Get parameter names and indices for each coefficient
        param_names = self._get_individual_parameter_names(base_data)
        n_customers = len(base_data['demands'])
        n_facilities = len(base_data['capacities'])

        results_list = [] # Use a list of dicts for efficiency

        # First, solve the base case
        print("Solving base case...")
        base_metrics = self._solve_instance_and_get_metrics(base_data)
        if base_metrics is None:
            print("Error: Base case failed to solve. Aborting OAT analysis.")
            return pd.DataFrame() # Return empty dataframe
        base_objective = base_metrics['objective_value']
        print(f"Base case objective: {base_objective:.2f}")

        # Create evenly spaced factors
        factors = np.linspace(factor_range[0], factor_range[1], n_samples)

        # For each parameter type
        for param_type, names in param_names.items():
            print(f"  Analyzing parameter type: {param_type}")

            # Determine indices based on parameter type
            if param_type == 'demands':
                indices = range(n_customers)
            elif param_type in ['capacities', 'fixed_costs']:
                indices = range(n_facilities)
            elif param_type == 'transportation_costs':
                indices = list(itertools.product(range(n_customers), range(n_facilities)))
            else:
                print(f"Warning: Skipping unknown parameter type {param_type}")
                continue

            # Iterate through coefficients with progress bar
            for idx in tqdm(indices, desc=f"  {param_type} coefficients", leave=False):
                # Get original value and parameter name
                try:
                    if param_type == 'transportation_costs':
                        i, j = idx
                        original_value = base_data[param_type][i, j]
                        param_name = f"{param_type}_{i}_{j}"
                    else:
                        original_value = base_data[param_type][idx]
                        param_name = f"{param_type}_{idx}"
                except (IndexError, KeyError) as e:
                    print(f"Error accessing base data for {param_type}, index {idx}: {e}")
                    continue # Skip this coefficient

                # For each factor
                for factor in factors:
                    # Skip the base case factor (or handle it if needed, e.g., for plotting)
                    if abs(factor - 1.0) < 1e-9: continue

                    # Modify single coefficient
                    modified_data = self._modify_single_coefficient(base_data, param_type, idx, factor)
                    if modified_data is None: continue # Skip if modification failed

                    # Get modified value
                    try:
                        if param_type == 'transportation_costs':
                            i, j = idx
                            modified_value = modified_data[param_type][i, j]
                        else:
                            modified_value = modified_data[param_type][idx]
                    except (IndexError, KeyError) as e:
                         print(f"Error accessing modified data for {param_type}, index {idx}: {e}")
                         continue # Skip this factor

                    # Solve and get metrics
                    metrics = self._solve_instance_and_get_metrics(modified_data)
                    if metrics is None: continue # Skip if solve failed

                    # Calculate relative change from base case
                    rel_change = (metrics['objective_value'] - base_objective) / base_objective if abs(base_objective) > 1e-9 else 0.0

                    # Store results
                    results_list.append({
                        'parameter_type': param_type,
                        'parameter_name': param_name,
                        'index': str(idx),
                        'factor': factor,
                        'original_value': original_value,
                        'modified_value': modified_value,
                        'objective_value': metrics['objective_value'],
                        'relative_change': rel_change,
                        'solve_time': metrics['solve_time'],
                        'total_time': metrics['total_time'],
                        'open_facilities_count': metrics['open_facilities_count'],
                        'avg_utilization': metrics['avg_utilization'],
                        'capacity_demand_ratio': metrics['capacity_demand_ratio']
                    })

        # Convert list of dicts to DataFrame
        if not results_list:
             print("Warning: No successful OAT results were generated.")
             df_results = pd.DataFrame()
        else:
            df_results = pd.DataFrame(results_list)

        # Save to file
        output_path = os.path.join(self.output_dir, "granular_oat_analysis.csv")
        try:
            df_results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving OAT results to CSV: {e}")


        # Calculate execution time
        execution_time = time.time() - start_time

        # Store in class
        self.results['granular_oat'] = df_results
        self.execution_times['granular_oat'] = execution_time

        print(f"Granular One-at-a-Time analysis completed in {execution_time:.2f} seconds")
        return df_results

    def elasticity_analysis_granular(self, base_data, delta_pct=0.01):
        """
        Perform elasticity analysis for each individual coefficient.

        Args:
            base_data: Base data dictionary
            delta_pct: Percentage change for elasticity calculation (e.g., 0.01 for 1%)

        Returns:
            pd.DataFrame: Elasticity results for individual coefficients
        """
        print("Running Granular Elasticity Analysis...")
        start_time = time.time()

        # Get parameter names and indices for each coefficient
        param_names = self._get_individual_parameter_names(base_data)
        n_customers = len(base_data['demands'])
        n_facilities = len(base_data['capacities'])

        # Solve base case
        print("Solving base case...")
        base_metrics = self._solve_instance_and_get_metrics(base_data)
        if base_metrics is None:
            print("Error: Base case failed to solve. Aborting Elasticity analysis.")
            return pd.DataFrame()
        base_objective = base_metrics['objective_value']
        print(f"Base case objective: {base_objective:.2f}")

        results_list = [] # Use a list of dicts

        # For each parameter type
        for param_type, names in param_names.items():
            print(f"  Analyzing elasticity for parameter type: {param_type}")

            # Determine indices based on parameter type
            if param_type == 'demands':
                indices = range(n_customers)
            elif param_type in ['capacities', 'fixed_costs']:
                indices = range(n_facilities)
            elif param_type == 'transportation_costs':
                indices = list(itertools.product(range(n_customers), range(n_facilities)))
            else:
                print(f"Warning: Skipping unknown parameter type {param_type}")
                continue

            # Iterate through coefficients with progress bar
            for idx in tqdm(indices, desc=f"  {param_type} coefficients", leave=False):
                # Get original value and parameter name
                try:
                    if param_type == 'transportation_costs':
                        i, j = idx
                        original_value = base_data[param_type][i, j]
                        param_name = f"{param_type}_{i}_{j}"
                    else:
                        original_value = base_data[param_type][idx]
                        param_name = f"{param_type}_{idx}"
                except (IndexError, KeyError) as e:
                    print(f"Error accessing base data for {param_type}, index {idx}: {e}")
                    continue # Skip this coefficient

                # Avoid calculation if original value is zero or very small, elasticity is ill-defined or zero
                if abs(original_value) < 1e-9:
                    elasticity = 0.0
                    objective_change = 0.0
                    perturbed_value = original_value
                    perturbed_objective = base_objective
                else:
                    # Modify single coefficient with small increase
                    modified_data = self._modify_single_coefficient(base_data, param_type, idx, 1.0 + delta_pct)
                    if modified_data is None: continue # Skip if modification failed

                    # Get perturbed value
                    try:
                        if param_type == 'transportation_costs':
                            i, j = idx
                            perturbed_value = modified_data[param_type][i, j]
                        else:
                            perturbed_value = modified_data[param_type][idx]
                    except (IndexError, KeyError) as e:
                         print(f"Error accessing modified data for {param_type}, index {idx}: {e}")
                         continue # Skip this factor

                    # Evaluate model
                    perturbed_metrics = self._solve_instance_and_get_metrics(modified_data)
                    if perturbed_metrics is None: continue # Skip if solve failed

                    perturbed_objective = perturbed_metrics['objective_value']

                    # Calculate elasticity: (% change in output) / (% change in input)
                    objective_change = perturbed_objective - base_objective
                    if abs(base_objective) > 1e-9 and abs(delta_pct) > 1e-9:
                        elasticity = (objective_change / base_objective) / delta_pct
                    elif abs(objective_change) < 1e-9:
                        elasticity = 0.0 # No change in output means zero elasticity
                    else:
                        # Base objective is zero but output changed, or delta is zero
                        elasticity = np.sign(objective_change) * float('inf')

                # Store results
                results_list.append({
                    'parameter_type': param_type,
                    'parameter_name': param_name,
                    'index': str(idx),
                    'original_value': original_value,
                    'perturbed_value': perturbed_value,
                    'elasticity': elasticity,
                    'objective_change': objective_change,
                    'base_objective': base_objective,
                    'perturbed_objective': perturbed_objective
                })

        # Convert to DataFrame
        if not results_list:
             print("Warning: No successful elasticity results were generated.")
             df_results = pd.DataFrame()
        else:
            df_results = pd.DataFrame(results_list)


        # Save to file
        output_path = os.path.join(self.output_dir, "granular_elasticity_analysis.csv")
        try:
            df_results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving Elasticity results to CSV: {e}")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Store in class
        self.results['granular_elasticity'] = df_results
        self.execution_times['granular_elasticity'] = execution_time

        print(f"Granular Elasticity analysis completed in {execution_time:.2f} seconds")
        return df_results

    def morris_method_granular(self, base_data, factor_range=(0.7, 1.3), num_trajectories=10, num_levels=4):
        """
        Perform Morris method for global sensitivity analysis at the coefficient level.

        Args:
            base_data: Base data dictionary
            factor_range: Tuple (min_factor, max_factor) for scaling coefficients
            num_trajectories: Number of trajectories (N in SALib)
            num_levels: Number of levels for discretization

        Returns:
            pd.DataFrame: Results of the analysis (mu, mu_star, sigma)
        """
        print("Running Granular Morris Method Sensitivity Analysis...")
        start_time = time.time()

        n_customers = len(base_data['demands'])
        n_facilities = len(base_data['capacities'])

        # Create a flat list of all coefficient names and their base values/indices
        all_params = []
        param_indices = {}
        param_base_values = {}

        # --- Demands ---
        for i in range(n_customers):
            param_name = f'demand_{i}'
            all_params.append(param_name)
            param_indices[param_name] = ('demands', i)
            param_base_values[param_name] = base_data['demands'][i]

        # --- Capacities ---
        for j in range(n_facilities):
            param_name = f'capacity_{j}'
            all_params.append(param_name)
            param_indices[param_name] = ('capacities', j)
            param_base_values[param_name] = base_data['capacities'][j]

        # --- Fixed Costs ---
        for j in range(n_facilities):
            param_name = f'fixed_cost_{j}'
            all_params.append(param_name)
            param_indices[param_name] = ('fixed_costs', j)
            param_base_values[param_name] = base_data['fixed_costs'][j]

        # --- Transportation Costs ---
        for i in range(n_customers):
            for j in range(n_facilities):
                param_name = f'trans_cost_{i}_{j}'
                all_params.append(param_name)
                param_indices[param_name] = ('transportation_costs', (i, j))
                param_base_values[param_name] = base_data['transportation_costs'][i, j]

        num_vars = len(all_params)
        print(f"  Number of parameters (coefficients): {num_vars}")

        # Set up problem definition for SALib
        # Bounds represent the *range* of the scaling factor, not the parameter value itself
        problem = {
            'num_vars': num_vars,
            'names': all_params,
            'bounds': [factor_range] * num_vars
        }

        # Generate samples (these are scaling factors)
        # Total samples = num_trajectories * (num_vars + 1)
        total_samples = num_trajectories * (num_vars + 1)
        print(f"  Generating {total_samples} Morris samples...")
        # Use optimal_trajectories=None as default, can explore others if needed
        param_values_factors = morris_sample.sample(problem, N=num_trajectories, num_levels=num_levels, optimal_trajectories=None)

        # Results storage
        y = np.zeros(param_values_factors.shape[0]) * np.nan # Initialize with NaN

        print(f"  Evaluating {len(param_values_factors)} Morris samples...")
        valid_results_count = 0
        # Iterate through the sampled scaling factor combinations
        for i, X_factors in enumerate(tqdm(param_values_factors, desc="Evaluating Morris Samples")):
            # Create a fresh copy of base_data for this sample using _modify_single_coefficient logic
            # This is less efficient than modifying in place but safer for complex types
            modified_data_i = base_data.copy() # Start fresh each time

            # Apply parameter scaling factors
            try:
                current_modified_data = base_data # Start with base for this sample
                for j, param_name in enumerate(problem['names']):
                    factor = X_factors[j] # Scaling factor for this param in this sample
                    param_type, idx = param_indices[param_name]

                    # Apply the modification using the helper function
                    # Pass the *current* state of modified data for this sample
                    temp_data = self._modify_single_coefficient(current_modified_data, param_type, idx, factor)

                    if temp_data is None:
                         print(f"Error modifying {param_name} with factor {factor} in sample {i}. Skipping sample.")
                         raise ValueError("Modification failed") # Break inner loop for this sample

                    current_modified_data = temp_data # Update for next parameter in this sample


                # Evaluate model with the fully modified data for this sample
                metrics = self._solve_instance_and_get_metrics(current_modified_data)
                if metrics is not None:
                    y[i] = metrics['objective_value']
                    valid_results_count += 1
                # else: y[i] remains NaN

            except Exception as e:
                # Log error for the specific sample, y[i] will remain NaN
                print(f"Error processing Morris sample {i}: {e}")


        print(f"  Finished evaluating samples. {valid_results_count} valid results obtained out of {len(param_values_factors)}.")

        # Handle any NaN values - crucial for Morris analysis
        nan_mask = np.isnan(y)
        if np.all(nan_mask):
             print("Error: All Morris evaluations failed. Cannot perform analysis.")
             self.results['granular_morris'] = {'df': pd.DataFrame(), 'raw_results': None}
             self.execution_times['granular_morris'] = time.time() - start_time
             return pd.DataFrame()

        if np.any(nan_mask):
            num_nan = np.sum(nan_mask)
            print(f"Warning: {num_nan} Morris evaluations resulted in errors or non-optimal solutions.")
            # Option 1: Replace NaN with mean/median of valid results
            mean_y = np.nanmean(y)
            y = np.nan_to_num(y, nan=mean_y)
            print(f"  NaN values replaced with mean objective value: {mean_y:.2f}")

        # Analyze samples
        print("  Performing Morris analysis...")
        try:
            # Use the scaling factors (param_values_factors) for analysis
            morris_results = morris_analyze.analyze(problem, param_values_factors, y,
                                                    print_to_console=False, # Quieter output
                                                    num_levels=num_levels)
        except Exception as e:
             print(f"Error during SALib Morris analysis: {e}")
             self.results['granular_morris'] = {'df': pd.DataFrame(), 'raw_results': None}
             self.execution_times['granular_morris'] = time.time() - start_time
             return pd.DataFrame()


        # Create DataFrame from results
        df_results = pd.DataFrame({
            'parameter': problem['names'],
            'mu': morris_results.get('mu', np.nan), # Use .get for robustness
            'mu_star': morris_results.get('mu_star', np.nan),
            'sigma': morris_results.get('sigma', np.nan),
            'mu_star_conf': morris_results.get('mu_star_conf', np.nan) # Confidence interval for mu_star
        })

        # Add parameter type and index columns
        df_results['parameter_type'] = df_results['parameter'].apply(
            lambda x: param_indices.get(x, (None, None))[0]
        )
        df_results['original_index'] = df_results['parameter'].apply(
            lambda x: str(param_indices.get(x, (None, None))[1])
        )

        # Save to file
        output_path = os.path.join(self.output_dir, "granular_morris_analysis.csv")
        try:
            df_results.sort_values('mu_star', ascending=False, inplace=True) # Sort by importance
            df_results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving Morris results to CSV: {e}")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Store in class
        self.results['granular_morris'] = {
            'df': df_results,
            'raw_results': morris_results # Store the raw dictionary too
        }
        self.execution_times['granular_morris'] = execution_time

        print(f"Granular Morris method completed in {execution_time:.2f} seconds")
        return df_results

    def sobol_analysis_granular(self, base_data, factor_range=(0.7, 1.3), n_samples=100, calc_second_order=False, parallel=False, n_processors=None):
        """
        Perform Sobol sensitivity analysis at the coefficient level.

        Args:
            base_data: Base data dictionary
            factor_range: Tuple (min_factor, max_factor) for scaling coefficients
            n_samples: Base number of samples (N in SALib). Total samples = N * (D + 2) or N * (2D + 2)
            calc_second_order: Whether to calculate second-order indices ( computationally expensive!)

        Returns:
            dict: Dictionary containing DataFrames for first-order, total, and optionally second-order indices.
        """
        print("Running Granular Sobol Sensitivity Analysis...")
        print("WARNING: This can be VERY computationally intensive with many coefficients!")
        start_time = time.time()

        n_customers = len(base_data['demands'])
        n_facilities = len(base_data['capacities'])

        # Create a flat list of all coefficient names, indices, and base values (similar to Morris)
        all_params = []
        param_indices = {}
        param_base_values = {}

        # --- Demands ---
        for i in range(n_customers):
            param_name = f'demand_{i}'
            all_params.append(param_name)
            param_indices[param_name] = ('demands', i)
            param_base_values[param_name] = base_data['demands'][i]
        # --- Capacities ---
        for j in range(n_facilities):
            param_name = f'capacity_{j}'
            all_params.append(param_name)
            param_indices[param_name] = ('capacities', j)
            param_base_values[param_name] = base_data['capacities'][j]
        # --- Fixed Costs ---
        for j in range(n_facilities):
            param_name = f'fixed_cost_{j}'
            all_params.append(param_name)
            param_indices[param_name] = ('fixed_costs', j)
            param_base_values[param_name] = base_data['fixed_costs'][j]
        # --- Transportation Costs ---
        for i in range(n_customers):
            for j in range(n_facilities):
                param_name = f'trans_cost_{i}_{j}'
                all_params.append(param_name)
                param_indices[param_name] = ('transportation_costs', (i, j))
                param_base_values[param_name] = base_data['transportation_costs'][i, j]

        num_vars = len(all_params)
        print(f"  Number of parameters (coefficients): {num_vars}")
        if calc_second_order and num_vars > 50: # Heuristic warning
             print(f"Warning: Calculating second-order Sobol indices for {num_vars} parameters will be extremely slow.")


        # Set up problem definition for SALib
        # Bounds define the *distribution* of the scaling factor.
        # For Sobol, we often assume uniform distribution over the factor range.
        problem = {
            'num_vars': num_vars,
            'names': all_params,
            'bounds': [factor_range] * num_vars, # Assumes uniform distribution [min_factor, max_factor]
            # 'dists': ['unif'] * num_vars # Can explicitly specify distribution if needed
        }

        # Generate samples using Saltelli's scheme
        # Total samples = N * (D + 2) without 2nd order, N * (2D + 2) with 2nd order
        total_samples = n_samples * (num_vars + 2) if not calc_second_order else n_samples * (2 * num_vars + 2)
        print(f"  Generating {total_samples} Sobol samples (N={n_samples})...")
        param_values_factors = saltelli.sample(problem, N=n_samples, calc_second_order=calc_second_order)

        # Results storage
        y = np.zeros(param_values_factors.shape[0]) * np.nan # Initialize with NaN

        print(f"  Evaluating {len(param_values_factors)} Sobol samples...")
        valid_results_count = 0
        # Iterate through the sampled scaling factor combinations
        for i, X_factors in enumerate(tqdm(param_values_factors, desc="Evaluating Sobol Samples")):
            # Apply parameter scaling factors - similar logic to Morris
            try:
                current_modified_data = base_data # Start with base for this sample
                for j, param_name in enumerate(problem['names']):
                    factor = X_factors[j] # Scaling factor for this param in this sample
                    param_type, idx = param_indices[param_name]

                    # Apply the modification using the helper function
                    temp_data = self._modify_single_coefficient(current_modified_data, param_type, idx, factor)

                    if temp_data is None:
                         print(f"Error modifying {param_name} with factor {factor} in sample {i}. Skipping sample.")
                         raise ValueError("Modification failed") # Break inner loop for this sample

                    current_modified_data = temp_data # Update for next parameter in this sample

                # Evaluate model with the fully modified data for this sample
                metrics = self._solve_instance_and_get_metrics(current_modified_data)
                if metrics is not None:
                    y[i] = metrics['objective_value']
                    valid_results_count += 1
                # else: y[i] remains NaN

            except Exception as e:
                 # Log error for the specific sample, y[i] will remain NaN
                print(f"Error processing Sobol sample {i}: {e}")


        print(f"  Finished evaluating samples. {valid_results_count} valid results obtained out of {len(param_values_factors)}.")

        # Handle NaN values before analysis
        nan_mask = np.isnan(y)
        if np.all(nan_mask):
             print("Error: All Sobol evaluations failed. Cannot perform analysis.")
             self.results['granular_sobol'] = {'first_order': pd.DataFrame(), 'total': pd.DataFrame(), 'second_order': None, 'raw_results': None}
             self.execution_times['granular_sobol'] = time.time() - start_time
             return {'first_order': pd.DataFrame(), 'total': pd.DataFrame(), 'second_order': None}

        if np.any(nan_mask):
            num_nan = np.sum(nan_mask)
            print(f"Warning: {num_nan} Sobol evaluations resulted in errors or non-optimal solutions.")
            mean_y = np.nanmean(y)
            y = np.nan_to_num(y, nan=mean_y)
            print(f"  NaN values replaced with mean objective value: {mean_y:.2f}")

        # Analyze samples
        print("  Performing Sobol analysis...")
        try:
             # Use the scaling factors (param_values_factors) for analysis
            sobol_results = sobol.analyze(problem, y,
                                          calc_second_order=calc_second_order,
                                          print_to_console=False,
                                          parallel=parallel,
                                          n_processors=n_processors)
        except Exception as e:
             print(f"Error during SALib Sobol analysis: {e}")
             self.results['granular_sobol'] = {'first_order': pd.DataFrame(), 'total': pd.DataFrame(), 'second_order': None, 'raw_results': None}
             self.execution_times['granular_sobol'] = time.time() - start_time
             return {'first_order': pd.DataFrame(), 'total': pd.DataFrame(), 'second_order': None}

        # --- Process Results ---

        # Create DataFrame for first-order (S1) and total-order (ST) indices
        df_main = pd.DataFrame({
            'parameter': problem['names'],
            'S1': sobol_results.get('S1', np.nan),
            'S1_conf': sobol_results.get('S1_conf', np.nan),
            'ST': sobol_results.get('ST', np.nan),
            'ST_conf': sobol_results.get('ST_conf', np.nan)
        })
        df_main['parameter_type'] = df_main['parameter'].apply(lambda x: param_indices.get(x, (None, None))[0])
        df_main['original_index'] = df_main['parameter'].apply(lambda x: str(param_indices.get(x, (None, None))[1]))

        # Separate DataFrames for clarity if desired, though df_main contains both
        df_first_order = df_main[['parameter', 'S1', 'S1_conf', 'parameter_type', 'original_index']].copy()
        df_total = df_main[['parameter', 'ST', 'ST_conf', 'parameter_type', 'original_index']].copy()

        # Create DataFrame for second-order indices (S2) if calculated
        df_second_order = None
        if calc_second_order and 'S2' in sobol_results:
            S2 = sobol_results['S2']
            S2_conf = sobol_results['S2_conf']
            S2_values = []
            num_vars = problem['num_vars'] # Get num_vars again

            # Ensure S2 and S2_conf are numpy arrays for correct indexing
            S2 = np.array(S2)
            S2_conf = np.array(S2_conf)

            # Check dimensions before iterating
            if S2.shape == (num_vars, num_vars) and S2_conf.shape == (num_vars, num_vars):
                # Iterate through the upper triangle of the interaction matrix
                for i in range(num_vars):
                    for j in range(i + 1, num_vars):
                        param_i = problem['names'][i]
                        param_j = problem['names'][j]
                        interaction = f"{param_i} x {param_j}" # Use 'x' for interaction
                        s2_val = S2[i, j]
                        s2_conf_val = S2_conf[i, j]

                        S2_values.append({
                            'parameter_combination': interaction,
                            'parameter1': param_i,
                            'parameter2': param_j,
                            'parameter1_type': param_indices.get(param_i, (None, None))[0],
                            'parameter2_type': param_indices.get(param_j, (None, None))[0],
                            'parameter1_index': str(param_indices.get(param_i, (None, None))[1]),
                            'parameter2_index': str(param_indices.get(param_j, (None, None))[1]),
                            'S2': s2_val,
                            'S2_conf': s2_conf_val
                        })
                if S2_values:
                    df_second_order = pd.DataFrame(S2_values)
                    df_second_order.sort_values('S2', ascending=False, inplace=True) # Sort by interaction strength
            else:
                 print(f"Warning: Mismatch in dimensions for S2/S2_conf. Expected ({num_vars},{num_vars}), Got S2:{S2.shape}, S2_conf:{S2_conf.shape}")


        # Save to files
        output_path_s1 = os.path.join(self.output_dir, "granular_sobol_first_order.csv")
        output_path_st = os.path.join(self.output_dir, "granular_sobol_total.csv")
        try:
            df_first_order.sort_values('S1', ascending=False, inplace=True)
            df_total.sort_values('ST', ascending=False, inplace=True)
            df_first_order.to_csv(output_path_s1, index=False)
            df_total.to_csv(output_path_st, index=False)
            print(f"Sobol S1 results saved to {output_path_s1}")
            print(f"Sobol ST results saved to {output_path_st}")
        except Exception as e:
            print(f"Error saving Sobol S1/ST results to CSV: {e}")

        if df_second_order is not None:
            output_path_s2 = os.path.join(self.output_dir, "granular_sobol_second_order.csv")
            try:
                df_second_order.to_csv(output_path_s2, index=False)
                print(f"Sobol S2 results saved to {output_path_s2}")
            except Exception as e:
                print(f"Error saving Sobol S2 results to CSV: {e}")


        # Calculate execution time
        execution_time = time.time() - start_time

        # Store in class
        self.results['granular_sobol'] = {
            'first_order': df_first_order,
            'total': df_total,
            'second_order': df_second_order,
            'raw_results': sobol_results # Store raw results
        }
        self.execution_times['granular_sobol'] = execution_time

        print(f"Granular Sobol analysis completed in {execution_time:.2f} seconds")
        return {
            'first_order': df_first_order,
            'total': df_total,
            'second_order': df_second_order
        }


    def plot_granular_results(self, top_n=20):
        """
        Generate plots for visualization of granular sensitivity analysis results.

        Args:
            top_n (int): Number of top parameters/interactions to display in plots.
        """
        print("Generating plots for granular sensitivity results...")
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        print(f"Plots will be saved in: {plots_dir}")

        # --- Plot Elasticity Analysis results ---
        if 'granular_elasticity' in self.results and not self.results['granular_elasticity'].empty:
            df = self.results['granular_elasticity'].copy()
            # Handle potential infinite values before calculating absolute
            df['elasticity'] = df['elasticity'].replace([np.inf, -np.inf], np.nan)
            df.dropna(subset=['elasticity'], inplace=True) # Remove rows where elasticity couldn't be calculated

            if not df.empty:
                df['abs_elasticity'] = df['elasticity'].abs() # Use absolute value for ranking
                df_sorted = df.sort_values('abs_elasticity', ascending=False).head(top_n)

                if not df_sorted.empty:
                    plt.figure(figsize=(12, 8))
                    # Use a sequential colormap
                    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(df_sorted)))
                    plt.barh(df_sorted['parameter_name'], df_sorted['elasticity'], color=colors)
                    plt.xlabel('Elasticity (% Change in Objective / % Change in Parameter)')
                    plt.ylabel('Parameter Coefficient')
                    plt.title(f'Top {top_n} Most Elastic Parameters (Granular)')
                    plt.gca().invert_yaxis() # Show most elastic at the top
                    plt.axvline(x=0, color='grey', linestyle='--', alpha=0.7)
                    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plot_path = os.path.join(plots_dir, "granular_elasticity_topN.png")
                    try:
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved: {plot_path}")
                    except Exception as e:
                        print(f"  Error saving plot {plot_path}: {e}")
                    plt.close()

                    # Plot elasticity by parameter type (box plot)
                    plt.figure(figsize=(10, 6))
                    # Filter out extreme outliers for better visualization if needed
                    # Use non-infinite values for quantile calculation
                    valid_elasticity = df['elasticity'][np.isfinite(df['elasticity'])]
                    if not valid_elasticity.empty:
                        q_low = valid_elasticity.quantile(0.01)
                        q_hi  = valid_elasticity.quantile(0.99)
                        # Apply filter to the original dataframe (with finite values)
                        df_filtered = df[np.isfinite(df['elasticity']) & (df['elasticity'] >= q_low) & (df['elasticity'] <= q_hi)]

                        if not df_filtered.empty:
                            sns.boxplot(x='parameter_type', y='elasticity', data=df_filtered, palette='viridis')
                            plt.title('Elasticity Distribution by Parameter Type (Granular, 1st-99th Percentile)')
                            plt.xlabel('Parameter Type')
                            plt.ylabel('Elasticity')
                            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            plot_path = os.path.join(plots_dir, "granular_elasticity_by_type_boxplot.png")
                            try:
                                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                                print(f"  Saved: {plot_path}")
                            except Exception as e:
                                print(f"  Error saving plot {plot_path}: {e}")
                            plt.close()
                        else:
                             print("  Skipping elasticity boxplot (no data within percentile range).")
                    else:
                        print("  Skipping elasticity boxplot (no finite elasticity values).")
                else:
                    print("  Skipping elasticity plots (no data after filtering infinities).")
            else:
                 print("  Skipping elasticity plots (no finite elasticity values found).")
        else:
            print("  Skipping elasticity plots (no results found or DataFrame empty).")


        # --- Plot Morris results ---
        if ('granular_morris' in self.results and
            self.results['granular_morris'] is not None and
            'df' in self.results['granular_morris'] and
            not self.results['granular_morris']['df'].empty):

            df = self.results['granular_morris']['df'].copy()
            # Drop NaN values which can interfere with plotting/sorting
            df.dropna(subset=['mu_star', 'sigma', 'mu_star_conf'], inplace=True)

            if not df.empty:
                df_sorted = df.sort_values('mu_star', ascending=False).head(top_n)

                if not df_sorted.empty:
                    # Plot top N most important parameters by mu_star
                    plt.figure(figsize=(12, 8))
                    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(df_sorted)))
                    plt.barh(df_sorted['parameter'], df_sorted['mu_star'], xerr=df_sorted['mu_star_conf'],
                             color=colors, capsize=3, alpha=0.8, error_kw=dict(alpha=0.5))
                    plt.xlabel('μ* (Overall Importance)')
                    plt.ylabel('Parameter Coefficient')
                    plt.title(f'Morris Method: Top {top_n} Most Important Parameters (μ*)')
                    plt.gca().invert_yaxis()
                    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plot_path = os.path.join(plots_dir, "granular_morris_topN_mu_star.png")
                    try:
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved: {plot_path}")
                    except Exception as e:
                        print(f"  Error saving plot {plot_path}: {e}")
                    plt.close()

                    # Plot mu_star vs sigma (Importance vs Interaction/Non-linearity)
                    plt.figure(figsize=(10, 8))
                    # Use parameter type for coloring/markers
                    param_types = df['parameter_type'].unique()
                    palette = sns.color_palette("husl", len(param_types))
                    type_color_map = dict(zip(param_types, palette))

                    for p_type in param_types:
                        subset = df[df['parameter_type'] == p_type]
                        if not subset.empty:
                            plt.scatter(subset['mu_star'], subset['sigma'],
                                        label=p_type, color=type_color_map.get(p_type, 'grey'), alpha=0.7, s=50, edgecolors='w', linewidth=0.5) # s is marker size

                    # Optional: Annotate top N points by mu_star
                    df_annot = df.sort_values('mu_star', ascending=False).head(top_n)
                    texts = []
                    for i, row in df_annot.iterrows():
                         # Use adjust_text later if labels overlap significantly
                         texts.append(plt.text(row['mu_star'] * 1.02, row['sigma'], row['parameter'], fontsize=8))

                    # Add adjust_text if needed and installed:
                    # try:
                    #     from adjustText import adjust_text
                    #     adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                    # except ImportError:
                    #     print("  adjustText library not found, skipping text adjustment.")
                    #     pass


                    plt.xlabel('μ* (Overall Importance)')
                    plt.ylabel('σ (Interaction / Non-linear Effects)')
                    plt.title('Morris Method: Importance vs Interaction (Granular)')
                    # Place legend outside plot
                    plt.legend(title='Parameter Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
                    plot_path = os.path.join(plots_dir, "granular_morris_mu_star_vs_sigma.png")
                    try:
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved: {plot_path}")
                    except Exception as e:
                        print(f"  Error saving plot {plot_path}: {e}")
                    plt.close()

                else:
                    print("  Skipping Morris plots (no data after dropping NaNs).")
            else:
                 print("  Skipping Morris plots (no data after dropping NaNs).")
        else:
            print("  Skipping Morris plots (no results found or DataFrame empty).")


        # --- Plot Sobol results ---
        if ('granular_sobol' in self.results and
            self.results['granular_sobol'] is not None and
            'first_order' in self.results['granular_sobol'] and
            'total' in self.results['granular_sobol'] and
            not self.results['granular_sobol']['first_order'].empty and
            not self.results['granular_sobol']['total'].empty):

            df_s1 = self.results['granular_sobol']['first_order'].copy()
            df_st = self.results['granular_sobol']['total'].copy()
            df_s2 = self.results['granular_sobol'].get('second_order') # Might be None or empty

            # Drop NaNs for reliable plotting
            df_s1.dropna(subset=['S1', 'S1_conf'], inplace=True)
            df_st.dropna(subset=['ST', 'ST_conf'], inplace=True)

            # Plot top N parameters by First-Order Index (S1)
            if not df_s1.empty:
                df_s1_sorted = df_s1.sort_values('S1', ascending=False).head(top_n)
                if not df_s1_sorted.empty:
                    plt.figure(figsize=(12, 8))
                    colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(df_s1_sorted)))
                    plt.barh(df_s1_sorted['parameter'], df_s1_sorted['S1'], xerr=df_s1_sorted['S1_conf'],
                             color=colors, capsize=3, alpha=0.8, error_kw=dict(alpha=0.5))
                    plt.xlabel('First-Order Index (S1)')
                    plt.ylabel('Parameter Coefficient')
                    plt.title(f'Sobol Method: Top {top_n} Parameters by Main Effect (S1)')
                    plt.gca().invert_yaxis()
                    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plot_path = os.path.join(plots_dir, "granular_sobol_topN_S1.png")
                    try:
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved: {plot_path}")
                    except Exception as e:
                        print(f"  Error saving plot {plot_path}: {e}")
                    plt.close()
                else:
                     print("  Skipping Sobol S1 plot (no data after sorting/filtering).")
            else:
                 print("  Skipping Sobol S1 plot (no data after dropping NaNs).")


            # Plot top N parameters by Total-Order Index (ST)
            if not df_st.empty:
                df_st_sorted = df_st.sort_values('ST', ascending=False).head(top_n)
                if not df_st_sorted.empty:
                    plt.figure(figsize=(12, 8))
                    colors = plt.cm.magma(np.linspace(0.1, 0.9, len(df_st_sorted)))
                    plt.barh(df_st_sorted['parameter'], df_st_sorted['ST'], xerr=df_st_sorted['ST_conf'],
                             color=colors, capsize=3, alpha=0.8, error_kw=dict(alpha=0.5))
                    plt.xlabel('Total-Order Index (ST)')
                    plt.ylabel('Parameter Coefficient')
                    plt.title(f'Sobol Method: Top {top_n} Parameters by Total Effect (ST)')
                    plt.gca().invert_yaxis()
                    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plot_path = os.path.join(plots_dir, "granular_sobol_topN_ST.png")
                    try:
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved: {plot_path}")
                    except Exception as e:
                        print(f"  Error saving plot {plot_path}: {e}")
                    plt.close()
                else:
                     print("  Skipping Sobol ST plot (no data after sorting/filtering).")
            else:
                 print("  Skipping Sobol ST plot (no data after dropping NaNs).")


            # Plot top N Second-Order Interactions (S2) if available
            if df_s2 is not None and not df_s2.empty:
                df_s2_c = df_s2.copy()
                df_s2_c.dropna(subset=['S2', 'S2_conf'], inplace=True)

                if not df_s2_c.empty:
                    df_s2_sorted = df_s2_c.sort_values('S2', ascending=False).head(top_n)
                    # Filter out S2 values very close to zero for plotting clarity
                    df_s2_sorted = df_s2_sorted[df_s2_sorted['S2'] > 1e-4] # Keep only potentially meaningful interactions

                    if not df_s2_sorted.empty:
                        plt.figure(figsize=(12, 8))
                        # Use a diverging colormap for interactions
                        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(df_s2_sorted)))
                        plt.barh(df_s2_sorted['parameter_combination'], df_s2_sorted['S2'], xerr=df_s2_sorted['S2_conf'],
                                 color=colors, capsize=3, alpha=0.8, error_kw=dict(alpha=0.5))
                        plt.xlabel('Second-Order Index (S2)')
                        plt.ylabel('Parameter Interaction')
                        plt.title(f'Sobol Method: Top {top_n} Parameter Interactions (S2 > 1e-4)')
                        plt.gca().invert_yaxis()
                        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
                        plt.tight_layout()
                        plot_path = os.path.join(plots_dir, "granular_sobol_topN_S2.png")
                        try:
                            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                            print(f"  Saved: {plot_path}")
                        except Exception as e:
                            print(f"  Error saving plot {plot_path}: {e}")
                        plt.close()
                    else:
                        print("  Skipping Sobol S2 plot (no significant interactions found after filtering).")
                else:
                    print("  Skipping Sobol S2 plot (no data after dropping NaNs).")
            else:
                print("  Skipping Sobol S2 plot (no results found or not calculated).")
        else:
            print("  Skipping Sobol plots (no results found or DataFrames empty).")

        print("Finished generating plots.")


    def summarize_results(self, top_n=10):
        """
        Summarize the key findings from the different sensitivity analyses performed.

        Args:
            top_n (int): Number of top parameters to show in summaries.

        Returns:
            dict: A dictionary containing summary DataFrames or key findings.
        """
        print("\n--- Granular Sensitivity Analysis Summary ---")
        summary = {}

        # --- Elasticity Summary ---
        print("\n--- Elasticity ---")
        if 'granular_elasticity' in self.results and not self.results['granular_elasticity'].empty:
            df_el = self.results['granular_elasticity'].copy()
            df_el['elasticity'] = df_el['elasticity'].replace([np.inf, -np.inf], np.nan)
            df_el.dropna(subset=['elasticity'], inplace=True)
            if not df_el.empty:
                df_el['abs_elasticity'] = df_el['elasticity'].abs()
                summary_el = df_el.sort_values('abs_elasticity', ascending=False).head(top_n)
                summary['top_elasticity'] = summary_el[['parameter_name', 'parameter_type', 'elasticity', 'original_value']]
                print(f"Top {min(top_n, len(summary_el))} Most Elastic Parameters (excluding NaN/inf):")
                print(summary['top_elasticity'].to_string(index=False, float_format="%.4f"))
            else:
                print("No finite elasticity results available.")
        else:
            print("Elasticity analysis results not available or empty.")

        # --- Morris Summary ---
        print("\n--- Morris Method ---")
        if ('granular_morris' in self.results and
            self.results['granular_morris'] is not None and
            'df' in self.results['granular_morris'] and
            not self.results['granular_morris']['df'].empty):

            df_mo = self.results['granular_morris']['df'].copy()
            df_mo.dropna(subset=['mu_star', 'sigma'], inplace=True) # Ensure values exist

            if not df_mo.empty:
                summary_mo_mustar = df_mo.sort_values('mu_star', ascending=False).head(top_n)
                summary_mo_sigma = df_mo.sort_values('sigma', ascending=False).head(top_n)
                summary['top_morris_mu_star'] = summary_mo_mustar[['parameter', 'parameter_type', 'mu_star', 'sigma']]
                summary['top_morris_sigma'] = summary_mo_sigma[['parameter', 'parameter_type', 'mu_star', 'sigma']]
                print(f"Top {min(top_n, len(summary_mo_mustar))} Parameters by Morris Importance (mu*):")
                print(summary['top_morris_mu_star'].to_string(index=False, float_format="%.4f"))
                print(f"\nTop {min(top_n, len(summary_mo_sigma))} Parameters by Morris Interaction/Non-linearity (sigma):")
                print(summary['top_morris_sigma'].to_string(index=False, float_format="%.4f"))
            else:
                print("No Morris results available after dropping NaN.")
        else:
            print("Morris analysis results not available or empty.")

        # --- Sobol Summary ---
        print("\n--- Sobol Method ---")
        if ('granular_sobol' in self.results and
            self.results['granular_sobol'] is not None and
            'first_order' in self.results['granular_sobol'] and
            'total' in self.results['granular_sobol'] and
            not self.results['granular_sobol']['first_order'].empty and
            not self.results['granular_sobol']['total'].empty):

            df_s1 = self.results['granular_sobol']['first_order'].copy()
            df_st = self.results['granular_sobol']['total'].copy()
            df_s2 = self.results['granular_sobol'].get('second_order')

            df_s1.dropna(subset=['S1'], inplace=True)
            df_st.dropna(subset=['ST'], inplace=True)

            if not df_s1.empty:
                summary_s1 = df_s1.sort_values('S1', ascending=False).head(top_n)
                summary['top_sobol_s1'] = summary_s1[['parameter', 'parameter_type', 'S1']]
                print(f"Top {min(top_n, len(summary_s1))} Parameters by Sobol Main Effect (S1):")
                print(summary['top_sobol_s1'].to_string(index=False, float_format="%.4f"))
            else:
                print("No Sobol S1 results available after dropping NaN.")

            if not df_st.empty:
                summary_st = df_st.sort_values('ST', ascending=False).head(top_n)
                summary['top_sobol_st'] = summary_st[['parameter', 'parameter_type', 'ST']]
                print(f"\nTop {min(top_n, len(summary_st))} Parameters by Sobol Total Effect (ST):")
                print(summary['top_sobol_st'].to_string(index=False, float_format="%.4f"))
            else:
                 print("No Sobol ST results available after dropping NaN.")

            if df_s2 is not None and not df_s2.empty:
                df_s2_c = df_s2.copy()
                df_s2_c.dropna(subset=['S2'], inplace=True)
                if not df_s2_c.empty:
                    summary_s2 = df_s2_c.sort_values('S2', ascending=False).head(top_n)
                    # Filter for display
                    summary_s2_filtered = summary_s2[summary_s2['S2'] > 1e-4] # Show potentially meaningful interactions
                    if not summary_s2_filtered.empty:
                        summary['top_sobol_s2'] = summary_s2_filtered[['parameter_combination', 'S2']]
                        print(f"\nTop {min(top_n, len(summary_s2_filtered))} Parameter Interactions by Sobol Second Order Effect (S2 > 1e-4):")
                        print(summary['top_sobol_s2'].to_string(index=False, float_format="%.4f"))
                    else:
                        print("\nNo significant Sobol second-order interactions found (S2 > 1e-4).")
                else:
                    print("\nNo Sobol second-order interactions available after dropping NaN.")
            else:
                print("\nSobol second-order interaction results not available or not calculated.")
        else:
            print("Sobol analysis results not available or empty.")

        print("\n--- End of Summary ---")
        return summary

    def save_results(self, filename="granular_sensitivity_all_results.pkl"):
        """Saves the results and execution times to a pickle file."""
        output_path = os.path.join(self.output_dir, filename)
        # Ensure DataFrames are included in the saved results
        data_to_save = {
            'results': self.results, # This should contain the DataFrames
            'execution_times': self.execution_times
        }
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"All granular sensitivity results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to pickle file: {e}")

    def load_results(self, filename="granular_sensitivity_all_results.pkl"):
        """Loads results and execution times from a pickle file."""
        load_path = os.path.join(self.output_dir, filename)
        try:
            with open(load_path, 'rb') as f:
                loaded_data = pickle.load(f)
            self.results = loaded_data.get('results', {})
            self.execution_times = loaded_data.get('execution_times', {})
            print(f"Granular sensitivity results loaded from {load_path}")
            # Verify loaded data structure if necessary
            print("Loaded result keys:", self.results.keys())
        except FileNotFoundError:
            print(f"Error: Could not find results file {load_path}")
        except Exception as e:
            print(f"Error loading results from pickle file: {e}")


# ==============================================================================
# Example Usage Block
# ==============================================================================
if __name__ == "__main__":

    from MILP_models.capfac import CapacitatedFacilityLocation
    # --- 2. Define Baseline Data ---
    # Using the specific baseline instance provided by the user
    baseline_instance = {
        'demands': np.array([20, 25, 30, 18, 22], dtype=np.int32), # 5 customers
        'capacities': np.array([80, 90, 70, 100, 85], dtype=np.float64), # 5 facilities
        'fixed_costs': np.array([150, 180, 160, 170, 155], dtype=np.float64), # Fixed cost for each facility
        'transportation_costs': np.array([[10, 12, 15, 20, 18],
                                          [14, 11, 13, 16, 19],
                                          [13, 17, 12, 14, 15],
                                          [12, 15, 10, 18, 16],
                                          [11, 13, 14, 15, 17]],
                                         dtype=np.float64)} # Shape (5, 5)

    N_CUST = len(baseline_instance['demands'])
    N_FAC = len(baseline_instance['capacities'])

    print("\n--- Using Baseline Data ---")
    print(f"Customers: {N_CUST}, Facilities: {N_FAC}")
    # print(baseline_instance) # Optionally print the data
    print("---------------------------\n")

    # --- 3. Initialize CFLP Model and Sensitivity Analysis ---
    # Create parameters dict needed by CapacitatedFacilityLocation __init__
    # We need 'continuous_assignment', others are less critical if not generating data
    cflp_params = {
        'n_customers': N_CUST, # Pass actual dimensions
        'n_facilities': N_FAC,
        'continuous_assignment': True # As per the class default/usage
        # Other params like intervals are not needed here
    }
    cflp_model_instance = CapacitatedFacilityLocation(parameters=cflp_params, seed=42)

    # IMPORTANT: Ensure the instance passed to GranularSensitivityAnalysis has n_customers/n_facilities set
    # The __init__ of CapacitatedFacilityLocation already sets these based on cflp_params
    # Verify:
    assert hasattr(cflp_model_instance, 'n_customers') and cflp_model_instance.n_customers == N_CUST
    assert hasattr(cflp_model_instance, 'n_facilities') and cflp_model_instance.n_facilities == N_FAC

    # Create the sensitivity analysis object using the actual CFLP model instance
    granular_sa = GranularSensitivityAnalysis(cflp_model_instance, output_dir="actual_cflp_granular_sensitivity")

    # --- 4. Run Specific Analyses ---
    # Choose which analyses to run by uncommenting

    print("\n--- Running OAT ---")
    oat_results = granular_sa.one_at_a_time_analysis_granular(baseline_instance, factor_range=(0.8, 1.2), n_samples=10)
    if oat_results is not None:
        print(f"OAT Results Shape: {oat_results.shape}")

    print("\n--- Running Elasticity ---")
    elasticity_results = granular_sa.elasticity_analysis_granular(baseline_instance, delta_pct=0.01) # 1% change
    if elasticity_results is not None:
        print(f"Elasticity Results Shape: {elasticity_results.shape}")

    print("\n--- Running Morris ---")
    # Adjust parameters for reasonable runtime with actual solver
    morris_results = granular_sa.morris_method_granular(baseline_instance, factor_range=(0.8, 1.2), num_trajectories=20, num_levels=4)
    if morris_results is not None and not morris_results.empty:
        print(f"Morris Results Shape: {morris_results.shape}")
    else:
        print("Morris analysis did not produce results.")


    print("\n--- Running Sobol ---")
    # Sobol is very expensive, use small N for testing
    sobol_results = granular_sa.sobol_analysis_granular(baseline_instance, factor_range=(0.8, 1.2), n_samples=2650, calc_second_order=True,parallel=True,n_processors=10)
    if sobol_results is not None:
         if not sobol_results['first_order'].empty:
             print(f"Sobol S1 Results Shape: {sobol_results['first_order'].shape}")
         if not sobol_results['total'].empty:
             print(f"Sobol ST Results Shape: {sobol_results['total'].shape}")
    else:
         print("Sobol analysis did not produce results.")


    # --- 5. Generate Plots and Summary ---
    print("\n--- Generating Plots ---")
    granular_sa.plot_granular_results(top_n=15)

    print("\n--- Generating Summary ---")
    summary = granular_sa.summarize_results(top_n=10)

    # --- 6. Save/Load ---
    print("\n--- Saving Results ---")
    granular_sa.save_results(filename="cflp_granular_sensitivity_results.pkl")

    # # Example of loading back
    # print("\n--- Loading Results ---")
    # granular_sa_loaded = GranularSensitivityAnalysis(cflp_model_instance, output_dir="actual_cflp_granular_sensitivity")
    # granular_sa_loaded.load_results(filename="cflp_granular_sensitivity_results.pkl")
    # # You could then call plot or summarize on granular_sa_loaded

    print("\nExample Run Finished.")
