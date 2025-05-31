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
from MILP_model_generation.capfac import CapacitatedFacilityLocation

class TraditionalSensitivityAnalysis:
    """
    A class implementing traditional sensitivity analysis methods for
    the Capacitated Facility Location Problem.
    """
    
    def __init__(self, cflp_model, output_dir="sensitivity_results"):
        """
        Initialize the sensitivity analysis framework.
        
        Args:
            cflp_model: An instance of CapacitatedFacilityLocation class
            output_dir: Directory to save results
        """
        self.cflp_model = cflp_model
        self.output_dir = output_dir
        self.results = {}
        self.execution_times = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _solve_instance_and_get_metrics(self, data):
        """
        Solve a single instance of the problem and return key metrics.
        
        Args:
            data: Data dictionary for the CFLP instance
            
        Returns:
            dict: Metrics including objective value, solve time, and solution details
        """
        start_time = time.time()
        model = self.cflp_model.generate_instance(data)
        status, solve_time = self.cflp_model.solve(model)
        total_time = time.time() - start_time
        
        # Extract solution details
        objective_value = model.objective.value()
        open_facilities = []
        for j in range(self.cflp_model.n_facilities):
            var_name = f"Open_{j}"
            var = model.variablesDict()[var_name]
            if var.value() > 0.5:  # Binary variable threshold
                open_facilities.append(j)
        
        # Calculate utilization of open facilities
        utilization = {}
        total_capacity = 0
        for j in open_facilities:
            capacity = data['capacities'][j]
            total_capacity += capacity
            used_capacity = 0
            for i in range(self.cflp_model.n_customers):
                var_name = f"Serve_{i}_{j}"
                var = model.variablesDict()[var_name]
                if var.value() > 0:
                    used_capacity += data['demands'][i] * var.value()
            utilization[j] = used_capacity / capacity
        
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        total_demand = sum(data['demands'])
        capacity_demand_ratio = total_capacity / total_demand if total_demand > 0 else float('inf')
        
        return {
            'objective_value': objective_value,
            'solve_time': solve_time,
            'total_time': total_time,
            'status': status,
            'open_facilities_count': len(open_facilities),
            'avg_utilization': avg_utilization,
            'capacity_demand_ratio': capacity_demand_ratio
        }
    
    def _generate_modified_data(self, base_data, param_name, param_values):
        """
        Generate modified data by changing a specific parameter.
        
        Args:
            base_data: Base data dictionary
            param_name: Name of the parameter to modify
            param_values: List of values for the parameter
            
        Returns:
            list: List of modified data dictionaries
        """
        modified_data_list = []
        
        for value in param_values:
            modified_data = {k: v.copy() if isinstance(v, np.ndarray) else v 
                             for k, v in base_data.items()}
            
            if param_name == 'demands':
                # Scale all demands by the given factor
                modified_data['demands'] = np.round(base_data['demands'] * value).astype(np.int32)
            elif param_name == 'capacities':
                # Scale all capacities by the given factor
                modified_data['capacities'] = np.round(base_data['capacities'] * value).astype(np.int32)
            elif param_name == 'fixed_costs':
                # Scale all fixed costs by the given factor
                modified_data['fixed_costs'] = np.round(base_data['fixed_costs'] * value).astype(np.int32)
            elif param_name == 'transportation_costs':
                # Scale all transportation costs by the given factor
                modified_data['transportation_costs'] = np.round(base_data['transportation_costs'] * value).astype(np.int32)
            
            modified_data_list.append(modified_data)
        
        return modified_data_list
    
    def one_at_a_time_analysis(self, base_data, param_ranges, n_samples=10):
        """
        Perform one-at-a-time sensitivity analysis.
        
        Args:
            base_data: Base data dictionary
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            n_samples: Number of samples per parameter
            
        Returns:
            dict: Results of the analysis
        """
        print("Running One-at-a-Time Sensitivity Analysis...")
        start_time = time.time()
        
        results = {
            'parameter': [],
            'value': [],
            'relative_change': [],
            'objective_value': [],
            'solve_time': [],
            'total_time': [],
            'open_facilities_count': [],
            'avg_utilization': [],
            'capacity_demand_ratio': []
        }
        
        # First, solve the base case
        base_metrics = self._solve_instance_and_get_metrics(base_data)
        base_objective = base_metrics['objective_value']
        
        # For each parameter, vary it and solve the problem
        for param_name, (min_val, max_val) in param_ranges.items():
            print(f"  Analyzing parameter: {param_name}")
            # Generate evenly spaced values in the range
            param_values = np.linspace(min_val, max_val, n_samples)
            
            modified_data_list = self._generate_modified_data(base_data, param_name, param_values)
            
            for i, modified_data in enumerate(tqdm(modified_data_list, desc=f"  {param_name} samples")):
                metrics = self._solve_instance_and_get_metrics(modified_data)
                
                # Calculate relative change from base case
                rel_change = (metrics['objective_value'] - base_objective) / base_objective if base_objective != 0 else float('inf')
                
                # Store results
                results['parameter'].append(param_name)
                results['value'].append(param_values[i])
                results['relative_change'].append(rel_change)
                results['objective_value'].append(metrics['objective_value'])
                results['solve_time'].append(metrics['solve_time'])
                results['total_time'].append(metrics['total_time'])
                results['open_facilities_count'].append(metrics['open_facilities_count'])
                results['avg_utilization'].append(metrics['avg_utilization'])
                results['capacity_demand_ratio'].append(metrics['capacity_demand_ratio'])
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save to file
        df_results.to_csv(f"{self.output_dir}/oat_analysis.csv", index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store in class
        self.results['oat'] = df_results
        self.execution_times['oat'] = execution_time
        
        print(f"One-at-a-Time analysis completed in {execution_time:.2f} seconds")
        return df_results
    
    def morris_method(self, base_data, param_ranges, num_trajectories=10, num_levels=4):
        """
        Perform Morris method for global sensitivity analysis.
        
        Args:
            base_data: Base data dictionary
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            num_trajectories: Number of trajectories
            num_levels: Number of levels
            
        Returns:
            dict: Results of the analysis
        """
        print("Running Morris Method Sensitivity Analysis...")
        start_time = time.time()
        
        # Set up problem definition for SALib
        problem = {
            'num_vars': len(param_ranges),
            'names': list(param_ranges.keys()),
            'bounds': [param_ranges[p] for p in param_ranges.keys()]
        }
        
        # Generate samples
        param_values = morris_sample.sample(problem, N=num_trajectories, num_levels=num_levels)
        
        # Results storage
        y = np.zeros(param_values.shape[0])
        
        print(f"  Evaluating {len(param_values)} samples...")
        for i, X in enumerate(tqdm(param_values)):
            # Create modified data
            modified_data = {k: v.copy() if isinstance(v, np.ndarray) else v 
                          for k, v in base_data.items()}
            
            # Apply parameter values
            for j, param_name in enumerate(problem['names']):
                factor = X[j]  # Scaling factor from Morris sampling
                
                if param_name == 'demands':
                    modified_data['demands'] = np.round(base_data['demands'] * factor).astype(np.int32)
                elif param_name == 'capacities':
                    modified_data['capacities'] = np.round(base_data['capacities'] * factor).astype(np.int32)
                elif param_name == 'fixed_costs':
                    modified_data['fixed_costs'] = np.round(base_data['fixed_costs'] * factor).astype(np.int32)
                elif param_name == 'transportation_costs':
                    modified_data['transportation_costs'] = np.round(base_data['transportation_costs'] * factor).astype(np.int32)
            
            # Evaluate model
            metrics = self._solve_instance_and_get_metrics(modified_data)
            y[i] = metrics['objective_value']
        
        # Analyze samples
        morris_results = morris_analyze.analyze(problem, param_values, y, print_to_console=True)
        
        # Create DataFrame from results
        df_results = pd.DataFrame({
            'parameter': problem['names'],
            'mu': morris_results['mu'],
            'mu_star': morris_results['mu_star'],
            'sigma': morris_results['sigma'],
            'mu_star_conf': morris_results['mu_star_conf']
        })
        
        # Save to file
        df_results.to_csv(f"{self.output_dir}/morris_analysis.csv", index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store in class
        self.results['morris'] = {
            'df': df_results,
            'raw_results': morris_results
        }
        self.execution_times['morris'] = execution_time
        
        print(f"Morris method completed in {execution_time:.2f} seconds")
        return df_results
    
    def sobol_analysis(self, base_data, param_ranges, n_samples=500, calc_second_order=True):
        """
        Perform Sobol sensitivity analysis.
        
        Args:
            base_data: Base data dictionary
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            n_samples: Number of samples
            calc_second_order: Whether to calculate second-order indices
            
        Returns:
            dict: Results of the analysis
        """
        print("Running Sobol Sensitivity Analysis...")
        start_time = time.time()
        
        # Set up problem definition for SALib
        problem = {
            'num_vars': len(param_ranges),
            'names': list(param_ranges.keys()),
            'bounds': [param_ranges[p] for p in param_ranges.keys()]
        }
        
        # Generate samples
        param_values = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
        
        # Results storage
        y = np.zeros(param_values.shape[0])
        
        print(f"  Evaluating {len(param_values)} samples...")
        for i, X in enumerate(tqdm(param_values)):
            # Create modified data
            modified_data = {k: v.copy() if isinstance(v, np.ndarray) else v 
                          for k, v in base_data.items()}
            
            # Apply parameter values
            for j, param_name in enumerate(problem['names']):
                factor = X[j]  # Scaling factor from Sobol sampling
                
                if param_name == 'demands':
                    modified_data['demands'] = np.round(base_data['demands'] * factor).astype(np.int32)
                elif param_name == 'capacities':
                    modified_data['capacities'] = np.round(base_data['capacities'] * factor).astype(np.int32)
                elif param_name == 'fixed_costs':
                    modified_data['fixed_costs'] = np.round(base_data['fixed_costs'] * factor).astype(np.int32)
                elif param_name == 'transportation_costs':
                    modified_data['transportation_costs'] = np.round(base_data['transportation_costs'] * factor).astype(np.int32)
            
            # Evaluate model
            try:
                metrics = self._solve_instance_and_get_metrics(modified_data)
                y[i] = metrics['objective_value']
            except Exception as e:
                print(f"Error in sample {i}: {e}")
                y[i] = np.nan
        
        # Handle any NaN values by replacing with the mean
        if np.isnan(y).any():
            mean_y = np.nanmean(y)
            y = np.nan_to_num(y, nan=mean_y)
        
        # Analyze samples
        sobol_results = sobol.analyze(problem, y, calc_second_order=calc_second_order, print_to_console=True)
        
        # Create DataFrame for first-order indices
        df_first_order = pd.DataFrame({
            'parameter': problem['names'],
            'S1': sobol_results['S1'],
            'S1_conf': sobol_results['S1_conf']
        })
        
        # Create DataFrame for total indices
        df_total = pd.DataFrame({
            'parameter': problem['names'],
            'ST': sobol_results['ST'],
            'ST_conf': sobol_results['ST_conf']
        })
        
        # Create DataFrame for second-order indices if calculated
        if calc_second_order:
            S2 = sobol_results['S2']
            S2_conf = sobol_results['S2_conf']
            
            # Convert to DataFrame with parameter combinations
            S2_values = []
            for i in range(problem['num_vars']):
                for j in range(i+1, problem['num_vars']):
                    param_i = problem['names'][i]
                    param_j = problem['names'][j]
                    interaction = f"{param_i} × {param_j}"
                    S2_values.append({
                        'parameter_combination': interaction,
                        'S2': S2[i][j],
                        'S2_conf': S2_conf[i][j]
                    })
            
            df_second_order = pd.DataFrame(S2_values)
        else:
            df_second_order = None
        
        # Save to files
        df_first_order.to_csv(f"{self.output_dir}/sobol_first_order.csv", index=False)
        df_total.to_csv(f"{self.output_dir}/sobol_total.csv", index=False)
        if df_second_order is not None:
            df_second_order.to_csv(f"{self.output_dir}/sobol_second_order.csv", index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store in class
        self.results['sobol'] = {
            'first_order': df_first_order,
            'total': df_total,
            'second_order': df_second_order,
            'raw_results': sobol_results
        }
        self.execution_times['sobol'] = execution_time
        
        print(f"Sobol analysis completed in {execution_time:.2f} seconds")
        return {
            'first_order': df_first_order,
            'total': df_total,
            'second_order': df_second_order
        }
    
    def _generate_triangular_distribution(self, min_val, mode_val, max_val, size=1):
        """
        Generate samples from a triangular distribution.
        
        Args:
            min_val: Minimum value
            mode_val: Mode value (most frequent)
            max_val: Maximum value
            size: Number of samples
            
        Returns:
            numpy.ndarray: Samples from triangular distribution
        """
        # Convert to c parameter for scipy triangular distribution
        # c = (mode - min) / (max - min)
        c = (mode_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        return stats.triang.rvs(c, loc=min_val, scale=max_val - min_val, size=size)
    
    def scenario_based_analysis(self, base_data, scenario_params, n_scenarios=100):
        """
        Perform scenario-based analysis using triangular distributions.
        
        Args:
            base_data: Base data dictionary
            scenario_params: Dictionary mapping parameters to (min, mode, max) values
            n_scenarios: Number of scenarios to generate
            
        Returns:
            dict: Results of the analysis
        """
        print("Running Scenario-Based Analysis...")
        start_time = time.time()
        
        # Results storage
        results = {
            'scenario_id': [],
            'objective_value': [],
            'solve_time': [],
            'total_time': [],
            'open_facilities_count': [],
            'avg_utilization': [],
            'capacity_demand_ratio': []
        }
        
        # Add parameter columns
        for param in scenario_params.keys():
            results[f"{param}_factor"] = []
        
        # Generate scenarios
        for scenario_id in range(n_scenarios):
            # Create modified data
            modified_data = {k: v.copy() if isinstance(v, np.ndarray) else v 
                          for k, v in base_data.items()}
            
            # Apply parameter values
            for param_name, (min_val, mode_val, max_val) in scenario_params.items():
                # Generate a single value from triangular distribution
                factor = self._generate_triangular_distribution(min_val, mode_val, max_val)[0]
                
                if param_name == 'demands':
                    modified_data['demands'] = np.round(base_data['demands'] * factor).astype(np.int32)
                elif param_name == 'capacities':
                    modified_data['capacities'] = np.round(base_data['capacities'] * factor).astype(np.int32)
                elif param_name == 'fixed_costs':
                    modified_data['fixed_costs'] = np.round(base_data['fixed_costs'] * factor).astype(np.int32)
                elif param_name == 'transportation_costs':
                    modified_data['transportation_costs'] = np.round(base_data['transportation_costs'] * factor).astype(np.int32)
                
                # Store the factor
                results[f"{param_name}_factor"].append(factor)
            
            # Evaluate model
            try:
                metrics = self._solve_instance_and_get_metrics(modified_data)
                
                # Store results
                results['scenario_id'].append(scenario_id)
                results['objective_value'].append(metrics['objective_value'])
                results['solve_time'].append(metrics['solve_time'])
                results['total_time'].append(metrics['total_time'])
                results['open_facilities_count'].append(metrics['open_facilities_count'])
                results['avg_utilization'].append(metrics['avg_utilization'])
                results['capacity_demand_ratio'].append(metrics['capacity_demand_ratio'])
                
            except Exception as e:
                print(f"Error in scenario {scenario_id}: {e}")
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save to file
        df_results.to_csv(f"{self.output_dir}/scenario_analysis.csv", index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store in class
        self.results['scenario'] = df_results
        self.execution_times['scenario'] = execution_time
        
        print(f"Scenario-based analysis completed in {execution_time:.2f} seconds")
        return df_results
    
    def run_comprehensive_analysis(self, base_data, param_ranges=None, scenario_params=None, 
                             n_oat_samples=10, n_scenarios=100, n_morris=10, n_sobol=500):
        """
        Run a comprehensive sensitivity analysis using multiple methods.
        
        Args:
            base_data: Base data dictionary
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            scenario_params: Dictionary mapping parameters to (min, mode, max) values
            n_oat_samples: Number of samples for OAT analysis
            n_scenarios: Number of scenarios for scenario analysis
            n_morris: Number of trajectories for Morris method
            n_sobol: Number of samples for Sobol analysis
            
        Returns:
            dict: Summary of all analyses
        """
        print("Running Comprehensive Sensitivity Analysis...")
        
        if param_ranges is None:
            # Default parameter ranges (±30% of base values)
            param_ranges = {
                'demands': (0.7, 1.3),
                'capacities': (0.7, 1.3),
                'fixed_costs': (0.7, 1.3),
                'transportation_costs': (0.7, 1.3)
            }
        
        if scenario_params is None:
            # Default scenario parameters (triangular distributions)
            scenario_params = {
                'demands': (0.7, 1.0, 1.3),
                'capacities': (0.7, 1.0, 1.3),
                'fixed_costs': (0.7, 1.0, 1.3),
                'transportation_costs': (0.7, 1.0, 1.3)
            }
        
        # Run One-at-a-Time analysis
        self.one_at_a_time_analysis(base_data, param_ranges, n_samples=n_oat_samples)
        
        # Run Morris method
        self.morris_method(base_data, param_ranges, num_trajectories=n_morris)
        
        # Run Sobol analysis
        self.sobol_analysis(base_data, param_ranges, n_samples=n_sobol)
        
        # Run Scenario-based analysis
        self.scenario_based_analysis(base_data, scenario_params, n_scenarios=n_scenarios)
        
        # Run Elasticity analysis
        self.elasticity_analysis(base_data, {k: 1.0 for k in param_ranges.keys()})
        
        # Generate report and plots
        self.generate_summary_report()
        self.plot_results()
        
        print("Comprehensive analysis completed.")
        return self.results
    
    def elasticity_analysis(self, base_data, param_ranges, delta_pct=0.01):
        """
        Perform elasticity analysis to determine sensitivity of objective to small parameter changes.
        
        Args:
            base_data: Base data dictionary
            param_ranges: Dictionary mapping parameter names to base values
            delta_pct: Percentage change for elasticity calculation
            
        Returns:
            pd.DataFrame: Elasticity results
        """
        print("Running Elasticity Analysis...")
        start_time = time.time()
        
        # Solve base case
        base_metrics = self._solve_instance_and_get_metrics(base_data)
        base_objective = base_metrics['objective_value']
        
        results = {
            'parameter': [],
            'elasticity': [],
            'base_value': [],
            'perturbed_value': [],
            'base_objective': [],
            'perturbed_objective': []
        }
        
        # For each parameter, calculate elasticity
        for param_name, base_value in param_ranges.items():
            print(f"  Analyzing elasticity for: {param_name}")
            
            # Create modified data with small increase
            modified_data = {k: v.copy() if isinstance(v, np.ndarray) else v 
                          for k, v in base_data.items()}
            
            # Apply parameter values with delta_pct increase
            perturbed_value = base_value * (1 + delta_pct)
            
            if param_name == 'demands':
                modified_data['demands'] = np.round(base_data['demands'] * perturbed_value / base_value).astype(np.int32)
            elif param_name == 'capacities':
                modified_data['capacities'] = np.round(base_data['capacities'] * perturbed_value / base_value).astype(np.int32)
            elif param_name == 'fixed_costs':
                modified_data['fixed_costs'] = np.round(base_data['fixed_costs'] * perturbed_value / base_value).astype(np.int32)
            elif param_name == 'transportation_costs':
                modified_data['transportation_costs'] = np.round(base_data['transportation_costs'] * perturbed_value / base_value).astype(np.int32)
            
            # Evaluate model
            perturbed_metrics = self._solve_instance_and_get_metrics(modified_data)
            perturbed_objective = perturbed_metrics['objective_value']
            
            # Calculate elasticity: % change in output / % change in input
            if base_objective != 0 and delta_pct != 0:
                elasticity = ((perturbed_objective - base_objective) / base_objective) / delta_pct
            else:
                elasticity = float('inf')
            
            # Store results
            results['parameter'].append(param_name)
            results['elasticity'].append(elasticity)
            results['base_value'].append(base_value)
            results['perturbed_value'].append(perturbed_value)
            results['base_objective'].append(base_objective)
            results['perturbed_objective'].append(perturbed_objective)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save to file
        df_results.to_csv(f"{self.output_dir}/elasticity_analysis.csv", index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store in class
        self.results['elasticity'] = df_results
        self.execution_times['elasticity'] = execution_time
        
        print(f"Elasticity analysis completed in {execution_time:.2f} seconds")
        return df_results
    
    def generate_summary_report(self):
        """
        Generate a summary report of all sensitivity analyses.
        
        Returns:
            dict: Summary report
        """
        summary = {
            'execution_times': self.execution_times,
            'parameter_rankings': {}
        }
        
        # Rank parameters by importance from different methods
        if 'morris' in self.results:
            morris_df = self.results['morris']['df']
            summary['parameter_rankings']['morris'] = morris_df.sort_values('mu_star', ascending=False)['parameter'].tolist()
        
        if 'sobol' in self.results:
            sobol_df = self.results['sobol']['total']
            summary['parameter_rankings']['sobol'] = sobol_df.sort_values('ST', ascending=False)['parameter'].tolist()
        
        if 'elasticity' in self.results:
            elasticity_df = self.results['elasticity']
            summary['parameter_rankings']['elasticity'] = elasticity_df.sort_values('elasticity', ascending=False)['parameter'].tolist()
        
        # Save summary
        with open(f"{self.output_dir}/summary_report.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        # Create a text file summary
        with open(f"{self.output_dir}/summary_report.txt", 'w') as f:
            f.write("Sensitivity Analysis Summary Report\n")
            f.write("=================================\n\n")
            
            f.write("Execution Times:\n")
            for method, time_taken in self.execution_times.items():
                f.write(f"  {method}: {time_taken:.2f} seconds\n")
            
            f.write("\nParameter Rankings by Method:\n")
            for method, ranking in summary.get('parameter_rankings', {}).items():
                f.write(f"  {method}: {', '.join(ranking)}\n")
        
        return summary
    
    def plot_results(self):
        """
        Generate plots for visualization of sensitivity analysis results.
        """
        # Create plots directory
        plots_dir = f"{self.output_dir}/plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot One-at-a-Time results if available
        if 'oat' in self.results:
            df = self.results['oat']
            plt.figure(figsize=(12, 8))
            
            for param in df['parameter'].unique():
                param_df = df[df['parameter'] == param]
                plt.plot(param_df['value'], param_df['relative_change'], marker='o', label=param)
            
            plt.xlabel('Parameter Value (Scaled)')
            plt.ylabel('Relative Change in Objective')
            plt.title('One-at-a-Time Sensitivity Analysis')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{plots_dir}/oat_analysis.png", dpi=300)
            plt.close()
        
        # Plot Morris Method results if available
        if 'morris' in self.results:
            df = self.results['morris']['df']
            plt.figure(figsize=(10, 8))
            
            # Sort by mu_star
            df_sorted = df.sort_values('mu_star')
            
            plt.barh(df_sorted['parameter'], df_sorted['mu_star'], xerr=df_sorted['mu_star_conf'])
            plt.xlabel('μ* (Parameter Importance)')
            plt.title('Morris Method: Parameter Importance')
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/morris_importance.png", dpi=300)
            plt.close()
            
            # Plot mu vs sigma
            plt.figure(figsize=(10, 8))
            plt.scatter(df['mu'], df['sigma'], s=50)
            
            # Add parameter labels
            for i, param in enumerate(df['parameter']):
                plt.annotate(param, (df['mu'].iloc[i], df['sigma'].iloc[i]), 
                             xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('μ (Mean Effect)')
            plt.ylabel('σ (Interaction Effects)')
            plt.title('Morris Method: Mean vs. Standard Deviation of Effects')
            plt.grid(True)
            plt.savefig(f"{plots_dir}/morris_mu_sigma.png", dpi=300)
            plt.close()
        
        # Plot Sobol results if available
        if 'sobol' in self.results:
            # First order indices
            df_first = self.results['sobol']['first_order']
            plt.figure(figsize=(10, 8))
            
            # Sort by S1
            df_sorted = df_first.sort_values('S1')
            
            plt.barh(df_sorted['parameter'], df_sorted['S1'], xerr=df_sorted['S1_conf'])
            plt.xlabel('S1 (First-Order Sensitivity Index)')
            plt.title('Sobol Method: First-Order Effects')
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/sobol_first_order.png", dpi=300)
            plt.close()
            
            # Total indices
            df_total = self.results['sobol']['total']
            plt.figure(figsize=(10, 8))
            
            # Sort by ST
            df_sorted = df_total.sort_values('ST')
            
            plt.barh(df_sorted['parameter'], df_sorted['ST'], xerr=df_sorted['ST_conf'])
            plt.xlabel('ST (Total Sensitivity Index)')
            plt.title('Sobol Method: Total Effects')
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/sobol_total.png", dpi=300)
            plt.close()
            
            # Second order indices if available
            if self.results['sobol']['second_order'] is not None:
                df_second = self.results['sobol']['second_order']
                plt.figure(figsize=(12, 10))
                
                # Sort by S2
                df_sorted = df_second.sort_values('S2')
                
                plt.barh(df_sorted['parameter_combination'], df_sorted['S2'], xerr=df_sorted['S2_conf'])
                plt.xlabel('S2 (Second-Order Sensitivity Index)')
                plt.title('Sobol Method: Interaction Effects')
                plt.tight_layout()
                plt.savefig(f"{plots_dir}/sobol_second_order.png", dpi=300)
                plt.close()
        
        # Plot Scenario Analysis results if available
        if 'scenario' in self.results:
            df = self.results['scenario']
            
            # Histogram of objective values
            plt.figure(figsize=(10, 6))
            plt.hist(df['objective_value'], bins=20, alpha=0.7)
            plt.xlabel('Objective Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Objective Values Across Scenarios')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{plots_dir}/scenario_objective_distribution.png", dpi=300)
            plt.close()
            
            # Correlation between parameters and objective
            plt.figure(figsize=(12, 10))
            
            # Get correlation columns
            corr_cols = [col for col in df.columns if col.endswith('_factor') or col == 'objective_value']
            corr_df = df[corr_cols].corr()
            
            # Create heatmap
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Between Parameters and Objective Value')
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/scenario_correlation.png", dpi=300)
            plt.close()
            
            # Scatter plots for each parameter vs objective
            for col in [c for c in df.columns if c.endswith('_factor')]:
                plt.figure(figsize=(10, 6))
                plt.scatter(df[col], df['objective_value'], alpha=0.7)
                plt.xlabel(col)
                plt.ylabel('Objective Value')
                plt.title(f'Objective Value vs {col}')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{plots_dir}/scenario_{col}_vs_objective.png", dpi=300)
                plt.close()
        
        # Plot Elasticity Analysis results if available
        if 'elasticity' in self.results:
            df = self.results['elasticity']
            plt.figure(figsize=(10, 8))
            
            # Sort by absolute elasticity
            df_sorted = df.sort_values('elasticity', key=abs, ascending=False)
            
            plt.barh(df_sorted['parameter'], df_sorted['elasticity'])
            plt.xlabel('Elasticity (% Change in Objective / % Change in Parameter)')
            plt.title('Parameter Elasticities')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/elasticity_analysis.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    BASE_PARAMS = {
        "n_customers": 10,
        "n_facilities": 10,
        "demand_interval": (5, 36),
        "capacity_interval": (10, 161),
        "transportation_cost_interval": (10, 20),
        "fixed_cost_interval": (100, 200),
        "continuous_assignment": True,
    }
    # Lightweight sample sizes so the test finishes in < 2 min on a laptop
    OAT_SAMPLES_PER_PARAM = 20
    MORRIS_TRAJ = 4
    SOBOL_BASE_SAMPLES = 60  # Saltelli will expand this internally

    print("▶️ Generating base problem instance…")
    cfl = CapacitatedFacilityLocation(BASE_PARAMS, seed=123)
    base_data = cfl.generate_data()

    tsa = TraditionalSensitivityAnalysis(cfl)

    # Define ±25 % parameter ranges for quick tests
    param_ranges = {
        "demands": (0.75, 1.25),
        "capacities": (0.75, 1.25),
        "fixed_costs": (0.75, 1.25),
        "transportation_costs": (0.75, 1.25),
    }

    # ------------------------------------------------------------------
    # 3a. Run One‑at‑a‑time (local) analysis
    # ------------------------------------------------------------------
    tsa.one_at_a_time_analysis(base_data, param_ranges, n_samples=OAT_SAMPLES_PER_PARAM)

    # ------------------------------------------------------------------
    # 3b. Run Morris elementary‑effects method
    # ------------------------------------------------------------------
    tsa.morris_method(base_data, param_ranges, num_trajectories=MORRIS_TRAJ, num_levels=4)

    # ------------------------------------------------------------------
    # 3c. Run Sobol first‑order + total‑order indices (no 2nd order to save time)
    # ------------------------------------------------------------------
    tsa.sobol_analysis(base_data, param_ranges, n_samples=SOBOL_BASE_SAMPLES, calc_second_order=False)

    # ------------------------------------------------------------------
    # 4. Generate summary report + plots
    # ------------------------------------------------------------------
    tsa.generate_summary_report()
    tsa.plot_results()

    print("✅ All analyses completed.")

    # Optional: pretty‑print execution times
    print("Execution times (s):")
    print(tsa.execution_times)