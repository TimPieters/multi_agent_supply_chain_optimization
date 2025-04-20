import pulp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy import stats
import random
import copy
from tqdm import tqdm
import seaborn as sns


class SensitivityAnalysis:
    def __init__(self, base_model, parameters=None):
        """
        Initialize sensitivity analysis with a base model and parameters to analyze
        
        Args:
            base_model: A dictionary containing the original model parameters
            parameters: List of parameter names to analyze; if None, all parameters are analyzed
        """
        self.base_model = base_model
        self.base_solution = self.solve_model(self.base_model)
        
        # If no specific parameters provided, analyze all
        if parameters is None:
            self.parameters = ['demands', 'capacities', 'fixed_costs', 'transportation_costs']
        else:
            self.parameters = parameters
    
    def solve_model(self, model_params):
        """
        Solve the MILP model with given parameters
        
        Args:
            model_params: Dictionary containing model parameters
            
        Returns:
            Dictionary with solution details
        """
        start_time = time.time()
        
        # Extract parameters
        demands = model_params['demands']
        capacities = model_params['capacities']
        fixed_costs = model_params['fixed_costs']
        transportation_costs = model_params['transportation_costs']
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        # Create the model
        model = pulp.LpProblem("CapacitatedFacilityLocation", pulp.LpMinimize)
        
        # Decision variables
        facility_open = {j: pulp.LpVariable(f"Open_{j}", cat=pulp.LpBinary) 
                         for j in range(n_facilities)}
        assignment = {(i, j): pulp.LpVariable(f"Serve_{i}_{j}", lowBound=0, cat=pulp.LpContinuous)
                      for i in range(n_customers) for j in range(n_facilities)}
        
        # Objective function
        model += (
            pulp.lpSum(fixed_costs[j] * facility_open[j] for j in range(n_facilities)) +
            pulp.lpSum(transportation_costs[i][j] * assignment[i, j] 
                      for i in range(n_customers) for j in range(n_facilities)),
            "TotalCost"
        )
        
        # Constraints
        # Each customer must be served at least once
        for i in range(n_customers):
            model += pulp.lpSum(assignment[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}"
        
        # Capacity constraints
        for j in range(n_facilities):
            model += (pulp.lpSum(assignment[i, j] * demands[i] for i in range(n_customers)) 
                     <= capacities[j] * facility_open[j]), f"Capacity_{j}"
        
        # Total open capacity must meet total demand
        model += (pulp.lpSum(capacities[j] * facility_open[j] for j in range(n_facilities)) 
                 >= sum(demands)), "TotalDemand"
        
        # Linking constraints
        for i in range(n_customers):
            for j in range(n_facilities):
                model += assignment[i, j] <= facility_open[j], f"Link_{i}_{j}"
        
        # Solve the model
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        solve_time = time.time() - start_time
        
        # Extract solution
        solution = {
            'status': pulp.LpStatus[model.status],
            'objective': pulp.value(model.objective),
            'facilities_open': [j for j in range(n_facilities) if pulp.value(facility_open[j]) > 0.5],
            'assignments': {(i, j): pulp.value(assignment[i, j]) 
                           for i in range(n_customers) for j in range(n_facilities) 
                           if pulp.value(assignment[i, j]) > 0.001},
            'solve_time': solve_time
        }
        
        return solution
    
    def modify_parameter(self, model_params, param_name, new_value):
        """
        Create a new model with a modified parameter
        
        Args:
            model_params: Current model parameters
            param_name: Name of parameter to modify
            new_value: New value for the parameter
            
        Returns:
            Modified model parameters
        """
        # Create a deep copy to avoid modifying original
        modified_params = copy.deepcopy(model_params)
        
        # Update the specified parameter
        modified_params[param_name] = new_value
        
        return modified_params
    
    def one_at_a_time_sensitivity(self, variations):
        """
        One-at-a-time sensitivity analysis
        
        Args:
            variations: Dictionary mapping parameter names to lists of variation values
            
        Returns:
            Dictionary with sensitivity results
        """
        results = {}
        
        for param in self.parameters:
            param_results = []
            
            for value in variations[param]:
                # Create modified model with new parameter value
                modified_params = self.modify_parameter(self.base_model, param, value)
                solution = self.solve_model(modified_params)
                
                # Calculate percent change in objective value
                percent_change = ((solution['objective'] - self.base_solution['objective']) / 
                                 self.base_solution['objective'] * 100)
                
                param_results.append({
                    'param_value': value,
                    'objective': solution['objective'],
                    'percent_change': percent_change,
                    'solve_time': solution['solve_time']
                })
            
            results[param] = param_results
        
        return results
    
    def triangular_scenario_analysis(self, min_values, median_values, max_values, n_samples=100):
        """
        Scenario analysis using triangular distributions
        
        Args:
            min_values: Dictionary mapping parameters to minimum values
            median_values: Dictionary mapping parameters to median values
            max_values: Dictionary mapping parameters to maximum values
            n_samples: Number of scenarios to generate
            
        Returns:
            List of scenario results
        """
        scenarios = []
        results = []
        
        # Generate n_samples scenarios
        for i in range(n_samples):
            scenario = {}
            for param in self.parameters:
                # Check if the parameter is a 2D list first.
                if isinstance(self.base_model[param], list) and self.base_model[param]:
                    if isinstance(self.base_model[param][0], list):
                        # For 2D list parameters (transportation_costs)
                        scenario[param] = [
                            [
                                random.triangular(min_values[param][i][j],
                                                max_values[param][i][j],
                                                median_values[param][i][j])
                                for j in range(len(self.base_model[param][i]))
                            ]
                            for i in range(len(self.base_model[param]))
                        ]
                    else:
                        # For 1D list parameters (demands, capacities, fixed_costs)
                        scenario[param] = [
                            random.triangular(min_values[param][j],
                                            max_values[param][j],
                                            median_values[param][j])
                            for j in range(len(self.base_model[param]))
                        ]
                else:
                    # For scalar parameters
                    scenario[param] = random.triangular(
                        min_values[param], max_values[param], median_values[param]
                    )
            scenarios.append(scenario)
        
        # Evaluate each scenario
        for scenario in tqdm(scenarios, desc="Analyzing scenarios"):
            modified_params = copy.deepcopy(self.base_model)
            for param, value in scenario.items():
                modified_params[param] = value
            
            solution = self.solve_model(modified_params)
            results.append({
                'scenario': scenario,
                'objective': solution['objective'],
                'facilities_open': solution['facilities_open'],
                'solve_time': solution['solve_time']
            })
        
        return results
    
    def elasticity_analysis(self, variation_percentage=10):
        """
        Calculate elasticity for each parameter
        
        Args:
            variation_percentage: Percentage by which to vary parameters
            
        Returns:
            Dictionary of elasticities sorted by impact
        """
        elasticities = {}
        base_objective = self.base_solution['objective']
        
        for param in self.parameters:
            # Check parameter type and handle accordingly
            if param == 'demands':
                # Vary each demand by the percentage
                base_value = self.base_model[param]
                
                # Test parameter increase
                increase_value = [d * (1 + variation_percentage/100) for d in base_value]
                increase_params = self.modify_parameter(self.base_model, param, increase_value)
                increase_solution = self.solve_model(increase_params)
                increase_objective = increase_solution['objective']
                
                # Test parameter decrease
                decrease_value = [d * (1 - variation_percentage/100) for d in base_value]
                decrease_params = self.modify_parameter(self.base_model, param, decrease_value)
                decrease_solution = self.solve_model(decrease_params)
                decrease_objective = decrease_solution['objective']
                
            elif param == 'capacities':
                # Vary each capacity by the percentage
                base_value = self.base_model[param]
                
                # Test parameter increase
                increase_value = [c * (1 + variation_percentage/100) for c in base_value]
                increase_params = self.modify_parameter(self.base_model, param, increase_value)
                increase_solution = self.solve_model(increase_params)
                increase_objective = increase_solution['objective']
                
                # Test parameter decrease
                decrease_value = [c * (1 - variation_percentage/100) for c in base_value]
                decrease_params = self.modify_parameter(self.base_model, param, decrease_value)
                decrease_solution = self.solve_model(decrease_params)
                decrease_objective = decrease_solution['objective']
                
            elif param == 'fixed_costs':
                # Vary each fixed cost by the percentage
                base_value = self.base_model[param]
                
                # Test parameter increase
                increase_value = [fc * (1 + variation_percentage/100) for fc in base_value]
                increase_params = self.modify_parameter(self.base_model, param, increase_value)
                increase_solution = self.solve_model(increase_params)
                increase_objective = increase_solution['objective']
                
                # Test parameter decrease
                decrease_value = [fc * (1 - variation_percentage/100) for fc in base_value]
                decrease_params = self.modify_parameter(self.base_model, param, decrease_value)
                decrease_solution = self.solve_model(decrease_params)
                decrease_objective = decrease_solution['objective']
                
            elif param == 'transportation_costs':
                # Vary each transportation cost by the percentage
                base_value = self.base_model[param]
                
                # Test parameter increase
                increase_value = [[tc * (1 + variation_percentage/100) for tc in row] for row in base_value]
                increase_params = self.modify_parameter(self.base_model, param, increase_value)
                increase_solution = self.solve_model(increase_params)
                increase_objective = increase_solution['objective']
                
                # Test parameter decrease
                decrease_value = [[tc * (1 - variation_percentage/100) for tc in row] for row in base_value]
                decrease_params = self.modify_parameter(self.base_model, param, decrease_value)
                decrease_solution = self.solve_model(decrease_params)
                decrease_objective = decrease_solution['objective']
            
            # Calculate elasticity
            elasticity = abs(increase_objective - decrease_objective) / (2 * variation_percentage/100 * base_objective)
            
            elasticities[param] = {
                'elasticity': elasticity,
                'increase_impact': (increase_objective - base_objective) / base_objective * 100,
                'decrease_impact': (decrease_objective - base_objective) / base_objective * 100,
                'increase_solve_time': increase_solution['solve_time'],
                'decrease_solve_time': decrease_solution['solve_time']
            }
        
        # Sort parameters by elasticity
        sorted_elasticities = {k: v for k, v in sorted(elasticities.items(), 
                                                      key=lambda item: item[1]['elasticity'], reverse=True)}
        
        return sorted_elasticities
    
    def monte_carlo_sensitivity(self, param_ranges, n_samples=1000, correlations=None):
        """
        Monte Carlo sensitivity analysis with optional parameter correlations
        
        Args:
            param_ranges: Dictionary mapping parameters to (min, max) tuples
            n_samples: Number of samples to generate
            correlations: Optional correlation matrix between parameters
            
        Returns:
            Dictionary with Monte Carlo results
        """
        results = []
        scenario_params = []
        
        # Generate parameter samples
        if correlations is not None:
            # Use multivariate normal with correlations
            param_list = list(self.parameters)  # Ordered list of parameters
            mean = [0] * len(param_list)
            
            # Generate correlated standard normal samples
            mvn_samples = np.random.multivariate_normal(
                mean=mean,
                cov=correlations,
                size=n_samples
            )
            
            # Transform to uniform using probability integral transform
            uniform_samples = stats.norm.cdf(mvn_samples)
            
            # Transform to target ranges
            for i in range(n_samples):
                scenario = {}
                for j, param in enumerate(param_list):
                    u = uniform_samples[i, j]  # Uniform sample for this parameter
                    
                    if param in ['demands', 'capacities', 'fixed_costs']:
                        # List parameters
                        min_vals, max_vals = param_ranges[param]
                        scenario[param] = [
                            min_val + u * (max_val - min_val)
                            for min_val, max_val in zip(min_vals, max_vals)
                        ]
                    elif param == 'transportation_costs':
                        # 2D list parameter - use the same u for all elements for simplicity
                        # In a real application, you might want more granular correlation control
                        min_vals, max_vals = param_ranges[param]
                        scenario[param] = [
                            [min_vals[i][j] + u * (max_vals[i][j] - min_vals[i][j])
                             for j in range(len(min_vals[i]))]
                            for i in range(len(min_vals))
                        ]
                scenario_params.append(scenario)
        else:
            # Generate independent samples
            for _ in range(n_samples):
                scenario = {}
                for param in self.parameters:
                    if param in ['demands', 'capacities', 'fixed_costs']:
                        # List parameters
                        min_vals, max_vals = param_ranges[param]
                        scenario[param] = [
                            random.uniform(min_val, max_val)
                            for min_val, max_val in zip(min_vals, max_vals)
                        ]
                    elif param == 'transportation_costs':
                        # 2D list parameter
                        min_vals, max_vals = param_ranges[param]
                        scenario[param] = [
                            [random.uniform(min_vals[i][j], max_vals[i][j])
                             for j in range(len(min_vals[i]))]
                            for i in range(len(min_vals))
                        ]
                scenario_params.append(scenario)
        
        # Evaluate scenarios
        for scenario in tqdm(scenario_params, desc="Running Monte Carlo analysis"):
            modified_params = copy.deepcopy(self.base_model)
            for param, value in scenario.items():
                modified_params[param] = value
            
            solution = self.solve_model(modified_params)
            results.append({
                'parameters': scenario,
                'objective': solution['objective'],
                'facilities_open': solution['facilities_open'],
                'solve_time': solution['solve_time']
            })
        
        return results
    
    def plot_oat_results(self, oat_results, parameter):
        """
        Plot one-at-a-time sensitivity results for a specific parameter
        
        Args:
            oat_results: Results from one_at_a_time_sensitivity
            parameter: Parameter name to plot
        """
        results = oat_results[parameter]
        values = [r['param_value'] for r in results]
        objectives = [r['objective'] for r in results]
        percent_changes = [r['percent_change'] for r in results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot objective values
        color = 'tab:blue'
        ax1.set_xlabel(f'{parameter} Values')
        ax1.set_ylabel('Objective Value', color=color)
        ax1.plot(values, objectives, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for percent change
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Percent Change (%)', color=color)
        ax2.plot(values, percent_changes, marker='s', linestyle='--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'Sensitivity Analysis for {parameter}')
        fig.tight_layout()
        
        return fig
    
    def plot_elasticity_results(self, elasticity_results):
        """
        Plot elasticity analysis results as a tornado diagram
        
        Args:
            elasticity_results: Results from elasticity_analysis
        """
        # Extract parameters and elasticities
        params = list(elasticity_results.keys())
        elasticities = [elasticity_results[p]['elasticity'] for p in params]
        increase_impacts = [elasticity_results[p]['increase_impact'] for p in params]
        decrease_impacts = [elasticity_results[p]['decrease_impact'] for p in params]
        
        # Sort by elasticity
        sorted_indices = np.argsort(elasticities)
        params = [params[i] for i in sorted_indices]
        increase_impacts = [increase_impacts[i] for i in sorted_indices]
        decrease_impacts = [decrease_impacts[i] for i in sorted_indices]
        
        # Create tornado diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(params))
        ax.barh(y_pos, increase_impacts, height=0.4, color='red', alpha=0.6, label='+10%')
        ax.barh(y_pos, decrease_impacts, height=0.4, color='blue', alpha=0.6, label='-10%')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel('Percent Change in Objective (%)')
        ax.set_title('Tornado Diagram: Parameter Sensitivity')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_monte_carlo_results(self, mc_results, key_params=None):
        """
        Plot Monte Carlo sensitivity analysis results
        
        Args:
            mc_results: Results from monte_carlo_sensitivity
            key_params: Optional list of key parameters to highlight
        """
        # Extract objectives
        objectives = [r['objective'] for r in mc_results]
        
        # Plot histogram of objectives
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(objectives, bins=30, color='skyblue', edgecolor='black')
        ax1.axvline(self.base_solution['objective'], color='red', linestyle='--', 
                   label=f'Base objective: {self.base_solution["objective"]:.2f}')
        ax1.set_xlabel('Objective Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Objective Values from Monte Carlo Analysis')
        ax1.legend()
        
        # If key parameters specified, create scatter plots
        if key_params and len(key_params) > 0:
            # Create scatter plots for key parameters vs objective
            fig2, axes = plt.subplots(len(key_params), 1, figsize=(10, 4*len(key_params)))
            
            # Handle case of single parameter
            if len(key_params) == 1:
                axes = [axes]
            
            for i, param in enumerate(key_params):
                # Extract parameter values (handle different parameter types)
                if param in ['demands', 'capacities', 'fixed_costs']:
                    # Use sum or average for list parameters
                    param_values = [sum(r['parameters'][param]) for r in mc_results]
                    param_label = f'Sum of {param}'
                else:
                    # For transportation_costs, use average
                    param_values = [
                        np.mean([np.mean(row) for row in r['parameters'][param]])
                        for r in mc_results
                    ]
                    param_label = f'Average {param}'
                
                # Create scatter plot
                axes[i].scatter(param_values, objectives, alpha=0.5)
                axes[i].set_xlabel(param_label)
                axes[i].set_ylabel('Objective Value')
                axes[i].set_title(f'{param} vs Objective')
                
                # Add trend line
                z = np.polyfit(param_values, objectives, 1)
                p = np.poly1d(z)
                axes[i].plot(param_values, p(param_values), "r--")
            
            plt.tight_layout()
            return fig1, fig2
        
        return fig1
    
    def critical_scenario_identification(self, scenario_results, percentile=95):
        """
        Identify critical scenarios based on high objective values
        
        Args:
            scenario_results: Results from scenario analysis or Monte Carlo
            percentile: Percentile threshold for critical scenarios
            
        Returns:
            Dictionary with critical scenario information
        """
        # Sort scenarios by objective value
        sorted_results = sorted(scenario_results, key=lambda x: x['objective'], reverse=True)
        
        # Calculate threshold for critical scenarios
        objectives = [r['objective'] for r in scenario_results]
        threshold = np.percentile(objectives, percentile)
        
        # Identify critical scenarios
        critical_scenarios = [r for r in sorted_results if r['objective'] >= threshold]
        
        # Analyze parameter patterns in critical scenarios
        parameter_patterns = {}
        for param in self.parameters:
            if param in ['demands', 'capacities', 'fixed_costs']:
                # For list parameters, calculate average values
                avg_values = [np.mean(r['parameters'][param]) for r in critical_scenarios]
                parameter_patterns[param] = {
                    'mean': np.mean(avg_values),
                    'std': np.std(avg_values),
                    'min': min(avg_values),
                    'max': max(avg_values)
                }
            elif param == 'transportation_costs':
                # For 2D list, calculate overall average
                avg_values = [
                    np.mean([np.mean(row) for row in r['parameters'][param]])
                    for r in critical_scenarios
                ]
                parameter_patterns[param] = {
                    'mean': np.mean(avg_values),
                    'std': np.std(avg_values),
                    'min': min(avg_values),
                    'max': max(avg_values)
                }
        
        return {
            'critical_scenarios': critical_scenarios,
            'parameter_patterns': parameter_patterns,
            'threshold': threshold,
            'count': len(critical_scenarios)
        }

    def run_comprehensive_analysis(self, variation_percentage=10, n_monte_carlo=500):
        """
        Run a comprehensive sensitivity analysis using multiple methods
        
        Args:
            variation_percentage: Percentage for OAT and elasticity analysis
            n_monte_carlo: Number of Monte Carlo samples
            
        Returns:
            Dictionary with results from all analysis methods
        """
        results = {}
        start_time = time.time()
        
        # 1. Elasticity Analysis
        print("Running elasticity analysis...")
        elasticity_results = self.elasticity_analysis(variation_percentage)
        results['elasticity'] = elasticity_results
        
        # 2. Define parameter ranges for Monte Carlo based on elasticity results
        # Focus more sampling on parameters with high elasticity
        param_ranges = {}
        for param in self.parameters:
            if param == 'demands':
                base_values = self.base_model[param]
                min_values = [0.7 * val for val in base_values]  # 30% decrease
                max_values = [1.3 * val for val in base_values]  # 30% increase
                param_ranges[param] = (min_values, max_values)
            elif param == 'capacities':
                base_values = self.base_model[param]
                min_values = [0.7 * val for val in base_values]
                max_values = [1.3 * val for val in base_values]
                param_ranges[param] = (min_values, max_values)
            elif param == 'fixed_costs':
                base_values = self.base_model[param]
                min_values = [0.7 * val for val in base_values]
                max_values = [1.3 * val for val in base_values]
                param_ranges[param] = (min_values, max_values)
            elif param == 'transportation_costs':
                base_values = self.base_model[param]
                min_values = [[0.7 * val for val in row] for row in base_values]
                max_values = [[1.3 * val for val in row] for row in base_values]
                param_ranges[param] = (min_values, max_values)
        
        # 3. Monte Carlo Analysis
        print(f"Running Monte Carlo analysis with {n_monte_carlo} samples...")
        mc_results = self.monte_carlo_sensitivity(param_ranges, n_samples=n_monte_carlo)
        results['monte_carlo'] = mc_results
        
        # 4. Critical Scenario Identification
        print("Identifying critical scenarios...")
        critical_scenarios = self.critical_scenario_identification(mc_results)
        results['critical_scenarios'] = critical_scenarios
        
        # 5. Generate triangular distribution parameters based on critical scenarios
        # For parameters with high elasticity
        sorted_params = list(elasticity_results.keys())
        key_params = sorted_params[:2]  # Focus on top 2 most sensitive parameters
        
        min_values = {}
        median_values = {}
        max_values = {}
        
        for param in self.parameters:
            if param in ['demands', 'capacities', 'fixed_costs']:
                base_values = self.base_model[param]
                
                if param in key_params:
                    # For key parameters, use wider triangular distributions
                    min_values[param] = [0.6 * val for val in base_values]  # 40% decrease
                    median_values[param] = base_values
                    max_values[param] = [1.4 * val for val in base_values]  # 40% increase
                else:
                    # For less important parameters, use narrower distributions
                    min_values[param] = [0.8 * val for val in base_values]  # 20% decrease
                    median_values[param] = base_values
                    max_values[param] = [1.2 * val for val in base_values]  # 20% increase
            elif param == 'transportation_costs':
                base_values = self.base_model[param]
                
                if param in key_params:
                    min_values[param] = [[0.6 * val for val in row] for row in base_values]
                    median_values[param] = base_values
                    max_values[param] = [[1.4 * val for val in row] for row in base_values]
                else:
                    min_values[param] = [[0.8 * val for val in row] for row in base_values]
                    median_values[param] = base_values
                    max_values[param] = [[1.2 * val for val in row] for row in base_values]
        
        # 6. Triangular Scenario Analysis
        print("Running triangular scenario analysis...")
        triangle_results = self.triangular_scenario_analysis(
            min_values, median_values, max_values, n_samples=100
        )
        results['triangular'] = triangle_results
        
        # Calculate total analysis time
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        print(f"Comprehensive sensitivity analysis completed in {total_time:.2f} seconds")
        return results
        
    def generate_performance_metrics(self, results):
        """
        Generate performance metrics for the sensitivity analysis
        
        Args:
            results: Results from run_comprehensive_analysis
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # 1. Quality metrics
        # 1.1 Range of objective values discovered
        mc_objectives = [r['objective'] for r in results['monte_carlo']]
        metrics['objective_range'] = max(mc_objectives) - min(mc_objectives)
        metrics['objective_min'] = min(mc_objectives)
        metrics['objective_max'] = max(mc_objectives)
        metrics['objective_mean'] = np.mean(mc_objectives)
        metrics['objective_std'] = np.std(mc_objectives)
        
        # 1.2 Parameter importance ranking
        metrics['parameter_importance'] = {
            param: results['elasticity'][param]['elasticity'] 
            for param in results['elasticity']
        }
        
        # 1.3 Critical scenario identification quality
        critical_params = results['critical_scenarios']['parameter_patterns']
        metrics['critical_scenarios_count'] = results['critical_scenarios']['count']
        metrics['critical_parameters'] = critical_params
        
        # 2. Efficiency metrics
        # 2.1 Total analysis time
        metrics['total_analysis_time'] = results['total_time']
        
        # 2.2 Average time per scenario
        mc_times = [r['solve_time'] for r in results['monte_carlo']]
        metrics['avg_solve_time'] = np.mean(mc_times)
        metrics['max_solve_time'] = max(mc_times)
        metrics['min_solve_time'] = min(mc_times)
        
        # 2.3 Number of scenarios analyzed
        metrics['monte_carlo_scenarios'] = len(results['monte_carlo'])
        metrics['triangular_scenarios'] = len(results['triangular'])
        metrics['total_scenarios'] = len(results['monte_carlo']) + len(results['triangular'])
        
        # 3. Decision quality metrics
        # 3.1 Facility selection stability
        # Calculate how often each facility is selected across scenarios
        facilities_freq = {}
        for r in results['monte_carlo']:
            for facility in r['facilities_open']:
                if facility not in facilities_freq:
                    facilities_freq[facility] = 0
                facilities_freq[facility] += 1
        
        metrics['facility_selection_frequency'] = {
            facility: count / len(results['monte_carlo']) * 100
            for facility, count in facilities_freq.items()
        }
        
        # 3.2 Robust facilities (selected in >80% of scenarios)
        metrics['robust_facilities'] = [
            facility for facility, freq in metrics['facility_selection_frequency'].items()
            if freq > 80
        ]
        
        # 3.3 Uncertain facilities (selected in 20-80% of scenarios)
        metrics['uncertain_facilities'] = [
            facility for facility, freq in metrics['facility_selection_frequency'].items()
            if 20 <= freq <= 80
        ]
        
        return metrics
        
    def plot_comprehensive_results(self, results, metrics):
        """
        Create visualizations for comprehensive sensitivity analysis results
        
        Args:
            results: Results from run_comprehensive_analysis
            metrics: Metrics from generate_performance_metrics
            
        Returns:
            Dictionary of plot figures
        """
        plots = {}
        
        # 1. Elasticity tornado diagram
        plots['elasticity_tornado'] = self.plot_elasticity_results(results['elasticity'])
        
        # 2. Monte Carlo objective distribution
        mc_objectives = [r['objective'] for r in results['monte_carlo']]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(mc_objectives, kde=True, ax=ax)
        ax.axvline(self.base_solution['objective'], color='red', linestyle='--', 
                  label=f'Base objective: {self.base_solution["objective"]:.2f}')
        ax.set_xlabel('Objective Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Objective Values from Monte Carlo Analysis')
        ax.legend()
        plots['objective_distribution'] = fig
        
        # 3. Facility selection frequency
        facility_freq = metrics['facility_selection_frequency']
        fig, ax = plt.subplots(figsize=(10, 6))
        facilities = list(facility_freq.keys())
        frequencies = list(facility_freq.values())
        
        # Sort by frequency
        sorted_indices = np.argsort(frequencies)[::-1]
        sorted_facilities = [facilities[i] for i in sorted_indices]
        sorted_frequencies = [frequencies[i] for i in sorted_indices]
        
        ax.bar(sorted_facilities, sorted_frequencies, color='skyblue')
        ax.axhline(80, color='green', linestyle='--', label='Robust threshold (80%)')
        ax.axhline(20, color='red', linestyle='--', label='Uncertain threshold (20%)')
        ax.set_xlabel('Facility')
        ax.set_ylabel('Selection Frequency (%)')
        ax.set_title('Facility Selection Frequency Across Scenarios')
        ax.legend()
        plots['facility_frequency'] = fig
        
        # 4. Parameter correlation heatmap with objective
        # Extract parameter values and objectives from Monte Carlo results
        param_data = {}
        for param in self.parameters:
            if param in ['demands', 'capacities', 'fixed_costs']:
                # Use sum for list parameters
                param_data[f'sum_{param}'] = [sum(r['parameters'][param]) for r in results['monte_carlo']]
            elif param == 'transportation_costs':
                # Use average for 2D list
                param_data[f'avg_{param}'] = [
                    np.mean([np.mean(row) for row in r['parameters'][param]])
                    for r in results['monte_carlo']
                ]
        
        param_data['objective'] = mc_objectives
        df = pd.DataFrame(param_data)
        
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Parameter-Objective Correlation Matrix')
        plots['correlation_heatmap'] = fig
        
        # 5. Critical vs Non-Critical Scenarios Comparison
        critical_scenarios = results['critical_scenarios']['critical_scenarios']
        critical_objectives = [r['objective'] for r in critical_scenarios]
        non_critical_indices = [
            i for i, r in enumerate(results['monte_carlo']) 
            if r['objective'] < results['critical_scenarios']['threshold']
        ]
        non_critical_objectives = [results['monte_carlo'][i]['objective'] for i in non_critical_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(critical_objectives, label='Critical Scenarios', ax=ax)
        sns.kdeplot(non_critical_objectives, label='Non-Critical Scenarios', ax=ax)
        ax.axvline(self.base_solution['objective'], color='red', linestyle='--', 
                  label=f'Base objective: {self.base_solution["objective"]:.2f}')
        ax.set_xlabel('Objective Value')
        ax.set_ylabel('Density')
        ax.set_title('Comparing Critical vs Non-Critical Scenarios')
        ax.legend()
        plots['critical_comparison'] = fig
        
        return plots
    
    def export_report(self, results, metrics, plots, filename="sensitivity_analysis_report.html"):
        """
        Export sensitivity analysis results as an HTML report
        
        Args:
            results: Results from run_comprehensive_analysis
            metrics: Metrics from generate_performance_metrics
            plots: Plots from plot_comprehensive_results
            filename: Output HTML filename
            
        Returns:
            Path to the generated report
        """
        import base64
        from io import BytesIO
        
        # Convert matplotlib figures to base64 for HTML embedding
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('ascii')
            return img_str
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sensitivity Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                .metrics-table { border-collapse: collapse; width: 100%; }
                .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .metrics-table tr:nth-child(even) { background-color: #f2f2f2; }
                .metrics-table th { padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }
                .figure { margin-bottom: 20px; text-align: center; }
                .figure img { max-width: 100%; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Supply Chain Optimization Sensitivity Analysis Report</h1>
                <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
            
            <div class="section">
                <h2>1. Executive Summary</h2>
                <p>This report presents the results of a comprehensive sensitivity analysis for a capacitated 
                facility location supply chain optimization model. The analysis aims to identify the most influential 
                parameters and critical scenarios that significantly impact the model's objective function.</p>
                
                <h3>Key Findings:</h3>
                <ul>
                    <li>Base solution objective value: """ + f"{self.base_solution['objective']:.2f}" + """</li>
                    <li>Objective range: """ + f"{metrics['objective_min']:.2f} to {metrics['objective_max']:.2f}" + """</li>
                    <li>Most sensitive parameter: """ + list(metrics['parameter_importance'].keys())[0] + """</li>
                    <li>Number of critical scenarios identified: """ + str(metrics['critical_scenarios_count']) + """</li>
                    <li>Robust facilities (selected in >80% of scenarios): """ + str(metrics['robust_facilities']) + """</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>2. Parameter Sensitivity Analysis</h2>
                
                <div class="figure">
                    <h3>Parameter Elasticity (Tornado Diagram)</h3>
                    <img src="data:image/png;base64,""" + fig_to_base64(plots['elasticity_tornado']) + """" alt="Elasticity Tornado Diagram">
                    <p>This diagram shows the elasticity of each parameter, indicating how sensitive the objective function is to changes in that parameter.</p>
                </div>
                
                <h3>Parameter Importance Ranking:</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Elasticity</th>
                    </tr>
        """
        
        # Add parameter importance rows
        for param, elasticity in sorted(metrics['parameter_importance'].items(), key=lambda x: x[1], reverse=True):
            html_content += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{elasticity:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>3. Monte Carlo Analysis Results</h2>
                
                <div class="figure">
                    <h3>Distribution of Objective Values</h3>
                    <img src="data:image/png;base64,""" + fig_to_base64(plots['objective_distribution']) + """" alt="Objective Distribution">
                    <p>This histogram shows the distribution of objective values across all Monte Carlo scenarios.</p>
                </div>
                
                <div class="figure">
                    <h3>Parameter-Objective Correlation Matrix</h3>
                    <img src="data:image/png;base64,""" + fig_to_base64(plots['correlation_heatmap']) + """" alt="Correlation Heatmap">
                    <p>This heatmap shows the correlation between parameter values and the objective function.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>4. Facility Analysis</h2>
                
                <div class="figure">
                    <h3>Facility Selection Frequency</h3>
                    <img src="data:image/png;base64,""" + fig_to_base64(plots['facility_frequency']) + """" alt="Facility Frequency">
                    <p>This chart shows how frequently each facility is selected across all scenarios.</p>
                </div>
                
                <h3>Facility Decision Robustness:</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Category</th>
                        <th>Facilities</th>
                    </tr>
                    <tr>
                        <td>Robust Facilities (>80%)</td>
                        <td>""" + ", ".join(map(str, metrics['robust_facilities'])) + """</td>
                    </tr>
                    <tr>
                        <td>Uncertain Facilities (20-80%)</td>
                        <td>""" + ", ".join(map(str, metrics['uncertain_facilities'])) + """</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>5. Critical Scenario Analysis</h2>
                
                <div class="figure">
                    <h3>Critical vs Non-Critical Scenarios</h3>
                    <img src="data:image/png;base64,""" + fig_to_base64(plots['critical_comparison']) + """" alt="Critical Scenarios">
                    <p>This chart compares the distribution of objective values for critical versus non-critical scenarios.</p>
                </div>
                
                <h3>Critical Scenario Parameter Patterns:</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
        """
        
        # Add critical parameter rows
        for param, stats in metrics['critical_parameters'].items():
            html_content += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{stats['mean']:.2f}</td>
                        <td>{stats['std']:.2f}</td>
                        <td>{stats['min']:.2f}</td>
                        <td>{stats['max']:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>6. Performance Metrics</h2>
                
                <h3>Analysis Efficiency:</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Analysis Time</td>
                        <td>""" + f"{metrics['total_analysis_time']:.2f} seconds" + """</td>
                    </tr>
                    <tr>
                        <td>Average Solve Time per Scenario</td>
                        <td>""" + f"{metrics['avg_solve_time']:.4f} seconds" + """</td>
                    </tr>
                    <tr>
                        <td>Total Scenarios Analyzed</td>
                        <td>""" + str(metrics['total_scenarios']) + """</td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        # Write HTML content to file
        with open(filename, 'w') as f:
            f.write(html_content)
        
        return filename
    

# For demonstration purposes, let's run a smaller analysis
if __name__ == "__main__":
    # Define the base model parameters
    base_model = {
        'demands': [20, 25, 30, 18, 22],  # For 5 customers
        'capacities': [80, 90, 70, 100, 85],  # For 5 facilities
        'fixed_costs': [150, 180, 160, 170, 155],  # Fixed cost for each facility
        'transportation_costs': [
            [10, 12, 15, 20, 18],
            [14, 11, 13, 16, 19],
            [13, 17, 12, 14, 15],
            [12, 15, 10, 18, 16],
            [11, 13, 14, 15, 17]
        ]
    }
    
    # Initialize sensitivity analysis
    print("Initializing sensitivity analysis...")
    sensitivity_analyzer = SensitivityAnalysis(base_model)
    
    # Get the base solution
    base_solution = sensitivity_analyzer.base_solution
    print(f"Base solution objective value: {base_solution['objective']:.2f}")
    print(f"Facilities open in base solution: {base_solution['facilities_open']}")
    
    # Run a comprehensive analysis
    print("\nRunning comprehensive sensitivity analysis...")
    results = sensitivity_analyzer.run_comprehensive_analysis(
        variation_percentage=15,  # 15% variation for elasticity analysis
        n_monte_carlo=1000  # Use 200 Monte Carlo samples for demonstration (increase for real analysis)
    )
    
    # Generate performance metrics
    print("\nGenerating performance metrics...")
    metrics = sensitivity_analyzer.generate_performance_metrics(results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plots = sensitivity_analyzer.plot_comprehensive_results(results, metrics)
    
    # Export report
    print("\nExporting HTML report...")
    report_path = sensitivity_analyzer.export_report(results, metrics, plots)
    print(f"Report saved to: {report_path}")
    
    # Display some key findings
    print("\n--- KEY FINDINGS ---")
    print("Parameter importance ranking:")
    for param, elasticity in sorted(metrics['parameter_importance'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  {param}: {elasticity:.4f}")
    
    print("\nRobust facilities (selected in >80% of scenarios):")
    print(f"  {metrics['robust_facilities']}")
    
    print("\nUncertain facilities (selected in 20-80% of scenarios):")
    print(f"  {metrics['uncertain_facilities']}")
    
    print("\nObjective value range:")
    print(f"  Min: {metrics['objective_min']:.2f}")
    print(f"  Max: {metrics['objective_max']:.2f}")
    print(f"  Mean: {metrics['objective_mean']:.2f}")
    print(f"  Std Dev: {metrics['objective_std']:.2f}")
    
    # Show one of the plots
    print("\nDisplaying elasticity tornado diagram...")
    plots['elasticity_tornado'].show()