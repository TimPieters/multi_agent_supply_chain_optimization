import math
import json

def relative_difference_list(baseline_list, perturbed_list):
    """
    Computes the average relative difference between two lists.
    For each element, it computes: abs(b - p) / (abs(b) if b != 0 else 1)
    """
    differences = []
    for b, p in zip(baseline_list, perturbed_list):
        differences.append(abs(b - p) / (abs(b) if b != 0 else 1))
    return sum(differences) / len(differences) if differences else 0

def relative_difference_matrix(baseline_matrix, perturbed_matrix):
    """
    Computes the average relative difference between two matrices.
    For each element, it computes: abs(b - p) / (abs(b) if b != 0 else 1)
    """
    differences = []
    for i in range(len(baseline_matrix)):
        for j in range(len(baseline_matrix[0])):
            b = baseline_matrix[i][j]
            p = perturbed_matrix[i][j]
            differences.append(abs(b - p) / (abs(b) if b != 0 else 1))
    return sum(differences) / len(differences) if differences else 0

def compute_total_perturbation(baseline_params, scenario_params, weights=None):
    """
    Computes a weighted average of the relative differences across all parameters.
    
    Args:
        baseline_params (dict): Dictionary with keys "demands", "capacities", "fixed_costs", "transportation_costs".
        scenario_params (dict): Same structure as baseline_params.
        weights (dict, optional): Weights for each parameter. Defaults to 1 for all.
    
    Returns:
        float: Total weighted perturbation magnitude.
    """
    if weights is None:
        weights = {
            "demands": 1,
            "capacities": 1,
            "fixed_costs": 1,
            "transportation_costs": 1
        }
    
    rel_diff_demands = relative_difference_list(baseline_params["demands"], scenario_params["demands"])
    rel_diff_capacities = relative_difference_list(baseline_params["capacities"], scenario_params["capacities"])
    rel_diff_fixed_costs = relative_difference_list(baseline_params["fixed_costs"], scenario_params["fixed_costs"])
    rel_diff_transport = relative_difference_matrix(baseline_params["transportation_costs"], scenario_params["transportation_costs"])
    
    total = (
        weights["demands"] * rel_diff_demands +
        weights["capacities"] * rel_diff_capacities +
        weights["fixed_costs"] * rel_diff_fixed_costs +
        weights["transportation_costs"] * rel_diff_transport
    )
    total /= sum(weights.values())
    return total

def grade_scenario(baseline_params, scenario_params, baseline_obj, scenario_obj, comp_time, benchmark_rank=None, top_k=None):
    """
    Grades a single scenario based on several metrics:
      - Impact: Change in objective value from the baseline.
      - Total Perturbation: Weighted average relative difference in parameters.
      - Normalized Impact Efficiency: |Impact| / Total Perturbation.
      - Computation Efficiency: 1 / Computation Time.
      - Top-k Bonus: A bonus if the scenario ranks in the top-k benchmarks (optional).
      - Overall Score: A weighted aggregation of the above metrics.
      
    Args:
        baseline_params (dict): Baseline parameters.
        scenario_params (dict): Perturbed scenario parameters.
        baseline_obj (float): Objective value for the baseline.
        scenario_obj (float): Objective value for the scenario.
        comp_time (float): Computation time in seconds.
        benchmark_rank (int, optional): The rank of this scenario among all scenarios, lower is better.
        top_k (int, optional): The top-k threshold for awarding a bonus.
        
    Returns:
        dict: A dictionary containing all computed metrics and the overall score.
    """
    # 1. Impact: Difference in objective value
    impact = scenario_obj - baseline_obj
    
    # 2. Total perturbation magnitude (relative change in parameters)
    total_perturbation = compute_total_perturbation(baseline_params, scenario_params)
    
    # 3. Normalized Impact Efficiency: Impact per unit of perturbation
    normalized_efficiency = abs(impact) / total_perturbation if total_perturbation > 0 else 0
    
    # 4. Computation Efficiency: Faster scenarios get a higher score.
    comp_efficiency = 1 / comp_time if comp_time > 0 else 0
    
    # 5. Top-k Bonus: If benchmark rank is provided and within top_k, add bonus.
    top_k_bonus = 0
    if benchmark_rank is not None and top_k is not None:
        if benchmark_rank <= top_k:
            top_k_bonus = 1  # A bonus value (this can be tuned)
    
    # 6. Aggregated Score: Combine metrics with chosen weights.
    # You can adjust these weights as needed.
    weights = {
        "normalized_efficiency": 0.5,
        "comp_efficiency": 0.3,
        "top_k_bonus": 0.2
    }
    
    overall_score = (
        weights["normalized_efficiency"] * normalized_efficiency +
        weights["comp_efficiency"] * comp_efficiency +
        weights["top_k_bonus"] * top_k_bonus
    )
    
    return {
        "Impact": impact,
        "Total Perturbation": total_perturbation,
        "Normalized Efficiency": normalized_efficiency,
        "Computation Time": comp_time,
        "Computation Efficiency": comp_efficiency,
        "Top-k Bonus": top_k_bonus,
        "Overall Score": overall_score
    }

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == '__main__':
    # Example baseline parameters (hard-coded)
    baseline_params = {
        "demands": [20, 25, 30, 18, 22],
        "capacities": [80, 90, 70, 100, 85],
        "fixed_costs": [150, 180, 160, 170, 155],
        "transportation_costs": [
            [10, 12, 15, 20, 18],
            [14, 11, 13, 16, 19],
            [13, 17, 12, 14, 15],
            [12, 15, 10, 18, 16],
            [11, 13, 14, 15, 17]
        ]
    }
    
    # Example agent-generated (perturbed) parameters
    # (For instance, Facility 2 is shut off and demands are increased by 10%)
    scenario_params = {
        "demands": [22, 27.5, 33, 19.8, 24.2],  # 10% increase
        "capacities": [80, 90, 0, 100, 85],  # Facility 2 shut off (set to 0)
        "fixed_costs": [150, 180, 160, 170, 155],  # unchanged
        "transportation_costs": [
            [10, 12, 15, 20, 18],
            [14, 11, 13, 16, 19],
            [13, 17, 12, 14, 15],
            [12, 15, 10, 18, 16],
            [11, 13, 14, 15, 17]
        ]
    }
    
    # Example baseline and scenario objective values from solving the models
    baseline_obj = 366.1  # example baseline total cost
    scenario_obj = 366.3  # example scenario total cost
    
    # Example computation time (in seconds) for solving the scenario
    comp_time = 0.08895870001288131  # seconds
    
    # Optionally, if you know the benchmark rank and top-k threshold:
    benchmark_rank = 3   # For example, this scenario ranked 3rd among many
    top_k = 5            # Top 5 scenarios get a bonus
    
    # Grade the scenario
    grading_results = grade_scenario(
        baseline_params,
        scenario_params,
        baseline_obj,
        scenario_obj,
        comp_time,
        benchmark_rank,
        top_k
    )
    
    # Print the detailed grading results
    print("Grading Results:")
    print(json.dumps(grading_results, indent=4))
