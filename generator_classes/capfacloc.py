from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous, LpStatus, value
import numpy as np
import pandas as pd
import time
import json

def build_cflp_model(demands, capacities, fixed_costs, transportation_costs, continuous_assignment=True):
    """
    Build a Capacitated Facility Location Problem (CFLP) model.
    
    Args:
        demands (list): Demand values for each customer.
        capacities (list): Capacity values for each facility.
        fixed_costs (list): Fixed costs to open each facility.
        transportation_costs (list of lists): Transportation cost matrix (customers x facilities).
        continuous_assignment (bool): Whether to use continuous assignment variables.
    
    Returns:
        model (LpProblem): The constructed MILP model.
        x (dict): Assignment decision variables.
        y (dict): Facility open/close decision variables.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    model = LpProblem("CapacitatedFacilityLocation", LpMinimize)

    # Decision variables
    y = {j: LpVariable(f"Open_{j}", cat=LpBinary) for j in range(n_facilities)}
    x = {(i, j): LpVariable(f"Serve_{i}_{j}", lowBound=0,
                              cat=LpContinuous if continuous_assignment else LpBinary)
         for i in range(n_customers) for j in range(n_facilities)}

    # Objective: minimize fixed + transportation costs
    model += (
        lpSum(fixed_costs[j] * y[j] for j in range(n_facilities)) +
        lpSum(transportation_costs[i][j] * x[i, j] for i in range(n_customers) for j in range(n_facilities)),
        "TotalCost"
    )

    # Each customer must be served at least once
    for i in range(n_customers):
        model += lpSum(x[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}"

    # Capacity constraints: a facility's total service cannot exceed its capacity if it is open
    for j in range(n_facilities):
        model += lpSum(x[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * y[j], f"Capacity_{j}"

    # Total open capacity must meet total demand
    model += lpSum(capacities[j] * y[j] for j in range(n_facilities)) >= sum(demands), "TotalDemand"

    # Linking: assignment only if facility is open
    for i in range(n_customers):
        for j in range(n_facilities):
            model += x[i, j] <= y[j], f"Link_{i}_{j}"

    return model, x, y

def solve_model(model):
    """
    Solves the given model and returns the solver status and objective value.
    """
    status = model.solve()
    status_str = LpStatus[status]
    obj_value = None
    if status_str == "Optimal":
        obj_value = value(model.objective)
    return status_str, obj_value

def apply_perturbation(baseline, multiplier):
    """
    Applies a multiplicative perturbation to a list of baseline values.
    """
    return [v * multiplier for v in baseline]

def apply_matrix_perturbation(matrix, multiplier):
    """
    Applies a multiplicative perturbation to each element of a 2D list.
    """
    return [[elem * multiplier for elem in row] for row in matrix]

if __name__ == '__main__':
    # -------------------------
    # Define the fixed baseline problem parameters
    # -------------------------
    demands = [20, 25, 30, 18, 22]  # For 5 customers
    capacities = [80, 90, 70, 100, 85]  # For 5 facilities
    fixed_costs = [150, 180, 160, 170, 155]  # Fixed cost for each facility
    transportation_costs = [
        [10, 12, 15, 20, 18],
        [14, 11, 13, 16, 19],
        [13, 17, 12, 14, 15],
        [12, 15, 10, 18, 16],
        [11, 13, 14, 15, 17]
    ]
    
    print("=== BASELINE PROBLEM ===")
    print("Demands:", demands)
    print("Capacities:", capacities)
    print("Fixed Costs:", fixed_costs)
    print("Transportation Costs:")
    for i, row in enumerate(transportation_costs):
        print(f"  Customer {i}:", row)
    
    # Build and solve baseline model, and measure computation time
    start_time = time.perf_counter()
    baseline_model, x_baseline, y_baseline = build_cflp_model(demands, capacities, fixed_costs, transportation_costs)
    base_status, base_obj = solve_model(baseline_model)
    base_comp_time = time.perf_counter() - start_time
    
    print("\n=== BASELINE SOLUTION ===")
    print("Solve Status:", base_status)
    if base_status == "Optimal":
        print(f"Total Cost: {base_obj:.2f}")
    else:
        print("Baseline model did not find an optimal solution.")
    
    # Create a list to log all scenarios, including the baseline.
    scenario_logs = []
    
    # Log the baseline scenario with a specific title.
    baseline_log_entry = {
        "Scenario Title": "Baseline Scenario",
        "Solver Status": base_status,
        "Objective Value": base_obj,
        "Impact (Delta from baseline)": 0.0,  # Baseline impact is zero by definition.
        "Computation Time (sec)": base_comp_time,
        "Perturbed Demands": json.dumps(demands),
        "Perturbed Capacities": json.dumps(capacities),
        "Perturbed Fixed Costs": json.dumps(fixed_costs),
        "Perturbed Transportation Costs": json.dumps(transportation_costs),
        "Multipliers": json.dumps({"demands": 1.0, "capacities": 1.0, "fixed_costs": 1.0, "transportation_costs": 1.0}),
        "Specific Capacity Changes": json.dumps({})
    }
    scenario_logs.append(baseline_log_entry)
    
    # -------------------------
    # Define perturbation scenarios
    # -------------------------
    scenarios = [
        {
            "name": "Increase Demands by 10%",
            "multipliers": {
                "demands": 1.10,
                "capacities": 1.0,
                "fixed_costs": 1.0,
                "transportation_costs": 1.0
            }
        },
        {
            "name": "Decrease Demands by 10%",
            "multipliers": {
                "demands": 0.90,
                "capacities": 1.0,
                "fixed_costs": 1.0,
                "transportation_costs": 1.0
            }
        },
        {
            "name": "Increase Capacities by 20%",
            "multipliers": {
                "demands": 1.0,
                "capacities": 1.20,
                "fixed_costs": 1.0,
                "transportation_costs": 1.0
            }
        },
        {
            "name": "Increase Transportation Costs by 15%",
            "multipliers": {
                "demands": 1.0,
                "capacities": 1.0,
                "fixed_costs": 1.0,
                "transportation_costs": 1.15
            }
        },
        {
            "name": "Shut Off Facility 2",
            "multipliers": {
                "demands": 1.0,
                "capacities": 1.0,
                "fixed_costs": 1.0,
                "transportation_costs": 1.0
            },
            "specific_capacity_changes": {2: 0}
        },
        {
            "name": "Decrease Facility 1 Capacity by 50%",
            "multipliers": {
                "demands": 1.0,
                "capacities": 1.0,
                "fixed_costs": 1.0,
                "transportation_costs": 1.0
            },
            "specific_capacity_changes": {1: capacities[1] * 0.5}
        }
    ]
    
    # -------------------------
    # Perturb the baseline problem and log results
    # -------------------------
    print("\n=== PERTURBATION ANALYSIS ===")
    for scenario in scenarios:
        name = scenario["name"]
        multipliers = scenario["multipliers"]
        
        # Apply multiplicative perturbations to each parameter
        new_demands = apply_perturbation(demands, multipliers["demands"])
        new_capacities = apply_perturbation(capacities, multipliers["capacities"])
        new_fixed_costs = apply_perturbation(fixed_costs, multipliers["fixed_costs"])
        new_transportation_costs = apply_matrix_perturbation(transportation_costs, multipliers["transportation_costs"])
        
        # Apply specific facility changes (e.g., shutting off a facility or modifying a single capacity)
        specific_changes = scenario.get("specific_capacity_changes", {})
        for facility_index, new_value in specific_changes.items():
            new_capacities[facility_index] = new_value
        
        # Build and solve the perturbed model, and measure computation time
        start_time = time.perf_counter()
        perturbed_model, x_pert, y_pert = build_cflp_model(new_demands, new_capacities, new_fixed_costs, new_transportation_costs)
        status, obj = solve_model(perturbed_model)
        comp_time = time.perf_counter() - start_time
        
        # Compute impact relative to baseline (if available)
        impact = None
        if base_obj is not None and obj is not None:
            impact = obj - base_obj
        
        # Log scenario details and metrics
        log_entry = {
            "Scenario Title": name,
            "Solver Status": status,
            "Objective Value": obj,
            "Impact (Delta from baseline)": impact,
            "Computation Time (sec)": comp_time,
            "Perturbed Demands": json.dumps(new_demands),
            "Perturbed Capacities": json.dumps(new_capacities),
            "Perturbed Fixed Costs": json.dumps(new_fixed_costs),
            "Perturbed Transportation Costs": json.dumps(new_transportation_costs),
            "Multipliers": json.dumps(multipliers),
            "Specific Capacity Changes": json.dumps(scenario.get("specific_capacity_changes", {}))
        }
        scenario_logs.append(log_entry)
        
        # Print scenario results
        print(f"\nScenario: {name}")
        print("Solve Status:", status)
        if status == "Optimal":
            print(f"Total Cost: {obj:.2f}")
            print(f"Impact (Delta from baseline): {impact:.2f}")
        print(f"Computation Time: {comp_time:.4f} sec")
    
    # Save all scenario logs (including baseline) to CSV
    df = pd.DataFrame(scenario_logs)
    csv_filename = "perturbation_analysis_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nScenario logs saved to '{csv_filename}'")
