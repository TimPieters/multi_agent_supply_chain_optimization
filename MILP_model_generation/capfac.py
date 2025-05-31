import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, LpContinuous, lpSum, LpStatus
import pulp
import os
import json

class CapacitatedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    

    def generate_data(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.randint(self.n_facilities, self.fixed_cost_interval) 

        transportation_costs = self.randint(size=(self.n_customers, self.n_facilities),
                                            interval=self.transportation_cost_interval)

        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        return res
    
    def generate_instance(self, data):
        """
        Generate a model instance for the capacitated facility location problem.
        
        Args:
            data: A dictionary containing demands, capacities, fixed_costs, and transportation_costs
            
        returns:
            model: A PuLP model instance for the capacitated facility location problem
        """
            
        demands = data['demands']
        capacities = data['capacities']
        fixed_costs = data['fixed_costs']
        transportation_costs = data['transportation_costs']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        # Create the model
        model = LpProblem(name="CapacitatedFacilityLocation", sense=LpMinimize)
        
        # Decision variables
        # Binary variables for whether to open facility j
        open_facilities = {j: LpVariable(f"Open_{j}", cat=LpBinary) for j in range(n_facilities)}
        
        # Variables for customer-facility assignments
        if self.continuous_assignment:
            serve = {(i, j): LpVariable(f"Serve_{i}_{j}", lowBound=0, upBound=1, cat=LpContinuous) 
                    for i in range(n_customers) for j in range(n_facilities)}
        else:
            serve = {(i, j): LpVariable(f"Serve_{i}_{j}", cat=LpBinary) 
                    for i in range(n_customers) for j in range(n_facilities)}
        
        # Objective function: minimize total cost
        model += (
            lpSum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) +
            lpSum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        )
        
        # Constraints: each customer's demand must be met
        for i in range(n_customers):
            model += (lpSum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")
        
        # Constraints: respect facility capacities
        for j in range(n_facilities):
            model += (
                lpSum(serve[i, j] * demands[i] for i in range(n_customers)) <= 
                capacities[j] * open_facilities[j], 
                f"Capacity_{j}"
            )
        
        # Constraint: total capacity must meet total demand
        total_demand = np.sum(demands)
        model += (
            lpSum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, 
            "TotalDemand"
        )
        
        # Tightening constraints: can't assign to closed facilities
        for i in range(n_customers):
            for j in range(n_facilities):
                model += (serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}")

        return model

    ################# PuLP modeling #################
    def solve(self, model):
        """
        Solve the capacitated facility location problem using PuLP
        
        Args:
            model: A PuLP model instance for the capacitated facility location problem
            
        Returns:
            tuple: (solve_status, solve_time)
        """
        
        # Solve the model and track time
        start_time = time.time()
        solver = pulp.SCIP_CMD(msg=False,
                               options=[
                                "-c", "threads=1"])
        model.solve(solver)
        end_time = time.time()
        
        status = model.status
        print("Total Cost:", model.objective.value())
        model.writeLP("facility_location_solution.lp")

        return status, end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 50,
        'n_facilities': 50,
        'demand_interval': (5, 36),
        'capacity_interval': (80, 120),
        'transportation_cost_interval': (10, 20),
        'fixed_cost_interval': (150, 200),
        'continuous_assignment': True,
    }

    facility_location = CapacitatedFacilityLocation(parameters)
    data = facility_location.generate_data()

    instance = facility_location.generate_instance(data)

# === Export the data to a JSON file ===
    OUTPUT_DIR = "data"
    # Convert numpy arrays to lists
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data[k] = v.tolist()
    
    n_cust = parameters['n_customers']
    n_fac = parameters['n_facilities']
        
    # Filename and path
    fname = f"capfacloc_data_{n_cust}cust_{n_fac}fac.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(data)    
    # Write JSON, in custom format
    with open(fpath, 'w') as f:
        f.write('{\n')
        f.write(f'  "demands": {json.dumps(data["demands"])},\n')
        f.write(f'  "capacities": {json.dumps(data["capacities"])},\n')
        f.write(f'  "fixed_costs": {json.dumps(data["fixed_costs"])},\n')
        f.write('  "transportation_costs": [\n')
        rows = data["transportation_costs"]
        for i, row in enumerate(rows):
            comma = ',' if i < len(rows) - 1 else ''
            f.write(f'    {json.dumps(row)}{comma}\n')
        f.write('  ]\n')
        f.write('}\n')

    print(f"Saved instance: {fpath}")
# ==================================================

    # baseline_instance = {
    #     'demands': np.array([20, 25, 30, 18, 22], dtype=np.int32), # For 5 customers
    #     'capacities': np.array([80, 90, 70, 100, 85], dtype=np.float64), # For 5 facilities
    #     'fixed_costs': np.array([150, 180, 160, 170, 155], dtype=np.float64), # Fixed cost for each facility
    #     'transportation_costs': np.array([[10, 12, 15, 20, 18],
    #                                       [14, 11, 13, 16, 19],
    #                                       [13, 17, 12, 14, 15],
    #                                       [12, 15, 10, 18, 16],
    #                                       [11, 13, 14, 15, 17]],
    #                                       dtype=np.float64)}

    # solve_status, solve_time = facility_location.solve(instance)

    # print(f"Solve Status: {solve_status}")
    # print(f"Solve Time: {solve_time:.2f} seconds")

    #print(model.to_dict)

    # Solve the model
    #status = model.solve()
    # print("Total Cost:", model.objective.value())
    # # Decision Variables
    # for v in model.variables():
    #     try:
    #         print(v.name,"=", v.value())
    #     except:
    #         print("error couldnt find value")