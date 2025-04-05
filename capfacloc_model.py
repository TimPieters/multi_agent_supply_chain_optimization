from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous, LpStatus, value
import numpy as np

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
n_customers = 5
n_facilities = 5

### DATA MANIPULATION CODE HERE ###

model = LpProblem("CapacitatedFacilityLocation", LpMinimize)

# Decision variables: facility_open indicates if a facility is open;
# assignment represents the service level from facility j to customer i.
facility_open = {j: LpVariable(f"Open_{j}", cat=LpBinary) for j in range(n_facilities)}
assignment = {(i, j): LpVariable(f"Serve_{i}_{j}", lowBound=0,
                                   cat=LpContinuous)
              for i in range(n_customers) for j in range(n_facilities)}

# Objective: minimize fixed costs plus transportation costs
model += (
    lpSum(fixed_costs[j] * facility_open[j] for j in range(n_facilities)) +
    lpSum(transportation_costs[i][j] * assignment[i, j] for i in range(n_customers) for j in range(n_facilities)),
    "TotalCost"
)

# Each customer must be served at least once
for i in range(n_customers):
    model += lpSum(assignment[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}"

# Capacity constraints: a facility's total service cannot exceed its capacity if it is open
for j in range(n_facilities):
    model += lpSum(assignment[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * facility_open[j], f"Capacity_{j}"

# Total open capacity must meet total demand
model += lpSum(capacities[j] * facility_open[j] for j in range(n_facilities)) >= sum(demands), "TotalDemand"

# Linking: assignment is allowed only if the facility is open
for i in range(n_customers):
    for j in range(n_facilities):
        model += assignment[i, j] <= facility_open[j], f"Link_{i}_{j}"

### CONSTRAINT CODE HERE ###


# result = get_results()
# print(result)

print(model.to_dict)

# Solve the model
status = model.solve()
print("Total Cost:", model.objective.value())
# Decision Variables
for v in model.variables():
    try:
        print(v.name,"=", v.value())
    except:
        print("error couldnt find value")
# status_str = LpStatus[status]
# total_cost = None
# if status_str == "Optimal":
#     total_cost = value(model.objective)

# print("Solve status:", status_str)
# if total_cost is not None:
#     print("Total Cost:", total_cost)

# print("\nAssignments:")
# for (i, j), var in assignment.items():
#     if var.varValue is not None and var.varValue > 1e-6:
#         print(f"  Customer {i} -> Facility {j}: {var.varValue}")

# print("\nOpen Facilities:")
# for j, var in facility_open.items():
#     if var.varValue is not None and var.varValue > 0.5:
#         print(f"  Facility {j} is open")