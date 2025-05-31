from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous, LpStatus, value
import numpy as np
import json

with open(DATA_FILE_PATH, "r") as f:
    data = json.load(f)

demands                  = data["demands"]                   # list of length n_customers
capacities               = data["capacities"]                # list of length n_facilities
fixed_costs              = data["fixed_costs"]               # list of length n_facilities
transportation_costs     = data["transportation_costs"]      # matrix (n_customers × n_facilities)
n_customers              = len(demands)
n_facilities             = len(capacities)

### DATA MANIPULATION CODE HERE ###

model = LpProblem("CapacitatedFacilityLocation", LpMinimize)

# Decision variables: facility_open indicates if a facility is open;
# assignment represents the service level from facility j to customer i.
facility_open = {j: LpVariable(f"Open_{j}", cat=LpBinary) for j in range(n_facilities)}
assignment = {(i, j): LpVariable(f"Serve_{i}_{j}", lowBound=0, upBound=1,
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


# Solve the model
#model.solve()
#print(model.to_json("capfacloc_model.JSON"))
# print("Total Cost:", model.objective.value())
# # Decision Variables
# for v in model.variables():
#     try:
#         print(v.name,"=", v.value())
#     except:
#         print("error couldnt find value")
