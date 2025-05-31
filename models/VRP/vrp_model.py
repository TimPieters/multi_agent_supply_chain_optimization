from pulp import (
    LpProblem, LpMinimize,
    LpVariable, lpSum,
    LpBinary, LpContinuous,
    LpStatus,LpStatusOptimal, value,
    PULP_CBC_CMD
)
import json
import time

# Load data from the specified JSON file
try:
    with open(DATA_FILE_PATH, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE_PATH}. Please ensure DATA_FILE_PATH is correctly set.")
    raise

# Unpack data for model construction
coords = data["coords"]
distance = {i: {j: data["distance"][i][j] for j in range(len(data["distance"]))}
            for i in range(len(data["distance"]))}
demand = {i: data["demand"][i] for i in range(len(data["demand"]))}
Q = data["vehicle_capacity"]
K = data["num_vehicles"]
depot = data.get("depot", 0)

nodes = list(distance.keys())
customers = [i for i in nodes if i != depot]

### DATA MANIPULATION CODE HERE ###

# Create model
model = LpProblem("CVRP_MTZ", LpMinimize)

# Decision variables
x = LpVariable.dicts("x", (nodes, nodes), cat=LpBinary)
u = LpVariable.dicts("u", nodes,
                     lowBound=0, upBound=len(customers),
                     cat=LpContinuous)
load = LpVariable.dicts("load", nodes,
                        lowBound=0, upBound=Q,
                        cat=LpContinuous)

# Objective: minimize total distance
model += lpSum(distance[i][j] * x[i][j]
               for i in nodes for j in nodes), "TotalDistance"

# Degree constraints for customers
for i in customers:
    model += lpSum(x[i][j] for j in nodes if j != i) == 1, f"OutDeg_{i}"
    model += lpSum(x[j][i] for j in nodes if j != i) == 1, f"InDeg_{i}"

# Depot degree <= K
model += lpSum(x[depot][j] for j in customers) <= K, "DepotOut"
model += lpSum(x[i][depot] for i in customers) <= K, "DepotIn"

# MTZ subtour elimination
model += u[depot] == 0, "MTZ_depot"
for i in customers:
    for j in customers:
        if i != j:
            model += (
                u[i] - u[j] + len(customers) * x[i][j]
                <= len(customers) - 1
            ), f"MTZ_{i}_{j}"

# Total demand must be less than or equal to total vehicle capacity
model += lpSum(demand) <= K * Q, "TotalVehicleCapacity"

# Capacity linking constraints
for i in customers:
    for j in customers:
        if i != j:
            model += (
                load[i] + demand[j] - load[j]
                <= Q * (1 - x[i][j])
            ), f"CapLink_{i}_{j}"
# Initialize load when leaving depot
for j in customers:
    model += load[j] >= demand[j] * x[depot][j], f"InitLoad_{j}"
# Fix depot load to zero
model += load[depot] == 0, "Load_depot"

### CONSTRAINT CODE HERE ###

# start = time.time()
# solver = PULP_CBC_CMD()
# model.solve(solver)
# end = time.time()

# status = LpStatus[model.status]
# total_cost = value(model.objective) if model.status == LpStatusOptimal else None
# runtime = end - start

# # Reconstruct routes for potential plotting or detailed logging
# routes = []
# if model.status == LpStatusOptimal:
#     succ = {i: [] for i in nodes}
#     for var in model.variables():
#         if var.name.startswith('x_') and value(var) > 0.5:
#             _, i, j = var.name.split('_')
#             succ[int(i)].append(int(j))

#     used = set()
#     for v in range(K): # Use K (num_vehicles) from the loaded data
#         route = [0] # Start with depot
#         cur = 0
#         while True:
#             if cur not in succ or not succ[cur]:
#                 break
#             nxt = succ[cur].pop(0)
#             if nxt == 0:
#                 route.append(0)
#                 break
#             if nxt in used:
#                 break
#             route.append(nxt)
#             used.add(nxt)
#             cur = nxt
#         if len(route) > 1:
#             routes.append(route)

# # Print results
# print(f"Status: {status}")
# print(f"Total distance: {total_cost}")
# print(f"Runtime: {runtime:.2f} seconds")
# print("Routes:")
# for i, route in enumerate(routes):
#     print(f" Vehicle {i+1}: {' -> '.join(map(str, route))}")