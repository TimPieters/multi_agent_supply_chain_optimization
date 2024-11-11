# A simple supply chain optimization model using PuLP

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpStatusOptimal

# Initialize supply, demand, and cost data
supply = [100, 150, 200]  # Supply from three suppliers
demand = [120, 130, 100, 100]  # Demand from four demand centers
costs = [
    [2, 3, 1, 4],  # Costs from supplier 0 to demand centers
    [3, 2, 5, 2],  # Costs from supplier 1 to demand centers
    [4, 1, 3, 2]   # Costs from supplier 2 to demand centers
]


### DATA MANIPULATION CODE HERE ###


# Create the problem instance
problem = LpProblem("SimpleSupplyChainProblem", LpMinimize)

# Create variables for each supply-demand pair
variables = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous')
             for i in range(len(supply))
             for j in range(len(demand))}

# Objective function: minimize transportation costs
problem += lpSum(costs[i][j] * variables[i, j] for i in range(len(supply)) for j in range(len(demand))), "Total Cost"

# Supply constraints
for i in range(len(supply)):
    problem += lpSum(variables[i, j] for j in range(len(demand))) <= supply[i], f"Supply_{i}"

# Demand constraints
for j in range(len(demand)):
    problem += lpSum(variables[i, j] for i in range(len(supply))) >= demand[j], f"Demand_{j}"


### CONSTRAINT CODE HERE ###


# Solve the problem
status = problem.solve()

# Print the results
print(LpStatus[status])
if status == LpStatusOptimal:
    print("Optimal Solution:")
    for (i, j), var in variables.items():
        if var.varValue > 0:
            print(f"Units from Supplier {i} to Demand Center {j}: {var.varValue}")

    print(f"Total Cost: {problem.objective.value()}")
else:
    print("Not solved to optimality. Optimization status:", status)
