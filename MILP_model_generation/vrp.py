import random
import time
import numpy as np
import json
import os
from pulp import (
    LpProblem, LpMinimize,
    LpVariable, lpSum,
    LpBinary, LpContinuous,
    LpStatus, value,
    PULP_CBC_CMD
)
import matplotlib.pyplot as plt

class VehicleRouting:
    def __init__(self, parameters, seed=None):
        # Load parameters as attributes
        for key, val in parameters.items():
            setattr(self, key, val)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        """
        Generate random integers in [interval[0], interval[1]) of given shape.
        """
        return np.random.randint(interval[0], interval[1], size)

    def generate_data(self):
        """
        Generate random VRP instance data:
          - coordinates for depot + customers
          - Euclidean distance matrix
          - demands per node (depot demand = 0)
          - vehicle capacity and number of vehicles

        Returns:
            dict with keys: 'distance', 'demand', 'vehicle_capacity', 'num_vehicles', 'depot'
        """
        # Total nodes = customers + depot
        total_nodes = self.n_customers + 1

        # Generate random coordinates in given interval
        coords = np.random.uniform(
            low=self.coordinate_interval[0],
            high=self.coordinate_interval[1],
            size=(total_nodes, 2)
        )

        # Compute Euclidean distance matrix
        dist = np.zeros((total_nodes, total_nodes))
        for i in range(total_nodes):
            for j in range(total_nodes):
                dist[i, j] = np.hypot(*(coords[i] - coords[j]))

        # Round distances to two decimals
        distance = dist.round(2).tolist()

        # Demands: first node is depot with zero demand
        demands = np.zeros(total_nodes, dtype=int)
        demands[1:] = self.randint(self.n_customers, self.demand_interval)

        data = {
            "coords": coords.tolist(),
            "distance": distance,
            "demand": demands.tolist(),
            "vehicle_capacity": self.vehicle_capacity,
            "num_vehicles": self.num_vehicles,
            "depot": 0
        }
        return data

    def generate_instance(self, data):
        """
        Build a PuLP model for the CVRP using MTZ subtour-elimination.
        """
        # Unpack data
        distance = {i: {j: data["distance"][i][j] for j in range(len(data["distance"]))}
                    for i in range(len(data["distance"]))}
        demand = {i: data["demand"][i] for i in range(len(data["demand"]))}
        Q = data["vehicle_capacity"]
        K = data["num_vehicles"]
        depot = data.get("depot", 0)

        nodes = list(distance.keys())
        customers = [i for i in nodes if i != depot]

        # Create model
        model = LpProblem("CVRP_MTZ", LpMinimize)

        # Decision variables
        x = LpVariable.dicts("x", (nodes, nodes), cat=LpBinary)
        u = LpVariable.dicts("u", nodes,
                             lowBound=0, upBound=len(customers),
                             cat=LpContinuous)
        # Include depot in load variables to allow setting load[depot] = 0
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

        return model

    def solve(self, model):
        """
        Solve the VRP and return status and solve time.
        """
        start = time.time()
        solver = PULP_CBC_CMD()
        model.solve(solver)
        end = time.time()

        print("Status:", LpStatus[model.status])
        print("Total distance:", value(model.objective))
        
        # Print decision variables
        print("\nRoute arcs (x[i][j]=1):")
        for var in model.variables():
            if var.name.startswith('x_') and value(var) > 0.5:
                print(f" {var.name} = {value(var)}")
        print("\nNode loads:")
        for var in model.variables():
            if var.name.startswith('load_'):
                print(f" {var.name} = {value(var)}")

        # Build successor list for chosen arcs
        succ = {i: [] for i in model.variables()[0].name and []}
        # Actually retrieve node list
        nodes = sorted({int(var.name.split('_')[1]) for var in model.variables() if var.name.startswith('x_')})
        succ = {i: [] for i in nodes}
        for var in model.variables():
            if var.name.startswith('x_') and value(var) > 0.5:
                _, i, j = var.name.split('_')
                succ[int(i)].append(int(j))

        # Reconstruct routes up to num_vehicles
        routes = []
        used = set()
        for v in range(self.num_vehicles):
            route = [0] # Start with depot
            cur = 0
            while True:
                if cur not in succ or not succ[cur]:
                    break
                nxt = succ[cur].pop(0)
                if nxt == 0:
                    route.append(0)
                    break
                if nxt in used:
                    break
                route.append(nxt)
                used.add(nxt)
                cur = nxt
            if len(route) > 1:
                routes.append(route)

        # Print reconstructed routes
        print("\nVehicle routes:")
        for idx, r in enumerate(routes, 1):
            print(f" Vehicle {idx}: {' -> '.join(map(str, r))}")

        model.writeLP("vrp_solution.lp")

        # (Optional) Route reconstruction can be added here

        return model.status, end - start, routes
    
    def plot_routes(self, data, routes):
        """
        Plot the CVRP solution: nodes and vehicle routes.
        """
        coords = np.array(data['coords'])
        depot = data.get('depot', 0)
        num_vehicles = data.get('num_vehicles', len(routes))
        total_nodes = len(coords) # Get total number of nodes

        plt.figure(figsize=(8, 8))
        # Plot customers and depot
        plt.scatter(coords[1:, 0], coords[1:, 1], c='blue', label='Customers')
        plt.scatter(coords[depot, 0], coords[depot, 1], c='red', marker='s', s=100, label='Depot')

        # Add labels for each node
        for i in range(total_nodes):
            plt.annotate(str(i), (coords[i, 0], coords[i, 1]),
                         textcoords="offset points", xytext=(5,5), ha='center')

        # Plot each route
        for idx, route in enumerate(routes, 1):
            pts = coords[route]
            plt.plot(pts[:, 0], pts[:, 1], linestyle='-', marker='o', label=f'Vehicle {idx}')

        plt.title(f'Vehicle Routing Solution, {num_vehicles} Vehicles')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 10,
        'coordinate_interval': (0, 100),
        'demand_interval': (1, 10),
        'vehicle_capacity': 50,
        'num_vehicles': 2
    }

    vrp = VehicleRouting(parameters, seed)
    data = vrp.generate_data()

    # Ensure output directory exists
    OUTPUT_DIR = os.path.join("models", "VRP", "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"vrp_data_{parameters['n_customers']}cust_{parameters['num_vehicles']}veh.json"
    fpath = os.path.join(OUTPUT_DIR, fname)

    # Export to JSON
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved VRP instance: {fpath}")

    # Build and solve model
    instance = vrp.generate_instance(data)
    status, runtime, routes = vrp.solve(instance)
    print(f"Solve status: {status}, time: {runtime:.2f}s")
    # vrp.plot_routes(data, routes)
