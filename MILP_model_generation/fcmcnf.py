"""
Reformat from https://github.com/ds4dm/learn2comparenodes/tree/master/problem_generation
"""
import random
import time
import numpy as np
import networkx as nx
import pulp
import json
import os

class FCMCNF:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(*self.f_range)
            u_ij = np.random.uniform(*self.u_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_commodities(self, G):
        commodities = []
        for k in range(self.n_commodities):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            # integer demands
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_instance(self):
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings
        }
        
        return res

    ################# Export costs data #################
    def export_costs_data(self, instance, filename="costs_data.json"):
        """Export the cost data to JSON format"""
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        
        costs_data = {
            'edges': [],
            'commodities': []
        }
        
        # Export edge costs
        for i, j in edge_list:
            c_ij, f_ij, u_ij = adj_mat[i, j]
            costs_data['edges'].append({
                'from_node': int(i),
                'to_node': int(j),
                'variable_cost': float(c_ij),
                'fixed_cost': float(f_ij),
                'capacity': float(u_ij)
            })
        
        # Export commodity data
        for k, (origin, destination, demand) in enumerate(instance['commodities']):
            costs_data['commodities'].append({
                'commodity_id': k,
                'origin': int(origin),
                'destination': int(destination),
                'demand': int(demand)
            })
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Export to JSON
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as f:
            json.dump(costs_data, f, indent=2)
        
        print(f"Costs data exported to {filepath}")
        return filepath

    ################# PuLP Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        # Create the model
        model = pulp.LpProblem("FCMCNF", pulp.LpMinimize)
        
        # Decision variables - create all at once for efficiency
        x_vars = pulp.LpVariable.dicts("x", 
                                       [(i, j, k) for (i, j) in edge_list for k in range(self.n_commodities)],
                                       lowBound=0, cat='Continuous')
        
        y_vars = pulp.LpVariable.dicts("y", 
                                       [(i, j) for (i, j) in edge_list],
                                       cat='Binary')


        # Objective function - build as list for efficiency
        obj_terms = []
        # Variable costs
        for (i, j) in edge_list:
            c_ij = adj_mat[i, j][0]
            for k in range(self.n_commodities):
                demand_k = commodities[k][2]
                obj_terms.append(demand_k * c_ij * x_vars[(i, j, k)])
        # Fixed costs
        for (i, j) in edge_list:
            f_ij = adj_mat[i, j][1]
            obj_terms.append(f_ij * y_vars[(i, j)])
        
        model += pulp.lpSum(obj_terms)


        # Flow conservation constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                # 1 if source, -1 if sink, 0 if else
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                
                flow_terms = []
                # Outgoing flow
                for j in outcommings[i]:
                    flow_terms.append(x_vars[(i, j, k)])
                # Incoming flow (subtract)
                for j in incommings[i]:
                    flow_terms.append(-x_vars[(j, i, k)])
                    
                if flow_terms:  # Only add if there are flow terms
                    model += pulp.lpSum(flow_terms) == delta_i

        # Capacity constraints
        for (i, j) in edge_list:
            u_ij = adj_mat[i, j][2]
            capacity_terms = [commodities[k][2] * x_vars[(i, j, k)] for k in range(self.n_commodities)]
            model += pulp.lpSum(capacity_terms) <= u_ij * y_vars[(i, j)]

        # Solve the model
        start_time = time.time()
        model.solve(pulp.SCIP_PY())  # Use CBC solver with no output
        end_time = time.time()
        
        return pulp.LpStatus[model.status], end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 25,
        'n_commodities': 35,
        'c_range': (11, 50),
        'f_range': (1100, 5000),  # Fixed costs range
        'u_range': (100, 1000),   # Capacity range
        'd_range': (10, 100),
        'er_prob': 0.3,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    
    # Export costs data to JSON
    fcmcnf.export_costs_data(instance)
    
    # Solve the problem
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")