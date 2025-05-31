# config.py

# Configuration for the LangGraph Sensitivity Analysis system

# Path to the optimization model Python file
MODEL_FILE_PATH = "models/VRP/vrp_model.py"

# Path to the JSON data file for the model
MODEL_DATA_PATH = "models/VRP/data/vrp_data_10cust_2veh.json"

# Path to the model description file (for Planner agent context)
MODEL_DESCRIPTION_PATH = "models/VRP/description.txt"

# Define data bounds for scenario validation (example structure)
# These bounds can be used by a validator agent to check proposed scenarios
DATA_BOUNDS = {
    "demands": {"min_multiplier": 0.5, "max_multiplier": 2.0},
    "capacities": {"min_multiplier": 0.5, "max_multiplier": 2.0},
    "fixed_costs": {"min_multiplier": 0.5, "max_multiplier": 2.0},
    "transportation_costs": {"min_multiplier": 0.5, "max_multiplier": 2.0},
    # Add specific bounds for individual parameters if needed, e.g.:
    # "demands_customer_0": {"min_value": 10, "max_value": 100},
}

# Define parameters to extract for each model
# This dictionary maps model file names to a list of parameter variable names
MODEL_PARAMETERS = {
    "models/CFLP/capfacloc_model.py": [
        "demands",
        "capacities",
        "fixed_costs",
        "transportation_costs",
        "n_customers",
        "n_facilities"
    ],
    "models/VRP/vrp_model.py": [
        "coords",
        "distance",
        "demand",
        "depot",
        "K", # Number of vehicles
        "Q"  # Vehicle capacity
    ],
    "simple_model.py": [
        "supply",
        "demand",
        "costs",
        "routes"
        # Add other relevant parameters for simple_model.py here
    ]
}

# Mapping of models to their available data files
MODEL_DATA_MAPPING = {
    "models/CFLP/capfacloc_model.py": [
        "capfacloc_data_10cust_10fac.json",
        "capfacloc_data_25cust_25fac.json",
        "capfacloc_data_50cust_50fac.json",
        "capfacloc_data.json" # Add the generic one too
    ],
    "models/VRP/vrp_model.py": [
        "vrp_data_10cust_2veh.json",
        "vrp_data_10cust_5veh.json",
        "vrp_data_20cust_5veh.json"
    ]
}
