import json
import time
import pandas as pd
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm # Import tqdm for progress bar

# Ensure project root is on sys.path so we can import utils
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

from utils import _read_source_code, _apply_model_modification, _run_with_exec, _get_optimization_result, _prepare_run_data, write_run_data_to_csv
from config import MODEL_PARAMETERS # Import MODEL_PARAMETERS for dynamic parameter extraction

# CONFIGURATION
default_model_path = "models/VRP/vrp_model.py"
default_data_path  = "models/VRP/data/vrp_data_10cust_2veh.json"
final_log_filepath = "benchmark/VRP/vrp_benchmark_10cust_2veh.csv"
max_workers       = 14  # number of threads

# Load base data
data      = json.load(open(default_data_path))
nodes     = list(range(len(data["distance"])))
depot     = data.get("depot", 0)
customers = [i for i in nodes if i != depot]
Q         = data["vehicle_capacity"]
K         = data["num_vehicles"]

# Read the base model code once
base_model_code = _read_source_code(default_model_path)

# Inject DATA_FILE_PATH into the model code
data_path_injection = f'DATA_FILE_PATH = "{default_data_path}"\n'
model_code_with_data_path = data_path_injection + base_model_code

# 1) Define scenario macros
scenario_macros = [
    {
        "question": "What would happen if the demand at customer {C} increased by {P}%?",
        "placeholders": {
            "C": customers,              # FULL list of customers
            "P": list(range(5, 51))      # 5% through 50%
        },
        "code_type": "ADD DATA",
        "code_template": "demand[{C}] = demand[{C}] * (1 + {P}/100)",
        "scenario_type": "demand-increase-customer-pct"
    },
    {
        "question": "What would happen if we changed the demand of customer {C} to {V}?",
        "placeholders": {
            "C": customers,              # FULL list of customers
            "V": list(range(1, 26))      # 1 to 25 units
        },
        "code_type": "ADD DATA",
        "code_template": "demand[{C}] = {V}",
        "scenario_type": "demand-increase-customer-int"
    },
    {
        "question": "What if demand at all customers increased by {P}%?",
        "placeholders": {
            "P": list(range(5, 51))
        },
        "code_type": "ADD DATA",
        "code_template": "for i in customers: demand[i] = demand[i] * (1 + {P}/100)",
        "scenario_type": "demand-increase-all"
    },
    {
        "question": "Suppose vehicle capacity changed to {CAP}. How does that affect total distance?",
        "placeholders": {
            "CAP": list(range(Q//2, Q*2 + 1)) # from half to double the original capacity 25 to 100
        },
        "code_type": "ADD DATA",
        "code_template": "Q = {CAP}",
        "scenario_type": "capacity-change"
    },
    {
        "question": "What if we change the fleet size to {KNEW} vehicles?",
        "placeholders": {
            "KNEW": list(range(1, K + 5 + 1)) # from 1 to K+5 (e.g., 1 to 7)
        },
        "code_type": "ADD DATA",
        "code_template": "K = {KNEW}",
        "scenario_type": "fleetsize-change"
    },
    {
        "question": "Why isn't the arc from node {I} to node {J} used?",
        "placeholders": {
            "I": nodes,
            "J": nodes
        },
        "code_type": "ADD CONSTRAINT",
        "code_template": "model += lpSum([x[{I}][{J}]]) == 0, \"forbid_{I}_{J}\"",
        "scenario_type": "forbid-arc"
    },
    {
        "question": "What if vehicle must traverse arc ({I}, {J})?",
        "placeholders": {
            "I": nodes,
            "J": nodes
        },
        "code_type": "ADD CONSTRAINT",
        "code_template": "model += lpSum([x[{I}][{J}]]) == 1, \"force_{I}_{J}\"",
        "scenario_type": "force-arc"
    }
    # Add additional macros as needed
]

# 2) Build full Cartesian product of all placeholder values
from itertools import product

tasks = []
seq = 0

for macro in scenario_macros:
    keys        = list(macro["placeholders"].keys())
    value_lists = [macro["placeholders"][k] for k in keys]

    for combo in product(*value_lists):
        seq += 1
        vals = { keys[i]: combo[i] for i in range(len(keys)) }
        tasks.append((macro, seq, vals))



# 3) Worker: each thread prepares run data
def run_scenario(macro, seq, vals, model_code_with_data_path):
    question   = macro["question"].format(**vals)
    code_snip  = macro["code_template"].format(**vals)
    mod_json   = {macro["code_type"]: code_snip}
    run_id     = f"vrp_bench_{time.strftime('%Y%m%d_%H%M%S')}_{seq}"

    try:
        modified_code = _apply_model_modification(model_code_with_data_path, mod_json)
        
        # Execute the modified code and get the locals dictionary
        locals_dict = _run_with_exec(modified_code)

        result = _get_optimization_result(locals_dict)
        
        if not locals_dict or "model" not in locals_dict:
            return {
                'run_id': run_id,
                'modification': json.dumps(mod_json),
                'scenario_text': question,
                'scenario_type': macro['scenario_type'],
                'status': 'Error',
                'message': 'Model execution failed or model not found.'
            }

        model = locals_dict["model"]
        execution_time = locals_dict.get('execution_time')

        # Dynamically define parameters based on the current model
        parameters = {}
        current_model_params = MODEL_PARAMETERS.get(default_model_path, [])
        for param_name in current_model_params:
            parameters[param_name] = locals_dict.get(param_name)
        
        # Fallback for models not explicitly listed in MODEL_PARAMETERS
        if not parameters and hasattr(model, 'parameters'):
            parameters = model.parameters
        elif not parameters:
            if 'demands' in locals_dict: parameters['demands'] = locals_dict['demands']
            if 'capacities' in locals_dict: parameters['capacities'] = locals_dict['capacities']
            if 'fixed_costs' in locals_dict: parameters['fixed_costs'] = locals_dict['fixed_costs']
            if 'transportation_costs' in locals_dict: parameters['transportation_costs'] = locals_dict['transportation_costs']
            if 'supply' in locals_dict: parameters['supply'] = locals_dict['supply']
            if 'costs' in locals_dict: parameters['costs'] = locals_dict['costs']
            if 'routes' in locals_dict: parameters['routes'] = locals_dict['routes']

        # Prepare the run data dictionary
        run_data = _prepare_run_data(
            model=model,
            parameters=parameters,
            run_id=run_id,
            model_file_path=default_model_path,
            model_data_path=default_data_path,
            execution_time=execution_time
        )
        
        # Add scenario metadata to run_data
        run_data.update({
            'modification': json.dumps(mod_json),
            'scenario_text': question,
            'scenario_type': macro['scenario_type'],
            'placeholder_values': json.dumps(vals) # Add placeholder_values
        })
        
        return run_data

    except Exception as e:
        print(f"Error during scenario run '{run_id}': {str(e)}")
        return {
            'run_id': run_id,
            'modification': json.dumps(mod_json),
            'scenario_text': question,
            'scenario_type': macro['scenario_type'],
            'status': 'Error',
            'message': f"Error: {str(e)}"
        }

# 4) Dispatch in parallel and collect all run data
all_benchmark_runs = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_scenario, m, s, v, model_code_with_data_path) for (m, s, v) in tasks]
    for f in tqdm(as_completed(futures), total=len(tasks), desc="Running VRP Benchmark Scenarios"):
        all_benchmark_runs.append(f.result())

# 5) Write all collected run data to the final log file
write_run_data_to_csv(all_benchmark_runs, final_log_filepath)
print(f"Benchmark complete: {len(all_benchmark_runs)} runs logged to {final_log_filepath}.")
