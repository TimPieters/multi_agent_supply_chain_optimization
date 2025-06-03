import json
import time
import pandas as pd
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm # Import tqdm for progress bar
from itertools import product
import itertools

# Ensure project root is on sys.path so we can import utils
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

from utils import _read_source_code, _apply_model_modification, _run_with_exec, _get_optimization_result, _prepare_run_data, write_run_data_to_csv, write_run_data_to_parquet
from config import MODEL_PARAMETERS # Import MODEL_PARAMETERS for dynamic parameter extraction

# CONFIGURATION
default_model_path = "models/VRP/vrp_model.py"
config_str = "15cust_2veh_100cap"
default_data_path  = f"models/VRP/data/vrp_data_{config_str}.json"
csv_log_filepath = f"benchmark/VRP/vrp_benchmark_{config_str}.csv"
parquet_log_filepath = f"benchmark/VRP/vrp_benchmark_{config_str}.parquet"
max_workers       = 14  # number of threads

# Load base data
try:
    with open(default_data_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Data file not found at {default_data_path}")
    sys.exit(1)

nodes     = list(range(len(data["distance"])))
depot     = data.get("depot", 0)
customers = [i for i in nodes if i != depot]
Q         = data["vehicle_capacity"]
K         = data["num_vehicles"]

# Read the base model code once
base_model_code = _read_source_code(default_model_path)

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

# 2) Generator for all scenario tasks
def generate_all_scenario_tasks(scenario_macros, nodes, customers):
    """
    Generates all scenario tasks iteratively, yielding one at a time.
    """
    seq = 0
    for macro in scenario_macros:
        keys = list(macro["placeholders"].keys())
        value_lists = [macro["placeholders"][k] for k in keys]
        
        for combo in product(*value_lists):
            seq += 1
            vals = {keys[i]: combo[i] for i in range(len(keys))}
            yield (macro, seq, vals)

# 3) Worker: each thread prepares run data
def run_scenario(macro, seq, vals, base_model_code, default_data_path, default_model_path):
    question   = macro["question"].format(**vals)
    code_snip  = macro["code_template"].format(**vals)
    mod_json   = {macro["code_type"]: code_snip}
    run_id     = f"vrp_bench_{time.strftime('%Y%m%d_%H%M%S')}_{seq}"

    try:
        # Inject DATA_FILE_PATH into the model code
        data_path_injection = f'DATA_FILE_PATH = "{default_data_path}"\n'
        model_code_with_data_path = data_path_injection + base_model_code

        modified_code = _apply_model_modification(model_code_with_data_path, mod_json)
        
        # Execute the modified code and get the locals dictionary
        locals_dict = _run_with_exec(modified_code)
        
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

        # Retrieve results
        result = _get_optimization_result(locals_dict)

        # Dynamically define parameters based on the current model
        parameters = {}
        current_model_params = MODEL_PARAMETERS.get(default_model_path, [])
        for param_name in current_model_params:
            parameters[param_name] = locals_dict.get(param_name)
        
        # Fallback for models not explicitly listed in MODEL_PARAMETERS
        if not parameters and hasattr(model, 'parameters'):
            parameters = model.parameters
        elif not parameters:
            if 'distance' in locals_dict: parameters['distance'] = locals_dict['distance']
            if 'demand' in locals_dict: parameters['demand'] = locals_dict['demand']
            if 'vehicle_capacity' in locals_dict: parameters['vehicle_capacity'] = locals_dict['vehicle_capacity']
            if 'num_vehicles' in locals_dict: parameters['num_vehicles'] = locals_dict['num_vehicles']
            # Add more common parameter names as needed for VRP if they exist in the model's locals_dict
            if 'nodes' in locals_dict: parameters['nodes'] = locals_dict['nodes']
            if 'depot' in locals_dict: parameters['depot'] = locals_dict['depot']
            if 'customers' in locals_dict: parameters['customers'] = locals_dict['customers']


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
            'placeholder_values' : json.dumps(vals)
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

# 4) Dispatch in parallel and collect all run data in batches
total_scenarios = 0
# Estimate total scenarios for tqdm, if possible, otherwise it will be dynamic
# This is a rough estimate, actual count might differ if placeholder lists are dynamic
for macro in scenario_macros:
    num_combinations = 1
    for key in macro["placeholders"]:
        num_combinations *= len(macro["placeholders"][key])
    total_scenarios += num_combinations

BATCH_SIZE = 200 # Process 500 scenarios at a time to manage memory

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    processed_count = 0
    
    # Use tqdm with a generator for tasks
    task_generator = generate_all_scenario_tasks(scenario_macros, nodes, customers)
    
    pbar = tqdm(total=total_scenarios, desc="Running VRP Benchmark Scenarios")
    
    while True:
        batch_tasks = list(itertools.islice(task_generator, BATCH_SIZE))
        if not batch_tasks:
            break # No more tasks
        
        for macro, seq, vals in batch_tasks:
            futures.append(executor.submit(run_scenario, macro, seq, vals, base_model_code, default_data_path, default_model_path))
        
        batch_results = []
        for future in as_completed(futures):
            batch_results.append(future.result())
            pbar.update(1) # Update progress bar for each completed scenario
        
        # Clear futures for the next batch
        futures = [] 
        
        # Write results of the current batch incrementally
        if batch_results:
            if parquet_log_filepath.endswith('.parquet'):
                write_run_data_to_parquet(batch_results, parquet_log_filepath)
            if csv_log_filepath.endswith('.csv'):
                write_run_data_to_csv(batch_results, csv_log_filepath)
            else:
                print("Warning: No valid log file extension found. Results not saved for this batch.")
        
        processed_count += len(batch_results)
        
    pbar.close()

print(f"Benchmark complete: {processed_count} runs logged.")
print(f"Results saved to Parquet: {parquet_log_filepath}")
print(f"Results saved to CSV (if applicable): {csv_log_filepath}")
