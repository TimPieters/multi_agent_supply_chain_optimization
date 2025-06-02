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
default_model_path = "models/CFLP/capfacloc_model.py"
default_data_path  = "models/CFLP/data/capfacloc_data_50cust_50fac.json"
data_options = ["capfacloc_data_10cust_10fac.json",
                "capfacloc_data_25cust_25fac.json",
                "capfacloc_data_50cust_50fac.json",
                ]
csv_log_filepath = "benchmark/CFLP/cflp_benchmark_50cust_50fac.csv"
parquet_log_filepath = "benchmark/CFLP/cflp_benchmark_50cust_50fac.parquet"
max_workers       = 14  # number of threads

# Load base data
try:
    with open(default_data_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Data file not found at {default_data_path}")
    sys.exit(1)

# Extract base parameters for CFLP
customers = list(range(len(data["demands"])))
facilities = list(range(len(data["capacities"])))
demands = data["demands"]
capacities = data["capacities"]
fixed_costs = data["fixed_costs"]
transportation_costs = data["transportation_costs"]

# 1) Define scenario macros for CFLP
scenario_macros = [
    {
        "question": "What if demand at customer {C} changed by {P}%?",
        "placeholders": {
            "C": customers,
            "P": list(range(1, 51))
        },
        "code_type": "ADD DATA",
        "code_template": "demands[{C}] = demands[{C}] * (1 + {P}/100)",
        "scenario_type": "demand-change-customer-pct"
    },
    {
        "question": "What would happen if we changed the demand of customer {C} to {V}?",
        "placeholders": {
            "C": customers,              # FULL list of customers
            "V": list(range(5, 37))      # 5 to 36 units
        },
        "code_type": "ADD DATA",
        "code_template": "demands[{C}] = {V}",
        "scenario_type": "demand-increase-customer-int"
    },
    {
        "question": "What if all demands changed by {P}%?",
        "placeholders": {
            "P": list(range(1, 51))
        },
        "code_type": "ADD DATA",
        "code_template": "demands = [d * (1 + {P}/100) for d in demands]",
        "scenario_type": "demand-change-all"
    },
    {
        "question": "What if capacity at facility {F} changed by {P}%?",
        "placeholders": {
            "F": facilities,
            "P": list(range(-20, 21))
        },
        "code_type": "ADD DATA",
        "code_template": "capacities[{F}] = capacities[{F}] * (1 + {P}/100)",
        "scenario_type": "capacity-change-facility-pct"
    },
    {
        "question": "What if capacity at facility {F} changed to {V}?",
        "placeholders": {
            "F": facilities,
            "V": list(range(80, 121)) # 80 to 120 units
        },
        "code_type": "ADD DATA",
        "code_template": "capacities[{F}] = {V}",
        "scenario_type": "capacity-change-facility-int"
    },
    {
        "question": "What if all capacities changed by {P}%?",
        "placeholders": {
            "P": list(range(-20, 21))
        },
        "code_type": "ADD DATA",
        "code_template": "capacities = [c * (1 + {P}/100) for c in capacities]",
        "scenario_type": "capacity-pct-change-all"
    },
    {
        "question": "What if fixed cost at facility {F} changed by {P}%?",
        "placeholders": {
            "F": facilities,
            "P": list(range(-20, 21))
        },
        "code_type": "ADD DATA",
        "code_template": "fixed_costs[{F}] = fixed_costs[{F}] * (1 + {P}/100)",
        "scenario_type": "fixed-cost-change-facility-pct"
    },
    {
        "question": "What if fixed cost at facility {F} changed to {V}?",
        "placeholders": {
            "F": facilities,
            "V": list(range(150, 201)) # 150 to 200 units
        },
        "code_type": "ADD DATA",
        "code_template": "fixed_costs[{F}] = {V}",
        "scenario_type": "fixed-cost-change-facility-int"
    },
    {
        "question": "What if all fixed costs changed by {P}%?",
        "placeholders": {
            "P": list(range(-20, 21))
        },
        "code_type": "ADD DATA",
        "code_template": "fixed_costs = [f * (1 + {P}/100) for f in fixed_costs]",
        "scenario_type": "fixed-cost-pct-change-all"
    },
    {
        "question": "What if transportation cost from facility {F} to customer {C} changed by {P}%?",
        "placeholders": {
            "F": facilities,
            "C": customers,
            "P": list(range(-20, 21))
        },
        "code_type": "ADD DATA",
        "code_template": "transportation_costs[{F}][{C}] = transportation_costs[{F}][{C}] * (1 + {P}/100)",
        "scenario_type": "transportation-cost-change-pct"
    },
    {
        "question": "What if transportation cost from facility {F} to customer {C} changed to {V}?",
        "placeholders": {
            "F": facilities,
            "C": customers,
            "V": list(range(10, 21)) # 10 to 20 units
        },
        "code_type": "ADD DATA",
        "code_template": "transportation_costs[{F}][{C}] = {V}",
        "scenario_type": "transportation-cost-change-int"
    },
    {
        "question": "Force facility {F} to be open.",
        "placeholders": {
            "F": facilities
        },
        "code_type": "ADD CONSTRAINT",
        "code_template": "model += facility_open[{F}] == 1, \"Force_Open_Facility_{F}\"",
        "scenario_type": "force-open-facility"
    },
    {
        "question": "Force facility {F} to be closed.",
        "placeholders": {
            "F": facilities
        },
        "code_type": "ADD CONSTRAINT",
        "code_template": "model += facility_open[{F}] == 0, \"Force_Close_Facility_{F}\"",
        "scenario_type": "force-close-facility"
    }
]

# 2) Generator for all scenario tasks
def generate_all_scenario_tasks(scenario_macros, customers, facilities):
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
    run_id     = f"cflp_bench_{time.strftime('%Y%m%d_%H%M%S')}_{seq}"

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
            if 'demands' in locals_dict: parameters['demands'] = locals_dict['demands']
            if 'capacities' in locals_dict: parameters['capacities'] = locals_dict['capacities']
            if 'fixed_costs' in locals_dict: parameters['fixed_costs'] = locals_dict['fixed_costs']
            if 'transportation_costs' in locals_dict: parameters['transportation_costs'] = locals_dict['transportation_costs']
            # Add more common parameter names as needed for CFLP if they exist in the model's locals_dict
            if 'customers' in locals_dict: parameters['customers'] = locals_dict['customers']
            if 'facilities' in locals_dict: parameters['facilities'] = locals_dict['facilities']


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

BATCH_SIZE = 1000 # Process 1000 scenarios at a time to manage memory

base_model_code = _read_source_code(default_model_path)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    processed_count = 0
    
    # Use tqdm with a generator for tasks
    task_generator = generate_all_scenario_tasks(scenario_macros, customers, facilities)
    
    pbar = tqdm(total=total_scenarios, desc="Running CFLP Benchmark Scenarios")
    
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
            elif csv_log_filepath.endswith('.csv'):
                write_run_data_to_csv(batch_results, csv_log_filepath)
            else:
                print("Warning: No valid log file extension found. Results not saved for this batch.")
        
        processed_count += len(batch_results)
        
    pbar.close()

print(f"Benchmark complete: {processed_count} runs logged.")
print(f"Results saved to Parquet: {parquet_log_filepath}")
print(f"Results saved to CSV (if applicable): {csv_log_filepath}")
