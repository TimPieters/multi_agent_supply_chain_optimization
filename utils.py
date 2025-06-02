import json
from pulp import LpStatus, LpStatusOptimal, LpStatusInfeasible, LpProblem, LpVariable
import time
import os
import traceback
import re
import io
import pulp
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from treed import TreeD
from pyscipopt import Model
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from config import MODEL_FILE_PATH, MODEL_DATA_PATH, MODEL_PARAMETERS # Import from config

import logging

# --- extracting and modifying model code. ---

# Placeholder in the source code where constraints will be inserted
DATA_CODE_STR = "### DATA MANIPULATION CODE HERE ###"
# Placeholder in the source code where constraints will be inserted
CONSTRAINT_CODE_STR = "### CONSTRAINT CODE HERE ###"

def _read_source_code(file_path: str) -> str:
    """
    Reads the source code of a Python model file.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        str: The source code as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        raise ValueError(f"Error reading the file '{file_path}': {e}")


def _replace(src_code: str, old_code: str, new_code: str) -> str:
    """
    Replaces an old code snippet with new code inside a source string.

    Args:
        src_code (str): The source code to modify.
        old_code (str): The code block to be replaced.
        new_code (str): The new code block to insert.

    Returns:
        str: The modified source code.
    """
    # Escape special characters in old_code to match literally
    old_code_escaped = re.escape(old_code)

    # Find the correct indentation level
    pattern = rf"(\s*){old_code_escaped}"
    match = re.search(pattern, src_code)

    if not match:
        raise ValueError(f"The specified old_code was not found in the source code.")

    head_spaces = match.group(1)  # Capture leading spaces
    indented_new_code = "\n".join([head_spaces + line for line in new_code.split("\n")])

    # Replace the old code with the correctly formatted new code
    return re.sub(pattern, indented_new_code, src_code)

def _run_with_exec(src_code: str, run_id: str = None, log_filepath: str = 'utils_main_run_log.csv') -> dict:
    """
    Executes a dynamically modified PuLP model and returns the locals dictionary.

    Args:
        src_code (str): The source code containing the modified PuLP model.
        run_id (str, optional): The run ID (not used directly in this function, but kept for signature consistency).
        log_filepath (str, optional): Path to the CSV file for logging (not used directly in this function, but kept for signature consistency).

    Returns:
        dict: The locals dictionary after executing the model, or an empty dictionary if an error occurs.
    """
    logging.info("Running optimization model...")  

    locals_dict = {}
    locals_dict.update(globals())  
    locals_dict.update(locals())   

    try:
        logging.info("Executing model source code...")  
        start_time = time.time() # Start timing
        exec(src_code, locals_dict, locals_dict)
        end_time = time.time() # End timing
        execution_time = end_time - start_time # Calculate execution time
        locals_dict['execution_time'] = execution_time # Store execution time in locals_dict

        logging.info("Model execution completed.")  
        
        return locals_dict

    except Exception as e:
        logging.error("Execution Error:", exc_info=True)  
        return {} # Return empty dict on error

def _get_optimization_result(locals_dict: dict) -> dict:
    """
    Extracts results from a solved PuLP optimization model.

    Args:
        locals_dict (dict): Dictionary containing execution context with `model` and `variables`.

    Returns:
        dict: A dictionary containing:
            - 'status': Solver status (Optimal, Infeasible, etc.).
            - 'solution': Non-zero decision variable values.
            - 'total_cost': Objective function value if solved optimally.
    """
    logging.info("Extracting optimization results...")  

    if "model" not in locals_dict:
        logging.error("`model` not found in execution context.")
        return {"status": "Error", "message": "model not found in execution context."}

    model = locals_dict["model"]
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)

    result = {
        "status": LpStatus[status],
        "raw_status": status,  # PuLP’s internal status code
        "solution": {},
        "total_cost": None
    }

    # Check if the model is infeasible and return immediately
    if status == LpStatusInfeasible:
        result["message"] = "The model is infeasible. The constraints are conflicting."
        return result

    if status == LpStatusOptimal:
        result["solution"] = {
            var.name: var.value() for var in model.variables()
            if var.value() is not None and var.value() >= 0
        }
        result["total_cost"] = model.objective.value()

    logging.info("Optimization results extracted.")  
    return result


def format_constraint_input(constraint_code: str) -> str:
    """
    Ensures the constraint is formatted correctly before inserting it into the model.
    
    Args:
        constraint_code (str): The constraint generated by the agent.
    
    Returns:
        str: A properly formatted constraint statement.
    """
    logging.debug(f"CONSTRAINT CODE {constraint_code}")
    # Ensure there are no unnecessary backticks or formatting issues
    constraint_code = constraint_code.strip().strip("`")
    logging.debug(f"STRIPPED CONSTRAINT CODE {constraint_code}")
    # Remove backticks (`, ```) and strip enclosing quotes if the whole string is quoted
    if constraint_code.startswith(("'", '"', "`")) and constraint_code.endswith(("'", '"', "`")):
        constraint_code = constraint_code[1:-1].strip()

        # Remove unmatched trailing or leading quote
    if constraint_code.endswith('"') and not constraint_code.startswith('"'):
        constraint_code = constraint_code[:-1].strip()
    if constraint_code.startswith('"') and not constraint_code.endswith('"'):
        constraint_code = constraint_code[1:].strip()

    # Ensure the constraint is a valid Python statement
    if not constraint_code.startswith("model +="):
        constraint_code = f"model += {constraint_code}"
    
    return constraint_code

def _clean_agent_code(raw_code: str, code_type: str) -> str:
    """
    Cleans syntax issues from LLM-generated code, adds prefix for constraints.

    Args:
        raw_code (str): Agent output string.
        code_type (str): One of 'ADD DATA' or 'ADD CONSTRAINT'.

    Returns:
        str: Clean and executable Python code.
    """
    code = raw_code.strip().strip("`")

    if code.startswith(("'", '"')) and code.endswith(("'", '"')):
        code = code[1:-1].strip()

    if code.endswith('"') and not code.startswith('"'):
        code = code[:-1].strip()
    if code.startswith('"') and not code.endswith('"'):
        code = code[1:].strip()

    if code_type == "ADD CONSTRAINT" and not code.startswith("model +="):
        code = f"model += {code}"

    return code


def _apply_model_modification(source_code: str, operations: dict) -> str:
    """
    Modifies the model's source code by inserting agent-generated data or constraint blocks.

    Args:
        source_code (str): Original code as string.
        operations (dict): Dictionary with keys like 'ADD DATA' or 'ADD CONSTRAINT'
                           and string or list-of-strings as values.

    Returns:
        str: Modified source code with inserted code blocks.
    """
    updated_code = source_code

    if not isinstance(operations, dict):
        raise ValueError("Operations must be a dictionary with keys like 'ADD DATA' or 'ADD CONSTRAINT'.")

    for op_type, code_blocks in operations.items():
        if not isinstance(code_blocks, list):
            code_blocks = [code_blocks]

        for block in code_blocks:
            cleaned_block = block #_clean_agent_code(block, op_type)

            if op_type == "ADD DATA":
                updated_code = _replace(updated_code, DATA_CODE_STR, cleaned_block)
            elif op_type == "ADD CONSTRAINT":
                updated_code = _replace(updated_code, CONSTRAINT_CODE_STR, cleaned_block)
            else:
                raise ValueError(f"Unsupported operation: {op_type}")

    return updated_code

def modify_and_run_model(modification_json: dict, model_file_path: str, model_data_path: str, run_id: str = None, log_filepath: str = None) -> dict:
    """
    Applies a structured modification (e.g., adding data or constraints), executes the supply chain model, and returns results.

    Args:
        modification_json (dict or str): A JSON specifying the operation(s).
        model_file_path (str): The path to the model's Python file.
        model_data_path (str): The path to the model's JSON data file.
        run_id (str, optional): The run ID to use for logging. If None, a default will be generated.
        log_filepath (str, optional): Path to the CSV file for logging. Defaults to None.

    Returns:
        dict: The optimization results, or an error message/empty dict if an error occurs.
    """
    try:
        if isinstance(modification_json, str):
            modification_json = modification_json.strip().strip("`")
            try:
                modification_json = json.loads(modification_json)
            except json.JSONDecodeError as json_err:
                return {"status": "Error", "message": f"JSON decoding error: {json_err.msg}. Please ensure keys and strings use double quotes. Make sure you don't include ```json ... ``` in the tool input."}

        if not isinstance(modification_json, dict):
            raise ValueError("Parsed input is not a dictionary. Ensure proper JSON format with double quotes.")

        model_code = _read_source_code(model_file_path)

        # Inject DATA_FILE_PATH into the model code
        data_path_injection = f'DATA_FILE_PATH = "{model_data_path}"\n'
        model_code_with_data_path = data_path_injection + model_code

        modified_code = _apply_model_modification(model_code_with_data_path, modification_json)
        
        # Execute the modified code and get the locals dictionary
        locals_dict = _run_with_exec(modified_code, run_id=run_id, log_filepath=log_filepath)
        
        if not locals_dict or "model" not in locals_dict:
            return {"status": "Error", "message": "Model execution failed or 'model' not found in locals_dict."}

        model = locals_dict["model"]
        execution_time = locals_dict.get('execution_time')

        # Retrieve results
        result = _get_optimization_result(locals_dict)

        # Dynamically define parameters based on the current model
        parameters = {}
        current_model_params = MODEL_PARAMETERS.get(MODEL_FILE_PATH, [])
        for param_name in current_model_params:
            parameters[param_name] = locals_dict.get(param_name)
        
        # Fallback for models not explicitly listed in MODEL_PARAMETERS
        if not parameters and hasattr(model, 'parameters'): # If model itself has a parameters attribute
            parameters = model.parameters
        elif not parameters: # As a last resort, try to capture common data structures
            if 'demands' in locals_dict: parameters['demands'] = locals_dict['demands']
            if 'capacities' in locals_dict: parameters['capacities'] = locals_dict['capacities']
            if 'fixed_costs' in locals_dict: parameters['fixed_costs'] = locals_dict['fixed_costs']
            if 'transportation_costs' in locals_dict: parameters['transportation_costs'] = locals_dict['transportation_costs']
            if 'supply' in locals_dict: parameters['supply'] = locals_dict['supply']
            if 'costs' in locals_dict: parameters['costs'] = locals_dict['costs']
            if 'routes' in locals_dict: parameters['routes'] = locals_dict['routes']
            # Add more common parameter names as needed

        # Log the run
        # Use provided run_id or generate a default if not provided (for standalone tests)
        actual_run_id = run_id if run_id is not None else f'unspecified_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        run_data = _prepare_run_data(
            model=model,
            parameters=parameters,
            run_id=actual_run_id,
            model_file_path=MODEL_FILE_PATH, # Pass the dynamically loaded model file path
            model_data_path=MODEL_DATA_PATH, # Pass the dynamically loaded model data path
            execution_time=execution_time # Pass execution time
        )

        if log_filepath is not None:
            if run_id is not None:
                logging.info(f"Logging run '{actual_run_id}' to '{log_filepath}'...")
            else:
                logging.info(f"When ID not specified, logging unspecified run '{actual_run_id}' to '{log_filepath}'...")
            write_run_data_to_csv(
                run_data_list = [run_data],
                log_filepath = log_filepath)
        else:
            if run_id is not None:
                logging.warning(f"Log file path not specified. Skipping logging run '{actual_run_id}' to CSV.")
            else:
                logging.warning(f"Log file path not specified. Skipping logging unspecified run '{actual_run_id}' to CSV.")
        
        # Display results
        logging.info("Optimization Completed.")
        # logging.info(f"Status: {result['status']}")
        # logging.info(f"Total Cost: {result['total_cost']}")
        # logging.info("Solution:")
        # for key, value in result['solution'].items():
        #     logging.info(f"  - {key}: {value}")

        return result

    except Exception as e:
        logging.error(f"Error during modification or execution: {str(e)}", exc_info=True)
        return {"status": "Error", "message": f"Error during modification or execution: {str(e)}"}
    

# ======== PuLP Model Util Functions =========

def get_objective_function(model):
    """
    Returns the objective function of the PuLP model.
    
    Args:
        model: A PuLP LpProblem instance
        
    Returns:
        The objective function object
    """
    return model.objective

def get_constraints(model):
    """
    Returns all constraints in the PuLP model.
    
    Args:
        model: A PuLP LpProblem instance
        
    Returns:
        A dictionary of constraints where keys are constraint names
    """
    return model.constraints

def get_variables(model):
    """
    Returns all variables in the PuLP model.
    
    Args:
        model: A PuLP LpProblem instance
        
    Returns:
        A list of all variables in the model
    """
    return model.variables()

def get_mps_format(model):
    """
    Exports the model to MPS format and returns the filename.
    
    Args:
        model: A PuLP LpProblem instance
        
    Returns:
        The filename of the exported MPS file
    """
    # Export the model to MPS format
    filename = f"{model.name}.mps"
    model.writeMPS(filename)
    
    logging.info(f"Model exported to MPS format: {filename}")
    return filename

def get_lp_format(model):
    """
    Exports the model to LP format and returns the filename.
    
    Args:
        model: A PuLP LpProblem instance
        
    Returns:
        The filename of the exported LP file
    """
    # Export the model to LP format
    filename = f"{model.name}.lp"
    model.writeLP(filename)
    
    logging.info(f"Model exported to LP format: {filename}")
    return filename

def build_model_from_lp(lp_file):
    """
    Builds a PuLP model from an LP file.
    
    Args:
        lp_file: Path to the LP file
        
    Returns:
        A new PuLP LpProblem instance
    """
    # Create a new empty model
    model = pulp.LpProblem(name=lp_file.replace('.lp', ''))
    
    logging.info(f"Model built from LP file: {lp_file}")
    return model

def build_model_from_mps(mps_file):
    """
    Builds a PuLP model from an MPS file.
    
    Args:
        mps_file: Path to the MPS file
        
    Returns:
        A new PuLP LpProblem instance
    """
    variables_dict, model = pulp.LpProblem.fromMPS(mps_file, sense=pulp.LpMinimize)
    
    logging.info(f"Model built from MPS file: {mps_file}")
    return model


def pulp_model_to_networkx(model):
    """
    Convert a PuLP model to a NetworkX bipartite graph for analysis and visualization.

    Args:
        model: A PuLP LpProblem instance

    Returns:
        G: A NetworkX graph representing the model structure
    """
    # Create an empty graph
    G = nx.Graph()

    # Get variables and constraints
    variables = model.variables()
    constraints = model.constraints

    # Add variable nodes
    for var in variables:
        var_type = "binary" if var.cat == pulp.LpBinary else "continuous"
        G.add_node(var.name, bipartite=0, type='variable', var_type=var_type)

    # Add constraint nodes
    for constraint_name, constraint in constraints.items():
        G.add_node(constraint_name, bipartite=1, type='constraint')

    # Add objective node
    objective_name = "Objective"
    G.add_node(objective_name, bipartite=1, type='objective')

    # Build edges between variables and constraints
    for constraint_name, constraint in constraints.items():
        # constraint.expr is an LpAffineExpression
        for var, coef in constraint.expr.items():
            # Add edge only if coefficient is nonzero
            if abs(coef) > 1e-15:
                G.add_edge(var.name, constraint_name, weight=coef)

    # Build edges for the objective function
    # model.objective is also an LpAffineExpression
    for var, coef in model.objective.items():
        if abs(coef) > 1e-15:
            G.add_edge(var.name, objective_name, weight=coef)

    return G

def plot_pulp_model_graph(G):
    """
    Plot the bipartite PuLP model graph using NetworkX and matplotlib.

    Args:
        G: A NetworkX graph returned by pulp_model_to_networkx.
    """
    # Separate nodes by 'bipartite' set or by 'type' attribute
    variable_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    other_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]

    # Use a bipartite layout, placing all 'bipartite=0' nodes on one side
    pos = nx.bipartite_layout(G, variable_nodes)

    # Create the plot
    plt.figure()
    nx.draw(G, pos, with_labels=True)

    # Draw coefficient labels on edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show the figure
    plt.show()


def get_model_stats(G):
    """
    Calculate statistics from a model graph.
    
    Args:
        G: NetworkX graph of the model
        
    Returns:
        stats: Dictionary of model statistics
    """
    variable_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'variable']
    binary_vars = [n for n, d in G.nodes(data=True) 
                  if d.get('type') == 'variable' and d.get('var_type') == 'binary']
    continuous_vars = [n for n, d in G.nodes(data=True) 
                      if d.get('type') == 'variable' and d.get('var_type') == 'continuous']
    
    constraint_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'constraint']
    
    stats = {
        "num_variables": len(variable_nodes),
        "num_binary_vars": len(binary_vars),
        "num_continuous_vars": len(continuous_vars),
        "num_constraints": len(constraint_nodes),
        "num_connections": G.number_of_edges(),
        "graph_density": nx.density(G),
        "average_constraint_degree": sum(G.degree(c) for c in constraint_nodes) / len(constraint_nodes) if constraint_nodes else 0,
        "average_variable_degree": sum(G.degree(v) for v in variable_nodes) / len(variable_nodes) if variable_nodes else 0
    }
    
    return stats


def extract_model_data(scip_model: Model):
    """
    Extracts from a solved PySCIPOpt model:
      • model_props: {'name','objective'}
      • variables: [{'name','value','obj_coef'}, …]
      • constraints: linear only: [{'name','lhs','rhs','sense','shadow_price'}, …]
    """
    # 1) Pull the current best solution
    sol = scip_model.getBestSol()

    # 2) Model‐level info
    model_props = {
        'name':      "SupplyChainModel",
        'objective': scip_model.getObjVal()
    }

    # 3) Variables
    variables = []
    for var in scip_model.getVars():
        variables.append({
            'name':     var.name,
            'value':    scip_model.getSolVal(sol, var),
            'obj_coef': var.getObj()
        })

    # 4) *Only* linear constraints
    constraints = []
    inf = scip_model.infinity()
    for cons in scip_model.getConss():
        if not cons.isLinear():
            # skip varbounds, logic, indicators, etc.
            continue

        lhs = scip_model.getLhs(cons)
        rhs = scip_model.getRhs(cons)

        # infer sense
        if lhs == rhs:
            sense = '=='
        elif lhs <= -inf and rhs <  inf:
            sense = '<='
        elif lhs >  -inf and rhs >= inf:
            sense = '>='
        else:
            sense = '<='

        shadow = scip_model.getDualsolLinear(cons)

        coefs_in_cons = scip_model.getValsLinear(cons)

        constraints.append({
            'name':         cons.name,
            'lhs':          lhs,
            'rhs':          rhs,
            'sense':        sense,
            'shadow_price': shadow,
            'coefs':        coefs_in_cons
        })

    return model_props, variables, constraints


# Load environment variables from .env into os.environ
# find and load .env automatically
dotenv_path = find_dotenv()
if not dotenv_path:
    raise RuntimeError("Cannot find .env—make sure it's named '.env' and in your project root")
load_dotenv(dotenv_path, override=False)

def load_to_neo4j(model_props, variables, constraints):
    """
    Reads NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD from .env
    and loads the extracted model into Neo4j.
    """
    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd  = os.getenv("NEO4J_PASSWORD")

    if not (uri and user and pwd):
        raise RuntimeError("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as session:
        # 1) Create/Merge Model node
        session.run(
            "MERGE (m:Model {name:$name}) "
            "SET m.objective = $objective",
            name=model_props['name'],
            objective=model_props['objective']
        )

        # 2) Variables
        for v in variables:
            session.run(
                "MERGE (var:Variable {name:$name}) "
                "SET var.value = $value, var.obj_coef = $obj_coef "
                "WITH var "
                "MATCH (m:Model {name:$model_name}) "
                "MERGE (m)-[:HAS_VAR]->(var)",
                name=v['name'],
                value=v['value'],
                obj_coef=v['obj_coef'],
                model_name=model_props['name']
            )

        # 3) Constraints
        for c in constraints:
            session.run(
                "MERGE (con:Constraint {name:$cname}) "
                "SET con.sense = $sense, con.rhs = $rhs, con.shadow_price = $shadow_price "
                "WITH con "
                "MATCH (m:Model {name:$model_name}) "
                "MERGE (m)-[:HAS_CONS]->(con)",
                cname=c['name'],
                sense=c['sense'],
                rhs=c['rhs'],
                shadow_price=c['shadow_price'],
                model_name=model_props['name']
            )

            # 4) APPEARS_IN edges with coeffs
            for var_name, coef in c.get('coefs', {}).items():
                session.run(
                    "MATCH (var:Variable {name:$vname}), (con:Constraint {name:$cname}) "
                    "MERGE (var)-[r:APPEARS_IN]->(con) "
                    "SET r.coef = $coef",
                    vname=var_name,
                    cname=c['name'],
                    coef=coef
                )

    driver.close()


# ======== SCIP Branching Tree Visualization =========

def visualize_bnb_treed(lp_path: str,
                       nodelimit: int = 2000,
                       showcuts: bool = False,
                       use_3d: bool = True):
    """
    Visualize the B&B tree of the MILP at lp_path using TreeD.
    
    Args:
      lp_path   – path to the .lp or .mps file
      nodelimit – maximum number of nodes to visualize
      showcuts  – whether to color nodes by cut count
      use_3d    – if True, use the 3D projection (default); otherwise, classic 2D.
    
    Returns:
      A Plotly Figure object.
    """
    td = TreeD(
        probpath=lp_path,
        nodesize=nodelimit,
        showcuts=showcuts
    )
    td.solve()               # runs PySCIPOpt solve + data collection
    if use_3d:
        fig = td.draw()      # 3D projection of LP solutions + tree height
    else:
        fig = td.draw2d()    # classic abstract tree
    return fig


def _prepare_run_data(model: LpProblem, parameters: dict, run_id: str, 
                      model_file_path: str = None, model_data_path: str = None, execution_time: float = None) -> dict:
    """
    Prepares a dictionary of data for a single PuLP model run.

    Args:
        model (LpProblem): The solved PuLP model object.
        parameters (dict): A dictionary of the input parameters used for this run.
        run_id (str): A unique identifier for this run.
        model_file_path (str, optional): Path to the model's Python file. Defaults to None.
        model_data_path (str, optional): Path to the model's JSON data file. Defaults to None.
        execution_time (float, optional): The time taken to execute the model in seconds. Defaults to None.
        
    Returns:
        dict: A dictionary containing all relevant data for the run.
    """
    status = LpStatus[model.status]
    objective_value = None
    variables_dict = {}
    
    if status == 'Optimal':
        objective_value = model.objective.value()
        variables_dict = {v.name: v.varValue for v in model.variables()}
    elif model.objective is not None:
         try:
             objective_value = model.objective.value()
         except AttributeError:
             objective_value = None
         variables_dict = {v.name: v.varValue for v in model.variables() if v.varValue is not None}

    constraints_dict = {name: str(c) for name, c in model.constraints.items()}

    run_data = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'status': status,
        'objective_value': objective_value,
        'model_file_path': model_file_path,
        'model_data_path': model_data_path,
        'parameters': json.dumps(parameters),
        'constraints': json.dumps(constraints_dict),
        'variables': json.dumps(variables_dict),
        'pulp_model_execution_time': execution_time
    }
    return run_data

def write_run_data_to_csv(run_data_list: list[dict], log_filepath: str):
    """
    Writes a list of run data dictionaries to a Pandas DataFrame and saves it to a CSV file.
    If the file exists, new data is appended. If not, a new file is created.

    Args:
        run_data_list (list[dict]): A list of dictionaries, where each dictionary represents a single run's data.
        log_filepath (str): Path to the CSV file for logging.
    """
    if not run_data_list:
        logging.info("No run data to write.")
        return

    try:
        new_runs_df = pd.DataFrame(run_data_list)

        # Load existing DataFrame or create a new one
        if os.path.exists(log_filepath):
            df = pd.read_csv(log_filepath)
            df = pd.concat([df, new_runs_df], ignore_index=True)
        else:
            df = new_runs_df

        # Save updated DataFrame
        df.to_csv(log_filepath, index=False)
        logging.info(f"Successfully logged {len(run_data_list)} runs to {log_filepath}")

    except Exception as e:
        logging.error(f"Error writing run data to CSV: {e}", exc_info=True)

def write_run_data_to_parquet(run_data_list: list[dict], log_filepath: str):
    """
    Writes a list of run data dictionaries to a Pandas DataFrame and saves it to a Parquet file.
    If the file exists, new data is appended. If not, a new file is created.

    Args:
        run_data_list (list[dict]): A list of dictionaries, where each dictionary represents a single run's data.
        log_filepath (str): Path to the Parquet file for logging.
    """
    if not run_data_list:
        logging.info("No run data to write.")
        return

    try:
        new_runs_df = pd.DataFrame(run_data_list)

        # Load existing DataFrame or create a new one
        if os.path.exists(log_filepath):
            # Read existing parquet file
            df = pd.read_parquet(log_filepath)
            # Concatenate new data
            df = pd.concat([df, new_runs_df], ignore_index=True)
        else:
            df = new_runs_df

        # Save updated DataFrame to parquet
        df.to_parquet(log_filepath, index=False)
        logging.info(f"Successfully logged {len(run_data_list)} runs to {log_filepath}")

    except ImportError:
        logging.error("Error: 'pyarrow' and 'fastparquet' libraries are required to write Parquet files.\nPlease install them using: pip install pyarrow fastparquet")
    except Exception as e:
        logging.error(f"Error writing run data to Parquet: {e}", exc_info=True)


if __name__ == "__main__":
    # Read the source code from file
    print("Reading source code from 'capfacloc_model.py'...")
    src_code = _read_source_code("capfacloc_model.py")
    print(f"Source code read successfully. Length: {len(src_code)} characters")
    
    # Create locals and globals dictionaries
    print("Setting up execution environment...")
    locals_dict = {}
    locals_dict.update(globals())  
    locals_dict.update(locals())   
    print("Execution environment set up successfully")
    
    # Execute the source code
    print("Executing model source code...")
    start_time = time.time() # Start timing
    exec(src_code, locals_dict, locals_dict)
    end_time = time.time() # End timing
    execution_time = end_time - start_time # Calculate execution time
    print("Model source code executed successfully")
    
    # Get the model from the locals dictionary
    print("Retrieving model from execution context...")
    model = locals_dict["model"]
    print(f"Model retrieved successfully: {model.name}")

    # --- Test _prepare_run_data and write_run_data_to_csv using the loaded model ---
    print("\n--- Testing _prepare_run_data and write_run_data_to_csv ---")
    # Define baseline parameters (matching capfacloc_model.py)
    parameters = {
        "demands": locals_dict.get("demands"), # Use loaded or default
        "capacities": locals_dict.get("capacities"),
        "fixed_costs": locals_dict.get("fixed_costs"),
        "transportation_costs": locals_dict.get("transportation_costs"),
        "n_customers": locals_dict.get("n_customers"),
        "n_facilities": locals_dict.get("n_facilities")
    }

    # Ensure the model is solved before logging
    print("Solving the loaded model (if not already solved)...")
    if model.status == pulp.LpStatusNotSolved:
         model.solve(pulp.PULP_CBC_CMD(msg=False))
    print(f"Model status: {LpStatus[model.status]}")

    # Prepare run data
    log_file = 'utils_main_run_log.csv'
    run_id_1 = f'utils_main_test_1_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_data_1 = _prepare_run_data(
        model=model,
        parameters=parameters,
        run_id=run_id_1,
        model_file_path="capfacloc_model.py",
        model_data_path="data/capfacloc_data_10cust_10fac.json",
        execution_time=execution_time
    )

    # Simulate another run
    run_id_2 = f'utils_main_test_2_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_data_2 = _prepare_run_data(
        model=model, # Using the same model for simplicity in test
        parameters=parameters,
        run_id=run_id_2,
        model_file_path="capfacloc_model.py",
        model_data_path="data/capfacloc_data_10cust_10fac.json",
        execution_time=execution_time + 0.5 # Slightly different time
    )

    # Write both runs to CSV in one go
    print(f"Writing prepared run data to '{log_file}'...")
    write_run_data_to_csv([run_data_1, run_data_2], log_file)
    print("--- Logging test complete. Check the log file. ---")
    # --- End of _prepare_run_data and write_run_data_to_csv test ---

    # Existing example usage continues below:
    print("\n---------- OBJECTIVE FUNCTION ----------")
    objective = get_objective_function(model)
    print(f"Objective function: {objective}")
    print(f"Objective function type: {type(objective)}")
    print(f"Objective function name: {objective.name}")
    
    print("\n---------- CONSTRAINTS ----------")
    constraints = get_constraints(model)
    print(f"Number of constraints: {len(constraints)}")
    print(f"Constraint names: {list(constraints.keys())[:5]}... (showing first 5)")
    
    print("\n---------- VARIABLES ----------")
    variables = get_variables(model)
    print(f"Number of variables: {len(variables)}")
    print(f"First 5 variables: {[v.name for v in variables[:5]]}... (showing first 5)")
    
    # print("\n---------- MPS FORMAT ----------")
    # print("Exporting model to MPS format...")
    # mps_file = get_mps_format(model)
    # print(f"Model exported to MPS file: {mps_file}")
    
    # print("\n---------- LP FORMAT ----------")
    # print("Exporting model to LP format...")
    # lp_file = get_lp_format(model)
    # print(f"Model exported to LP file: {lp_file}")
    
    # print("\n---------- BUILDING MODEL FROM LP ----------")
    # print("Building new model from LP file...")
    # new_model_from_lp = build_model_from_lp(lp_file)
    # print(f"Model name: {new_model_from_lp.name}")
    # print(f"Number of constraints: {len(new_model_from_lp.constraints)}")
    # print(new_model_from_lp)
    
    # print("\n---------- BUILDING MODEL FROM MPS ----------")
    # print("Building new model from MPS file...")
    # new_model_from_mps = build_model_from_mps(mps_file)
    # print(f"Model name: {new_model_from_mps.name}")
    # print(f"Number of constraints: {len(new_model_from_mps.constraints)}")
    # # print(new_model_from_mps)
    # # print("Solving model... \n")
    # # new_model_from_mps.solve()

    # print("\nGenerate graph: \n\n")
    # G = pulp_model_to_networkx(new_model_from_mps)
    # plot_pulp_model_graph(G)

    # graph_stats = get_model_stats(G)
    # print(graph_stats)

    # print("\nExtracting MILP data to dictionary...")
    # m = Model()
    # m.readProblem("facility_location_solution.lp")
    # m.optimize()
    # model_props, vars, cons = extract_model_data(m)
    # print("\nExtracted model data:")
    # print("Model Properties:")
    # print(model_props)
    # print("\nVariables:")
    # print(vars)
    # print("\nConstraints:")
    # print(cons)
    # print("\nLoading model data to Neo4j...")
    # load_to_neo4j(model_props, vars, cons)


    # # Visualize the B&B tree
    # fig = visualize_bnb_treed(
    #     lp_path="facility_location_solution.lp",
    #     nodelimit=1000,
    #     showcuts=True,
    #     use_3d=False   # or True for the fancy 3D view
    # )
    # fig.show()        # in Jupyter or a desktop Python session
    # # — or save it —
    # fig.write_image("bnb_tree.png")

    # ========== Logging Run to DataFrame ==========
    # model = locals_dict["model"]
    # solver = pulp.PULP_CBC_CMD(msg=False)
    # status = model.solve(solver)
