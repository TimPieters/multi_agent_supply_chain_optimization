# Import necessary libraries and modules for LangChain V2
import ast
import json
import re
import traceback
from langchain.agents import create_react_agent,AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.tools import Tool
from pulp import LpStatus, LpStatusOptimal, LpStatusInfeasible
import time
import os

# Load environment variables
load_dotenv()

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

# Placeholder in the source code where constraints will be inserted
DATA_CODE_STR = "### DATA MANIPULATION CODE HERE ###"
# Placeholder in the source code where constraints will be inserted
CONSTRAINT_CODE_STR = "### CONSTRAINT CODE HERE ###"

def _insert_code(src_code: str, new_lines: str) -> str:
    """
    Inserts new constraint code into the supply chain model at the designated placeholder.

    Args:
        src_code (str): The full source code as a string.
        new_lines (str): The new constraint code to be inserted.

    Returns:
        str: The modified source code with the new constraint added.
    """
    return _replace(src_code, CONSTRAINT_CODE_STR, new_lines)

def _run_with_exec(src_code: str) -> str:
    """
    Executes a dynamically modified PuLP model and extracts the results.

    Args:
        src_code (str): The source code containing the modified PuLP model.

    Returns:
        str: The optimization result (objective value) or an error message.
    """
    print("\nlog - Running optimization model...")  

    locals_dict = {}
    locals_dict.update(globals())  
    locals_dict.update(locals())   

    try:
        print("\nlog - Executing model source code...")  
        exec(src_code, locals_dict, locals_dict)

        print("\nlog - Model execution completed.")  
        
        # Retrieve results
        result = _get_optimization_result(locals_dict)

        # Display results
        print("\nlog - Optimization Completed.")
        print(f"Status: {result['status']}")
        print(f"Total Cost: {result['total_cost']}")
        print("Solution:")
        for key, value in result['solution'].items():
            print(f"  - {key}: {value}")

        return result

    except Exception as e:
        print("\nExecution Error:", traceback.format_exc())  
        return f"Execution Error:\n{traceback.format_exc()}"

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
    print("\nlog - Extracting optimization results...")  

    if "model" not in locals_dict:
        print("Error: `model` not found in execution context.")
        return {"status": "Error", "message": "model not found in execution context."}

    model = locals_dict["model"]
    status = model.solve()

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
            if var.value() is not None and var.value() > 1e-6
        }
        result["total_cost"] = model.objective.value()

    print("\nlog - Optimization results extracted.")  
    return result


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


source_code = _read_source_code("multi_agent_supply_chain_optimization/simple_model.py")
new_source_code = _read_source_code("multi_agent_supply_chain_optimization/capfacloc_model.py")
original_result = _run_with_exec(new_source_code)

# Define a simple prompt to test the connection
prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    partial_variables={"source_code": new_source_code,
                       "original_result": original_result},
    template="""
    You are an AI assistant for supply chain optimization. You analyze the provided Python optimization model
    and modify it based on the user's questions. You explain solutions from a PuLP Python solver.
    You compare with the original objective value if you have it available. You clearly report the numbers and explain the impact to the user.

    You have access to the following tools:

    {tools}

    Below is the full source code of the supply chain model:
    ---SOURCE CODE---
    ```python
    {source_code}
    ```

    Before the modification, the model had the following results:
    ---ORIGINAL RESULT---
    {original_result}
    ---
    your written code will be added to the line with substring:
    "### DATA MANIPULATION CODE HERE ###"    
    "### CONSTRAINT CODE HERE ###"

    LOOK VERY WELL at these example questions and their answers and codes:
    --- EXAMPLES ---
    "Limit the total supply from supplier 0 to 80 units."
    model += lpSum(variables[0, j] for j in range(len(demand))) <= 80, 'Supply_Limit_Supplier_0'

    "Ensure that Supplier 1 supplies at least 50 units in total."
    model += lpSum(variables[1, j] for j in range(len(demand))) >= 50, 'Minimum_Supply_Supplier_1'
    ---

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
)

# Create a simple tool for testing
def echo_tool(input_text):
    return f"Echo: {input_text}"

echo_tool_obj = Tool(
    name="EchoTool",
    func=echo_tool,
    description="A tool that echoes back the user input."
)

def format_constraint_input(constraint_code: str) -> str:
    """
    Ensures the constraint is formatted correctly before inserting it into the model.
    
    Args:
        constraint_code (str): The constraint generated by the agent.
    
    Returns:
        str: A properly formatted constraint statement.
    """
    print("CONSTRAINT CODE " + constraint_code)
    # Ensure there are no unnecessary backticks or formatting issues
    constraint_code = constraint_code.strip().strip("`")
    print("STRIPPED CONSTRAINT CODE " + constraint_code)
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
            cleaned_block = _clean_agent_code(block, op_type)

            if op_type == "ADD DATA":
                updated_code = _replace(updated_code, DATA_CODE_STR, cleaned_block)
            elif op_type == "ADD CONSTRAINT":
                updated_code = _replace(updated_code, CONSTRAINT_CODE_STR, cleaned_block)
            else:
                raise ValueError(f"Unsupported operation: {op_type}")

    return updated_code


def modify_and_run_model(modification_json: dict) -> str:
    """
    Applies a structured modification (e.g., adding data or constraints), executes the model, and returns results.

    Args:
        modification_json (dict or str): A dictionary or JSON string specifying the operation(s).

    Returns:
        str: The optimization results or error message.
    """
    try:
        if isinstance(modification_json, str):
            modification_json = modification_json.strip().strip("`")
            try:
                modification_json = json.loads(modification_json)
            except json.JSONDecodeError as json_err:
                return f"JSON decoding error: {json_err.msg}. Please ensure keys and strings use double quotes. Make sure you don't include ```json ... ``` in the tool input."

        if not isinstance(modification_json, dict):
            raise ValueError("Parsed input is not a dictionary. Ensure proper JSON format with double quotes.")

        model_code = _read_source_code("multi_agent_supply_chain_optimization/capfacloc_model.py")
        modified_code = _apply_model_modification(model_code, modification_json)
        result = _run_with_exec(modified_code)

        return result

    except Exception as e:
        return f"Error during modification or execution: {str(e)}"

# Define the tool for the agent
modify_model_tool = Tool(
    name="ModifyAndRunModel",
    func=modify_and_run_model,
    description="""
        "Use this tool to modify the model by adding constraints or data. "
        You must provide the input as a valid JSON object using double quotes for both keys and values.
        Example:
        {
        "ADD CONSTRAINT": "model += lpSum(variables[0, j] for j in range(len(demand))) <= 80, \\"Supply_Limit_Supplier_0\\""
        }
        or
        {
        "ADD DATA": "supply = [200, 300, 300]"
        }
        Do not use single quotes or Python-style dictionaries.
        The tool updates and executes the model and returns results."
    """
)

# Add the tool to the tools list
tools = [modify_model_tool]

LOG_FILE = "agent_execution_log.json"

def log_execution(response, log_file=LOG_FILE):
    """
    Logs the agent's execution details in a JSON array stored in a file.
    Each run is appended as a separate object in the array.
    
    The logged details include:
      - 'run_count': A sequential count of runs.
      - 'timestamp': The time when the run was executed.
      - 'user_input': The input question.
      - 'final_output': The agent's final output.
      - 'intermediate_steps': A list of tool interactions, each with:
              - 'tool'
              - 'tool_input'
              - 'tool_output'
    """
    # Create a log entry based on the response.
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": response.get("input", ""),
        "final_output": response.get("output", ""),
        "intermediate_steps": []
    }
    
    for step in response.get("intermediate_steps", []):
        try:
            # Each step is expected to be a tuple: (AgentAction, tool_output)
            action, tool_output = step
            step_entry = {
                "tool": action.tool,
                "tool_input": action.tool_input,
                "tool_output": tool_output
            }
        except Exception as e:
            step_entry = {"step": str(step)}
        log_entry["intermediate_steps"].append(step_entry)
    
    # Read the existing log (if it exists), append the new entry, then write back the full array.
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            if not isinstance(log_data, list):
                log_data = []
        except Exception:
            log_data = []
    else:
        log_data = []
    
    # Append a new run number based on the current log length.
    log_entry["run_count"] = len(log_data) + 1
    log_data.append(log_entry)
    
    # Write the updated log data back to the file.
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Run {log_entry['run_count']} logged to {log_file}.")

def get_last_run(log_file=LOG_FILE):
    """
    Reads the log file and returns the last run log entry as a dictionary.
    
    Returns None if no run entries exist.
    """
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        if isinstance(log_data, list) and log_data:
            return log_data[-1]
    except Exception as e:
        print("Error reading log file:", e)
    return None


def extract_last_run_details(log_file=LOG_FILE):
    """
    Extracts details from the last log entry in the JSON log file.
    Returns a dictionary with:
      - "user_input": The original user input for the last run.
      - "steps": A list of dictionaries, each containing:
            "tool_input": The input given to a tool,
            "tool_output": The output received from that tool.
    If no log entries are found, returns None.
    """
    if not os.path.exists(log_file):
        print("Log file does not exist.")
        return None

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception as e:
        print("Error loading log file:", e)
        return None

    if not isinstance(logs, list) or not logs:
        print("Log file is empty or not in expected format.")
        return None

    # Get the last log entry
    last_run = logs[-1]
    result = {
        "user_input": last_run.get("user_input", ""),
        "steps": []
    }
    
    # Loop over intermediate_steps and pick out tool_input and tool_output.
    for step in last_run.get("intermediate_steps", []):
        step_details = {
            "tool_input": step.get("tool_input", ""),
            "tool_output": step.get("tool_output", "")
        }
        result["steps"].append(step_details)
    
    return result


# Example usage
if __name__ == "__main__":

    # Load pre-trained Large Language Model from OpenAI
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    router_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt = prompt
    )

    router_agent_executor = AgentExecutor(agent=router_agent, tools=tools, return_intermediate_steps=True, verbose=True, handle_parsing_errors=False)

    try:
        #simple_model.py:
        # response = router_agent_executor.invoke({"input": "What happens if supply at supplier 0 is limited to 80?"})
        # response = router_agent_executor.invoke({"input": "What happens if supply at supplier 0 decreases? And what if it's completely zero?"})
        # response = router_agent_executor.invoke({"input": "Limit supplier 0 to 100 and force demand center 2 to receive exactly 90 units."})
        #capfacloc_model.py:
        response = router_agent_executor.invoke({"input": "What happens if the capacity of the first facility is limited to 15?"})
        #response = router_agent_executor.invoke({"input": "Please perform in depth sensitivity analysis. Calculate the impact of multiple scenarios and then report your results."})
        print("API Connection Successful! Response:")
        print(response)
        # Optionally, also write the string representation to a text file
        with open("agent_verbose_log.txt", "w", encoding="utf-8") as f:
            f.write(str(response))
        print("Output:")
        print(response["output"])
        print("Intermediate Steps:")
        print(response["intermediate_steps"])

        # Log this run in the JSON file as an element in an array.
        log_execution(response)
        
        details = extract_last_run_details()
        if details:
            print("User Input:")
            print(details["user_input"])
            print("\nTool Steps:")
            for i, step in enumerate(details["steps"], start=1):
                print(f"Step {i}:")
                print("  Tool Input:", step["tool_input"])
                print("  Tool Output:", step["tool_output"])
        else:
            print("No log details extracted.")

    except Exception as e:
        print("API Connection Failed. Error:")
        print(e)

    # src_code = 'def hello_world():\n    print("Hello, world!")\n\n# Some other code here'
    # old_code = 'print("Hello, world!")'
    # new_code = 'print("Bonjour, monde!")\nprint("Hola, mundo!")'
    # modified_code = _replace(src_code, old_code, new_code)
    # print(modified_code)

    # agent_data_json = {
    #     "ADD DATA": "supply = [300, 300, 300]"
    # }

    # agent_constraint_json = {
    # "ADD CONSTRAINT": "model += lpSum(variables[0, j] for j in range(len(demand))) <= 100, 'Limit_Supplier_0'"
    # }

    # source_code = _read_source_code("multi_agent_supply_chain_optimization/simple_model.py")
    # modified_code = _apply_model_modification(source_code, agent_constraint_json)
    # result = _run_with_exec(modified_code)