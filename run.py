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
from langchain_core.tools import tool
from pulp import LpStatus, LpStatusOptimal, LpStatusInfeasible
import time
import os
from typing import List
import json
from langchain.agents.tools import InvalidTool
from utils import _replace, _read_source_code, _run_with_exec, _apply_model_modification, modify_and_run_model

# Load environment variables
load_dotenv()

# Placeholder in the source code where constraints will be inserted
DATA_CODE_STR = "### DATA MANIPULATION CODE HERE ###"
# Placeholder in the source code where constraints will be inserted
CONSTRAINT_CODE_STR = "### CONSTRAINT CODE HERE ###"


source_code = _read_source_code("multi_agent_supply_chain_optimization/simple_model.py")
new_source_code = _read_source_code("multi_agent_supply_chain_optimization/capfacloc_model.py")
original_result = "{'status': 'Optimal', 'raw_status': 1, 'solution': {'Open_0': 1.0, 'Open_1': 0.0, 'Open_2': 1.0, 'Open_3': 0.0, 'Open_4': 0.0, 'Serve_0_0': 1.0, 'Serve_0_1': 0.0, 'Serve_0_2': 0.0, 'Serve_0_3': 0.0, 'Serve_0_4': 0.0, 'Serve_1_0': 0.0, 'Serve_1_1': 0.0, 'Serve_1_2': 1.0, 'Serve_1_3': 0.0, 'Serve_1_4': 0.0, 'Serve_2_0': 0.1, 'Serve_2_1': 0.0, 'Serve_2_2': 0.9, 'Serve_2_3': 0.0, 'Serve_2_4': 0.0, 'Serve_3_0': 0.0, 'Serve_3_1': 0.0, 'Serve_3_2': 1.0, 'Serve_3_3': 0.0, 'Serve_3_4': 0.0, 'Serve_4_0': 1.0, 'Serve_4_1': 0.0, 'Serve_4_2': 0.0, 'Serve_4_3': 0.0, 'Serve_4_4': 0.0}, 'total_cost': 366.1}"


# === WHAT-IF AGENT ===

# Define a simple prompt to test the connection
prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    partial_variables={"source_code": new_source_code,
                       "original_result": original_result},
    template="""
    You are an AI assistant for supply chain optimization. You analyze the provided Python optimization model
    and modify it based on the user's questions. You explain solutions from a PuLP Python solver.
    You compare with the original objective value if you have it available.
    You clearly report the numbers and explain the impact to the user.

    your written code will be added to the line with substring:
    "### DATA MANIPULATION CODE HERE ###"    
    "### CONSTRAINT CODE HERE ###"

    You have access to the following tools:
    {tools}

    Below is the full source code of the supply chain model:
    ---SOURCE CODE---
    ```python
    {source_code}
    ```
    ---

    Before the modification, the model had the following results:
    ---ORIGINAL RESULT---
    {original_result}
    ---

    Use the following FORMAT:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the tool to use, MUST BE exactly one of [{tool_names}] (for example, "update_model")
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


# Define the tool for the agent
modify_model_tool = Tool(
    name="update_model",
    func=modify_and_run_model,
    description="""
        "Use this tool to modify the model by adding constraints or data. "
        You must provide the input as a valid JSON object using double quotes for both keys and values. NEVER add ```json ```
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
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    whatif_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt = prompt
    )

    whatif_agent_executor = AgentExecutor(agent=whatif_agent, tools=tools, return_intermediate_steps=True, verbose=True, handle_parsing_errors=True)

    try:
        #simple_model.py:
        # response = whatif_agent_executor.invoke({"input": "What happens if supply at supplier 0 is limited to 80?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if supply at supplier 0 decreases? And what if it's completely zero?"})
        # response = whatif_agent_executor.invoke({"input": "Limit supplier 0 to 100 and force demand center 2 to receive exactly 90 units."})
        
        #capfacloc_model.py:
        # response = whatif_agent_executor.invoke({"input": "What happens if the capacity of the first facility is limited to 50?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the second customer increases by 15 units, raising it from 25 to 40?"})       
        # response = whatif_agent_executor.invoke({"input": "What happens if the fixed cost of the third facility is increased by 25%?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the fourth customer increases by 30 units, raising it from 18 to 48?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the first customer increases by 50 units, raising it from 20 to 70?"})
        response = whatif_agent_executor.invoke({"input": "What happens if the capacity of the first facility is limited to 15?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the fifth customer increases by 40 units, raising it from 22 to 62?"})
        #response = whatif_agent_executor.invoke({"input": "Please perform in depth sensitivity analysis. Calculate the impact of multiple scenarios and then report your results."})
        #response = whatif_agent_executor.invoke({"input": "What scenarios did we test in the past?"})
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