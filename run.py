# Import necessary libraries and modules for LangChain V2
import ast
import json
import re
import traceback
from langchain.agents import create_react_agent, AgentExecutor
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
import logging
from utils import _read_source_code, modify_and_run_model # Removed _replace, _run_with_exec, _apply_model_modification as they are internal to modify_and_run_model
from config import MODEL_FILE_PATH, MODEL_DATA_PATH, MODEL_DESCRIPTION_PATH, LOGGING_LEVEL

# Load environment variables
load_dotenv()

# Dynamically load model source code, input data, and model description
source_code = _read_source_code(MODEL_FILE_PATH)
input_data = _read_source_code(MODEL_DATA_PATH)
model_description = _read_source_code(MODEL_DESCRIPTION_PATH)

# Get baseline result by running the model without modifications
# This will use the modify_and_run_model function with an empty modification
logging.info("Running baseline model to get original result...")
original_result_dict = modify_and_run_model({}, MODEL_FILE_PATH, MODEL_DATA_PATH)
original_result = json.dumps(original_result_dict) # Convert dict to string for prompt partial variable
logging.info("Baseline model run complete. Original result: %s", original_result)

# === WHAT-IF AGENT ===

# Define the prompt template for the agent
prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    partial_variables={"source_code": source_code, # Give source_code to agent
                       "original_result": original_result, # Give the baseline result to agent
                       "input_data": input_data, # Give input data to agent
                       "model_description": model_description # Add model description to context
                       },
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

    --- MODEL CONTEXT ---
    {model_description}

    Below is the full source code of the supply chain model:
    ---SOURCE CODE---
    ```python
    {source_code}
    ```
    ---

    --- Input Data ---
    {input_data}
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
# The func needs to be a lambda to pass additional arguments (model_file_path, model_data_path)
modify_model_tool = Tool(
    name="update_model",
    func=lambda modification_json: modify_and_run_model(modification_json, MODEL_FILE_PATH, MODEL_DATA_PATH),
    description="""
        "Use this tool to modify the model by adding constraints or data. "
        You must provide the input as a valid JSON object using double quotes for both keys and values. NEVER add ```json ```
        Example:
        {"ADD CONSTRAINT": "model += lpSum(variables[0, j] for j in range(len(demand))) <= 80, \\"Supply_Limit_Supplier_0\\""}
        or
        {"ADD DATA": "supply = [200, 300, 300]"}
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
    # Configure logging
    logging.basicConfig(level=LOGGING_LEVEL, format='%(levelname)s: %(message)s')

    # Load pre-trained Large Language Model from OpenAI
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

    whatif_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    whatif_agent_executor = AgentExecutor(agent=whatif_agent, tools=tools, return_intermediate_steps=True, verbose=True, handle_parsing_errors=True)

    try:
        # Example usage with dynamic model and data paths
        # To change the model or data, update MODEL_FILE_PATH and MODEL_DATA_PATH in config.py
        print(f"Running agent with model: {MODEL_FILE_PATH} and data: {MODEL_DATA_PATH}")

        # Example questions for capfacloc_model.py:
        # response = whatif_agent_executor.invoke({"input": "What happens if the capacity of the first facility is limited to 15?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the second customer increases by 15 units, raising it from 25 to 40?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the fixed cost of the third facility is increased by 25%?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the fourth customer increases by 30 units, raising it from 18 to 48?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the first customer increases by 50 units, raising it from 20 to 70?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if the demand for the fifth customer increases by 40 units, raising it from 22 to 62?"})

        # Example questions for simple_model.py (if MODEL_FILE_PATH is set to simple_model.py in config.py):
        # response = whatif_agent_executor.invoke({"input": "What happens if supply at supplier 0 is limited to 80?"})
        # response = whatif_agent_executor.invoke({"input": "What happens if supply at supplier 0 decreases? And what if it's completely zero?"})
        # response = whatif_agent_executor.invoke({"input": "Limit supplier 0 to 100 and force demand center 2 to receive exactly 90 units."})

        # Example questions for vrp_model.py:
        response = whatif_agent_executor.invoke({"input": "What happens if the capacity of the vehicle is reduced to 30?"})

        # print("API Connection Successful! Response:")
        # print(response)
        # # Optionally, also write the string representation to a text file
        # with open("agent_verbose_log.txt", "w", encoding="utf-8") as f:
        #     f.write(str(response))
        # print("Output:")
        # print(response["output"])
        # print("Intermediate Steps:")
        # print(response["intermediate_steps"])

        # Log this run in the JSON file as an element in an array.
        log_execution(response)
        
        details = extract_last_run_details()
        # if details:
        #     print("User Input:")
        #     print(details["user_input"])
        #     print("\nTool Steps:")
        #     for i, step in enumerate(details["steps"], start=1):
        #         print(f"Step {i}:")
        #         print("  Tool Input:", step["tool_input"])
        #         print("  Tool Output:", step["tool_output"])
        # else:
        #     print("No log details extracted.")

    except Exception as e:
        print("API Connection Failed. Error:")
        print(e)
