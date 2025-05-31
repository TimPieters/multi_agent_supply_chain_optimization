import ast
import json
import re
import traceback
import time
import os
from typing import List, Dict, Any, Tuple, Annotated, TypedDict
from typing_extensions import TypedDict
import operator
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain imports for compatibility
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool

# Environment and utility imports
from dotenv import load_dotenv
from pulp import LpStatus, LpStatusOptimal, LpStatusInfeasible

# Import utility functions from the original codebase
from utils import _replace, _read_source_code, _run_with_exec, _apply_model_modification, modify_and_run_model

# Load environment variables
load_dotenv()

# Constants
DATA_CODE_STR = "### DATA MANIPULATION CODE HERE ###"
CONSTRAINT_CODE_STR = "### CONSTRAINT CODE HERE ###"
LOG_FILE = "agent_execution_log.json"

# Read the source code files
new_source_code = _read_source_code("multi_agent_supply_chain_optimization/capfacloc_model.py")
original_result = "{'status': 'Optimal', 'raw_status': 1, 'solution': {'Open_0': 1.0, 'Open_1': 0.0, 'Open_2': 1.0, 'Open_3': 0.0, 'Open_4': 0.0, 'Serve_0_0': 1.0, 'Serve_0_1': 0.0, 'Serve_0_2': 0.0, 'Serve_0_3': 0.0, 'Serve_0_4': 0.0, 'Serve_1_0': 0.0, 'Serve_1_1': 0.0, 'Serve_1_2': 1.0, 'Serve_1_3': 0.0, 'Serve_1_4': 0.0, 'Serve_2_0': 0.1, 'Serve_2_1': 0.0, 'Serve_2_2': 0.9, 'Serve_2_3': 0.0, 'Serve_2_4': 0.0, 'Serve_3_0': 0.0, 'Serve_3_1': 0.0, 'Serve_3_2': 1.0, 'Serve_3_3': 0.0, 'Serve_3_4': 0.0, 'Serve_4_0': 1.0, 'Serve_4_1': 0.0, 'Serve_4_2': 0.0, 'Serve_4_3': 0.0, 'Serve_4_4': 0.0}, 'total_cost': 366.1}"

# Define the system prompt template
SYSTEM_PROMPT = """
You are an AI assistant for supply chain optimization. You analyze the provided Python optimization model
and modify it based on the user's questions. You explain solutions from a PuLP Python solver.
You compare with the original objective value if you have it available.
You clearly report the numbers and explain the impact to the user.

your written code will be added to the line with substring:
"### DATA MANIPULATION CODE HERE ###"    
"### CONSTRAINT CODE HERE ###"

Before the modification, the model had the following results:
---ORIGINAL RESULT---
{original_result}
---

Below is the full source code of the supply chain model:
---SOURCE CODE---
```python
{source_code}
```
---
"""

# Define the tool for model modification
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
    """,
)

tools=[modify_model_tool]

# Define the state for the graph
class AgentState(TypedDict):
    """State for the supply chain analysis agent"""
    messages: Annotated[List[Any], operator.add]
    input: str
    # intermediate_steps is not populated by ToolNode, removing for clarity

# Logging utilities
def log_execution(state, log_file=LOG_FILE):
    """
    Logs the agent's execution details in a JSON array stored in a file.
    Each run is appended as a separate object in the array.
    """
    # Extract info from state
    # Note: intermediate_steps is no longer tracked directly in the state with ToolNode
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": state.get("input", ""),
        "final_output": get_final_output(state),
        "messages_log": [msg.dict() for msg in state.get("messages", [])] # Log messages instead
    }

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
    log_entry["run_count"] = len(log_data) + 1
    log_data.append(log_entry)

    # Write the updated log data back to the file.
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    print(f"Run {log_entry['run_count']} logged to {log_file}.")
    # Return the state unmodified, logging is a side effect
    return state


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
    Returns a dictionary with key information about the last run.
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
    # Extract messages log instead of intermediate steps
    result = {
        "user_input": last_run.get("user_input", ""),
        "messages_log": last_run.get("messages_log", [])
    }
    return result

def get_final_output(state):
    """Extract the final output from the state"""
    messages = state.get("messages", [])
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
    return ""

# Function to initialize the agent state
def initialize_agent_state(human_input: str) -> AgentState:
    """Initialize the agent state with the human input and system message"""
    formatted_prompt = SYSTEM_PROMPT.format(
        source_code=new_source_code,
        original_result=original_result
    )
    return AgentState(
        messages=[
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=human_input)
        ],
        input=human_input
        # intermediate_steps removed
    )

def agent_step(state: AgentState):
    """
    1) Invokes the LLM with tools bound
    2) Appends the response to the messages list
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    response = llm.bind_tools(tools).invoke(state["messages"])
    # LangGraph will automatically append the ToolMessages if ToolNode is used
    return {"messages": [response]}


# Build the LangGraph state machine
def build_graph():
    """Build the LangGraph state machine using ToolNode"""
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("agent", agent_step)
    tool_node = ToolNode(tools)
    workflow.add_node("call_tool", tool_node)
    workflow.add_node("log", log_execution) # Keep log node

    # Set the entrypoint
    workflow.add_edge(START, "agent")

    # Add the conditional edge from agent to tool_node or log
    # This uses the built-in tools_condition to check the last message
    workflow.add_conditional_edges(
        "agent",
        tools_condition, # LangGraph's built-in condition
        {
            "tools": "call_tool", # If tool calls are present, go to tool_node
            END: "log"          # Otherwise, go to log
        }
    )

    # Add edge from tool_node back to agent
    workflow.add_edge("call_tool", "agent")

    # Add final edge from log to END
    workflow.add_edge("log", END)

    # Compile the graph
    # Add memory for potential multi-turn conversations if needed
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def run_agent(human_input: str):
    """Run the agent with the given input"""
    # Build the graph
    graph = build_graph()
    
    # Initialize the state
    state = initialize_agent_state(human_input)
    
    # Run the graph with configuration for the checkpointer
    # Using a fixed thread_id and increased recursion limit
    config = {"thread_id": "supply-chain-run-1", "recursion_limit": 50}
    result = graph.invoke(state, config=config)

    return result

if __name__ == "__main__":
    try:
        # Example usage
        human_input = "What happens if the capacity of the first facility is limited to 15?"

        print(f"Running analysis for: {human_input}")

        # Build graph and initialize state
        graph = build_graph()
        state = initialize_agent_state(human_input)

        # Define config for invoke/stream
        config = {
            "recursion_limit": 10,
            "configurable": {
                "thread_id": "supply-chain-run-1"
            }
        }

        # Use stream to observe execution step-by-step
        print("\n--- Agent Stream ---")
        final_state = None
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            print(chunk)
            print("---")
            # The last chunk before the stream ends usually contains the final state under '__end__'
            # or the state after the last node ran (e.g., 'log'). We capture the latest state.
            # The structure might vary slightly, adjust if needed based on observed output.
            if END in chunk:
                 # If END is explicitly in the chunk keys, use that state
                 final_state = chunk[END]
            else:
                 # Otherwise, assume the last chunk's value (state after the node ran) is the latest state
                 # Get the state from the last node that ran in the chunk
                 last_node_output = list(chunk.values())[-1]
                 if isinstance(last_node_output, dict) and 'messages' in last_node_output:
                     final_state = last_node_output # Update final_state with the latest valid state dict

        print("--- End Stream ---\n")

        if final_state is None:
             print("Error: Could not determine final state from stream.")
             # Attempt to get final state using invoke as fallback (might hit recursion limit again)
             # print("Attempting invoke as fallback...")
             # result = graph.invoke(state, config=config)
             # final_state = result # invoke returns the final state directly
             exit(1) # Exit if final state is still None


        # Print the final result from the captured state
        print("\nFinal Result (from captured state):")
        print(get_final_output(final_state))

        # Print final messages from the captured state
        print("\nFinal Messages (from captured state):")
        for msg in final_state.get("messages", []):
            print(f"{msg.type}: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  Tool Calls: {msg.tool_calls}")
            print("-" * 20)


        # Extract and print the details from the log file
        details = extract_last_run_details()
        if details:
            print("\nLog Details:")
            print("User Input:", details["user_input"])
            print("\nMessages Log:")
            for i, msg_log in enumerate(details.get("messages_log", []), start=1):
                 print(f"Message {i}: {msg_log}")

    except Exception as e:
        print("Error running agent:")
        print(e)
        traceback.print_exc()
