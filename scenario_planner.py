from typing import List
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from coder_agent import CoderAgent
from langchain.tools import Tool
from langchain.agents import create_react_agent,AgentExecutor

# === Scenario Planner ===
def compute_delta(original: float, new: float) -> float:
    return round(new - original, 2)

def build_scenario_log_from_runs(run_logs: List[dict], baseline_obj: float) -> List[str]:
    scenario_log = []
    
    # Accept both tool names: old ("ModifyAndRunModel") and new ("update_model")
    valid_tool_names = {"update_model", "modifyandrunmodel"}
    
    for i, log in enumerate(run_logs):
        user_input = log.get("user_input", "[No Input]")
        steps = log.get("intermediate_steps", [])
        
        # Check if any intermediate step uses one of the valid tool names.
        valid_run = any(
            isinstance(step, dict) and step.get("tool", "").strip().replace(" ", "").lower() in valid_tool_names
            for step in steps
        )
        
        # Skip this log if no step uses a valid tool name.
        if not valid_run:
            continue
        
        # Find the first step that uses a valid tool name.
        valid_step = next(
            (step for step in steps
             if isinstance(step, dict) and step.get("tool", "").strip().replace(" ", "").lower() in valid_tool_names),
            None
        )
        
        if valid_step is None:
            continue
        
        step_output = valid_step.get("tool_output", {})
        
        if step_output.get("status", "").lower() == "infeasible":
            entry = f"{i+1}. {user_input} → Infeasible"
        else:
            total_cost = step_output.get("total_cost", None)
            if total_cost is not None:
                delta = compute_delta(baseline_obj, total_cost)
                entry = f"{i+1}. {user_input} → ΔObj = {delta} (from {baseline_obj} to {round(total_cost, 2)})"
            else:
                entry = f"{i+1}. {user_input} → No cost found"
        
        scenario_log.append(entry)
    
    return scenario_log

# === Chain Scenario Planner ===

# Static description of the model (can be made dynamic later)
BASE_CONTEXT = """
You are working with a facility location supply chain optimization problem.

This is the baseline data:
demands = [20, 25, 30, 18, 22]  # For 5 customers
capacities = [80, 90, 70, 100, 85]  # For 5 facilities
fixed_costs = [150, 180, 160, 170, 155]  # Fixed cost for each facility
transportation_costs = [
    [10, 12, 15, 20, 18],
    [14, 11, 13, 16, 19],
    [13, 17, 12, 14, 15],
    [12, 15, 10, 18, 16],
    [11, 13, 14, 15, 17]
]
n_customers = 5
n_facilities = 5

- There are 5 suppliers and 5 customers.
- Customer demands vary between 10 and 70 units.
- Transport costs vary between 10 and 20.
- There are 5 facilities to choose from, and each has a fixed cost.
- Facilities can either be open or closed.
- The model minimizes total cost, including transport and facility opening costs.
"""

# LangChain prompt for generating new scenario proposals
planner_prompt = PromptTemplate(
    input_variables=["base_context", "scenario_log"],
    template="""
You are a scenario planning AI for supply chain optimization.
Your job is to intelligently explore the sensitivity of the optimization model.
You are *not* just suggesting random changes — your goal is to find scenarios that cause a *large impact* on the total cost or feasibility.
Given past results, propose a scenario likely to have higher impact.


--- MODEL CONTEXT ---
{base_context}

--- PAST SCENARIOS LOG ---
{scenario_log}

Review the scenarios and their objective function impacts (ΔObj). Then propose ONE new scenario that is likely to:
- Significantly increase or decrease the total cost
- Lead to infeasibility
- Reveal something critical about model behavior

Do NOT repeat similar changes (e.g., repeatedly changing the same facility capacity).
Avoid low-impact changes like small cost tweaks.


Use the following FORMAT:

Thought: you should always analyze the previous scenarios, take a deep breath and think about what to do
Scenario: Propose a *new*, *higher-impact* perturbation in one line of natural language.

--- NEW SCENARIO PROPOSAL ---
"""
)

# Set up the LLM engine
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Create LangChain chain
planner_chain = llm | planner_prompt | StrOutputParser()

def generate_new_scenario(scenario_log: list[str], base_context: str = BASE_CONTEXT) -> str:
    """
    Given a formatted list of previous scenario results, generate a new perturbation scenario as natural language.
    """
    joined_log = "\n".join(scenario_log)
    
    # Manually format the prompt
    prompt_str = planner_prompt.format(
        base_context=base_context,
        scenario_log=joined_log
    )

    # Pass the full prompt string to the LLM
    result = llm.invoke(prompt_str)
    
    return result.content.strip()

# === ReAct Agent Scenario Planner ===

# Set up the LLM engine
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Load execution log
with open("agent_execution_log.json", "r") as f:
    run_logs = json.load(f)

# Create a formatted scenario log
baseline_obj = 366.10
scenario_log = build_scenario_log_from_runs(run_logs, baseline_obj)
original_result = "{'status': 'Optimal', 'raw_status': 1, 'solution': {'Open_0': 1.0, 'Open_1': 0.0, 'Open_2': 1.0, 'Open_3': 0.0, 'Open_4': 0.0, 'Serve_0_0': 1.0, 'Serve_0_1': 0.0, 'Serve_0_2': 0.0, 'Serve_0_3': 0.0, 'Serve_0_4': 0.0, 'Serve_1_0': 0.0, 'Serve_1_1': 0.0, 'Serve_1_2': 1.0, 'Serve_1_3': 0.0, 'Serve_1_4': 0.0, 'Serve_2_0': 0.1, 'Serve_2_1': 0.0, 'Serve_2_2': 0.9, 'Serve_2_3': 0.0, 'Serve_2_4': 0.0, 'Serve_3_0': 0.0, 'Serve_3_1': 0.0, 'Serve_3_2': 1.0, 'Serve_3_3': 0.0, 'Serve_3_4': 0.0, 'Serve_4_0': 1.0, 'Serve_4_1': 0.0, 'Serve_4_2': 0.0, 'Serve_4_3': 0.0, 'Serve_4_4': 0.0}, 'total_cost': 366.1}"

# Define a simple prompt to test the connection
prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    partial_variables={"original_result": original_result, "base_context": BASE_CONTEXT, "scenario_log": scenario_log},
    template="""
    You are a scenario planning AI for supply chain optimization.
    Your job is to intelligently explore the sensitivity of the optimization model.
    You are *not* just suggesting random changes — your goal is to find scenarios that cause a *large impact* on the total cost or feasibility.
    Given past results, you propose and execute scenarios likely to have higher impact.

    Review the scenarios and their objective function impacts (ΔObj). Then propose ONE new scenario that is likely to:
        - Significantly increase or decrease the total cost
        - Lead to infeasibility
        - Reveal something critical about model behavior

        Do NOT repeat similar changes (e.g., repeatedly changing the same facility capacity).
        Avoid low-impact changes like small cost tweaks.

    You have access to the following tools:
    {tools}

    Before the modification, the model had the following results:
    ---ORIGINAL RESULT---
    {original_result}
    ---

    --- MODEL CONTEXT ---
    {base_context}

    --- PAST SCENARIOS LOG ---
    {scenario_log}

    --- FORMAT ---
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the tool to use, MUST BE exactly one of [{tool_names}] For example: run_scenario
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

def run_scenario(input_str: str):
    coder_agent = CoderAgent()
    result = coder_agent.run(input_str)
    return result

run_scenario = Tool(
    name="run_scenario",
    func=run_scenario,
    description="""Runs a scenario based on a natural language input string"""
)

tools = [run_scenario]

scenario_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt = prompt
)

scenario_agent_executor = AgentExecutor(agent=scenario_agent, tools=tools, return_intermediate_steps=True, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    result = scenario_agent_executor.invoke({"input": "Perform sensitivity analysis", "agent_scratchpad": ""})
    print(result)