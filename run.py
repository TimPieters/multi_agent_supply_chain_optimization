# Import necessary libraries and modules for LangChain V2
import re
import traceback
from langchain.agents import create_react_agent,AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Load pre-trained Large Language Model from OpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Define a simple prompt to test the connection
prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template="""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

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
# Create a runnable sequence to test the LLM
chain = llm

# Create a simple tool for testing
def echo_tool(input_text):
    return f"Echo: {input_text}"

echo_tool_obj = Tool(
    name="EchoTool",
    func=echo_tool,
    description="A tool that echoes back the user input."
)

# Add the tool to the tools list
tools = [echo_tool_obj]

router_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt = prompt
)

router_agent_executor = AgentExecutor(agent=router_agent, tools=tools, verbose=True)

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
        raise ValueError(f"❌ The specified old_code was not found in the source code.")

    head_spaces = match.group(1)  # Capture leading spaces
    indented_new_code = "\n".join([head_spaces + line for line in new_code.split("\n")])

    # Replace the old code with the correctly formatted new code
    return re.sub(pattern, indented_new_code, src_code)

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
    locals_dict = {}
    locals_dict.update(globals())  # Merge global variables
    locals_dict.update(locals())   # Merge local variables

    try:
        # Execute the modified source code
        exec(src_code, locals_dict, locals_dict)
    
    except Exception as e:
        return f"❌ Execution Error:\n{traceback.format_exc()}"


# Example usage
if __name__ == "__main__":
    # try:
    #     response = router_agent_executor.invoke({"input": "Hi how are you?"})
    #     print("API Connection Successful! Response:")
    #     print(response)
    # except Exception as e:
    #     print("API Connection Failed. Error:")
    #     print(e)

    src_code = 'def hello_world():\n    print("Hello, world!")\n\n# Some other code here'
    old_code = 'print("Hello, world!")'
    new_code = 'print("Bonjour, monde!")\nprint("Hola, mundo!")'
    modified_code = _replace(src_code, old_code, new_code)
    print(modified_code)

    example_model_code = """
# A simple supply chain optimization model using PuLP

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpStatusOptimal

# Initialize supply, demand, and cost data
supply = [100, 150, 200]  # Supply from three suppliers
demand = [120, 130, 100, 100]  # Demand from four demand centers
costs = [
    [2, 3, 1, 4],  # Costs from supplier 0 to demand centers
    [3, 2, 5, 2],  # Costs from supplier 1 to demand centers
    [4, 1, 3, 2]   # Costs from supplier 2 to demand centers
]


### DATA MANIPULATION CODE HERE ###


# Create the problem instance
problem = LpProblem("SimpleSupplyChainProblem", LpMinimize)

# Create variables for each supply-demand pair
variables = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous')
             for i in range(len(supply))
             for j in range(len(demand))}

# Objective function: minimize transportation costs
problem += lpSum(costs[i][j] * variables[i, j] for i in range(len(supply)) for j in range(len(demand))), "Total Cost"

# Supply constraints
for i in range(len(supply)):
    problem += lpSum(variables[i, j] for j in range(len(demand))) <= supply[i], f"Supply_{i}"

# Demand constraints
for j in range(len(demand)):
    problem += lpSum(variables[i, j] for i in range(len(supply))) >= demand[j], f"Demand_{j}"


### CONSTRAINT CODE HERE ###


# Solve the problem
status = problem.solve()

# Print the results
print(LpStatus[status])
if status == LpStatusOptimal:
    print("Optimal Solution:")
    for (i, j), var in variables.items():
        if var.varValue > 0:
            print(f"Units from Supplier {i} to Demand Center {j}: {var.varValue}")

    print(f"Total Cost: {problem.objective.value()}")
else:
    print("Not solved to optimality. Optimization status:", status)
"""

    extra_constraint = """problem += lpSum(variables[i, j] for i in [0,2] for j in range(len(demand))) <= 300, "Total_Shipment_Limit_S0_S2" """
    
    modified_example_model_code = _replace(example_model_code, CONSTRAINT_CODE_STR, extra_constraint)
    _run_with_exec(modified_example_model_code)