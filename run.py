# Import necessary libraries and modules for LangChain V2
import re
import traceback
from langchain.agents import create_react_agent,AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.tools import Tool
from pulp import LpStatus, LpStatusOptimal, LpStatusInfeasible


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
        locals_dict (dict): Dictionary containing execution context with `problem` and `variables`.

    Returns:
        dict: A dictionary containing:
            - 'status': Solver status (Optimal, Infeasible, etc.).
            - 'solution': Non-zero decision variable values.
            - 'total_cost': Objective function value if solved optimally.
    """
    print("\nlog - Extracting optimization results...")  

    if "problem" not in locals_dict or "variables" not in locals_dict:
        print("Error: `problem` or `variables` not found in execution context.")
        return {"status": "Error", "message": "Problem or variables not found in execution context."}

    problem = locals_dict["problem"]
    variables = locals_dict["variables"]
    status = problem.solve()

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
        for (i, j), var in variables.items():
            if var.varValue > 0:  
                result["solution"][(i, j)] = var.varValue
        result["total_cost"] = problem.objective.value()

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

# Load pre-trained Large Language Model from OpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Define a simple prompt to test the connection
prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    partial_variables={"source_code": _read_source_code("multi_agent_supply_chain_optimization/simple_model.py")},
    template="""
    You are an AI assistant for supply chain optimization. You analyze the provided Python optimization model
    and modify it based on the user's questions. You explain solutions from a PuLP Python solver.

    You have access to the following tools:

    {tools}

    Below is the full source code of the supply chain model:

    ```python
    {source_code}
    ```

    your written code will be added to the line with substring:
    "### CONSTRAINT CODE HERE ###"

    LOOK VERY WELL at these example questions and their answers and codes:
    --- EXAMPLES ---
    "Limit the total supply from supplier 0 to 80 units."
    problem += lpSum(variables[0, j] for j in range(len(demand))) <= 80, "Supply_Limit_Supplier_0"

    "Ensure that Supplier 1 supplies at least 50 units in total."
    problem += lpSum(variables[1, j] for j in range(len(demand))) >= 50, "Minimum_Supply_Supplier_1"
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

def format_constraint_input(constraint_code: str) -> str:
    """
    Ensures the constraint is formatted correctly before inserting it into the model.
    
    Args:
        constraint_code (str): The constraint generated by the agent.
    
    Returns:
        str: A properly formatted constraint statement.
    """
    # Ensure there are no unnecessary backticks or formatting issues
    constraint_code = constraint_code.strip().strip("`")

    # Ensure the constraint is a valid Python statement
    if not constraint_code.startswith("problem +="):
        constraint_code = f"problem += {constraint_code}"
    
    return constraint_code

def modify_and_run_model(constraint_code: str) -> str:
    """
    Inserts a new constraint into the model, executes it, and returns the results.

    Args:
        constraint_code (str): The constraint code generated by the agent.

    Returns:
        str: The optimization results.
    """
    try:
        # Read the original model
        model_code = _read_source_code("multi_agent_supply_chain_optimization/simple_model.py")

        # Format the constraint properly
        formatted_constraint = format_constraint_input(constraint_code)

        # Insert the constraint at the predefined placeholder
        modified_code = _replace(model_code, "### CONSTRAINT CODE HERE ###", formatted_constraint)

        # Execute the modified model
        result = _run_with_exec(modified_code)

        return result  # Return results to the agent

    except Exception as e:
        return f"Error while modifying the model: {str(e)}"

# Define the tool for the agent
modify_model_tool = Tool(
    name="ModifyAndRunModel",
    func=modify_and_run_model,
    description="""
    This tool **modifies the supply chain model by adding a constraint** and then **runs the updated model**.

    **How to use this tool:**
    - You must **generate a valid constraint** using the syntax of the provided PuLP model.
    - The generated constraint should be passed as input to this tool.
    - The tool will **insert the constraint, execute the model, and return the optimization results**.

    DO NOT wrap the constraint in triple backticks or Markdown-style code blocks.
    Simply return the raw Python expression as a string.
    """
)

# Add the tool to the tools list
tools = [echo_tool_obj, modify_model_tool]

router_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt = prompt
)

router_agent_executor = AgentExecutor(agent=router_agent, tools=tools, return_intermediate_steps=True, verbose=True)


# Example usage
if __name__ == "__main__":
    try:
        response = router_agent_executor.invoke({"input": "What happens if supply at supplier 0 is limited to 80?"})
        print("API Connection Successful! Response:")
        print(response)
    except Exception as e:
        print("API Connection Failed. Error:")
        print(e)

    # src_code = 'def hello_world():\n    print("Hello, world!")\n\n# Some other code here'
    # old_code = 'print("Hello, world!")'
    # new_code = 'print("Bonjour, monde!")\nprint("Hola, mundo!")'
    # modified_code = _replace(src_code, old_code, new_code)
    # print(modified_code)

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
"""

    # simple_model = _read_source_code("multi_agent_supply_chain_optimization/simple_model.py")


    # extra_constraint = """problem += lpSum(variables[i, j] for i in [0,2] for j in range(len(demand))) <= 300, "Total_Shipment_Limit_S0_S2" """
    
    # modified_example_model_code = _replace(simple_model, CONSTRAINT_CODE_STR, extra_constraint)
    # _run_with_exec(modified_example_model_code)