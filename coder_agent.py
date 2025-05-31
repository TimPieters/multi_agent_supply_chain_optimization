# Import necessary libraries and modules for LangChain V2
import json
from langchain.agents import create_react_agent,AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.tools import Tool
from langchain_core.tools import tool
from pulp import LpStatus, LpStatusOptimal, LpStatusInfeasible
from typing import List
import json
from utils import _replace, _read_source_code, _run_with_exec, modify_and_run_model
from config import MODEL_FILE_PATH, MODEL_DATA_PATH

class CoderAgent:
    def __init__(self, model_path: str = MODEL_FILE_PATH, llm_model_name: str = "gpt-4o", llm_temperature: float = 0):
        self.source_code = _read_source_code(model_path)
        self.tools = [self._create_safety_tool()]
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=llm_temperature)

        self.prompt_template = self._build_prompt()
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt_template
        )
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            return_intermediate_steps=True,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def _build_prompt(self) -> PromptTemplate:
        # Define a simple prompt to test the connection
        return PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            partial_variables={"source_code": self.source_code},
            template="""
            You are an AI assistant for supply chain optimization. You analyze the provided Python optimization model
            and modify it based on the user's questions. You just state your added code.
            You use the key "ADD CONSTRAINT" to add a constraint, and you use the key "ADD DATA" to add data.
            You can only do one modification and you always do a safety check.

            You must provide a valid JSON object using double quotes for both keys and values. NEVER add ```json ```
            Example:
            {{ "ADD CONSTRAINT": "model += lpSum(variables[0, j] for j in range(len(demand))) <= 80, \\"Supply_Limit_Supplier_0\\"" }}
            or
            {{ "ADD DATA": "supply = [200, 300, 300]" }}
            Do not use single quotes or Python-style dictionaries.

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

            ---FORMAT---

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the tool to use, MUST BE exactly one of [{tool_names}]
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
    def _create_safety_tool(self) -> Tool:
        def safety_check_tool(input_json: str) -> str:
            """
            Checks whether the supplied JSON object (as a string) representing the modification
            is safe (i.e., it is a dictionary containing either "ADD DATA" or "ADD CONSTRAINT").
            
            Args:
                input_json (str): The JSON string representing the modification.
                
            Returns:
                str: "SAFE" if the modification is considered safe; otherwise, an error message.
            """
            try:
                mod = json.loads(input_json)
            except Exception as e:
                return f"ERROR: Invalid JSON. {str(e)}"
            
            if not isinstance(mod, dict):
                return "ERROR: Modification is not a dictionary."
            
            valid_keys = {"ADD DATA", "ADD CONSTRAINT"}
            if not valid_keys.intersection(mod.keys()):
                return "ERROR: Modification does not contain a valid key. Must include one of 'ADD DATA' or 'ADD CONSTRAINT'."
            
            # Optionally, add other safety checks here (for example, check for disallowed content)
            
            return "SAFE"

        # Create the SafetyCheck tool.
        safety_check = Tool(
            name="SafetyCheck",
            func=safety_check_tool,
            description=(
                "Checks if the generated code modification is safe. "
                "Input should be a JSON string representing the modification. "
                "Returns 'SAFE' if the modification is valid; otherwise, returns an error message."
            )
        )
        return safety_check

    def run(self, input_str: str):
        print("\n" + "="*100)
        print(f"Running CoderAgent with input: {input_str}")
        print("\n" + "="*100)
        response = self.executor.invoke({"input": input_str})
        final_answer = response["output"]
        print("="*100)
        print("Executing supply chain model:")
        print("\n" + "="*100)
        result = modify_and_run_model(final_answer)
        print("\n" + "="*100)
        print(f"input question: {input_str}")
        print(f"code: {final_answer}")
        print(f"execution result: {result}")
        print("\n" + "="*100)
        return result

# Example usage
if __name__ == "__main__":

    coder = CoderAgent()
    #input_question = "What happens if the demand for the fifth customer increases by 40 units, raising it from 22 to 62?"
    input_question = "Increase the fixed cost of the second facility by 50%."
    output = coder.run(input_question)
    print(json.dumps(output, indent=2))
