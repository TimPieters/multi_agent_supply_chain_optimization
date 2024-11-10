# Import necessary libraries and modules for LangChain V2
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

# Example usage
if __name__ == "__main__":
    try:
        response = router_agent_executor.invoke({"input": "Hi how are you?"})
        print("API Connection Successful! Response:")
        print(response)
    except Exception as e:
        print("API Connection Failed. Error:")
        print(e)
