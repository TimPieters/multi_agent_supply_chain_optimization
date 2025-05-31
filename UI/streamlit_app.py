import streamlit as st
import json
import pandas as pd
import os
from datetime import datetime
import time

# Import the necessary components from your LangGraph script
from langgraph_sensitivity_analysis import (
    run_sensitivity_analysis,
    SensitivityAnalysisState,
    app, # The compiled LangGraph app
    print_workflow_graph, # Helper to print graph (optional for UI)
    # We will dynamically load these based on user selection, so remove initial load
    # source_code,
    # input_data,
    # model_description,
    BASELINE_OBJ as DEFAULT_BASELINE_OBJ, # Default baseline objective
    MODEL_FILE_PATH as DEFAULT_MODEL_FILE_PATH,
    MODEL_DATA_PATH as DEFAULT_MODEL_DATA_PATH,
    MODEL_DESCRIPTION_PATH as DEFAULT_MODEL_DESCRIPTION_PATH,
    planner_llm, # Default planner LLM
    coder_llm # Default coder LLM
)

# Import _run_with_exec from utils.py
from utils import _run_with_exec, modify_and_run_model

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="LangGraph Sensitivity Analysis")

# --- Sidebar ---
st.sidebar.title("About This App")
st.sidebar.info(
    "This application performs automated sensitivity analysis on optimization models "
    "using a LangGraph-powered multi-agent system. "
    "The Planner agent proposes scenarios, the Coder agent translates them into code modifications, "
    "and the Executor runs the modified model. The Analyzer then logs and summarizes the results."
)

st.sidebar.markdown("---")
st.sidebar.subheader("OpenAI API Configuration")
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    help="Your OpenAI API key. If not provided, it assumes the key is set as an environment variable (OPENAI_API_KEY)."
)

st.title("LangGraph Automated Sensitivity Analysis")
st.markdown("Run sensitivity analysis on your optimization model using an AI-powered LangGraph agent.")

# --- Configuration Section ---
st.header("Configuration")

# Define base paths for models, data, and descriptions
# Assuming the app is run from within the 'multi_agent_supply_chain_optimization' directory
BASE_DIR = "" # Keep as empty string as per user feedback
MODELS_DIR = "." # Models are directly in the current directory (e.g., capfacloc_model.py)
DATA_DIR = "data"
MODEL_DESCRIPTIONS_DIR = "model_descriptions"

# --- Model and Data Selection (Dropdowns side-by-side) ---
st.subheader("Model and Data Selection")
col_model, col_data = st.columns(2)

with col_model:
    available_models = ["capfacloc_model.py"] # Hardcoded as per user feedback
    selected_model_file = st.selectbox(
        "Select Optimization Model",
        options=available_models,
        index=0, # Only one option, so index 0
        help="Choose the Python file containing the optimization model."
    )
    # The actual path to the selected model file
    selected_model_full_path = selected_model_file

with col_data:
    available_data_files = [
        "capfacloc_data_10cust_10fac.json",
        "capfacloc_data_25cust_25fac.json",
        "capfacloc_data_50cust_50fac.json"
    ] # Hardcoded as per user feedback
    selected_data_file = st.selectbox(
        "Select Data File",
        options=available_data_files,
        index=0, # Default to the first data file
        help="Choose the JSON file containing the input data for the selected model."
    )
    # The actual path to the selected data file
    selected_data_full_path = os.path.join(DATA_DIR, selected_data_file)

# Dynamically determine model description path based on selected model
model_name_without_ext = os.path.splitext(selected_model_file)[0]
default_model_description_file = f"{model_name_without_ext}_description.txt"
selected_model_description_full_path = os.path.join(MODEL_DESCRIPTIONS_DIR, default_model_description_file)

# --- Display Model Description (in an expander) ---
with st.expander("View Model Description", expanded=False):
    try:
        with open(selected_model_description_full_path, "r") as f:
            model_description_content = f.read()
        st.markdown(model_description_content)
    except FileNotFoundError:
        st.warning(f"Model description file not found: {selected_model_description_full_path}")
    except Exception as e:
        st.error(f"Error loading model description: {e}")

st.markdown("---")

# --- Advanced Paths (User Override) ---
st.header("Advanced Paths (Override Defaults)")
with st.expander("Override Model and Data Paths", expanded=False):
    # These text inputs will allow users to override the dropdown selections
    # They will default to the selected dropdown values
    override_model_file_path = st.text_input(
        "Override Model File Path",
        value=selected_model_full_path, # Default to dropdown selection
        help="Enter a custom path to your optimization model file. This will override the dropdown selection."
    )
    override_model_data_path = st.text_input(
        "Override Model Data Path",
        value=selected_data_full_path, # Default to dropdown selection
        help="Enter a custom path to your model's JSON data file. This will override the dropdown selection."
    )
    override_model_description_path = st.text_input(
        "Override Model Description Path",
        value=selected_model_description_full_path, # Default to derived description path
        help="Enter a custom path to your model's description file. This will override the derived path."
    )

# Use the overridden paths if provided, otherwise use the dropdown selections
model_file_path = override_model_file_path
model_data_path = override_model_data_path
model_description_path = override_model_description_path

st.markdown("---")
st.header("Analysis Parameters")

col1, col2 = st.columns(2)
with col1:
    baseline_obj = st.number_input(
        "Baseline Objective Value",
        value=DEFAULT_BASELINE_OBJ,
        help="The objective value of the original, unmodified model. Used for calculating ΔObj."
    )
with col2:
    max_iterations = st.number_input(
        "Maximum Scenarios to Run",
        min_value=1,
        value=5,
        help="The maximum number of sensitivity scenarios the agent will explore."
    )

st.subheader("LLM Configuration")
col3, col4 = st.columns(2)
with col3:
    planner_model = st.text_input(
        "Planner LLM Model",
        value=planner_llm.model_name,
        help="The OpenAI model name for the Planner agent (e.g., 'gpt-4o-mini', 'gpt-4o')."
    )
    planner_temperature = st.slider(
        "Planner Temperature",
        min_value=0.0,
        max_value=1.0,
        value=planner_llm.temperature,
        step=0.1,
        help="Temperature for the Planner LLM. Higher values mean more creative scenarios."
    )
with col4:
    coder_model = st.text_input(
        "Coder LLM Model",
        value=coder_llm.model_name,
        help="The OpenAI model name for the Coder agent (e.g., 'gpt-4o', 'gpt-3.5-turbo')."
    )
    coder_temperature = st.slider(
        "Coder Temperature",
        min_value=0.0,
        max_value=1.0,
        value=coder_llm.temperature,
        step=0.1,
        help="Temperature for the Coder LLM. Lower values mean more deterministic code generation."
    )

st.markdown("---")

# --- Baseline Model Execution ---
st.header("Baseline Model Execution")
if st.button("Run Baseline Model", type="secondary"):
    st.info("Running baseline model... Please wait.")
    try:
        start_time = time.time()
        # Use _run_with_exec for baseline model execution
        baseline_result = modify_and_run_model({},model_file_path, model_data_path)
        end_time = time.time()
        execution_time = end_time - start_time

        st.subheader("Baseline Model Results")
        if baseline_result and isinstance(baseline_result, dict):
            st.metric(label="Status", value=baseline_result.get('status', 'N/A'))
            st.metric(label="Objective Value", value=f"{baseline_result.get('total_cost', 'N/A'):.2f}")
            st.metric(label="Execution Time", value=f"{execution_time:.2f} seconds")

            if 'solution' in baseline_result and baseline_result['solution']:
                with st.expander("View Decision Variables"):
                    solution_dict = baseline_result['solution']
                    # Convert dictionary of variables to a DataFrame for better display
                    var_data = [{"Variable": var_name, "Value": var_value} for var_name, var_value in solution_dict.items()]
                    
                    if var_data:
                        st.dataframe(pd.DataFrame(var_data), use_container_width=True)
                    else:
                        st.write("No decision variables found in the solution.")
                        st.json(solution_dict) # Show raw dict if empty or unexpected
            
            # if 'constraints' in baseline_result and baseline_result['constraints']:
            #     with st.expander("View Constraints"):
            #         st.json(baseline_result['constraints']) # Constraints might be complex, show as JSON
            
            # if 'parameters' in baseline_result and baseline_result['parameters']:
            #     with st.expander("View Input Parameters"):
            #         st.json(baseline_result['parameters'])

        else:
            st.warning("Baseline model execution did not return expected results.")
            st.json(baseline_result) # Show raw result for debugging

    except Exception as e:
        st.error(f"An error occurred during baseline model execution: {e}")
        st.exception(e)

st.markdown("---")

# --- Run Sensitivity Analysis Button ---
if st.button("Run Sensitivity Analysis", type="primary"):
    st.session_state['running_analysis'] = True
    st.session_state['scenario_log'] = []
    st.session_state['final_analysis_summary'] = ""
    st.session_state['run_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.info("Analysis started. Please wait...")

    progress_text = st.empty()
    scenario_log_display = st.empty()
    final_summary_display = st.empty()
    
    # Create a placeholder for the scenario log to update dynamically
    scenario_log_container = st.container()
    scenario_log_container.subheader("Scenario Log")
    scenario_log_area = scenario_log_container.empty()
    
    # Initialize progress bar
    progress_bar = st.progress(0)

    try:
        # To provide real-time updates, we need to modify the LangGraph nodes
        # to accept a callback function that updates Streamlit.
        # For this initial version, I'll run it as is and display results at the end.
        # If real-time updates are critical, we'll need to revisit the LangGraph script.

        # For now, let's just run the existing function and display the final state.
        # The `run_sensitivity_analysis` function already prints progress to console,
        # but Streamlit needs explicit updates.

        # This is a placeholder for actual real-time updates.
        # The actual LangGraph execution will happen in run_sensitivity_analysis.
        # To show progress, we need to modify the LangGraph nodes to report progress.
        # This is a more involved change. For now, I'll just show a "running" message
        # and then the final result.

        # The `run_sensitivity_analysis` function already handles logging to CSV/TXT.
        # We just need to display the final results.

        # Let's make a simple progress bar that updates based on max_iterations.
        # This will require modifying the `run_sensitivity_analysis` function to
        # accept a callback for progress updates.

        # I will modify `langgraph_sensitivity_analysis.py` to pass a callback to the nodes.
        # This will be the next step after creating this basic streamlit_app.py.

        # For now, I will just call the function and display the final result.
        # The user can then tell me if they want real-time updates.

        final_state = run_sensitivity_analysis(
            baseline_objective=baseline_obj,
            max_iterations=max_iterations,
            planner_model=planner_model,
            planner_temperature=planner_temperature,
            coder_model=coder_model,
            coder_temperature=coder_temperature,
            model_file_path=model_file_path,
            model_data_path=model_data_path,
            model_description_path=model_description_path,
            openai_api_key=openai_api_key # Pass the API key from the sidebar
        )

        st.session_state['final_analysis_summary'] = final_state.get('final_analysis_summary', 'No summary available.')
        st.session_state['scenario_log'] = final_state.get('scenario_log', [])

        progress_bar.progress(100)
        progress_text.success("Analysis Complete!")

        final_summary_display.subheader("Final Analysis Summary")
        final_summary_display.write(st.session_state['final_analysis_summary'])

        scenario_log_area.text_area(
            "Full Scenario Log",
            "\n".join(st.session_state['scenario_log']),
            height=300
        )

        st.subheader("Download Logs")
        run_id = st.session_state['run_id']
        log_csv_path = f"logs/run_log_{run_id}.csv"
        log_txt_path = f"logs/scenario_log_{run_id}.txt"

        if os.path.exists(log_csv_path):
            with open(log_csv_path, "rb") as f:
                st.download_button(
                    label="Download Run Log (CSV)",
                    data=f,
                    file_name=f"run_log_{run_id}.csv",
                    mime="text/csv"
                )
        if os.path.exists(log_txt_path):
            with open(log_txt_path, "rb") as f:
                st.download_button(
                    label="Download Scenario Log (TXT)",
                    data=f,
                    file_name=f"scenario_log_{run_id}.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.exception(e)

# Display logs if they exist in session state (for re-runs or persistence)
if 'scenario_log' in st.session_state and st.session_state['scenario_log']:
    st.subheader("Previous Scenario Log")
    st.text_area(
        "Full Scenario Log",
        "\n".join(st.session_state['scenario_log']),
        height=300,
        key="previous_scenario_log"
    )

if 'final_analysis_summary' in st.session_state and st.session_state['final_analysis_summary']:
    st.subheader("Previous Final Analysis Summary")
    st.write(st.session_state['final_analysis_summary'])
