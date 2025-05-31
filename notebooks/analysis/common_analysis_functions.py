import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os # Added for os.path.exists

# Ensure project root is on sys.path so we can import from other modules if needed
root_dir = Path(__file__).resolve().parents[3] # Adjust path as needed to reach multi_agent_supply_chain_optimization
sys.path.insert(0, str(root_dir))

from utils import modify_and_run_model # Import modify_and_run_model

# --- Baseline Objective Calculation ---
def get_baseline_objective(model_file_path: str, model_data_path: str, baseline_log_filepath: str = "notebooks/baseline_cflp_log.csv"):
    """
    Runs the baseline model to get its objective value and logs the run.

    Args:
        model_file_path (str): Path to the model's Python file.
        model_data_path (str): Path to the model's JSON data file.
        baseline_log_filepath (str): Path to the CSV file for logging the baseline run.

    Returns:
        float: The baseline objective value if optimal, None otherwise.
    """
    print(f"Running baseline model from {model_file_path} with data {model_data_path}...")
    result = modify_and_run_model(
        modification_json={},
        model_file_path=str(model_file_path),
        model_data_path=str(model_data_path),
        run_id="baseline_run",
        log_filepath=baseline_log_filepath
    )
    
    if isinstance(result, dict) and result.get('status') == 'Optimal':
        baseline_obj = result['total_cost']
        print(f"Baseline objective value: {baseline_obj}")
        return baseline_obj
    else:
        print(f"Failed to get optimal baseline objective: {result}")
        return None

# --- Read Baseline Log ---
def read_baseline_log(baseline_log_filepath: str) -> pd.DataFrame:
    """
    Reads the baseline log file into a Pandas DataFrame.

    Args:
        baseline_log_filepath (str): Path to the baseline CSV log file.

    Returns:
        pd.DataFrame: The loaded baseline data, or an empty DataFrame if not found/empty.
    """
    try:
        df_baseline = pd.read_csv(baseline_log_filepath)
        if not df_baseline.empty:
            print(f"Baseline log file loaded from {baseline_log_filepath}.")
            return df_baseline
        else:
            print("Baseline log file is empty.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"Baseline log file not found at {baseline_log_filepath}. Please run baseline model first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading baseline log: {e}")
        return pd.DataFrame()

def load_benchmark_data(filepath: str) -> pd.DataFrame:
    """
    Loads benchmark data from a CSV or Parquet file into a Pandas DataFrame.

    Args:
        filepath (str): The path to the benchmark file (CSV or Parquet).

    Returns:
        pd.DataFrame: The loaded benchmark data.
    """
    try:
        file_extension = Path(filepath).suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(filepath)
        elif file_extension == '.parquet':
            try:
                df = pd.read_parquet(filepath)
            except ImportError:
                print("Error: 'pyarrow' or 'fastparquet' library is required to read Parquet files.")
                print("Please install them using: pip install pyarrow")
                return pd.DataFrame()
        else:
            print(f"Error: Unsupported file format '{file_extension}'. Only .csv and .parquet are supported.")
            return pd.DataFrame()

        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Benchmark file not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading benchmark data from {filepath}: {e}")
        return pd.DataFrame()

def parse_json_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Parses specified JSON string columns in a DataFrame into Python objects.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to parse as JSON.

    Returns:
        pd.DataFrame: The DataFrame with specified columns parsed.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else x)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df

# --- Modification JSON Parsing ---
def parse_modification_json(df, col_name='modification'):
    parsed_data = []
    for _, row in df.iterrows():
        try:
            mod_json = json.loads(row[col_name])
            mod_type = list(mod_json.keys())[0] if mod_json else None
            mod_value = mod_json[mod_type] if mod_type else None
            parsed_data.append({'modification_type': mod_type, 'modification_value': mod_value})
        except (json.JSONDecodeError, TypeError, IndexError):
            parsed_data.append({'modification_type': None, 'modification_value': None})
    return df.assign(**pd.DataFrame(parsed_data))

def filter_optimal_solutions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows with 'Optimal' status.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df['status'] == 'Optimal'].copy()

def plot_objective_distribution(df: pd.DataFrame, title: str = "Distribution of Objective Values", 
                                 objective_col: str = 'objective_value', save_path: str = None):
    """
    Plots the distribution of objective values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        title (str): Title of the plot.
        objective_col (str): Name of the column containing objective values.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[objective_col], kde=True)
    plt.title(title)
    plt.xlabel("Objective Value")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_solve_time_distribution(df: pd.DataFrame, title: str = "Distribution of Solve Times", 
                                 time_col: str = 'pulp_model_execution_time', save_path: str = None):
    """
    Plots the distribution of solve times.

    Args:
        df (pd.DataFrame): The input DataFrame.
        title (str): Title of the plot.
        time_col (str): Name of the column containing solve times.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[time_col], kde=True)
    plt.title(title)
    plt.xlabel("Solve Time (seconds)")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# You can add more common functions here, e.g., for correlation analysis,
# extracting specific parameters from the 'parameters' JSON column, etc.
