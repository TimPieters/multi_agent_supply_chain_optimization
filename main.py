import json
from scenario_planner import generate_new_scenario
from scenario_planner import build_scenario_log_from_runs

# Load execution log
with open("agent_execution_log.json", "r") as f:
    run_logs = json.load(f)

# Create a formatted scenario log
baseline_obj = 366.10
scenario_log = build_scenario_log_from_runs(run_logs, baseline_obj)

# Optionally print past scenarios
for entry in scenario_log:
    print(entry)

# Generate new planner proposal
planner_output = generate_new_scenario(scenario_log)

print("Planner proposed:", planner_output)

# Optionally run it through the executor:
# response = router_agent_executor.invoke({"input": planner_output})
