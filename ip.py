import pulp
import argparse
import time
from util import load_knapsack_problem, save_results
import numpy as np

def solve_knapsack_with_ip(values, weights, capacities, time_limit=3600):
    item_count = len(values)
    knapsack_count = len(capacities)
    
    # Create the problem variable:
    prob = pulp.LpProblem("MultiKnapsack", pulp.LpMaximize)
    
    # Create decision variables
    x = pulp.LpVariable.dicts("item", [(i, k) for i in range(item_count) for k in range(knapsack_count)], cat=pulp.LpBinary)
    
    # Objective function: Maximize the total value of all items in all knapsacks
    prob += pulp.lpSum([values[i] * x[(i, k)] for i in range(item_count) for k in range(knapsack_count)])
    
    # Constraint: Do not exceed the capacity of any knapsack
    for k in range(knapsack_count):
        prob += pulp.lpSum([weights[i] * x[(i, k)] for i in range(item_count)]) <= capacities[k], f"Capacity_{k}"
    
    # Constraint: An item can be in at most one knapsack
    for i in range(item_count):
        prob += pulp.lpSum([x[(i, k)] for k in range(knapsack_count)]) <= 1, f"OneKnapsack_{i}"
    
    # Solver settings
    pulp_solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=True)
    
    # Solve the problem
    prob.solve(pulp_solver)
    
    # Check if the solution is optimal
    is_optimal = pulp.LpStatus[prob.status] == 'Optimal'
    
    # Extract the results
    result = {
        "Status": pulp.LpStatus[prob.status],
        "Total Value": pulp.value(prob.objective),
        "Items in Knapsacks": {(i, k): pulp.value(x[(i, k)]) for i in range(item_count) for k in range(knapsack_count) if pulp.value(x[(i, k)]) == 1},
        "Is Optimal": is_optimal
    }
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    parser.add_argument("--timelimit", type=int, default=300, help="Time limit in seconds")
    args = parser.parse_args()
    problem_name = args.problem_name
    time_limit = args.timelimit
    
    knapsack_df, item_df = load_knapsack_problem(problem_name)
    capacities = knapsack_df['capacity'].values
    values = item_df['value'].values
    weights = item_df['weight'].values
    
    print("Knapsack Problem:")
    print("Values:", values)
    print("Weights:", weights)
    print("Capacities:", capacities)
    
    print("\nSolving with Integer Programming...")
    
    start_time = time.time()
    ip_result = solve_knapsack_with_ip(values, weights, capacities, time_limit)
    end_time = time.time()
    inference_time = end_time - start_time
    
    total_value = ip_result["Total Value"]
    status = ip_result["Status"]
    is_optimal = ip_result["Is Optimal"]
    policy = [0]*len(values)
    for key in ip_result["Items in Knapsacks"]:
        policy[key[0]] = key[1] + 1
    
    result_df = save_results(
        problem_name=problem_name,
        method="Integer Programming",
        total_value=total_value,
        inference_time=inference_time,
        optimal=is_optimal
    )
    
    print("Inference results (the last one is the current result):")
    print(result_df)
