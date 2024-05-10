import pulp
import argparse
import time
import json
from util import make_directory, load_knapsack_problem
import numpy as np

def solve_knapsack_with_ip(values, weights, capacities):
    item_count = len(values)
    knapsack_count = len(capacities)
    
    # Create the problem variable:
    prob = pulp.LpProblem("MultiKnapsack", pulp.LpMaximize)
    
    # Create decision variables
    x = pulp.LpVariable.dicts("item", [(i, k) for i in range(item_count) for k in range(knapsack_count)], 
                              cat=pulp.LpBinary)
    
    # Objective function: Maximize the total value of all items in all knapsacks
    prob += pulp.lpSum([values[i] * x[(i, k)] for i in range(item_count) for k in range(knapsack_count)])
    
    # Constraint: Do not exceed the capacity of any knapsack
    for k in range(knapsack_count):
        prob += pulp.lpSum([weights[i] * x[(i, k)] for i in range(item_count)]) <= capacities[k], f"Capacity_{k}"
    
    # Constraint: An item can be in at most one knapsack
    for i in range(item_count):
        prob += pulp.lpSum([x[(i, k)] for k in range(knapsack_count)]) <= 1, f"OneKnapsack_{i}"
    
    # Solve the problem
    prob.solve()
    
    # Extract the results
    result = {"Status": pulp.LpStatus[prob.status],
              "Total Value": pulp.value(prob.objective),
              "Items in Knapsacks": {(i, k): pulp.value(x[(i, k)]) for i in range(item_count) for k in range(knapsack_count) if pulp.value(x[(i, k)]) == 1}}
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    args = parser.parse_args()
    problem_name = args.problem_name
    
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
    ip_result = solve_knapsack_with_ip(values, weights, capacities)
    end_time = time.time()
    train_time = end_time - start_time
    
    total_value = ip_result["Total Value"]
    status = ip_result["Status"]
    policy = [0]*len(values)
    for key in ip_result["Items in Knapsacks"]:
        policy[key[0]] = key[1] + 1
    
    result_dict = dict()
    result_dict["method"] = "Integer Programming"
    result_dict["n_knapsacks"] = len(knapsack_df)
    result_dict["n_items"] = len(item_df)
    result_dict["time"] = train_time
    result_dict["solution"] = {
        "total_value": int(total_value),
        "status": status
    }
    for knapsack_id in knapsack_df["id"]:
        result_dict["solution"][knapsack_id] = {"items": []}
    for i, a in enumerate(policy):
        knapsack_id = knapsack_df["id"][a - 1] if a > 0 else None
        if knapsack_id is not None:
            result_dict["solution"][knapsack_id]["items"].append(item_df["id"][i])
    for knapsack_id in knapsack_df["id"]:
        result_dict["solution"][knapsack_id]["value"] = np.sum([item_df["value"][item_df["id"] == i].values[0] for i in result_dict["solution"][knapsack_id]["items"]], dtype=np.int64).item()
        result_dict["solution"][knapsack_id]["weight"] = np.sum([item_df["weight"][item_df["id"] == i].values[0] for i in result_dict["solution"][knapsack_id]["items"]], dtype=np.int64).item()
        result_dict["solution"][knapsack_id]["capacity"] = knapsack_df["capacity"][knapsack_df["id"] == knapsack_id].values[0].item()
    
    directory = f"results/{problem_name}/integer_programming"
    make_directory(directory)
    with open(f"{directory}/result.json", 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print("\nInteger Programming Solution:")
    print("Status:", status)
    print("Total value from Q-learning:", total_value)
    print("Time:", train_time)