import argparse

import numpy as np
import pandas as pd

from src.util import make_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str, help="Name of the problem")
    parser.add_argument("n_knapsacks", type=int, help="Number of knapsacks")
    parser.add_argument("n_items", type=int, help="Number of items")
    parser.add_argument("--max_capacity", type=int, default=50, help="Maximum capacity of each knapsack")
    parser.add_argument("--max_value", type=int, default=20, help="Maximum value of each item")
    parser.add_argument("--max_weight", type=int, default=20, help="Maximum weight of each item")
    args = parser.parse_args()
    problem_name = args.problem_name
    n_knapsacks = args.n_knapsacks
    n_items = args.n_items
    max_capacity = args.max_capacity
    max_value = args.max_value
    max_weight = args.max_weight
    
    make_directory(f"problems/{problem_name}")
    
    capacities = np.random.randint(1, max_capacity + 1, n_knapsacks)
    knapsack_ids = [f"knapsack{i + 1}" for i in range(n_knapsacks)]
    knapsack_df = pd.DataFrame({"id": knapsack_ids, "capacity": capacities})
    knapsack_df.to_csv(f"problems/{problem_name}/knapsacks.csv", index=False)
    
    values = np.random.randint(1, max_value + 1, n_items)
    weights = np.random.randint(1, max_weight + 1, n_items)
    item_ids = [f"item{i + 1}" for i in range(n_items)]
    item_df = pd.DataFrame({"id": item_ids, "value": values, "weight": weights})
    item_df.to_csv(f"problems/{problem_name}/items.csv", index=False)