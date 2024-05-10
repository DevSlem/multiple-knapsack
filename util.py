import os
import pandas as pd

def make_directory(dir_path: str):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

def load_knapsack_problem(problem_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the knapsack problem and returns the knapsack and item dataframes."""
    knapsack_df = pd.read_csv(f"problems/{problem_name}/knapsacks.csv")
    item_df = pd.read_csv(f"problems/{problem_name}/items.csv")
    return knapsack_df, item_df