import os
import pandas as pd
from typing import Optional
import numpy as np

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

def moving_average(values: np.ndarray, n: Optional[int] = None, smooth: Optional[float] = None):
    if (n is None and smooth is None) or (n is not None and smooth is not None):
        raise ValueError("you must specify either n or smooth")
    if smooth is not None:
        if smooth < 0.0 or smooth > 1.0:
            raise ValueError(f"smooth must be in [0, 1], but got {smooth}")
        n = int((1.0 - smooth) * 1 + smooth * len(values))
    ret = np.cumsum(values, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    ret[:n] = ret[:n] / np.arange(1, n + 1)
    return ret

def save_results(
    problem_name: str,
    method: str,
    total_value: int,
    episodes: Optional[int] = None,
    train_time: Optional[float] = None,
    inference_time: Optional[float] = None,
    **kwargs,
) -> pd.DataFrame:
    """Save the result of the method to the results csv file."""
    result_dict = {
        "method": method,
        "total_value": total_value,
        "episodes": episodes,
        "train_time": train_time,
        "inference_time": inference_time,
        **kwargs,
    }
    
    result_df = pd.DataFrame([result_dict])
    directory = f"results/inference/{problem_name}"
    make_directory(directory)
    
    try:
        old_result_df = pd.read_csv(f"{directory}/results.csv")
        result_df = pd.concat([old_result_df, result_df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    result_df["total_value"] = result_df["total_value"].astype(pd.Int64Dtype())
    result_df["episodes"] = result_df["episodes"].astype(pd.Int64Dtype())
    result_df.to_csv(f"{directory}/results.csv", index=False)
    return result_df