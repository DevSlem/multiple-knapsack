import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import util
from collections import defaultdict
import os

parser = argparse.ArgumentParser()
parser.add_argument("problem_names", type=str, nargs="*")
args = parser.parse_args()
problem_names = args.problem_names

if len(problem_names) == 0:
    problem_names = os.listdir("problems")

knapsack_dict, item_dict = {}, {}
result_dict = {}

for problem_name in problem_names:
    try:
        knapsack_df, item_df = util.load_knapsack_problem(problem_name)
        result_df = pd.read_csv(f"results/inference/{problem_name}/results.csv")
    except FileNotFoundError:
        continue

    knapsack_dict[problem_name] = knapsack_df
    item_dict[problem_name] = item_df
    result_dict[problem_name] = result_df
    
problem_names = list(knapsack_dict.keys())
    
if len(problem_names) == 0:
    print("No results found.")
    exit()
    
n_knapsacks_x_items_dict = {}
total_values_dict = defaultdict(dict)
train_times_dict = defaultdict(dict)
inference_times_dict = defaultdict(dict)

for problem_name in problem_names:
    n_knapsacks = len(knapsack_dict[problem_name])
    n_items = len(item_dict[problem_name])
    n_knapsacks_x_items_dict[problem_name] = n_knapsacks * n_items
    
    for method in result_dict[problem_name]["method"].unique():
        result_row = result_dict[problem_name].loc[result_dict[problem_name]["method"] == method]
        total_value = result_row["total_value"].values[0]
        train_time = result_row["train_time"].values[0]
        inference_time = result_row["inference_time"].values[0]
        
        total_values_dict[method][problem_name] = total_value
        train_times_dict[method][problem_name] = train_time
        inference_times_dict[method][problem_name] = inference_time
        
        
def plot_summary(x_dict, y_dict, xlabel, ylabel, title, save_path, logscale=False):
    x_dict = {problem: value for problem, value in sorted(x_dict.items(), key=lambda x: x[1])}
    problem_to_index = {problem: index for index, problem in enumerate(x_dict.keys())}
    ticks = np.arange(len(x_dict))
    xlabels = list(x_dict.values())
    
    for method in y_dict.keys():
        valid_y = {problem: value for problem, value in y_dict[method].items() if value is not None}
        if not valid_y:
            continue
        
        # sort valid_y by x_dict
        valid_y = {problem: valid_y[problem] for problem in x_dict.keys() if problem in valid_y}
        valid_problems = valid_y.keys()
        index = [problem_to_index[problem] for problem in valid_problems]
        
        valid_x = [x_dict[problem] for problem in valid_problems]
        valid_y = list(valid_y.values())
        valid_y = [y for x, y in sorted(zip(valid_x, valid_y))]

        plt.plot(index, valid_y, label=method, marker='o')
        plt.xticks(ticks, xlabels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        if logscale:
            plt.yscale('log')
        
    plt.savefig(save_path)
    plt.close()
    
util.make_directory("results/summary")

normalized_total_values_dict = defaultdict(dict)
for problem_name in problem_names:
    max_total_value = max(total_values_dict[method][problem_name] for method in total_values_dict.keys() if problem_name in total_values_dict[method])
    for method in total_values_dict.keys():
        if problem_name not in total_values_dict[method]:
            continue
        total_value = total_values_dict[method][problem_name]
        normalized_total_value = total_value / max_total_value
        normalized_total_values_dict[method][problem_name] = normalized_total_value

plot_summary(
    n_knapsacks_x_items_dict,
    total_values_dict,
    r"$N_{K} \times N_{I}$",
    "Total Value",
    "Total Values over Problem Size",
    "results/summary/total_values.png",
)

plot_summary(
    n_knapsacks_x_items_dict,
    normalized_total_values_dict,
    r"$N_{K} \times N_{I}$",
    "Normalized Total Value",
    "Normalized Total Values over Problem Size",
    "results/summary/normalized_total_values.png"
)


plot_summary(
    n_knapsacks_x_items_dict,
    train_times_dict,
    r"$N_{K} \times N_{I}$",
    "Train Time",
    "Train Times over Problem Size",
    "results/summary/train_times.png",
    logscale=True
)

plot_summary(
    n_knapsacks_x_items_dict,
    inference_times_dict,
    r"$N_{K} \times N_{I}$",
    "Inference Time",
    "Inference Times over Problem Size",
    "results/summary/inference_times.png",
    logscale=True
)