# Multiple Knapsack

Optimize multiple knapsack problem using reinforcement learning. It's one of the most popular problems in combinatorial optimization.

## Setup

Create Python 3.9 environment and activate it:

```bash
conda create -n multi-knapsack python=3.9 -y
conda activate multi-knapsack
```

Install the required packages:

```bash
pip install torch==2.3.0
pip install matplotlib==3.8.4
pip install pandas==2.2.2
pip install PuLP==2.8.0
```

## Run

First of all, you should generate a multiple knapsack problem using the following command:

```bash
python generate_problem.py [PROBLEM_NAME] [NUM_KNAPSACKS] [NUM_ITEMS]
```

where `[PROBLEM_NAME]` is the name of the problem, `[NUM_KNAPSACKS]` is the number of knapsacks, and `[NUM_ITEMS]` is the number of items. Then, you can see the generated problem in the `problems/[PROBLEM_NAME]` directory. Specifically, there are two csv files generated: `knapsacks.csv` and `items.csv`. See details in the `generate_problem.py` file.

You can optimize the problems using below methods:

| Method | Command | Description |
| --- | --- | --- |
| Integer Programming | `python ip.py [PROBLEM_NAME]` | It gives the optimal solution. |
| Q-learning | `python qlearning.py [PROBLEM_NAME]` | Conventional tabular Q-learning. |
| DQN | `python dqn.py [PROBLEM_NAME]` | (Not implemented yet) Deep Q-network. |

See details in each python file.

## Results

If you optimize the problems using the above methods, you can see the results in the `results` directory. For example, if you optimize the problem named `example` using the Q-learning method, you can see the results in the `results/example/qlearning` directory. The most important file is `result.json` which contains all details of the optimization process. It requires the following structure:

```json
{
    "method": "METHOD_NAME",
    "n_knapsacks": 3,
    "n_items": 10,
    "time": 0.008505105972290039,
    "solution": {
        "total_value": 21,
    }
}
```

You can compare and visualize the results using the following command:

TODO: Not implemented yet.
