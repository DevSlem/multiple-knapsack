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

You can train the below methods using the following command (method link to the algorithm):

| Method | Command | Multiple Instances | Inference | 
| --- | --- | --- | --- |
| Integer Programming | `python ip.py [PROBLEM_NAME]` | ❌  | Immediate |
| Q-learning | `python qlearning.py [PROBLEM_NAME]` | ❌  | Immediate |
| DQN | `python dqn.py [PROBLEM_NAME]` | ❌  | Immediate |
| [DQN with Transformer](img/dqn-transformer-algorithm.png) | `python dqn_transformer.py` | ✅ | `--inference` or `-i` |
| [REINFORCE](img/reinforce-algorithm.png) | `python reinforce.py` | ✅ | `--inference` or `-i` |

where `[PROBLEM_NAME]` is the name of the problem.

> Note: The methods that can be trained on only one specific problem instance immediately evaluate the trained model on the same one as soon as the training is finished. On the other hand, the methods that can be trained on multiple problem instances require the `--inference [PROBLEM_NAME]` or `-i [PROBLEM_NAME]` option to evaluate the trained model on specific problem instances.

See detail hyperparameters in each python file.

## Results

If you optimize the problems using the above methods, you can see the results in the `results` directory. Specifically, there are `results/train` and `results/inference` directories. The former contains the results of the training process, and the latter contains the results of the evaluation process. 

Train results:

* `results/train/[PROBLEM_NAME]-[METHOD]`: Method that can be trained on only one specific problem instance.
* `results/train/[METHOD]`: Method that can be trained on multiple problem instances.

Inference results:

* `results/inference/[PROBLEM_NAME]`

For example, if you optimize the problem named `example` using "Q-learning" method, you can see the training results in the `results/train/example-qlearning` directory and the inference results in the `results/inference/example` directory. Meanwhile, if you train "DQN with Transformer" method, you can see the training results in the `results/train/dqn_transformer` directory. If you evaluate the trained method on the problem named `example`, you can see the inference results in the `results/inference/example` directory.

The most important file is `results.csv` in `results/inference/[PROBLEM_NAME]` directory. It contains the summary of the results. The csv table is like this:

| method | total_value | episodes | train_time | inference_time |
| --- | --- | --- | --- | --- |
| Integer Programming | 299 | | | 0.14815759658813477 |
| Q-learning | 171 | 10000 | 5.908851146697998 | 0.00016880035400390625 |
| DQN with Transformer | 235 | 300 | 8938.396562576294 | 0.2456543445587158 |

You can compare and visualize the results using the following command:

```bash
python summary.py [PROBLEM_NAMES...]
```

where `[PROBLEM_NAMES...]` is the names of the problems. For example, if you want to compare the results of the problems named `example1` and `example2`, you can use the following command: `$ python summary.py example1 example2`. Also, you can compare the results of all problems using the following command: `$ python summary.py`. The results are saved in the `results/summary` directory.