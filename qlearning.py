import numpy as np
import argparse
import time
import json
from util import make_directory, load_knapsack_problem
import matplotlib.pyplot as plt

import numpy as np

class MultiKnapsackQLearning:
    def __init__(self, values, weights, capacities, lr=0.1, gamma=0.95, episodes=1000, eps=1.0, eps_decay=0.995, eps_min=0.01):
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.initial_capacities = np.array(capacities)
        self.lr = lr
        self.gamma = gamma
        self.episodes = episodes
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_items = len(values)
        self.n_knapsacks = len(capacities)
        
        self.n_actions = self.n_knapsacks + 1  # Actions: do nothing (0) or put item in knapsack 1 to n_knapsacks
        self.Q = np.zeros((self.n_items, self.n_actions))

    def select_action(self, item):
        # epsilon-greedy policy
        if np.random.rand() < self.eps:
            return np.random.choice(self.n_actions)  # Choose any action randomly
        else:
            return np.argmax(self.Q[item])  # Choose the best action from Q-table

    def update_q_value(self, item, action, reward):
        max_next_q = np.max(self.Q[item + 1]) if item + 1 < self.n_items else 0
        self.Q[item, action] += self.lr * (reward + self.gamma * max_next_q - self.Q[item, action])
        
    def train(self):
        epsilons = []
        cumulative_rewards = []
        
        for episode in range(self.episodes):
            # reset the environment
            current_capacities = self.initial_capacities.copy()
            cumulative_reward = 0
            
            for item in range(self.n_items):
                # select an action
                action = self.select_action(item)
                # take the action and observe the reward
                reward = 0
                if action > 0 and self.weights[item] <= current_capacities[action - 1]:
                    reward = self.values[item]  # Reward for placing the item in the knapsack
                    current_capacities[action - 1] -= self.weights[item]  # Update the capacity
                elif action > 0 and self.weights[item] > current_capacities[action - 1]:
                    reward = -self.values[item] * 0.5  # Penalty for attempting to exceed capacity
                
                # update Q-value
                self.update_q_value(item, action, reward)
                
                cumulative_reward += reward

            epsilons.append(self.eps)
            cumulative_rewards.append(cumulative_reward)
            
            # epsilon decay
            self.eps = max(self.eps * self.eps_decay, self.eps_min)
        
        return epsilons, cumulative_rewards

    def get_policy(self):
        policy = []
        current_capacities = self.initial_capacities.copy()
        for item in range(self.n_items):
            action = np.argmax(self.Q[item])  # Always select the best action from Q-table for each item
            if action > 0 and self.weights[item] <= current_capacities[action - 1]:
                policy.append(action)  # Add the action to the policy if it is feasible
                current_capacities[action - 1] -= self.weights[item]  # Update capacity if the item is placed
            else:
                policy.append(0)  # Append 0 (do nothing) if the action is not feasible
        return policy


    def get_total_value(self):
        policy = self.get_policy()
        return sum(self.values[i] for i, a in enumerate(policy) if a > 0)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eps_min", type=float, default=0.01)
    args = parser.parse_args()
    problem_name = args.problem_name
    episodes = args.episodes
    gamma = args.gamma
    eps = args.eps
    eps_decay = args.eps_decay
    lr = args.lr
    eps_min = args.eps_min
    
    knapsack_df, item_df = load_knapsack_problem(problem_name)
    capacities = knapsack_df['capacity'].values
    values = item_df['value'].values
    weights = item_df['weight'].values
    
    print("Knapsack Problem:")
    print("Values:", values)
    print("Weights:", weights)
    print("Capacities:", capacities)
    
    learner = MultiKnapsackQLearning(values, weights, capacities, lr=lr, gamma=gamma, episodes=episodes, eps=eps, eps_decay=eps_decay, eps_min=eps_min)
    
    print("\nSolving with Q-Learning...")
    
    start_time = time.time()
    epsilons, cumulative_rewards = learner.train()
    end_time = time.time()
    train_time = end_time - start_time
    
    policy = learner.get_policy()
    total_value = learner.get_total_value()
    
    result_dict = dict()
    result_dict["method"] = "Q-learning"
    result_dict["n_knapsacks"] = len(knapsack_df)
    result_dict["n_items"] = len(item_df)
    result_dict["time"] = train_time
    result_dict["episodes"] = episodes
    result_dict["solution"] = {
        "total_value": total_value.item(),
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
    
    directory = f"results/{problem_name}/qlearning"
    make_directory(directory)
    with open(f"{directory}/result.json", 'w') as f:
        json.dump(result_dict, f, indent=4)
        
    plt.plot(epsilons)
    plt.title(f"Epsilon Decay ({eps_decay})")
    plt.xlabel("Episode")
    plt.ylabel(f"Epsilon")
    plt.savefig(f"{directory}/epsilon_decay.png")
    plt.close()
    
    plt.plot(cumulative_rewards)
    plt.title(f"Cumulative Rewards")
    plt.xlabel("Episode")
    plt.ylabel(f"Cumulative Reward")
    plt.savefig(f"{directory}/cumulative_rewards.png")
    plt.close()
    
    print("\nQ-Learning Solution:")
    print("Total value from Q-learning:", total_value)
    print("Time:", train_time)
    