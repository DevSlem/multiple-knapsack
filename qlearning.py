import numpy as np
import argparse
import time
import json
from util import make_directory, load_knapsack_problem, save_results
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
    
    # save train results
    directory = f"results/train/{problem_name}-qlearning"
    make_directory(directory)
        
    plt.plot(epsilons)
    plt.title(f"Epsilon Decay ({eps_decay})")
    plt.xlabel("Episode")
    plt.ylabel(f"Epsilon")
    plt.savefig(f"{directory}/epsilons.png")
    plt.close()
    
    plt.plot(cumulative_rewards)
    plt.title(f"Cumulative Rewards")
    plt.xlabel("Episode")
    plt.ylabel(f"Cumulative Reward")
    plt.savefig(f"{directory}/cumulative_rewards.png")
    plt.close()
    
    # policy = learner.get_policy()
    start_time = time.time()
    total_value = learner.get_total_value()
    inference_time = time.time() - start_time
    
    # save inference results
    result_df = save_results(
        problem_name=problem_name,
        method="Q-learning",
        total_value=total_value,
        episodes=episodes,
        train_time=train_time,
        inference_time=inference_time,
    )
    
    print("Inference results (the last one is the current result):")
    print(result_df)
