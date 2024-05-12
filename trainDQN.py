import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from multiKnapsackEnv import *
from dqn import *

import argparse
from util import make_directory, load_knapsack_problem

import time
import json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    args = parser.parse_args()
    problem_name = args.problem_name

    knapsack_df, item_df = load_knapsack_problem(problem_name)
    capacities = knapsack_df['capacity'].values
    values = item_df['value'].values
    weights = item_df['weight'].values

    items = []
    for i in range(len(values)):
        items.append((values[i], weights[i]))
        
    env = MultiKnapsackEnv(items, capacities)

    EPISODES = 3
    dqn = DQN(2 * len(values) + len(capacities) + len(values), len(values) * len(capacities))
    td_loss_list = []
    cumulative_reward_list = []
    start_time = time.time()
    for e in range(EPISODES):
        print(e)
        obs = env.reset()
        terminated = False
        cumulative_reward = 0.0
        while not terminated:
            # select action
            action = dqn.select_action(obs)
            
            # take the action then observe next state and reward
            next_obs, reward, terminated = env.step(action)
            
            action = np.array([action]) # (n_envs,)
            next_obs = next_obs[np.newaxis, :] # (n_envs, obs_dim)
            reward = np.array([reward]) # (n_envs,)
            terminated = np.array([terminated]) # (n_envs,)
            
            # update DQN
            td_losses = dqn.update(obs, action, next_obs, reward, terminated)
            if td_losses is not None:
                td_loss_list.append(np.mean(td_losses))
            
            # transition to the next state
            obs = next_obs
            cumulative_reward += reward[0]
            print(f'cumulative_reward : {cumulative_reward}')

        
        cumulative_reward_list.append(cumulative_reward)
        if cumulative_reward_list[e].shape == (1,):
            cumulative_reward_list[e] = cumulative_reward_list[e][0]
            # print(f'cumulative_reward : {cumulative_reward}')
        # plt.figure(figsize=(10, 5))
        
        # plt.subplot(1, 2, 1)
        # plt.plot(td_loss_list, label='Loss')
        # plt.title('Loss over Episodes')
        # plt.ylabel('Loss')
        # plt.legend()
        
        # plt.subplot(1, 2, 2)
        # plt.plot(cumulative_reward_list, label='Reward')
        # plt.title('Reward over Episodes')
        # plt.xlabel('Episodes')
        # plt.ylabel('Reward')
        # plt.legend()


        # plt.show()
    # TODO: plot td_loss_list and cumulative_reward_list

    end_time = time.time()
    train_time = end_time - start_time

    result_dict = dict()
    result_dict["method"] = "DQN"
    result_dict["n_knapsacks"] = len(knapsack_df)
    result_dict["n_items"] = len(item_df)
    result_dict["time"] = train_time
    result_dict["solution"] = {
        "total_value": int(cumulative_reward_list[-1])
    }

    directory = f"results/{problem_name}/dqn"
    make_directory(directory)
    with open(f"{directory}/result.json", 'w') as f:
        json.dump(result_dict, f, indent=4)

    with open(f"{directory}/reward.txt", 'w') as f:
        f.write(str(cumulative_reward_list))
        
    with open(f"{directory}/loss.txt", 'w') as f:
        f.write(str(td_loss_list))


