import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from multiKnapsackEnv import *
from dqn import *

def load_data(filename):
    data = {}
    itmes = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # 빈 라인 스킵
                continue
            key, value = line.strip().split(' = ')
            # 세미콜론 제거 및 공백 제거
            value = value.strip(';').strip()
            # eval 대신 안전한 ast.literal_eval 사용
            try:
                import ast
                data[key] = ast.literal_eval(value)
            except ValueError as e:
                print(f"Error parsing {value}: {e}")
    return data['NbItems'], data['NbKnapsacks'], np.asarray(data['capacity']), np.asarray(data['value']), np.asarray(data['weight'])

for i in range(1, 2):
    num_items, num_knapsacks, capacities, values, weights = load_data('data/data_%d.dat' % i)
    # num_items, num_knapsacks, capacities, values, weights = load_data('/content/data_1.dat')
    items = []
    for i in range(len(values)):
        items.append((values[i], weights[i]))
        
    env = MultiKnapsackEnv(items, capacities)

    EPISODES = 300
    dqn = DQN(2 * num_items + num_knapsacks + num_items, num_items * num_knapsacks)
    td_loss_list = []
    cumulative_reward_list = []
    value_list = []
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
            print(cumulative_reward)

        value = 0
        for i in range(num_items):
            if obs[0][i] > 0:
                value += values[i]
        print(value)
        value_list.append(value)
        cumulative_reward_list.append(cumulative_reward)
        if cumulative_reward_list[e].shape == (1,):
            cumulative_reward_list[e] = cumulative_reward_list[e][0]
        print(cumulative_reward)
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
    with open('reward/reward_%d.dat' % i, 'w') as f:
        f.write(cumulative_reward_list)
        
    with open('loss/loss_%d.dat' % i, 'w') as f:
        f.write(cumulative_reward_list)
    
    
    
    print("Test passed.")


