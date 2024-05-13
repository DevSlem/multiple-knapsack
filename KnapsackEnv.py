import numpy as np
import gym 
from gym import spaces


class KnapsackEnv(gym.Env):
    def __init__(self, items, capacities):
        super(KnapsackEnv, self).__init__()
        
        self.item_dim = 5
        self.items = np.array(items) # 아이템의 값, 무게
        self.num_items = len(items)
        
        self.capacities = np.array(capacities) #knapsack의 capacity
        self.n_bags = len(capacities)
        
        self.values = self.items[:, 0]
        self.weights = self.items[:, 1]
        self.remaining_capacities = np.array(capacities)
        self.selectable = np.ones(self.num_items)
        self.ratios = self.values / self.weights

        
        # 행동 공간
        self.action_space = np.array(self.num_items * self.n_bags)

        # 상태 공간
        # n_items x item_dim (profit, weight, q_value, selected, residual capacities
        self.state = np.empty((self.num_items * self.n_bags, self.item_dim), dtype=np.float32)
        
        self.time_step = 0
        self.reset()

    def reset(self):
        # Initialize the state with the items and initial capacity
        self.state = np.empty((self.num_items * self.n_bags, self.item_dim), dtype=np.float32)

        # Update each item's profit, weight, ratios, selectable, remaining capacities     
        self.state[:, 0] = np.tile(self.values, self.n_bags)
        self.state[:, 1] = np.tile(self.weights, self.n_bags)
        self.state[:, 2] = np.tile(self.ratios, self.n_bags)
        self.state[:, 3] = np.tile(np.ones(self.num_items), self.n_bags)
        self.state[:, 4] = np.tile(self.capacities, self.num_items)
        
        self.time_step = 0
        return self.state[np.newaxis, :, :] #add batch dimension

    def step(self, action):
        item_idx = action % self.num_items
        bag_idx = action // self.num_items
        
        done = False
        reward = 0

        if self.selectable[item_idx] == 1 and self.weights[item_idx] <= self.remaining_capacities[bag_idx]:
            self.selectable[item_idx] = 0
            self.remaining_capacities[bag_idx] -= self.weights[item_idx]
            reward += self.values[item_idx]

        self.time_step += 1
        # Check if the problem is done (either by running out of capacity or if all items are considered)
        if self.time_step >= self.num_items or np.all(self.remaining_capacities <= 0):
            done = True

        # set the 'state'
        next_state = self.state[np.newaxis, :, :]  # Add batch dimension
        return next_state, reward, done, {}