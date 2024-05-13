import numpy as np
import gymnasium as gym
from gymnasium import spaces


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
        self.ratios = self.values / self.weights
        
        # 행동 공간
        self.action_space = np.array(self.num_items * self.n_bags)

        # 상태 공간
        # n_items x item_dim (profit, weight, q_value, selected, residual capacities
        # self.state = np.empty((self.num_items * self.n_bags, self.item_dim), dtype=np.float32)
        
        self.time_step = 0
        self.reset()

    def reset(self):
        # Initialize the state with the items and initial capacity
        self.selectable = np.ones((self.num_items, self.n_bags)).astype(bool)
        # self.state = np.empty((self.num_items * self.n_bags, self.item_dim), dtype=np.float32)
        self.remaining_capacities = self.capacities.copy()

        # Update each item's profit, weight, ratios, selectable, remaining capacities     
        state = self._make_state()
        
        self.time_step = 0
        return state[np.newaxis, :, :] #add batch dimension

    def step(self, action):
        # item_idx = action % self.num_items
        # bag_idx = action // self.num_items
        item_idx = action // self.n_bags
        bag_idx = action % self.n_bags
        
        done = False
        reward = 0

        if self.selectable[item_idx, bag_idx] == True and self.weights[item_idx] <= self.remaining_capacities[bag_idx]:
            self.selectable[item_idx, :] = False
            self.remaining_capacities[bag_idx] -= self.weights[item_idx]
            reward += self.values[item_idx]

        # set selectable
        self._update_selctable_matrix()
        
        self.time_step += 1
        # Check if the problem is done (either by running out of capacity or if all items are considered)
        if np.all(~self.selectable):
            done = True

        # set the 'state'
        next_state = self._make_state()  # Add batch dimension
        return next_state[np.newaxis, ...], reward, done, {}
    
    def _make_state(self):
        state = np.empty((self.num_items * self.n_bags, self.item_dim), dtype=np.float32)
        state[:, 0] = np.repeat(self.values, self.n_bags)
        state[:, 1] = np.repeat(self.weights, self.n_bags)
        state[:, 2] = np.repeat(self.ratios, self.n_bags)
        state[:, 3] = self.selectable.astype(np.float32).flatten()
        state[:, 4] = np.tile(self.remaining_capacities, self.num_items)
        return state
    
    def _update_selctable_matrix(self):
        # selectable: (n_items, n_bags)
        # weights: (n_items,)
        # remaining_capacities: (n_bags,)
        # broadcast weights and remaining_capacities to (n_items, n_bags)
        # then, compare with selectable
        temp = self.weights[..., np.newaxis] <= self.remaining_capacities[np.newaxis, ...]
        self.selectable = np.logical_and(self.selectable, temp)
        