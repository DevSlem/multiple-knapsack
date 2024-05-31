import numpy as np
import gymnasium as gym
from gymnasium import spaces


class KnapsackEnv(gym.Env):
    def __init__(self, items = None, capacities = None):
        super(KnapsackEnv, self).__init__()
        
        self.item_dim = 5
        
        if items is not None and capacities is not None:
            self.items = np.array(items) # 아이템의 값, 무게
            self.capacities = np.array(capacities).astype(np.float32) #knapsack의 capacity
            self.num_items = len(items)
            self.num_bags = len(capacities)
        else:
            self.items = None 
            self.capacities = None 
            self.num_items = 0
            self.num_bags = 0 
                    
        # 상태 공간
        # n_items x item_dim (profit, weight, q_value, selected, residual capacities
        # self.state = np.empty((self.num_items * self.n_bags, self.item_dim), dtype=np.float32)
        
        self.time_step = 0
        self.reset(items, capacities)

    def reset(self, items=None, capacities=None):
        # generate problem that has different(random) item num and bag num... 
        # num_items : between 2 and 100
        # num_bags : ?? 10-30
        # profit, weights [10^-6, 1], normalization will be done just before return     
        
        if self.items is None or self.capacities is None:           #Random Generation
            max_value = 10
            max_weight = 10
            max_capacity = 20
            
            self.num_items = np.random.randint(2, 151)
            self.num_bags = np.random.randint(2, 31)
            self.values = np.random.randint(1, max_value + 1, self.num_items)
            self.weights = np.random.randint(1, max_weight + 1, self.num_items)
            self.capacities = np.random.randint(1, max_capacity + 1, self.num_bags)                        
        else:                                                       #Non Random 
            self.values = self.items[:, 0]
            self.weights = self.items[:, 1]
        
        self.ratios = self.values / self.weights
        self.remaining_capacities = np.array(self.capacities) 
        
        self.max_item_value = np.max(self.values)
        self.max_item_weight = np.max(self.weights)      
        
        self.action_space = np.array(self.num_items * self.num_bags)
        
        # Initialize the state with the items and initial capacity
        self.selectable = np.ones((self.num_items, self.num_bags)).astype(bool)
        # self.state = np.empty((self.num_items * self.n_bags, self.item_dim), dtype=np.float32)

        # Update each item's profit, weight, ratios, selectable, remaining capacities     
        state = self._make_state()
        norm_state = self._normalize_state(state) 
        self.time_step = 0
        
        return norm_state[np.newaxis, :, :] #add batch dimension

    def step(self, action):
        item_idx = action // self.num_bags
        bag_idx = action % self.num_bags
        
        done = False
        reward = 0
        norm_reward = 0

        if self.selectable[item_idx, bag_idx] == True and self.weights[item_idx] <= self.remaining_capacities[bag_idx]:
            self.selectable[item_idx, :] = False
            self.remaining_capacities[bag_idx] -= self.weights[item_idx]
            reward += self.values[item_idx]
            norm_reward = np.float32(reward) / np.float32(self.max_item_value) #normalization

        # set selectable
        self._update_selctable_matrix()
        
        self.time_step += 1
        # Check if the problem is done (either by running out of capacity or if all items are considered)
        if np.all(~self.selectable):
            done = True

        # set the 'state'
        next_state = self._make_state()  # Add batch dimension
        norm_next_state = self._normalize_state(next_state)
        
        return norm_next_state[np.newaxis, ...], norm_reward, done, {"value": reward} 
    
    def _make_state(self):
        state = np.empty((self.num_items * self.num_bags, self.item_dim), dtype=np.float32)
        state[:, 0] = np.repeat(self.values, self.num_bags)
        state[:, 1] = np.repeat(self.weights, self.num_bags)
        state[:, 2] = np.repeat(self.ratios, self.num_bags)
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
    
    def _normalize_state(self, state):
        normstate = state.copy()
        normstate[:, 0] /= self.max_item_value #value
        normstate[:, 1] /= self.max_item_weight #weight
        #ratio is same
        #selectable is same
        normstate[:, 4] /= self.max_item_weight #remaining_capacities
        return normstate
        