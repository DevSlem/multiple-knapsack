import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn

class MultiKnapsackEnv(gym.Env):
    def __init__(self, items, capacities):
        super(MultiKnapsackEnv, self).__init__()
        self.items = np.array(items)  # 아이템 (값, 무게)
        self.capacities = np.array(capacities)  # 배낭의 용량
        
        self.num_items = len(items)
        self.num_bags = len(capacities)
        
        # 행동 공간 (각 아이템을 각 배낭에 넣을 수 있는 선택지)
        self.action_space = spaces.Discrete(self.num_items * self.num_bags)
        
        # 상태 공간
        # 아이템의 값, 무게, 배낭의 남은 용량, 아이템의 선택 상태
        self.observation_space = spaces.Dict({
            'item_values': spaces.Box(low=0, high=np.inf, shape=(self.num_items,), dtype=np.float32),
            'item_weights': spaces.Box(low=0, high=np.inf, shape=(self.num_items,), dtype=np.float32),
            'remaining_capacities': spaces.Box(low=0, high=np.max(capacities), shape=(self.num_bags,), dtype=np.float32),
            'selection_status': spaces.MultiBinary(self.num_items)
        })
        
        self.time_step = -1
        
        self.state = None
        self.reset()

    def prepare_state(self):
        item_values = np.array(self.state['item_values'], dtype=np.float32)
        item_weights = np.array(self.state['item_weights'], dtype=np.float32)
        remaining_capacities = np.array(self.state['remaining_capacities'], dtype=np.float32)
        selection_status = np.array(self.state['selection_status'], dtype=np.float32)
        
        ret = np.concatenate([item_values, item_weights, remaining_capacities, selection_status])
        ret = ret.reshape(1, len(ret))
        return ret
        # item_values = torch.as_tensor(self.state['item_values'], dtype=torch.float32)
        # item_weights = torch.as_tensor(self.state['item_weights'], dtype=torch.float32)
        # remaining_capacities = torch.as_tensor(self.state['remaining_capacities'], dtype=torch.float32)
        # selection_status = torch.as_tensor(self.state['selection_status'], dtype=torch.float32)
        
        # ret = torch.cat([item_values, item_weights, remaining_capacities, selection_status])

        # return ret.reshape(1, len(ret))
    def reset(self):
        self.state = {
            'item_values': self.items[:, 0],
            'item_weights': self.items[:, 1],
            'remaining_capacities': self.capacities.copy(),
            'selection_status': np.zeros(self.num_items, dtype=int)
        }
        self.time_step = -1

        return self.prepare_state()
    
    def step(self, action):
        item_idx = action % self.num_items
        bag_idx = action // self.num_items
        
        item_value = self.items[item_idx, 0]
        item_weight = self.items[item_idx, 1]
        
        if self.state['selection_status'][item_idx] == 0 and self.state['remaining_capacities'][bag_idx] >= item_weight:
            self.state['remaining_capacities'][bag_idx] -= item_weight
            self.state['selection_status'][item_idx] = 1
            reward = item_value
        else:
            reward = 0  # 아이템을 추가할 수 없는 경우 보상 없음
        
        # 모든 아이템이 선택되었는지, 또는 더 이상 아이템을 추가할 수 없는지 확인
        no_more_fits = np.all(self.state['remaining_capacities'] < np.min(self.items[self.state['selection_status'] == 0, 1]))
        done = np.all(self.state['selection_status']) or no_more_fits
        
        mask = self.valid_actions()
        # self.time_step += 1
        # if self.time_step > 100:
        #     done = True

        return self.prepare_state(), reward, done, mask

    def valid_actions(self):
        mask = np.zeros(self.num_items * self.num_bags, dtype=bool)
        for i in range(self.num_items):
            for j in range(self.num_bags):
                if self.state["selection_status"][i] == 0 and self.state['remaining_capacities'][j] >= self.items[i][1]:
                    mask[i + j * self.num_items] = True
        return mask.reshape(1, len(mask))