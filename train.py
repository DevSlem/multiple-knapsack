import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gym
from gym import spaces

import argparse
from util import make_directory, load_knapsack_problem

import time
import json

class ReplayBuffer:
    def __init__(self, obs_dim: int, max_size: int, batch_size: int) -> None:
        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0
        self.idx = 0
        
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size,), dtype=np.int64)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size,), dtype=np.float32)
        self.terminated_buf = np.zeros((max_size,), dtype=np.float32)
        
    def store(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, terminated: np.ndarray):
        batch_size = obs.shape[0]
        idxs = np.arange(self.idx, self.idx + batch_size) % self.max_size
        
        self.obs_buf[idxs] = obs
        self.action_buf[idxs] = action
        self.next_obs_buf[idxs] = next_obs
        self.reward_buf[idxs] = reward
        self.terminated_buf[idxs] = terminated
        
        self.idx = (self.idx + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
        
    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        
        obs = self.obs_buf[idxs]
        action = self.action_buf[idxs]
        next_obs = self.next_obs_buf[idxs]
        reward = self.reward_buf[idxs]
        terminated = self.terminated_buf[idxs]
        
        return obs, action, next_obs, reward, terminated
        
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=256) -> None:
        super(QNetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

class DQN:
    def __init__(self, obs_dim, n_actions, device="cpu", epoch=3, lr=1e-3, gamma=0.99, max_size=1000, batch_size=32, eps=1.0, eps_decay=0.995, min_eps=0.01) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.epoch = epoch
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        
        self.q_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(obs_dim, max_size, batch_size)
        
    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Select an action based on the current observation.

        Args:
            obs (ndarray): `(n_envs, obs_dim)`

        Returns:
            action (ndarray): `(n_envs,)`
        """
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            # 이미 텐서일 경우, clone().detach()를 사용하여 연산 그래프로부터 분리
            obs = obs.clone().detach().to(self.device)
        # obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        q_values = self.q_network(obs)
        return torch.argmax(q_values, dim=-1).cpu().numpy()
    
    def update(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, terminated: np.ndarray):
        """
        Update DQN.

        Args:
            obs (ndarray): `(n_envs, obs_dim)`
            action (ndarray): `(n_envs,)`
            next_obs (ndarray): `(n_envs, obs_dim)`
            reward (ndarray): `(n_envs,)`
            terminated (ndarray): `(n_envs,)`
        """
        self.replay_buffer.store(obs, action, next_obs, reward, terminated)
        
        if self.replay_buffer.size < self.replay_buffer.batch_size:
            return
        
        return self._train()
        
    def _train(self):
        td_losses = []
        for _ in range(self.epoch):
            # sample from replay buffer
            obs, action, next_obs, reward, terminated = self.replay_buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            terminated = torch.tensor(terminated, dtype=torch.float32).to(self.device)
            
            # feedforward
            q_values = self.q_network(obs) # Q(s, :)
            with torch.no_grad():
                next_q_values = self.target_network(next_obs) # Q(s', :) 
                
            # compute td loss
            q_values = q_values.gather(dim=1, index=action.unsqueeze(-1)).squeeze(-1) # Q(s, a)
            q_target_values = reward + (1 - terminated) * self.gamma * torch.max(next_q_values, dim=-1).values
            td_loss = F.mse_loss(q_values, q_target_values)
            
            # training step
            self.optimizer.zero_grad()
            td_loss.backward()
            self.optimizer.step()

            td_losses.append(td_loss.item())
        
        self.eps = max(self.eps * self.eps_decay, self.min_eps)
        
        return td_losses

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
        item_values = torch.as_tensor(self.state['item_values'], dtype=torch.float32)
        item_weights = torch.as_tensor(self.state['item_weights'], dtype=torch.float32)
        remaining_capacities = torch.as_tensor(self.state['remaining_capacities'], dtype=torch.float32)
        selection_status = torch.as_tensor(self.state['selection_status'], dtype=torch.float32)
        
        return torch.cat([item_values, item_weights, remaining_capacities, selection_status])
    
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
        
        self.time_step += 1
        if self.time_step > 50000:
            done = True

        return self.prepare_state(), reward, done

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    args = parser.parse_args()
    _problem_name = args.problem_name
    for i in range(1, 11):
        problem_name = _problem_name + str(i)
        knapsack_df, item_df = load_knapsack_problem(problem_name)
        capacities = knapsack_df['capacity'].values
        values = item_df['value'].values
        weights = item_df['weight'].values

        items = []
        for i in range(len(values)):
            items.append((values[i], weights[i]))
            
        env = MultiKnapsackEnv(items, capacities)

        EPISODES = 30
        dqn = DQN(2 * len(values) + len(capacities) + len(values), len(values) * len(capacities))
        td_loss_list = []
        cumulative_reward_list = []
        value_list = []
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
                print(f'td_losses : {td_loss_list[-1]}')

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

        train_time = time.time() - start_time

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
        
        
        


