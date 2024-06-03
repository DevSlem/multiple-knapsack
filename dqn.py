import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from multiKnapsackEnv import *

import argparse
from util import make_directory, load_knapsack_problem, save_results

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
    def __init__(self, obs_dim, n_actions, n_nonselectability_flags, device="cpu", epoch=3, lr=1e-3, gamma=0.99, max_size=1000, batch_size=32, eps=1.0, eps_decay=0.995, min_eps=0.01, target_net_update_freq=100) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_nonselectability_flags = n_nonselectability_flags
        self.device = torch.device(device)
        self.epoch = epoch
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.target_net_update_freq = target_net_update_freq
        self.time_step = -1
        
        self.q_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(obs_dim, max_size, batch_size)
        
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, mask) -> np.ndarray:
        """
        Select an action based on the current observation.

        Args:
            obs (ndarray): `(n_envs, obs_dim)`

        Returns:
            action (ndarray): `(n_envs,)`
        """
        # mask = np.zeros(self.n_actions, dtype=bool)
        # for i in range(self.n_nonselectability_flags):
        #     for j in range(self.n_actions / self.n_nonselectability_flags):
        #         if self.selection_status[i] == 0 and self.bag_capacities[j] >= self.item_weights[i]:
        #             mask[i + j * self.num_items] = True
        # return mask

        # nonselectability_flags = obs[:, -self.n_nonselectability_flags:]
        # nonselectable_mask = np.concatenate([nonselectability_flags for _ in range(self.n_actions // self.n_nonselectability_flags)], axis=-1).astype(bool)
        
        if np.random.rand() < self.eps:
            rand_logits = np.random.rand(obs.shape[0], self.n_actions)
            rand_logits[~torch.BoolTensor(mask)] = -float('inf')
            return np.argmax(rand_logits, axis=-1)
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            # 이미 텐서일 경우, clone().detach()를 사용하여 연산 그래프로부터 분리
            obs = obs.clone().detach().to(self.device)
        # obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        q_values = self.q_network(obs)
        q_values[~torch.BoolTensor(mask)] = -float('inf')
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
        self.time_step += 1
        self.replay_buffer.store(obs, action, next_obs, reward, terminated)
        
        if self.replay_buffer.size < self.replay_buffer.batch_size:
            return

        td_losses = self._train()
        return {
            "td_losses": td_losses,
            "eps": self.eps,
        }
        
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
        
        self._update_target_net()
        self.eps = max(self.eps * self.eps_decay, self.min_eps)
        
        return td_losses
    
    def _update_target_net(self):
        if self.time_step % self.target_net_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def inference(env, dqn):
    obs = env.reset()
    terminated = False
    total_value = 0.0
    mask = np.zeros((1, len(values) * len(capacities)), dtype=bool)
            
    while not terminated:
        action = dqn.select_action(obs, mask)
        obs, reward, terminated, mask = env.step(action)
        total_value += reward
        print(total_value)

    return total_value

def train(env, dqn, episodes):
    td_loss_list = []
    cumulative_reward_list = []
    epsilons = []
    start_time = time.time()
    for e in range(episodes):
        obs = env.reset()
        terminated = False
        cumulative_reward = 0.0
        mask = np.zeros((1, len(values) * len(capacities)), dtype=bool)
        while not terminated:
            # select action
            action = dqn.select_action(obs, mask)
            
            # take the action then observe next state and reward
            next_obs, reward, terminated, _mask = env.step(action)
            
            # action = np.array([action]) # (n_envs,)
            # next_obs = next_obs[np.newaxis, :] # (n_envs, obs_dim)
            # reward = np.array([reward]) # (n_envs,)
            # terminated = np.array([terminated]) # (n_envs,)
            
            # update DQN
            dqn_info = dqn.update(obs, action, next_obs, reward, terminated)
            if dqn_info is not None:
                td_loss_list.append(np.mean(dqn_info["td_losses"]))
                epsilons.append(dqn_info["eps"])
            # transition to the next state
            obs = next_obs
            mask = _mask
            cumulative_reward += reward
            # print(f'cumulative_reward : {cumulative_reward}')

        
        cumulative_reward_list.append(cumulative_reward)
        if cumulative_reward_list[e].shape == (1,):
            cumulative_reward_list[e] = cumulative_reward_list[e][0]
        print(f'episode: {e}, cumulative_reward : {cumulative_reward[0]}')
    return cumulative_reward_list, td_loss_list, epsilons

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    parser.add_argument("--inference", "-i", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    args = parser.parse_args()
    inference_problem = args.inference
    problem_name = args.problem_name
    episodes = args.episodes
    
    if inference_problem == False:
        # dqn = DQN(2 * len(values) + len(capacities) + len(values), len(values) * len(capacities), len(values))
        knapsack_df, item_df = load_knapsack_problem(problem_name)
        num_items = len(item_df)
        num_bags = len(knapsack_df)
        values = item_df['value'].values
        weights = item_df['weight'].values
        capacities = knapsack_df['capacity'].values
        items = []
        for i in range(num_items):
            items.append((values[i], weights[i]))
        env = MultiKnapsackEnv(items, capacities)
        
        
        dqn = DQN(3 * num_items + num_bags, num_bags * num_items, num_items)
        start_time = time.time()
        cumulative_reward_list, td_loss_list, epsilons = train(env, dqn, episodes = episodes)
        end_time = time.time()
        train_time = end_time - start_time

        directory = "results/train/dqn/" + problem_name
        make_directory(directory)
        ckpt_dict = {
            "agent": dqn.q_network.state_dict(),
            "train_time": train_time,
            "episodes": episodes,
        }
        torch.save(ckpt_dict, f"{directory}/checkpoint.pt")

        plt.plot(cumulative_reward_list)
        plt.title("Cumulative Rewards")
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.savefig(f"{directory}/cumulative_rewards.png")
        plt.close()
        
        plt.plot(td_loss_list)
        plt.title("TD Losses")
        plt.xlabel('Steps')
        plt.ylabel('TD Loss')
        plt.savefig(f"{directory}/td_losses.png")
        plt.close()
        
        plt.plot(epsilons)
        plt.title(f"Epsilon Decay {args.eps_decay}")
        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        plt.savefig(f"{directory}/epsilons.png")
        plt.close()

    else:
        problem_name = args.problem_name

        knapsack_df, item_df = load_knapsack_problem(problem_name)
        capacities = knapsack_df['capacity'].values
        values = item_df['value'].values
        weights = item_df['weight'].values

        items = []
        for i in range(len(values)):
            items.append((values[i], weights[i]))
            
        env = MultiKnapsackEnv(items, capacities)
        dqn = DQN(2 * len(values) + len(capacities) + len(values), len(values) * len(capacities), len(values), eps=0.0, min_eps=0.0)
        ckpt_dict = torch.load(f"results/train/dqn/" + problem_name + "/checkpoint.pt")
        dqn.q_network.load_state_dict(ckpt_dict["agent"])
        
        start_time = time.time()
        total_value = inference(env, dqn)
        inference_time = time.time() - start_time

        result_df = save_results(
            problem_name=problem_name,
            method="DQN",
            total_value=total_value,
            episodes=ckpt_dict["episodes"],
            train_time=ckpt_dict["train_time"],
            inference_time=inference_time,
        )

        print("Inference results (the last one is the current result):")
        print(result_df)
