import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from KnapsackEnv import KnapsackEnv
from util import (load_knapsack_problem, make_directory, moving_average,
                  save_results)


class KnapsackPolicyNetwork(nn.Module):
    def __init__(
        self,
        item_dim: int,
    ):
        super().__init__()
        
        self.obs_embedding_layer = nn.Sequential(
            nn.Linear(item_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=4,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        self.policy_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network. 

        Args:
            obs (Tensor): `(batch_size, n_knapsacks x n_items, item_dim)`
            
        Returns:
            q_value (Tensor): `(batch_size, n_knapsacks x n_items)`
        """
        obs_embedding = self.obs_embedding_layer(obs)
        transformer_output = self.transformer_layer(obs_embedding)
        logits = self.policy_layer(transformer_output)
        logits = 10.0 * logits.tanh()
        return logits.squeeze(-1)
    
class KnapsackTransformerREINFORCEAgent:
    """
    In this problem, the observation of a state is represented by the matrix obtained stacking all the item vectors together. 
    The observation shape is `(n_knapsacks x n_items, item_dim)`.
    Both `n_knapsacks` and `n_items` are not constant and can vary because they're represented as a sequence. 
    Transformer is permutation-invariant, so their order doesn't matter. 
    The item vector can be represented as [value, weight, value/weight, selectability_flag, remaining_knapsack_capacity].
    
    Reference: https://research.tudelft.nl/en/publications/reinforcement-learning-for-the-knapsack-problem
    """
    def __init__(
        self, 
        item_dim: int,
        selectability_flag_idx: int,
        gamma: float = 1.0,
        entropy_coef: float = 0.001,
        device: str = "cuda",
        inference: bool = False,
    ):
        self.item_dim = item_dim
        self.selectability_flag_idx = selectability_flag_idx
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        self.inference = inference
        
        self.policy_network = KnapsackPolicyNetwork(item_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=1e-6)
        
        self._reward_buffer = []
        self._action_log_prob_buffer = []
        self._entropy_buffer = []
        
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Select an action based on the current observation.

        Args:
            obs (Tensor): `(num_envs, n_knapsacks x n_items, item_dim)`

        Returns:
            action (Tensor): `(num_envs,)`, the index of the selected knapsack and item
        """
        nonselectable_mask = obs[:, :, self.selectability_flag_idx] < 0.5
           
        logits = self.policy_network(obs.to(self.device))
        logits[nonselectable_mask] = -float('inf')
        policy_dist = Categorical(logits=logits)

        action = policy_dist.sample()
        
        if not self.inference:
            self._action_log_prob = policy_dist.log_prob(action)
            self._entropy = policy_dist.entropy()
        
        return action.detach().cpu()
        
    def update(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor):
        """
        Update the agent with a single step of experience.

        Args:
            obs (Tensor): `(num_envs, n_knapsacks x n_items, item_dim)`
            action (Tensor): `(num_envs,)`
            next_obs (Tensor): `(num_envs, n_knapsacks x n_items, item_dim)`
            reward (Tensor): `(num_envs,)`
            terminated (Tensor): `(num_envs,)`
        """
        if self.inference:
            return
        
        self._reward_buffer.append(reward)
        self._action_log_prob_buffer.append(self._action_log_prob)
        self._entropy_buffer.append(self._entropy)
        
        if terminated.item():
            loss, entropy = self._train()       
            return {
                "policy_loss": loss,
                "entropy": entropy,
            }
            
    def _train(self):
        # (1, episode_len, *shape)
        reward, action_log_prob, entropy = self._buffer_to_tensor()
        ret = self._compute_return(reward.squeeze(0))
        
        std = ret.std() + 1e-8 if len(ret) > 1 else 1.0
        ret = (ret - ret.mean()) / std
        policy_loss = -(ret * action_log_prob.squeeze(0)).mean()
        entropy = entropy.mean()
        
        loss = policy_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
        return loss.item(), entropy.item()
    
    def _buffer_to_tensor(self):
        reward = torch.stack(self._reward_buffer, dim=1).to(self.device)
        action_log_prob = torch.stack(self._action_log_prob_buffer, dim=1).to(self.device)
        entropy = torch.stack(self._entropy_buffer, dim=1).to(self.device)
        
        self._reward_buffer.clear()
        self._action_log_prob_buffer.clear()
        self._entropy_buffer.clear()
        return reward, action_log_prob, entropy
    
    def _compute_return(self, reward: torch.Tensor) -> torch.Tensor:
        ret = torch.empty_like(reward)
        G = 0.0
        for t in reversed(range(len(ret))):
            G = reward[t] + self.gamma * G
            ret[t] = G
        return ret
    
    
def train(env: KnapsackEnv, agent: KnapsackTransformerREINFORCEAgent, episodes: int, summary_freq: int):
    cumulative_reward_list = []
    policy_losses = []
    entropies = []
    total_values = []
    _start_time = time.time()
    
    for e in range(episodes):
        obs = env.reset()
        obs = torch.from_numpy(obs)
        done = False
        cumulative_reward = 0.0
        total_value = 0.0
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action.item())
            
            next_obs = torch.from_numpy(next_obs)
            reward = torch.tensor([reward], dtype=torch.float32)
            terminated = torch.tensor([done], dtype=torch.float32)
            
            agent_info = agent.update(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated
            )
            
            obs = next_obs
            cumulative_reward += reward.item()
            total_value += info["value"]
            
            if agent_info is not None:
                policy_losses.append(agent_info["policy_loss"])
                entropies.append(agent_info["entropy"])
        
        cumulative_reward_list.append(cumulative_reward)
        total_values.append(total_value)
        if e % summary_freq == 0:
            print(f"Training time: {time.time() - _start_time:.2f}, Episode: {e}, Cumulative Reward: {cumulative_reward}")
            
    return cumulative_reward_list, policy_losses, entropies, total_values

def inference(env: KnapsackEnv, agent: KnapsackTransformerREINFORCEAgent):
    obs = env.reset()
    terminated = False
    total_value = 0
    
    while not terminated:
        action = agent.select_action(torch.from_numpy(obs))
        obs, reward, terminated, info = env.step(action.item())
        total_value += info["value"]
        
    return total_value
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("problem_name", type=str)
    parser.add_argument("--inference", "-i", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--summary_freq", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    # problem_name = args.problem_name
    inference_problem = args.inference
    episodes = args.episodes
    summary_freq = args.summary_freq

    
    if inference_problem is None:
        env = KnapsackEnv()
        
        agent = KnapsackTransformerREINFORCEAgent(
            item_dim=5,
            selectability_flag_idx=3,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            device=args.device,
        )
        
        start_time = time.time()
        cumulative_reward_list, policy_losses, entropies, _ = train(env, agent, episodes, summary_freq)
        end_time = time.time()
        train_time = end_time - start_time
        
        directory = "results/train/reinforce"
        make_directory(directory)
        
        ckpt_dict = {
            "agent": agent.policy_network.state_dict(),
            "train_time": train_time,
            "episodes": episodes,
        }
        torch.save(ckpt_dict, f"{directory}/checkpoint.pt")
        
        plt.plot(moving_average(cumulative_reward_list, smooth=0.01))
        plt.title("Cumulative Rewards")
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.savefig(f"{directory}/cumulative_rewards.png")
        plt.close()
        
        plt.plot(moving_average(policy_losses, smooth=0.01))
        plt.title("Policy Losses")
        plt.xlabel('Steps')
        plt.ylabel('Policy Loss')
        plt.savefig(f"{directory}/policy_losses.png")
        plt.close()
        
        plt.plot(moving_average(entropies, smooth=0.01))
        plt.title(f"Entropies")
        plt.xlabel('Steps')
        plt.ylabel('Entropy')
        plt.savefig(f"{directory}/entropies.png")
        plt.close()
    
    else:
        knapsack_df, item_df = load_knapsack_problem(inference_problem)
        env = KnapsackEnv(
            items=item_df[['value', 'weight']].values,
            capacities=knapsack_df['capacity'].values,
        )
        
        agent = KnapsackTransformerREINFORCEAgent(
            item_dim=5,
            selectability_flag_idx=3,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            device=args.device,
            inference=True,
        )
        
        ckpt_dict = torch.load(f"results/train/reinforce/checkpoint.pt")
        agent.policy_network.load_state_dict(ckpt_dict["agent"])
        
        start_time = time.time()
        total_value = inference(env, agent)
        end_time = time.time()
        inference_time = end_time - start_time
        
        result_df = save_results(
            problem_name=inference_problem,
            method="REINFORCE",
            total_value=total_value,
            episodes=ckpt_dict["episodes"],
            train_time=ckpt_dict["train_time"],
            inference_time=inference_time,
        )
        
        print("Inference results (the last one is the current result):")
        print(result_df)
