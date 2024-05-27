import torch
import torch.nn as nn
import torch.optim as optim
from KnapsackEnv import KnapsackEnv
from util import load_knapsack_problem, make_directory
import argparse
import matplotlib.pyplot as plt
import time
import json
from torch.distributions import Categorical
import pandas as pd

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
        
        self.q_value_layer = nn.Sequential(
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
        q_value = self.q_value_layer(transformer_output)
        return q_value.squeeze(-1)
    
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
    ):
        self.item_dim = item_dim
        self.selectability_flag_idx = selectability_flag_idx
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        
        self.policy_network = KnapsackPolicyNetwork(item_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=1e-6)
        
        self._obs_buffer = []
        self._action_buffer = []
        self._next_obs_buffer = []
        self._reward_buffer = []
        self._terminated_buffer = []
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
        nonselectable_mask = obs[:, :, self.selectability_flag_idx] == 0
           
        logits = self.policy_network(obs.to(self.device))
        logits[nonselectable_mask] = -float('inf')
        policy_dist = Categorical(logits=logits)

        action = policy_dist.sample()
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
        
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        policy_loss = -(ret * action_log_prob.squeeze(0)).mean()
        entropy = entropy.mean()
        
        loss = policy_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._reward_buffer.clear()
        self._action_log_prob_buffer.clear()
        self._entropy_buffer.clear()
        
        return loss.item(), entropy.item()
    
    def _buffer_to_tensor(self):
        reward = torch.stack(self._reward_buffer, dim=1)
        action_log_prob = torch.stack(self._action_log_prob_buffer, dim=1)
        entropy = torch.stack(self._entropy_buffer, dim=1)
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
    summary_freq = 1
    
    for e in range(episodes):
        obs = env.reset()
        obs = torch.from_numpy(obs)
        terminated = False
        cumulative_reward = 0.0
        
        while not terminated:
            action = agent.select_action(obs)
            next_obs, reward, terminated, _ = env.step(action.item())
            
            next_obs = torch.from_numpy(next_obs)
            reward = torch.tensor([reward], dtype=torch.float32)
            terminated = torch.tensor([terminated], dtype=torch.float32)
            
            agent_info = agent.update(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated
            )
            
            obs = next_obs
            cumulative_reward += reward.item()
            
            if agent_info is not None:
                policy_losses.append(agent_info["policy_loss"])
                entropies.append(agent_info["entropy"])
        
        cumulative_reward_list.append(cumulative_reward)
        if e % summary_freq == 0:
            print(f"Episode: {e}, Cumulative Reward: {cumulative_reward}")
            
    return cumulative_reward_list, policy_losses, entropies

def inference(env: KnapsackEnv, agent: KnapsackTransformerREINFORCEAgent):
    obs = env.reset()
    terminated = False
    cumulative_reward = 0.0
    
    while not terminated:
        action = agent.select_action(torch.from_numpy(obs))
        obs, reward, terminated, _ = env.step(action.item())
        cumulative_reward += reward
        
    return cumulative_reward
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--summary_freq", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    problem_name = args.problem_name
    episodes = args.episodes
    summary_freq = args.summary_freq
    
    
    knapsack_df, item_df = load_knapsack_problem(problem_name)
    
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
    )
    
    start_time = time.time()
    cumulative_reward_list, policy_losses, entropies = train(env, agent, episodes, summary_freq)
    end_time = time.time()
    train_time = end_time - start_time
    
    total_value = inference(env, agent)
        
    directory = f"results/{problem_name}/transformer_reinforce"
    make_directory(directory)
    
    result_dict = dict()
    result_dict["method"] = "Transformer REINFORCE"
    result_dict["n_knapsacks"] = len(knapsack_df)
    result_dict["n_items"] = len(item_df)
    result_dict["time"] = train_time
    result_dict["episodes"] = episodes
    result_dict["solution"] = {
        "total_value": total_value,
    }
    
    with open(f"{directory}/result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    
    cumulative_reward_df = pd.DataFrame(cumulative_reward_list, columns=["cumulative_reward"])
    cumulative_reward_df.to_csv(f"{directory}/cumulative_rewards.csv", index=False)
    
    plt.plot(cumulative_reward_list)
    plt.title("Cumulative Rewards")
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.savefig(f"{directory}/cumulative_rewards.png")
    plt.close()
    
    plt.plot(policy_losses)
    plt.title("Policy Losses")
    plt.xlabel('Steps')
    plt.ylabel('Policy Loss')
    plt.savefig(f"{directory}/policy_losses.png")
    plt.close()
    
    plt.plot(entropies)
    plt.title(f"Entropies")
    plt.xlabel('Steps')
    plt.ylabel('Entropy')
    plt.savefig(f"{directory}/entropies.png")
    plt.close()
    