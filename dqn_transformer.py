import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from KnapsackEnv import KnapsackEnv
from util import load_knapsack_problem, make_directory, save_results
import argparse
import matplotlib.pyplot as plt
import time

class ReplayBuffer:
    def __init__(self, obs_shape: tuple[int, ...], max_size: int, batch_size: int):
        self.obs_shape = obs_shape
        self.max_size = max_size
        self.batch_size = batch_size
        
        self.obs_buf = torch.zeros((max_size, *obs_shape), dtype=torch.float32)
        self.action_buf = torch.zeros(max_size, dtype=torch.long)
        self.next_obs_buf = torch.zeros((max_size, *obs_shape), dtype=torch.float32)
        self.reward_buf = torch.zeros(max_size, dtype=torch.float32)
        self.terminated_buf = torch.zeros(max_size, dtype=torch.float32)
        
        self.idx = 0
        self.size = 0
        
    def store(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor):
        """
        Store a experience batch in the replay buffer.

        Args:
            obs (Tensor): `(batch_size, *obs_shape)`
            action (Tensor): `(batch_size,)`
            next_obs (Tensor): `(batch_size, *obs_shape)`
            reward (Tensor): `(batch_size,)`
            terminated (Tensor): `(batch_size,)`
        """
        batch_size = obs.shape[0]
        idxs = torch.arange(self.idx, self.idx + batch_size) % self.max_size
        
        self.obs_buf[idxs] = obs.to(dtype=torch.float32)
        self.action_buf[idxs] = action.to(dtype=torch.long)
        self.next_obs_buf[idxs] = next_obs.to(dtype=torch.float32)
        self.reward_buf[idxs] = reward.to(dtype=torch.float32)
        self.terminated_buf[idxs] = terminated.to(dtype=torch.float32)
        
        self.idx = (self.idx + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

        
    def sample(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the replay buffer.
        """
        idxs = torch.randint(0, self.size, (self.batch_size,))
        
        obs = self.obs_buf[idxs]
        action = self.action_buf[idxs]
        next_obs = self.next_obs_buf[idxs]
        reward = self.reward_buf[idxs]
        terminated = self.terminated_buf[idxs]
        
        return obs.to(device), action.to(device), next_obs.to(device), reward.to(device), terminated.to(device)

class KnapsackQNetwork(nn.Module):
    def __init__(
        self,
        item_dim: int,
        num_layers: int,
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
            num_layers=num_layers,
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
    
class KnapsackDQNTransformerAgent:
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
        epoch: int = 1,
        gamma: float = 1.0,
        eps: float = 1.0,
        eps_decay: float = 0.995,
        min_eps: float = 0.05,
        tau: float = 0.05,
        replay_buffer_max_size: int = 100000,
        batch_size: int = 512,
        num_layers: int = 2,
        device: str = "cuda",
        inference: bool = False,
    ):
        self.item_dim = item_dim
        self.selectability_flag_idx = selectability_flag_idx
        self.epoch = epoch
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.tau = tau
        self.replay_buffer_max_size = replay_buffer_max_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.inference = inference
        
        self.q_network = KnapsackQNetwork(item_dim, num_layers).to(self.device)
        self.target_network = KnapsackQNetwork(item_dim, num_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=1e-6)
        
        self.replay_buffer_dict = dict()
        
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Select an action based on the current observation.

        Args:
            obs (Tensor): `(num_envs, n_knapsacks x n_items, item_dim)`

        Returns:
            action (Tensor): `(num_envs,)`, the index of the selected knapsack and item
        """
        num_envs, n_knapsack_x_items, _ = obs.shape
        nonselectable_mask = obs[:, :, self.selectability_flag_idx] == 0
        
        # epsilon-greedy policy
        if not self.inference and random.random() < self.eps:
            rand_logits = torch.rand((num_envs, n_knapsack_x_items))
            rand_logits[nonselectable_mask] = -float('inf')
            return rand_logits.argmax(dim=-1).cpu()
        
        q_value = self.q_network(obs.to(self.device))
        q_value[nonselectable_mask] = -float('inf')
        return q_value.argmax(dim=-1).cpu()
        
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
        
        n_knapsack_x_items = obs.shape[1]
        self._replay_buffer(n_knapsack_x_items).store(obs, action, next_obs, reward, terminated)
        
        if self._replay_buffer(n_knapsack_x_items).size >= self.batch_size:
            td_losses = self._train(n_knapsack_x_items)
            return {
                "td_losses": td_losses,
                "eps": self.eps,
            }
            
    def _train(self, n_knapsack_x_items: int):
        td_losses = []
        
        for _ in range(self.epoch):
            obs, action, next_obs, reward, terminated = self._replay_buffer(n_knapsack_x_items).sample(self.device)
            
            # feedforward
            q_values = self.q_network(obs)
            with torch.no_grad():
                next_greedy_action = self.target_network(next_obs).argmax(dim=-1)
                next_q_values = self.q_network(next_obs)
                
            # compute td loss
            q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
            next_q_value = next_q_values.gather(1, next_greedy_action.unsqueeze(-1)).squeeze(-1)
            target_q_value = reward + (1 - terminated) * self.gamma * next_q_value
            td_loss = F.mse_loss(q_value, target_q_value)
            
            # training step
            self.optimizer.zero_grad()
            td_loss.backward()
            self.optimizer.step()
            
            td_losses.append(td_loss.item())
            
        self._update_target_net()
        self.eps = max(self.eps * self.eps_decay, self.min_eps)
        
        return td_losses
            
    def _update_target_net(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def _replay_buffer(self, n_knapsack_x_items: int) -> ReplayBuffer:
        if n_knapsack_x_items not in self.replay_buffer_dict:
            self.replay_buffer_dict[n_knapsack_x_items] = ReplayBuffer((n_knapsack_x_items, self.item_dim), self.replay_buffer_max_size, self.batch_size)
        return self.replay_buffer_dict[n_knapsack_x_items]
    
    
def train(env: KnapsackEnv, agent: KnapsackDQNTransformerAgent, episodes: int, summary_freq: int):
    cumulative_reward_list = []
    td_losses = []
    epsilons = []
    summary_freq = 1
    _start_time = time.time()
    
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
                td_losses += agent_info["td_losses"]
                epsilons.append(agent_info["eps"])
        
        cumulative_reward_list.append(cumulative_reward)
        if e % summary_freq == 0:
            print(f"Training time: {time.time() - _start_time:.2f}, Episode: {e}, Cumulative Reward: {cumulative_reward}")
            
    return cumulative_reward_list, td_losses, epsilons

def inference(env: KnapsackEnv, agent: KnapsackDQNTransformerAgent):
    obs = env.reset()
    terminated = False
    total_value = 0.0
    
    while not terminated:
        action = agent.select_action(torch.from_numpy(obs))
        obs, reward, terminated, info = env.step(action.item())
        total_value += info["value"]
        
    return round(total_value.item())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("problem_name", type=str)
    parser.add_argument("--inference", "-i", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--summary_freq", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--min_eps", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--replay_buffer_max_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    # problem_name = args.problem_name
    inference_problem = args.inference
    episodes = args.episodes
    summary_freq = args.summary_freq
    
    if inference_problem is None:
        env = KnapsackEnv()
        
        agent = KnapsackDQNTransformerAgent(
            item_dim=5,
            selectability_flag_idx=3,
            epoch=args.epoch,
            gamma=args.gamma,
            eps=args.eps,
            eps_decay=args.eps_decay,
            min_eps=args.min_eps,
            tau=args.tau,
            replay_buffer_max_size=args.replay_buffer_max_size,
            batch_size=args.batch_size,
            num_layers=2,
            device=args.device,
        )
        
        start_time = time.time()
        cumulative_reward_list, td_losses, epsilons = train(env, agent, episodes, summary_freq)
        end_time = time.time()
        train_time = end_time - start_time
        
        directory = "results/train/dqn_transformer"
        make_directory(directory)
        ckpt_dict = {
            "agent": agent.q_network.state_dict(),
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
        
        plt.plot(td_losses)
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
        knapsack_df, item_df = load_knapsack_problem(inference_problem)
        env = KnapsackEnv(
            items=item_df[['value', 'weight']].values,
            capacities=knapsack_df['capacity'].values,
        )
        
        agent = KnapsackDQNTransformerAgent(
            item_dim=5,
            selectability_flag_idx=3,
            epoch=0,
            gamma=args.gamma,
            eps=0.0,
            eps_decay=args.eps_decay,
            min_eps=0.0,
            tau=args.tau,
            replay_buffer_max_size=args.replay_buffer_max_size,
            batch_size=args.batch_size,
            num_layers=2,
            device=args.device,
            inference=True,
        )
        
        ckpt_dict = torch.load(f"results/train/dqn_transformer/checkpoint.pt")
        agent.q_network.load_state_dict(ckpt_dict["agent"])
    
        start_time = time.time()
        total_value = inference(env, agent)
        inference_time = time.time() - start_time

        result_df = save_results(
            problem_name=inference_problem,
            method="DQN with Transformer",
            total_value=total_value,
            episodes=ckpt_dict["episodes"],
            train_time=ckpt_dict["train_time"],
            inference_time=inference_time,
        )
        
        print("Inference results (the last one is the current result):")
        print(result_df)
