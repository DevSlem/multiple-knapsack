import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

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
                nhead=8,
                batch_first=True,
            ),
            num_layers=6,
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
    
class KnapsackTransformerDQNAgent:
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
        device: str = "cuda",
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
        
        self.q_network = KnapsackQNetwork(item_dim).to(self.device)
        self.target_network = KnapsackQNetwork(item_dim).to(self.device)
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
        if random.random() < self.eps:
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
        n_knapsack_x_items = obs.shape[1]
        self._replay_buffer(n_knapsack_x_items).store(obs, action, next_obs, reward, terminated)
        
        if self._replay_buffer(n_knapsack_x_items).size >= self.batch_size:
            self._train(n_knapsack_x_items)
            
    def _train(self, n_knapsack_x_items: int):
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
            
        self._update_target_net()
        self.eps = max(self.eps * self.eps_decay, self.min_eps)
            
    def _update_target_net(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def _replay_buffer(self, n_knapsack_x_items: int) -> ReplayBuffer:
        if n_knapsack_x_items not in self.replay_buffer_dict:
            self.replay_buffer_dict[n_knapsack_x_items] = ReplayBuffer((n_knapsack_x_items, self.item_dim), self.replay_buffer_max_size, self.batch_size)
        return self.replay_buffer_dict[n_knapsack_x_items]
    