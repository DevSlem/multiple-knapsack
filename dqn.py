import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, obs_dim, n_actions, device="cpu", epoch=3, lr=1e-3, gamma=0.99, max_size=1000, batch_size=32, eps=1.0, eps_decay=0.995, min_eps=0.01, target_net_update_freq=100) -> None:
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
        self.target_net_update_freq = target_net_update_freq
        self.time_step = -1
        
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
        self.time_step += 1
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
        
        self._update_target_net()
        self.eps = max(self.eps * self.eps_decay, self.min_eps)
        
        return td_losses
    
    def _update_target_net(self):
        if self.time_step % self.target_net_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())