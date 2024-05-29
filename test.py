import unittest
from dqn_transformer import KnapsackTransformerDQNAgent
import torch
import numpy as np
from KnapsackEnv import KnapsackEnv

class TestClass(unittest.TestCase):
    def test_single_knapsack_transformer_dqn(self):
        item_dim = 5
        n_knapsacks = 1
        n_items = 10
        episodes = 100
        num_envs = 1
        
        items = [(np.random.randint(1, 11), np.random.randint(1, 11)) for _ in range(n_items)]
        capacities = [np.random.randint(10, 50) for _ in range(n_knapsacks)]
        
        agent = KnapsackTransformerDQNAgent(
            item_dim=item_dim,
            selectability_flag_idx=3,
            replay_buffer_max_size=100,
            batch_size=16,
            device="cpu"
        )
        
        env = KnapsackEnv(items=items, capacities=capacities)
        
        for _ in range(episodes):
            #obs = torch.randn(num_envs, n_knapsacks * n_items, item_dim)
            obs = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)#added
            
            terminated = False
            
            while not terminated:
                action = agent.select_action(obs).item()
                next_obs, reward, terminated, _ = env.step(action)
                next_obs = torch.tensor(next_obs, dtype=torch.float32)#.unsqueeze(0)  # Add batch dimension
                reward= torch.tensor([reward], dtype=torch.float32)
                terminated = torch.tensor([terminated], dtype=torch.float32)
                
                """
                action = agent.select_action(obs)
                next_obs = torch.randn(num_envs, n_knapsacks * n_items, item_dim)
                reward = torch.randn(num_envs)
                terminated = torch.rand(num_envs) < 0.3
                """
                
                agent.update(
                    obs=obs,
                    #action=action,
                    action = torch.tensor([action], dtype=torch.float32), #added
                    next_obs=next_obs,
                    reward=reward,
                    terminated=terminated
                )
                
                obs = next_obs
                
if __name__ == '__main__':
    unittest.main()