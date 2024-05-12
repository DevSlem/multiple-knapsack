import unittest
from transformer_dqn import SingleKnapsackTransformerDQNAgent
import torch

class TestClass(unittest.TestCase):
    def test_single_knapsack_transformer_dqn(self):
        item_dim = 5
        n_items = 10
        episodes = 100
        num_envs = 1
        
        agent = SingleKnapsackTransformerDQNAgent(
            item_dim=item_dim,
            replay_buffer_max_size=100,
            batch_size=16,
        )
        
        for _ in range(episodes):
            obs = torch.randn(num_envs, n_items, item_dim)
            terminated = torch.tensor([False])
            
            while not terminated:
                action = agent.select_action(obs)
                next_obs = torch.randn(num_envs, n_items, item_dim)
                reward = torch.randn(num_envs)
                terminated = torch.rand(num_envs) < 0.3
                
                agent.update(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    reward=reward,
                    terminated=terminated
                )
                
                obs = next_obs
                
if __name__ == '__main__':
    unittest.main()