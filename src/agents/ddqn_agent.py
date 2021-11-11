'''
Code Adapted from PFRL Quickstart Guide
https://github.com/pfnet/pfrl/blob/master/examples/quickstart/quickstart.ipynb
'''

import torch
import pfrl
import numpy

DEFAULT_CONFIG = {
    'OPTIMIZER_LR': 1e-2,  # TORCH OPTIMIZER LEARNING RATE
    'GAMMA': 0.9,  # REWARD DISCOUNT FACTOR
    'EPSILON': 0.3, # EPISODE DISCOVERY FACTOR
    'REPLAYBUFFER_CAPACITY': 10 ** 6, # REPLAY BUFFER CAPACITY
    'TORCH_DEVICE': 0, # PYTORCH DEVICE
}
        

def DDQN_Agent(observation_space, action_space, config=DEFAULT_CONFIG):
    obs_size = observation_space.low.size
    n_actions = action_space.n

    class Q_NET(torch.nn.Module):
        def __init__(self, obs_size, n_actions):
            super().__init__()
            self.l1 = torch.nn.Linear(obs_size, 50)
            self.l2 = torch.nn.Linear(50, 50)
            self.l3 = torch.nn.Linear(50, n_actions)

        def forward(self, x):
            h = x
            h = torch.nn.functional.relu(self.l1(h))
            h = torch.nn.functional.relu(self.l2(h))
            h = self.l3(h)
            return pfrl.action_value.DiscreteActionValue(h)
            
    q_func = Q_NET(obs_size, n_actions)
    q_func2 = Q_NET(obs_size, n_actions)

    optimizer = torch.optim.Adam(q_func.parameters(), eps=config['OPTIMIZER_LR'])

    explorer = pfrl.explorers.ConstantEpsilonGreedy(
        epsilon=config['EPSILON'],
        random_action_func=action_space.sample
    )

    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=config['REPLAYBUFFER_CAPACITY'])
    return pfrl.agents.DoubleDQN(
        q_func,
        #pfrl.q_functions.DuelingDQN(n_actions, n_frames),
        optimizer,
        replay_buffer,
        config['GAMMA'],
        explorer,
        replay_start_size=500,
        update_interval=1,
        target_update_interval=100,
        gpu=config['TORCH_DEVICE'],
    )
