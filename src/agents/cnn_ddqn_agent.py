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
    n_frames = 3

    class Q_NET(torch.nn.Module):
        def __init__(self, n_frames, n_actions):
            super(Q_NET, self).__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_frames, 32, 8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, stride=1),
                torch.nn.Flatten(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, n_actions),
                pfrl.q_functions.DiscreteActionValueHead(),
            )      

        def forward(self, x):
            return self.net(x)

    q_func = Q_NET(n_frames, n_actions)
    q_func2 = Q_NET(n_frames, n_actions)

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
