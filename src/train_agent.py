'''
Code Adapted from PFRL Quickstart Guide
https://github.com/pfnet/pfrl/blob/master/examples/quickstart/quickstart.ipynb
'''
import gym
import pfrl
import logging
import sys
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

from agents.ddqn_agent import DDQN_Agent

env = gym.make('LunarLander-v2')
agent = DDQN_Agent(env.observation_space, env.action_space)

# Set up the logger to print info messages for understandability.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')


pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=50000,           # Train the agent for 5000 steps
    eval_n_steps=None,       # We evaluate for episodes, not time
    eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
    train_max_episode_len=200,  # Maximum length of each episode
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='weights/ddqn_agent',      # Save everything to 'weights/ddqn_agent' directory
)

#agent.load('weights/ddqn_agent/best')

state = env.reset()
for _ in range(1000):
    env.render('human')
    state, _, done, _ = env.step(agent.act(state))

    if done:
        break

env.close()