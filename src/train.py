import gym
import pfrl

from agents.ddqn_agent import DDQN_Agent

env = gym.make('LunarLander-v2')
agent = DDQN_Agent(env.observation_space, env.action_space)

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=2000,           # Train the agent for 2000 steps
    eval_n_steps=None,       # We evaluate for episodes, not time
    eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
    train_max_episode_len=200,  # Maximum length of each episode
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='weights/ddqn_agent',      # Save everything to 'weights/ddqn_agent' directory
)