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

'''
# Set up the logger to print info messages for understandability.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')


pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=5000,           # Train the agent for 5000 steps
    eval_n_steps=None,       # We evaluate for episodes, not time
    eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
    train_max_episode_len=200,  # Maximum length of each episode
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='weights/ddqn_agent',      # Save everything to 'weights/ddqn_agent' directory
)
'''

def get_screen(env):
    screen = env.render(mode='rgb_array')
    screen = screen.transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.Resize(64, interpolation=Image.CUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(screen)
    return screen


n_episodes = 300
max_episode_len = 200
for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        #env.render()
        obs = get_screen(env)
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')

with agent.eval_mode():
    for i in range(10):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t == 200
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)

agent.save('weights/ddqn_agent/best')

#agent.load('weights/ddqn_agent/best')

state = env.reset()
for _ in range(1000):
    env.render('human')
    state, _, done, _ = env.step(agent.act(state))

    if done:
        break

env.close()