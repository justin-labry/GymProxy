import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from examples.gamblers_problem import *

options = {
    'start_loc': 12,
    # goal_loc is not specified, so it will be randomly sampled
}


env = gym.make('GridWorld-v0')
print(env)
#
# obs, info = env.reset(seed=1, options=options)
# print(obs)
# print(info)

# for t in range(5):
#     # img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")
#
#     action = env.action_space.sample()
#     obs, rew, terminated, truncated, info = env.step(action)
#     print(obs, rew, terminated, truncated, info)

episodes = 1

for episode in range(1, episodes + 1):
    state, info = env.reset()
    done = False
    score = 0

    while not done:  # try alternatively while True to see full fail
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)  # in newer version five arguments (truncated between done and info)
        score += reward
        env.render()  # need pip install pyglet
        print(f"Episode {episode} obs: {obs} info {info}  Score: {score}")

env.close()