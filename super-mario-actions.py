import gym

import numpy as np
import gym

env = gym.make('SuperMarioBros-1-1-v0')

env.reset()

# forward
forward = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1],
]
jump = [
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
]

for a in a1:
    observation_new, reward, done, info = env.step(a)

print("done")
