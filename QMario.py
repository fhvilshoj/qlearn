from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from collections import deque
import numpy as np
import gym

env = gym.make('SuperMarioBros-1-1-v0')

import random

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

# TODO Shoot, fast forward

action_map = [forward, jump]

#
#
# action_map = [
#     [0,0,0,0,0,1],
#     [0,0,0,0,1,0],
#     [0,0,0,0,1,1],
#     [0,0,0,1,0,0],
#     [0,0,0,1,0,1],
#     [0,0,0,1,1,0],
#     [0,0,0,1,1,1],
#     [0,0,1,0,0,0],
#     [0,0,1,0,0,1],
#     [0,0,1,0,1,0],
#     [0,0,1,0,1,1],
#     [0,0,1,1,0,0],
#     [0,0,1,1,0,1],
#     [0,0,1,1,1,0],
#     [0,0,1,1,1,1],
#     [0,1,0,0,0,0],
#     [0,1,0,0,0,1],
#     [0,1,0,0,1,0],
#     [0,1,0,0,1,1],
#     [0,1,0,1,0,0],
#     [0,1,0,1,0,1],
#     [0,1,0,1,1,0],
#     [0,1,0,1,1,1],
#     [0,1,1,0,0,0],
#     [0,1,1,0,0,1],
#     [0,1,1,0,1,0],
#     [0,1,1,0,1,1],
#     [0,1,1,1,0,0],
#     [0,1,1,1,0,1],
#     [0,1,1,1,1,0],
#     [0,1,1,1,1,1],
#     [1,0,0,0,0,0],
#     [1,0,0,0,0,1],
#     [1,0,0,0,1,0],
#     [1,0,0,0,1,1],
#     [1,0,0,1,0,0],
#     [1,0,0,1,0,1],
#     [1,0,0,1,1,0],
#     [1,0,0,1,1,1],
#     [1,0,1,0,0,0],
#     [1,0,1,0,0,1],
#     [1,0,1,0,1,0],
#     [1,0,1,0,1,1],
#     [1,0,1,1,0,0],
#     [1,0,1,1,0,1],
#     [1,0,1,1,1,0],
#     [1,0,1,1,1,1],
#     [1,1,0,0,0,0],
#     [1,1,0,0,0,1],
#     [1,1,0,0,1,0],
#     [1,1,0,0,1,1],
#     [1,1,0,1,0,0],
#     [1,1,0,1,0,1],
#     [1,1,0,1,1,0],
#     [1,1,0,1,1,1],
#     [1,1,1,0,0,0],
#     [1,1,1,0,0,1],
#     [1,1,1,0,1,0],
#     [1,1,1,0,1,1],
#     [1,1,1,1,0,0],
#     [1,1,1,1,0,1],
#     [1,1,1,1,1,0],
#     [1,1,1,1,1,1]
# ]

# model = Sequential()
# model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
# model.add(Flatten())
# model.add(Dense(18, init='uniform', activation='relu'))
# model.add(Dense(10, init='uniform', activation='relu'))
action_space_size = len(action_map) #env.action_space.shape
# model.add(Dense(action_space_size, init='uniform', activation='linear'))

print(env.observation_space.shape)
shape_ = ((2,) + env.observation_space.shape)
print(shape_)

model = Sequential()
model.add(Conv2D(32, input_shape=shape_, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(action_space_size))

print('got here')

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

D = deque()
observetime = 500
epsilon = 0.7
gamma = 0.9
mb_size = 50

observation = env.reset()
print(observation.shape)
obs = np.expand_dims(observation, axis=0)
# state = np.stack((obs,obs), axis=1)
state = np.stack((observation,observation), axis=0)
print(state.shape)
done = False
for t in range(observetime):
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, action_space_size, size=1)[0]
    else:
        Q = model.predict(state)
        action = np.argmax(Q)

    reward = 0.
    # observation_new = 0
    for a in action_map[action]:
        # observation_new, reward, done, info = env.step(action_map[action])
        observation_new, r, d, i = env.step(a)
        reward += r
        if d:
            done = True
            break
    obs_new = np.expand_dims(observation_new, axis=0)
    # state_new = np.append(np.expand_dims(obs_new, axis=0), state[:,:1,:], axis=1)
    state_new = np.append(np.expand_dims(observation_new, axis=0), state[:1,:], axis=1)
    D.append((state, action, reward, state_new, done))

    state = state_new
    if done:
        # env.reset()
        obs = np.expand_dims(observation, axis=0)
        # state = np.stack((obs, obs), axis=1)
        state = np.stack((observation, observation), axis=1)
print('Observing Finished')

# SECOND STEP

minibatch = random.sample(D, mb_size)

inputs_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros(inputs_shape)
targets = np.zeros((mb_size, action_space_size))

for i in range(0, mb_size):
    state = minibatch[i][0]
    action = minibatch[i][1]
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]

    inputs[i:i+1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)

    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)

    model.train_on_batch(inputs, targets)

print('Learning Finished')

# THIRD STEP: PLAY!

observation = env.reset()
# obs = np.expand_dims(observation, axis=0)
# state = np.stack((obs, obs), axis=1)
state = np.stack((observation, observation), axis=0)
done = False
tot_reward = 0.0
    
while not done:
    # env.render()
    Q = model.predict(state)
    action = np.argmax(Q)

    reward = 0.
    # observation_new = 0
    for a in action_map[action]:
        # observation_new, reward, done, info = env.step(action_map[action])
        observation_new, r, d, i = env.step(a)
        reward += r
        if d:
            done = True
            break

    # observation, reward, done, info = env.step(action_map[action])
    obs = np.expand_dims(observation_new, axis=0)
    # state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    state = np.append(np.expand_dims(observation_new, axis=0), state[:1, :], axis=1)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(reward))

