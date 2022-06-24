# -*- coding: utf-8 -*-
"""Train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZwbjQBzlmyVTJ200kMANu9ikYd4BN2ja

Install and load all dependencies (first time only) \
NOTE: you may need to restart the runtime afterwards (CTRL+M .).
"""

!apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common

!apt-get install -y patchelf

!pip install gym
!pip install free-mujoco-py

"""Set up the custom Hopper environment



1.   Upload `classes.zip` to the current session's file storage
2.   Un-zip it by running cell below

"""

!unzip classes.zip

from google.colab import drive
drive.mount('/content/drive')

"""---

\

**Train an RL agent on the OpenAI Gym Hopper environment**

\


Choose which agent you want to train.
"""

import torch
import gym
import argparse

from classes.env.custom_hopper import *
from classes.actor_critic import Agent, Policy
#from classes.vanilla import Agent, Policy
import numpy as np
import matplotlib.pyplot as plt

"""# TRAINING FOR VANILLA POLICY GRADIENT AND ACTOR CRITIC"""

#env = gym.make('CustomHopper-source-v0')
env = gym.make('CustomHopper-target-v0')

print('Action space:', env.action_space)
print('State space:', env.observation_space)
print('Dynamics parameters:', env.get_parameters())

"""First we test the performance of the algorithm using different learning rates values, using just 20000 episodes to see which performs best."""

n_episodes = 20000
print_every = 2000
device = 'cpu'

observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.shape[-1]
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
lr_rewards = {}
for lr in learning_rates:
  policy = Policy(observation_space_dim, action_space_dim)
  agent = Agent(lr, 1, policy, device=device)
  total_rewards = []
  for episode in range(n_episodes):
    done = False
    train_reward = 0
    state = env.reset()  # Reset the environment and observe the initial state
    loss = []
    while not done:  # Loop until the episode is over

      action, action_probabilities = agent.get_action(state)
      previous_state = state

      state, reward, done, info = env.step(action.detach().cpu().numpy())

      agent.store_outcome(previous_state, state, action_probabilities, reward, done)

      train_reward += reward

    agent.update_policy()
    total_rewards.append(train_reward)
    agent.empty_outcome()
        
    if (episode+1)%print_every == 0:
      print('Training episode:', episode)
      print('Episode return:', train_reward)

  lr_rewards[lr] = total_rewards

from re import I
import numpy as np
import matplotlib.pyplot as plt

def cumulative_moving_average(x):
  cumsum = []
  for i in range(1, len(x)):
    cumsum.append(sum(x[i-100:i])/100)
  return cumsum

fig, ax = plt.subplots()
for key, value in lr_rewards.items():
  ax.plot(cumulative_moving_average(value), label = f'learning rate {key}')
ax.set_title(f'Actor Critic - source - different learning rates')
ax.set_xlabel('Episode')
ax.set_ylabel('Average reward of last 100 episodes')
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width, pos.height*0.85])
lgd = ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), ncol = 2, fancybox = True)
plt.savefig('/content/drive/MyDrive/ACPG_source_different_learning_rates_20000_ep.eps', format='eps', bbox_extra_artists = (lgd,), bbox_inches = 'tight')
plt.show

"""Now we use the value of the learning rate which performed better to train a model with 100000 episodes"""

n_episodes = 100000
print_every = 2000
device = 'cpu'

"""
  Training
"""
observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.shape[-1]
alpha = 0.01**(1/99999)
policy = Policy(observation_space_dim, action_space_dim)
agent = Agent(1e-2, 1, policy, device=device)
total_rewards = []
for episode in range(n_episodes):
  
  done = False
  train_reward = 0
  state = env.reset()  # Reset the environment and observe the initial state
  
  while not done:  # Loop until the episode is over

    action, action_probabilities = agent.get_action(state)
    previous_state = state

    state, reward, done, info = env.step(action.detach().cpu().numpy())

    agent.store_outcome(previous_state, state, action_probabilities, reward, done)

    train_reward += reward

  agent.update_policy()
  total_rewards.append(train_reward)
  agent.empty_outcome()

  if (episode+1)%print_every == 0:
    print('Training episode:', episode)
    print('Episode return:', train_reward)

torch.save(agent.policy.state_dict(), "/content/drive/My Drive/model_target_AC_lr_-2")