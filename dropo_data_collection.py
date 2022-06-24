# -*- coding: utf-8 -*-
"""DROPO_data_collection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yFDleBYolDBCb1LNHxK87Rr4kLkH8xWB
"""

import torch
import gym
import argparse

from classes.env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

model = PPO.load("/content/model.mdl")
number_of_states = 500
states = []
next_states = []
actions = []
terminals = []
counter = 0

while len(states) < number_of_states:
  done = False
  state = env_target.reset()
  while not done and len(states) < number_of_states:
    action, _ = model.predict(state)
    previous_state = state
    
    state, reward, done, info = env_target.step(action)

    states.append(previous_state)
    next_states.append(state)
    actions.append(action)
    terminals.append(done)
    counter = counter + 1

with open('data.npy', 'wb') as f:

    np.save(f, np.array(states))
    np.save(f, np.array(next_states))
    np.save(f, np.array(actions))
    np.save(f, np.array(terminals))