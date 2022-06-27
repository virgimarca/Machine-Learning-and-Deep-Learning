# -*- coding: utf-8 -*-
"""PPO_UDR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j1qfLg4E7tYixOmOaCg2_S3oJm6FlbIh
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
!pip install stable-baselines3

!unzip classes.zip

import torch
import gym
import argparse
import numpy as npy

from env.custom_hopper_3_10 import *
from env.custom_hopper_10_100 import *
from env.custom_hopper_100_1000 import *
#from env.custom_hopper_1000_10000 import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

device = 'cpu'

env_sources = ['CustomHopper_3_10-source-v0']
env_target= gym.make('CustomHopper_3_10-target-v0')

"""
Hyperparameter tuning design
"""

# 4 learning rates * 4 ranges for the masses 

class Model():
  def __init__(self,model,environment,learning_rate,number_of_episodes,mass):
    self.model              = model
    self.environment        = environment
    self.gym_environment    = gym.make(environment)
    self.learning_rate      = learning_rate
    self.number_of_episodes = number_of_episodes
    self.mass               = mass

  def model_saver(self,string):
    self.model_saved_name   = string

models = []

mean_rewards = []
std_rewards = []

"""
  Training
"""

learning_rates = [1e-3]
n_episodes = [100000]

for lr in learning_rates:
  for nEpisodes in n_episodes:
    for envs in env_sources:
      model = PPO("MlpPolicy", envs,  learning_rate=lr, batch_size=2048, gamma=0.99, verbose = 0, device = device)
      model.learn(total_timesteps= nEpisodes*50, n_eval_episodes = nEpisodes, eval_log_path = '/content/ppo')
      if envs == 'CustomHopper_3_10-source-v0':
        mod = Model(model,envs,lr,nEpisodes,"from3to10kg")
        models.append(mod)
        models[-1].model.save("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from3to10kg.mdl")
        models[-1].model_saver("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from3to10kg.mdl")
      elif envs == 'CustomHopper_10_100-source-v0':
        mod = Model(model,envs,lr,nEpisodes,"from10to100kg")
        models.append(mod)
        models[-1].model.save("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from10to100kg.mdl")
        models[-1].model_saver("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from10to100kg.mdl")
      elif envs == 'CustomHopper_100_1000-source-v0':
        mod = Model(model,envs,lr,nEpisodes,"from100to1000kg")
        models.append(mod)
        models[-1].model.save("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from100to1000kg.mdl")
        models[-1].model_saver("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from100to1000kg.mdl")
      else:
        mod = Model(model,envs,lr,nEpisodes,"from1000to10000kg")
        models.append(mod)
        models[-1].model.save("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from1000to10000kg.mdl")
        models[-1].model_saver("ppoHopper__lr-"+str(lr)+"__numEpisodes-"+str(nEpisodes)+"__masses:from1000to10000kg.mdl")

# Evaluate the trained agent
mockEnv   = 'CustomHopper_3_10-source-v0'
mockModel = PPO("MlpPolicy", mockEnv,  learning_rate=0.1, n_steps=50, batch_size=50, gamma=0.99, verbose = 0, device = device)

best_mean_reward_model_setup  = Model(mockModel, mockEnv, 0.1, 1, "from3to10kg")
best_std_reward_model_setup   = Model(mockModel, mockEnv, 0.1, 1, "from3to10kg")
best_mean_reward = [0,0]
best_std_reward = [0,0]

mean_rewards = []
std_rewards = []

for i in range(len(models)):
  mod = models[i].model
  env = models[i].gym_environment
  
  mean_reward, std_reward = evaluate_policy(models[i].model, env, n_eval_episodes=50)
  mean_rewards.append(mean_reward)
  std_rewards.append(std_reward)

  print(f'Evaluated '+str(i+1)+'/'+str(len(models))+' models.')
  
best_mean_reward_model_setup    = models[mean_rewards.index(max(mean_rewards))]
best_std_reward_model_setup     = models[std_rewards.index(max(std_rewards))]

for i in range(len(models)):
  print("TUNING#"+str(i+1)+": trained on env: "+models[i].environment+" --> "+f"mean_reward:{mean_rewards[i]:.2f} +/- {std_rewards[i]:.2f}")

print("Best setup for mean reward is:\n"+"TUNING#"+str(mean_rewards.index(max(mean_rewards))+1))
print("Best setup for std reward is:\n"+"TUNING#"+str(std_rewards.index(min(std_rewards))+1))

"""
  Testing on Source
"""

test_on_source_average_rewards = []
test_on_source_rewards = []
i = 1
for mod in models:
  model = PPO.load(mod.model_saved_name)
  test_rewards = []
  episodes = 1000

  for episode in range(episodes):
    done = False
    test_reward = 0
    state = mod.gym_environment.reset()

    while not done:
      action, _ = mod.model.predict(state)
      state, reward, done, info = mod.gym_environment.step(action)
      test_reward += reward
    test_rewards.append(test_reward)
    if episode%50==0:
      print(f'EPISODE #'+str(episode)+' from MODEL #'+str(i)+' - REWARD: '+str(test_reward))
  test_on_source_average_rewards.append(np.mean(test_rewards))
  test_on_source_rewards.append(test_rewards)
  print(f'TEST SS ON MODEL #'+str(i)+' DONE!')
  i+=1

print(f'SS TESTING COMPLETED!')

"""
  Testing on Target
"""

test_on_target_average_rewards = []
test_on_target_rewards = []
i = 1
for mod in models:
  model = PPO.load(mod.model_saved_name)
  test_rewards = []
  episodes = 1000

  for episode in range(episodes):
    done = False
    test_reward = 0
    state = env_target.reset()

    while not done:
      action, _ = mod.model.predict(state)
      state, reward, done, info = env_target.step(action)
      test_reward += reward
    test_rewards.append(test_reward)
    if episode%50==0:
      print(f'EPISODE #'+str(episode)+' from MODEL #'+str(i)+' - REWARD: '+str(test_reward))
  test_on_target_average_rewards.append(np.mean(test_rewards))
  test_on_target_rewards.append(test_rewards)
  print(f'TEST SS ON MODEL #'+str(i)+' DONE!')
  i+=1

print(f'ST TESTING COMPLETED!')

import numpy as np
import matplotlib.pyplot as plt

def cumulative_average(x):
  cumul = []
  for i in range(1, len(x)):
    cumul.append(sum(x[:i])/i)
  return cumul

for i in range(len(models)):
  plt.plot(cumulative_average(test_on_target_rewards[i]),label='source-target cumulative average')
  plt.plot(cumulative_average(test_on_source_rewards[i]),label='target-target cumulative average')
plt.title(f'source-target v. source-source')
plt.xlabel(f'Episode')
plt.ylabel(f'Reward')
plt.legend()
plt.savefig('SS v. ST testing using cumulative averages.eps', format='eps')
plt.show

"""
plt.plot(test_rewards, label = f'Episode Reward')
plt.plot(reward_moving_average, label = f'Cumulative Moving Average')
plt.title(f'target - target')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig('VPG-test_target-target_lr_1e-4_100000_ep.eps', format='eps')
plt.show()
"""

for i in range(len(models)):
  plt.plot(test_on_source_rewards[i],label='rewards fro source-source testing')
  plt.plot(cumulative_average(test_on_source_rewards[i]))
plt.title(f'source-source rewards and cumulative average')
plt.xlabel(f'Episode')
plt.ylabel(f'Reward')
plt.legend()
plt.savefig('Source-source testing.eps', format='eps')
plt.show

for i in range(len(models)):
  plt.plot(test_on_target_rewards[i],label='rewards for source-target testing')
  plt.plot(cumulative_average(test_on_target_rewards[i]))
plt.title(f'source-target rewards and cumulative average')
plt.xlabel(f'Episode')
plt.ylabel(f'Reward')
plt.legend()
plt.savefig('Source-target testing.eps', format='eps')
plt.show