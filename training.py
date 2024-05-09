import gymnasium as gym
import math
import random
import gym_puddle
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import (DQN, A2C, PPO, HER)
from stable_baselines3.dqn import MlpPolicy as DQNPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import time
import json
import numpy as np
import os

ENV_CONFIGS = [
   'pw1.json',
   'pw2.json',
   'pw3.json',
   'pw4.json',
   'pw5.json'
]

def get_env_setup(current_env_num):

  env_num = random.randint(0, 4)
  if isinstance(current_env_num, int):
    while env_num == current_env_num:
      env_num = random.randint(0, 4)
  env_name = ENV_CONFIGS[env_num]
  dir = os.getcwd()
  json_dir = os.path.join(dir,'gym_puddle', 'env_configs', env_name)
  with open(json_dir) as f:
    env_setup = json.load(f)
  env = gym.make(
    "PuddleWorld-v0",
    start=env_setup["start"],
    goal=env_setup["goal"],
    goal_threshold=env_setup["goal_threshold"],
    noise=env_setup["noise"],
    thrust=env_setup["thrust"],
    puddle_top_left=env_setup["puddle_top_left"],
    puddle_width=env_setup["puddle_width"],
  )

  return env_num, env

def train_model(total_episodes):
  # initialize
  current_env_num = None
  current_env_num, env = get_env_setup(current_env_num)
  model = DQN(DQNPolicy, env, verbose=1)

  for episode in range(1, total_episodes + 1):
    obs, _ = env.reset()
    done = False
    total_reward = 0 # total rewards achieved in an episode
    total_steps = 0 # total steps made in an episode

    # train model for 1 episode
    model.learn(total_timesteps=int(1e5))

    while not done and total_reward >= -10000 and total_steps <= 500:
      action, _states = model.predict(obs)
      obs, reward, done, trunc, _ = env.step(action)
      total_reward += reward
      total_steps += 1
    if episode % 10 == 0:
      env.close()
      current_env_num, env = get_env_setup(current_env_num)
      model.set_env(env)

  return model

def evaluate_model(model, eval_episodes, env):
  obs, info = env.reset()

  # Create an empty list to store the frames
  frames = []
  episode_rewards = []

  for episode in range(1, eval_episodes + 1):
    total_reward = 0
    done = False
    num_steps = 0

    while not done and num_steps <=500 and total_reward >= -10000: # to avoid infinite loops for the untuned DQN we set a truncation limit, but you should make your agent sophisticated enough to avoid infinite-step episodes
        num_steps +=1
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if done == True:
          print("here")

        image = env.render()
        frames.append(image)

        if done:
          print(f"total reward in this episode: {total_reward}")
          episode_rewards.append(total_reward)
          total_reward = 0
          break

  env.close()

  if episode_rewards == []:
    print("no episode finished in this run.")
  else:
    for i, reward in enumerate(episode_rewards):
      print(f"episode {i}: reward: {reward}")
  
  return frames, episode_rewards


if __name__ == '__main__':
  #train the model, and save the trained model
  total_episodes = 10000
  trained_model = train_model(total_episodes)
  # dqn_model.save("dqn_model")

  # test the trained model
  # dqn_model = DQN.load("dqn_model")
  env = gym.make("PuddleWorld-v0")
  obs, info = env.reset()
  frames, episode_rewards = evaluate_model(trained_model, 10, env)

