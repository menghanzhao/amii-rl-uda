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
import pandas as pd

ENV_CONFIGS = [
   'pw1.json',
   'pw2.json',
   'pw3.json',
   'pw4.json',
   'pw5.json'
]


def evaluate_model(model, eval_episodes, env):

  # Create an empty list to store the frames
  frames = []
  episode_rewards = []
  episode_steps = []

  for episode in range(1, eval_episodes + 1):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done: # to avoid infinite loops for the untuned DQN we set a truncation limit, but you should make your agent sophisticated enough to avoid infinite-step episodes
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
    episode_rewards.append(total_reward)
    image = env.render()
    frames.append(image)
    print(f"total reward in this episode: {total_reward}")
    print(f"total steps in this episode: {env.num_steps}")

  env.close()
  
  return frames, episode_rewards

def evaluate_submit(model, seed, config):
  dir = os.getcwd()
  json_dir = os.path.join(dir, 'gym_puddle', 'env_configs', config)
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
  env.evaluation = True
  obs, info = env.reset(seed=seed)
  total_reward = 0
  done = False
  while not done:
    action, _states = model.predict(obs)
    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward
  env.close()
  return total_reward


def generate_test_results_for_submission(model):
  pass


if __name__ == '__main__':
  #train the model, and save the trained model
  ## Initialize environment
  dir = os.getcwd()
  json_dir = os.path.join(dir,'gym_puddle', 'env_configs', 'pw1.json')
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

  ## Initialize model
  env.reset()
  model = DQN(DQNPolicy ,env, verbose=1)
  iter = 0
  while iter < 10:
    iter += 1
    model.learn(total_timesteps=20)
    model.save(f"dqn_model_iter{iter}")

  # test the trained model
  # dqn_model = DQN.load("dqn_model")
  # env = gym.make(
  #   "PuddleWorld-v0",
  #   # start=env_setup["start"],
  #   # goal=env_setup["goal"],
  #   # goal_threshold=env_setup["goal_threshold"],
  #   # noise=env_setup["noise"],
  #   # thrust=env_setup["thrust"],
  #   # puddle_top_left=env_setup["puddle_top_left"],
  #   # puddle_width=env_setup["puddle_width"],
  # )
  obs, info = env.reset()
  frames, episode_rewards = evaluate_model(model, 10, env)

  # print(frames)




  # SUBMISSION
  data = []
  seeds = range(1, 101)  # Seeds from 1 to 100
  configs = ['pw1.json', 'pw2.json', 'pw3.json', 'pw4.json', 'pw5.json']

  # Evaluate the model
  for seed in seeds:
    seed_data = {'seed_ID': seed}
    for i, config in enumerate(configs):
      reward = evaluate_submit(model, seed, config)
      seed_data[f'ep_reward_pw{i + 1}'] = reward
    data.append(seed_data)

  # Create DataFrame and save to CSV
  df = pd.DataFrame(data)
  df.to_csv('submission.csv', index=False)
  print("Evaluation completed and results saved.")

