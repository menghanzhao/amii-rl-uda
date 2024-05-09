#%%
import gymnasium as gym
import gym_puddle
env = gym.make("PuddleWorld-v0")

#obs, _ = env.reset()
start = env.start
env.reset()
distances = env.distance_to_puddle_edges()
escape_direction = env.closest_escape_direction()
# print(obs)
# %%
