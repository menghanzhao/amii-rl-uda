import gymnasium as gym
import gym_puddle
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy as DQNPolicy
import time
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend like 'Agg' for batch processing
import numpy as np
# from IPython import display
# import pyvirtualdisplay
# import cv2

# select agents
agent = 'human' # 'human' or 'random' or 'RL'

#some functions to help the visualization and interaction wit the environment
def visualize(frames, video_name = "video.mp4"):
    # Saves the frames as an mp4 video using cv2
    video_path = video_name
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def online_rendering(image):
    #Visualize one frame of the image in a display
    ax.axis('off')
    img_with_frame = np.zeros((image.shape[0]+2, image.shape[1]+2, 3), dtype=np.uint8)
    img_with_frame[1:-1, 1:-1, :] = image
    ax.imshow(img_with_frame)
    display.display(plt.gcf())
    display.clear_output(wait=True)

def prepare_display():
  #Prepares display for onine rendering of the frames in the game
  _display = pyvirtualdisplay.Display(visible=False,size=(1400, 900))
  _ = _display.start()
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.axis('off')


def get_action():
    action = None
    while action not in ["w", "a", "s", "d", "W", "A", "S", "D"]:
        action = input("Enter action (w/a/s/d): ")
    if action == "w":
        return 3
    elif action == "a":
        return 0
    elif action == "s":
        return 2
    elif action == "d":
        return 1


env = gym.make("PuddleWorld-v0", render_mode="human")  # you should set the render_mode to "human" to visualize the environment locally. If you are running this code snippet on colab, these lines won't work since colab doesn't support virtual display screens
env.reset(seed=12222222)    # reset the environment to start a new episode

env.render()   # this will open a window to visualize the environment
time.sleep(5)  # wait for 5 seconds so that you can see the window
env.close()    # you should close the environment to close the virtual window at the end of your code

# environment configuration from kaggle (5 cases pw1-pw5)
json_file = '/Users/jansenfs/PycharmProjects/competition/gym-puddle/gym_puddle/env_configs/pw2.json'  # include the path to the json file here
with open(json_file) as f:
    env_setup = json.load(f)  # load the json file with the environment configuration
#
# # initialize environment
# env = gym.make(  # initialize the environment with the corresponding values
#     "PuddleWorld-v0",
#     start=env_setup["start"],
#     goal=env_setup["goal"],
#     goal_threshold=env_setup["goal_threshold"],
#     noise=env_setup["noise"],
#     thrust=env_setup["thrust"],
#     puddle_top_left=env_setup["puddle_top_left"],
#     puddle_width=env_setup["puddle_width"],
# )
# obs, info = env.reset()
# image = env.render()
# time.sleep(5)  # wait for 5 seconds so that you can see the window
# fig, ax = plt.subplots(figsize=(5, 5))
# online_rendering(image)


### Compare random agent/human/RL agent
## 1. random action
if agent == 'random':
    # prepare_display() #uncomment this line to see the online rendering of the environment frame by frame
    env = gym.make("PuddleWorld-v0")
    obs, info = env.reset()
    total_reward = 0
    episode_rewards = []
    frames = []

    for time_step in range(10):
        action = env.action_space.sample()  # take a random action
        observation, reward, done, trunc, info = env.step(action)
        total_reward += reward

        image = env.render()
        #online_rendering(image) #uncomment this line to see the online rendering of the environment frame by frame
        frames.append(image)

        print(f" t: {time_step}, observation: {observation}, reward: {reward}") #uncomment this line to see the environment-agent interaction details

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

    visualize(frames, "random.mp4")

## 2. human agent
elif agent == 'human':
    #prepare_display() #uncomment this line to see the online rendering of the environment frame by frame
    env = gym.make("PuddleWorld-v0")

    obs, info = env.reset(seed=1)
    total_reward = 0
    episode_rewards = []
    frames = []

    for time_step in range(10):
        action = get_action()
        observation, reward, done, trunc, info = env.step(action)
        total_reward += reward

        image = env.render()
        time.sleep(5)  # wait for 5 seconds so that you can see the window
        fig, ax = plt.subplots(figsize=(5, 5))
        online_rendering(image) #uncomment this line to see the online rendering of the environment frame by frame
        frames.append(image)

        print(f" t: {time_step}, observation: {observation}, reward: {reward}")  #uncomment this line to see the environment-agent interaction details

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
    visualize(frames, "human.mp4")

## 3. RL agent
elif agent == 'RL':
    #train the model, and save the trained model
    env = gym.make("PuddleWorld-v0")
    dqn_model = DQN(DQNPolicy, env, verbose=1)
    dqn_model.learn(total_timesteps=int(1e5))
    dqn_model.save("dqn_model")

    # test the trained model
    dqn_model = DQN.load("dqn_model")
    env = gym.make("PuddleWorld-v0")
    obs, info = env.reset()

    # Create an empty list to store the frames
    frames = []
    episode_rewards = []

    for episode in range(1):
      total_reward = 0
      done = False
      num_steps = 0

      while not done and num_steps <=1000: # to avoid infinite loops for the untuned DQN we set a truncation limit, but you should make your agent sophisticated enough to avoid infinite-step episodes
          num_steps +=1
          action, _states = dqn_model.predict(obs)
          observation, reward, done, trunc, info = env.step(action)
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

    visualize(frames, "DQN.mp4")