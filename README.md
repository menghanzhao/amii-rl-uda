# Reviving Puddle World - Upper Bound (2024) AI Competition
Puddle World is an environment that got traction in the 1990s which was studied by [Boyan and Moore (1995)](https://www.ri.cmu.edu/pub_files/pub1/boyan_justin_1995_1/boyan_justin_1995_1.pdf) and then later picked up by Rich Sutton in the same year. [Rich (1995)](https://proceedings.neurips.cc/paper_files/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html) conducted various experiments on the environment, and we use this study as a reference for our implementation of the Puddle World.

The agent starts at an initial state (denoted in red) in the Puddle World and the task for the agent is to navigate around the puddles (denoted in black) to reach the goal state (denoted in green). 

<p align="center">
  <kbd>
    <img src='puddle_world.png'/>
  </kbd>
</p>

Building upon the details from Rich's study:

In Puddle World, there are four directional movements available to the agent: up, down, right, and left. Each movement shifts the agent by approximately 0.05 units in the specified direction, with adjustments made to prevent the agent from moving beyond the boundaries of the space. Additionally, a random Gaussian noise, with a standard deviation of 0.01, is introduced to the action along both dimensions. The task's reward imposes a penalty of -1 for each time step. Further penalties are incurred if the agent enters either or both of the two oval-shaped "puddles." These penalties are calculated as -400 times the distance into the puddle, measured from the nearest edge. The puddles are positioned at coordinates [0. , 0.85] and [0.35, 0.9], with respective widths and heights of [0.55, 0.2] and [0.2, 0.6]. The starting state for the agent is situated at [0.2, 0.4], and the ultimate objective is to reach the point [1.0, 1.0].

This repository is an extension of the previous open-source implementation of the environment. This implementation is compatible with the gymnasium library, making it easy to interact with the environment.


## Installation
Make a virtual env for your project

```python
python -m venv myenv
source myenv/bin/activate
```

First, you will need to clone the repo by the following command:

```
git clone https://github.com/Amii-Open-Source/gym-puddle.git
```

Then navigate to the repo directory by using the `cd` command, and run the following install command. 

```python
cd path/to/directory
pip install -e .
```

You can also find the details about the needed python and library versions in `setup.py`.

## Usage
```python
import gymnasium as gym
import gym_puddle # Don't forget this extra line!

env = gym.make('PuddleWorld-v0')
```

##  Configurations
Your task is to train an agent that can generalize well across different provided configurations of the environment. Each of these configurations feature different positions for puddles, which makes it challenging for the agent to find the most rewarding path to the goal.

You can find these configurations in the `env_configs` folder of the repository. 
You can specify one of the `.json` files for the various environment configurations provided, where `pw1.json` corresponds to the original Puddle World environment in the paper.
You can then intitialize the Puddle World as mentioned in the  `getting_started.ipynb` Colab guide.
Here is a snippet of how you can intitalize your environment with the desired configuration:

```python
json_file = 'path/to/json/' #include the path to the json file here

with open(json_file) as f:
  env_setup = json.load(f) #load the json file with the environment configuration
  
env = gym.make( #initialize the environment with the corresponding values
  "PuddleWorld-v0",
  start=env_setup["start"],
  goal=env_setup["goal"],
  goal_threshold=env_setup["goal_threshold"],
  noise=env_setup["noise"],
  thrust=env_setup["thrust"],
  puddle_top_left=env_setup["puddle_top_left"],
  puddle_width=env_setup["puddle_width"],
)

```


# More Details and Getting Started
For more details on how to get started with the environment, refer to `getting_started.ipynb` Colab file. In this guide, we go through how to install the environment, how to access the environment details, and how to initialize an instance with the desired configurations. 

Furthermore, we show you how to visualize the environment, and how you can run a random, human, and DQN agent on the environment. We illustrate how a simple training loop would look like and provide guidance on how to make a submission file for the Kaggle Competition.
