import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import copy
import pygame

import numpy as np
import random
import os
import json


class PuddleEnv(gymnasium.Env):
    def __init__(
            self,
            start: list[float] = [0.2, 0.4],
            goal: list[float] = [1.0, 1.0],
            goal_threshold: float = 0.1,
            noise: float = 0.01,
            thrust: float = 0.05,
            puddle_top_left: list[list[float]] = [[0, 0.85], [0.35, 0.9]],
            puddle_width: list[list[float]] = [[0.55, 0.2], [0.2, 0.6]],
            render_mode: str = "rgb_array",
    ) -> None:
        """
        Initialize the PuddleEnv environment.

        Args:
            start (list[float]): Starting position of the agent.
            goal (list[float]): Goal position.
            goal_threshold (float): Threshold distance to consider the agent has reached the goal.
            noise (float): Magnitude of the noise added to the agent's actions.
            thrust (float): Magnitude of the agent's thrust.
            puddle_top_left (list[list[float]]): List of puddle top left positions.
            puddle_width (list[list[float]]): List of puddle width values.
        """

        self.start = np.array(start)
        self.goal = np.array(goal)

        self.goal_threshold = goal_threshold

        self.noise = noise
        self.thrust = thrust

        self.puddle_top_left = [np.array(top_left) for top_left in puddle_top_left]
        self.puddle_width = [np.array(width) for width in puddle_width]

        self.action_space = spaces.Discrete(4)

        self.obs_low = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1, -1, 0, -1, -1])
        self.obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1, 1, 4, 1, 1])
        self.observation_space = spaces.Box(self.obs_low, self.obs_high, shape=(12,), dtype=np.float64)

        self.actions = [np.zeros(2) for i in range(4)]

        for i in range(4):
            self.actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

        self.num_steps = 0
        self.total_episodes = 0

        # Rendering
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 400
        self.min_reward = self.find_min_reward()
        self.heatmap = False
        self.env = 1
        self.evaluation = False
        self.noisehistory = []

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform a step in the environment.

        Args:
            action (int): Action to be taken by the agent.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Tuple containing the new position, reward, done flag, trunc flag, and additional information.
        """
        self.num_steps += 1
        trunc = False  # we don't have a truncation condition for this environment
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        noise = self.np_random.normal(loc=0.0, scale=self.noise, size=(2,))
        self.pos += self.actions[action] + noise
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = self._get_reward(self.pos)

        done = (np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold) or (self.num_steps > 5000)

        if done:
            self.total_episodes += 1
        distances = self.distance_to_puddle_edges()

        if len(self.noisehistoryx) >= 10:
            self.noisehistoryx.pop(0)
        self.noisehistoryx.append(noise[0])
        # Calculate the mean of elements in the list
        current_meanx = sum(self.noisehistoryx) / len(self.noisehistoryx)

        if len(self.noisehistoryy) >= 10:
            self.noisehistoryy.pop(0)
        self.noisehistoryy.append(noise[1])
        # Calculate the mean of elements in the list
        current_meany = sum(self.noisehistoryy) / len(self.noisehistoryy)

        obs_lst = list(self.pos)
        for item in distances:
            obs_lst.append(item)
        obs_lst.append(self.env)
        for item in noise:
            obs_lst.append(item)
        obs_lst.append(self.radar_guidance())
        obs_lst.append(current_meanx)
        obs_lst.append(current_meany)

        return np.array(obs_lst), reward, done, trunc, {}

    def distance_to_puddle_edges(self):
        x = self.pos[0]
        y = self.pos[1]
        min_left = min_right = min_top = min_bottom = 1

        # Iterate over each puddle
        for top_left, width in zip(self.puddle_top_left, self.puddle_width):
            x_left, y_top = top_left
            w, h = width
            x_right = x_left + w
            y_bottom = y_top + h

            # Calculate horizontal distances if the y-coordinate of the agent is within the vertical span of the puddle
            if y_top <= y <= y_bottom:
                if x_left < x:  # Puddle is to the left
                    min_left = min(min_left, x - x_left)
                if x_right > x:  # Puddle is to the right
                    min_right = min(min_right, x_right - x)

            # Calculate vertical distances if the x-coordinate of the agent is within the horizontal span of the puddle
            if x_left <= x <= x_right:
                if y_top < y:  # Puddle is above
                    min_top = min(min_top, y - y_top)
                if y_bottom > y:  # Puddle is below
                    min_bottom = min(min_bottom, y_bottom - y)

        # Prepare the distances dictionary, converting np.inf to None for clarity
        distances = {
            'left': 1 if min_left == 1 else min_left,
            'right': 1 if min_right == 1 else min_right,
            'top': 1 if min_top == 1 else min_top,
            'bottom': 1 if min_bottom == 1 else min_bottom
        }
        escape_dir = self.closest_escape_direction()
        if escape_dir != 0:
            distances['left'] = - distances['left']
            distances['right'] = - distances['right']
            distances['top'] = - distances['top']
            distances['bottom'] = - distances['bottom']

        return list(distances.values())

    def closest_escape_direction(self):
        x = self.pos[0]
        y = self.pos[1]
        escape_direction = 0  # 0 means agent is not inside any puddle

        for top_left, width in zip(self.puddle_top_left, self.puddle_width):
            x_left, y_top = top_left
            w, h = width
            x_right = x_left + w
            y_bottom = y_top + h

            # Check if the agent is inside this puddle
            if x_left <= x <= x_right and y_top <= y <= y_bottom:
                # Calculate distances to each side
                left_dist = x - x_left
                right_dist = x_right - x
                top_dist = y - y_top
                bottom_dist = y_bottom - y

                # Find the minimum distance and set escape direction
                min_escape_dist = min(left_dist, right_dist, top_dist, bottom_dist)
                if min_escape_dist == left_dist:
                    escape_direction = 1  # Left
                elif min_escape_dist == right_dist:
                    escape_direction = 2  # Right
                elif min_escape_dist == top_dist:
                    escape_direction = 3  # Top
                elif min_escape_dist == bottom_dist:
                    escape_direction = 4  # Bottom
                else:
                    escape_direction = 0

        return escape_direction

    def _get_reward(self, pos: np.ndarray) -> float:
        """
        Calculate the reward based on the agent's position.

        Args:
            pos (numpy.ndarray): Agent's position.

        Returns:
            float: Reward value.
        """
        reward = float("inf")  # Initialize reward with a large positive value
        reward_puddles = []
        for top_left, wid in zip(self.puddle_top_left, self.puddle_width):
            if (
                    top_left[0] <= pos[0] <= top_left[0] + wid[0]
                    and top_left[1] - wid[1] <= pos[1] <= top_left[1]
            ):
                # Calculate the distance from the nearest edge of the puddle to the agent
                dist_to_edge = max(
                    abs(pos[0] - top_left[0]),
                    abs(top_left[0] + wid[0] - pos[0]),
                    abs(pos[1] - top_left[1]),
                    abs(top_left[1] - wid[1] - pos[1]),
                )
                reward_puddle = min(reward, -400 * dist_to_edge)
                reward_puddles.append(reward_puddle)
        if (
                reward_puddles == []
                and np.linalg.norm((pos - self.goal), ord=1) < self.goal_threshold
        ):
            return 0  # If the agent is in the goal, return 0
        elif reward_puddles == []:
            return -1  # -1 for each timestep
        else:
            return min(reward_puddles)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Seed value for the random number generator.
            options (dict): Additional options.

        Returns:
            tuple[np.ndarray, dict]: Tuple containing the initial position and additional information.
        """
        self.np_random, seed = seeding.np_random(seed)
        self.num_steps = 0
        if self.start is None:
            self.pos = self.observation_space.sample()
            while (
                    np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold
            ):  # make sure the start position is not too close to the goal
                self.pos = self.observation_space.sample()
        else:
            self.pos = copy.copy(self.start)

        if self.total_episodes % 10 == 0 and not self.evaluation:
            self.env = random.randint(1, 5)
            dir = os.getcwd()
            json_dir = os.path.join(dir, 'gym_puddle', 'env_configs', f'pw{self.env}.json')
            with open(json_dir) as f:
                env_setup = json.load(f)

                self.puddle_top_left = [np.array(top_left) for top_left in env_setup["puddle_top_left"]]
                self.puddle_width = [np.array(width) for width in env_setup["puddle_width"]]

        distances = self.distance_to_puddle_edges()
        escape_direction = self.closest_escape_direction()
        obs_lst = list(self.pos)
        for item in distances:
            obs_lst.append(item)
        obs_lst.append(self.env)
        obs_lst.append(0) # noise xx
        obs_lst.append(0) # noise y
        obs_lst.append(0) # radar guide
        obs_lst.append(0) # historical noise meanx
        obs_lst.append(0) # historical noise mena y
        self.noisehistoryx = []
        self.noisehistoryy = []

        return np.array(obs_lst), {}
        # return self.pos, {}

    def radar_guidance(self):
        # Check right direction
        right_info = self.closest_puddle_edge('right')
        if right_info and right_info['distance'] > 0.1:
            return 2  # Move right if safe

        # Check upward direction
        up_info = self.closest_puddle_edge('up')
        if up_info and up_info['distance'] > 0.1:
            return 1  # Move up if safe

        # Check downward direction
        down_info = self.closest_puddle_edge('down')
        if down_info and down_info['distance'] > 0.1:
            return 3  # Move down if safe

        # Check leftward direction
        left_info = self.closest_puddle_edge('left')
        if left_info and left_info['distance'] > 0.1:
            return 4  # Move left if safe

        # If no clear path is found, possibly stay in place
        return 0  # Indicates no movement or a need to reevaluate

    def closest_puddle_edge_for_radar(self, direction):
        x = self.pos[0]
        y = self.pos[1]
        closest_distance = float('inf')
        puddle_info = None

        for i, (top_left, width) in enumerate(zip(self.puddle_top_left, self.puddle_width)):
            distance = None

            if direction == 'right' and top_left[1] <= y <= top_left[1] + width[1]:
                distance = top_left[0] - x
            elif direction == 'left' and top_left[1] <= y <= top_left[1] + width[1]:
                distance = x - (top_left[0] + width[0])
            elif direction == 'up' and top_left[0] <= x <= top_left[0] + width[0]:
                distance = y - (top_left[1] + width[1])
            elif direction == 'down' and top_left[0] <= x <= top_left[0] + width[0]:
                distance = top_left[1] - y

            if distance is not None and 0 <= distance < closest_distance:
                closest_distance = distance
                puddle_info = {'index': i, 'distance': distance, 'width': width, 'top_left': top_left}

        return puddle_info

    def radar_guidance(self):
        directions = ['up', 'right', 'down', 'left']
        movement_commands = [1, 2, 3, 4]  # Corresponding movement commands for each direction

        for direction, command in zip(directions, movement_commands):
            info = self.closest_puddle_edge_for_radar(direction)
            if info and info['distance'] > 0.1:
                return command  # Return the corresponding command if safe

        return 0

    def render(self) -> np.ndarray or None:  # type: ignore
        """
        Render the environment.

        Returns:
            numpy.ndarray or None: Rendered frame as an RGB array or None if the render mode is "human".
        """
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """
        Render a single frame of the environment.

        Returns:
            numpy.ndarray: Rendered frame as an RGB array.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # set teh window name
            pygame.display.set_caption("Puddle World")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        if self.heatmap:
            # color the window as a heatmap based on the value of the reward at each pixel
            for i in range(self.window_size):
                for j in range(self.window_size):
                    pos = np.array([i / self.window_size, 1 - j / self.window_size])
                    reward = self._get_reward(pos)
                    if reward < -1:
                        max_reward = -1
                        color = int(
                            255
                            * (reward - self.min_reward)
                            / (max_reward - self.min_reward)
                        )
                        pygame.draw.rect(canvas, (255, color, 0), (i, j, 1, 1))

        # Draw the goal
        goal_pos = (
            int(self.goal[0] * self.window_size) - 10,
            self.window_size - int(self.goal[1] * self.window_size) + 10,
        )
        pygame.draw.circle(canvas, (0, 255, 0), goal_pos, 10)

        # Draw the puddles
        for top_left, wid in zip(self.puddle_top_left, self.puddle_width):
            puddle_pos = (
                int(top_left[0] * self.window_size),
                self.window_size - int(top_left[1] * self.window_size),
            )
            puddle_size = (
                int(wid[0] * self.window_size),
                int(wid[1] * self.window_size),
            )
            pygame.draw.ellipse(canvas, (0, 0, 0), (puddle_pos, puddle_size))

        # Draw the agent
        agent_pos = (
            int(self.pos[0] * self.window_size),
            self.window_size - int(self.pos[1] * self.window_size),
        )
        pygame.draw.circle(canvas, (255, 0, 0), agent_pos, 5)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def close(self) -> None:
        """
        Close the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def find_min_reward(self) -> float:
        """
        Find the minimum reward value in the environment.

        Returns:
            float: Minimum reward value.
        """
        min_reward = float("inf")
        for i in range(100):
            for j in range(100):
                pos = np.array([i / 100, j / 100])
                reward = self._get_reward(pos)
                if reward < min_reward:
                    min_reward = reward

        return min_reward

    def close(self) -> None:
        """
        Close the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
