import cv2
import gym
import numpy
import numpy as np
import pygame
import torch
from gym import spaces

ACTIONS = list(
    iter(
        torch.tensor(
            [
                [[1, 0]],
                [[-1, 0]],
                [[0, 1]],
                [[1, 1]],
                [[-1, 1]],
                [[-1, -1]],
                [[1, -1]],
                [[0, -1]],
            ]
        )
    )
)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


class Feild:
    def __init__(self, grid):
        self.grid: numpy.ndarray = grid
        self.len = grid.flatten().shape[0]


class Drone:
    def __init__(self, position=Point(0, 0)):
        self.position = position
        self.actions = []


class DroneEnv(gym.Env):
    def __init__(self, p_feild: Feild):
        super(DroneEnv, self).__init__()

        # Action space: define based on the number of discrete actions
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        # Observation space: size of the flattened grid + 2 for the drone's position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(p_feild.grid.size,), dtype=np.float32
        )
        self.grid = p_feild.grid.copy()
        self.screen_is_initialized = False

    def init_screen(self):
        if not self.screen_is_initialized:
            pygame.init()
            self.cell_size = 100
            self.screen_size = (
                self.grid.shape[0] * self.cell_size,
                self.grid.shape[1] * self.cell_size,
            )
            self.screen = pygame.display.set_mode(
                (self.screen_size[0], self.screen_size[1])
            )
            self.screen_is_initialized = True

    def _mkobservation(self) -> np.ndarray:
        # Observation is the flattened grid + drone's x and y position
        observation = self.field.grid.flatten().tolist()
        observation.append(float(self.drone.position.x))
        observation.append(float(self.drone.position.y))
        return np.array(observation, dtype=np.float32)

    def _regularize(self, action):
        if action >= 0.5:
            return 1
        elif action <= -0.5:
            return -1
        return 0

    def drone_in_field(self):
        shape = self.field.grid.shape
        return (
            self.drone.position.x >= 0
            and self.drone.position.x < shape[0]
            and self.drone.position.y >= 0
            and self.drone.position.y < shape[1]
        )

    def step(self, action):
        """
        Update the drone's position based on the action and calculate the reward.
        Return the new observation, reward, done flag, and info.
        """
        reward = 1.0
        # Update drone position
        action = Point(
            self._regularize(action[0]),
            self._regularize(action[1]),
        )
        print("-------------------------------------")
        print("action", action)
        print("from", self.drone.position)
        self.drone.position = self.drone.position + action
        print(" to ", self.drone.position)
        if self.drone_in_field():
            reward = self.field.grid.sum(axis=1).sum(axis=0)
            if self.field.grid[self.drone.position.x, self.drone.position.y] == 1:
                reward = -10.0

            self.limit -= 1
            self.field.grid[self.drone.position.x, self.drone.position.y] = 1
            done = not np.any(self.field.grid == 0)
            if done:
                reward += 10.0
        else:
            reward = -100.0
            done = True

        print("reword", reward)
        print("-------------------------------------")
        return (
            self._mkobservation(),
            float(reward),
            bool(done),
            # bool(self.limit < 0),
            {"drone": self.drone, "grid": self.field.grid.copy()},
        )

    def reset(self, **kwargs):
        """
        Reset the environment by creating a new field and drone, then return the initial observation.
        """
        self.field = Feild(self.grid.copy())
        self.drone = Drone()
        self.limit = (
            self.field.grid.size + self.field.grid.size * 0.3
        )  # Update based on grid size
        return self._mkobservation()

    def render(self, mode="human"):
        self.init_screen()
        self.screen.fill((0, 0, 0))  # Fill with black
        pygame.time.delay(100)
        # Draw the grid
        for i in range(self.field.grid.shape[0]):
            for j in range(self.field.grid.shape[1]):
                color = (255, 255, 255) if self.field.grid[i, j] == 1 else (0, 0, 0)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

        # Draw the drone position (if needed)
        drone_x = int(self.drone.position.y * self.cell_size)
        drone_y = int(self.drone.position.x * self.cell_size)
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (drone_x + self.cell_size // 2, drone_y + self.cell_size // 2),
            self.cell_size // 4,
        )

        pygame.display.flip()  # Update the display

        # Handle events to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        pygame.quit()
