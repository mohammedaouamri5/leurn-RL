import os
from typing import Literal

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn, tensor

import Env as Env

models_dir = "models/eps-greedy"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize the environment
grid = np.zeros((10, 10))
env = Env.DroneEnv(Env.Feild(grid.copy()))
obs = env.reset()


class Agent(nn.Module):
    def __init__(self, eps=0.1, state_len=10, action_len=10):
        super(Agent, self).__init__()
        self.Value = nn.Sequential(
            nn.Linear(state_len + action_len, 2048),
            nn.Conv1d(1, 10, 5),
            nn.ReLU6(),
            nn.Conv1d(10, 100, 5),
            nn.ReLU6(),
            nn.Conv1d(100, 10, 5),
            nn.ReLU6(),
            nn.Conv1d(10, 1, 5),
            nn.Linear(2032, 1),
        )

    def forward(self, __type: Literal["value"] = "value", **kwargs):
        if __type == "value":
            state = kwargs["state"]
            action = kwargs["action"]
            x = torch.cat([state, action], dim=1)
            return self.Value(x)


agent = Agent(eps=0.1, state_len=102, action_len=2)
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
__loss = nn.MSELoss()


def get_train_action(obs, agent, eps=0.3):
    action = None
    reword = None
    __obs = torch.from_numpy(obs)
    __obs = __obs.float().unsqueeze(0)

    if np.random.random() < eps:
        action = Env.ACTIONS[np.random.randint(0, len(Env.ACTIONS))]
        reword = agent.forward(action=action, state=__obs)
    else:
        for __action in Env.ACTIONS:
            __reword = agent.forward(action=__action, state=__obs)
            if action is None or __reword > reword:
                action = __action
                reword = __reword
    return action[0].numpy(), reword


agent = agent.train()
for i in range(1000000):
    obs = env.reset()
    done = False
    while not done:
        action, old_reword = get_train_action(obs, agent, eps=0.1)
        obs, new_reward, done, info = env.step(action)
        new_reward = torch.tensor([[new_reward]])

        print("new_reward = ", new_reward)
        print("old_reword = ", old_reword)

        optimizer.zero_grad()
        loss = __loss(old_reword, new_reward)
        loss.backward()
        optimizer.step()

        env.render()

        #
