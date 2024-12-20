import os

import gym as gym
import numpy as np
from stable_baselines3 import DQN

import Env as Env

models_dir = "models/DQN"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

grid = np.zeros((10, 10))
env = Env.DroneEnv(Env.Feild(grid.copy()))
env.reset()
model = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=logdir,
    policy_kwargs=dict(net_arch=[102, 1024, 1024, 2]),
)

TIMESTEPS = 10000
iters = 0
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    env.render()
