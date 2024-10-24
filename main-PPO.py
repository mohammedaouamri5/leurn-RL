import os

import gym
import numpy as np
from stable_baselines3 import PPO

import Env as Env

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize the environment
grid = np.zeros((10, 10))
env = Env.DroneEnv(Env.Feild(grid.copy()))
env.reset()

# Create the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    policy_kwargs=dict(net_arch=[102, 1024, 2]),
    gamma=0.999,
)

TIMESTEPS = 1000
iters = 0
for i in range(1000000):
    # Train the model
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

    # Save the model periodically
    model.save(f"{models_dir}/{TIMESTEPS * i}")

    # Render the environment (for visualization during training)
    env.render()  # Ensure that DroneEnv has a render method
