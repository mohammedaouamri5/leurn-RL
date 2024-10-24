import os
import sys

import gym as gym
import numpy as np
from stable_baselines3 import PPO

import Env as Env

# Define directories
models_dir = "models/PPO"
logdir = "logs"
model_path = f"{models_dir}/{ sys.argv[1] }"  # Specify the model path (change this based on your saved model)

# Create directories if they don't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Setup environment
grid = np.zeros((10, 10))
env = Env.DroneEnv(Env.Feild(grid.copy()))
env.reset()

# Load the model
model = PPO.load(model_path, env=env, tensorboard_log=logdir)

# Interaction with environment
obs = env.reset()
while True:
    step = 0
    action, _states = model.predict(obs)  # PPO model prediction
    obs, reward, done, info = env.step(action)
    env.render()
    print(
        f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}"
    )
    step += 1  # Update the step count

    if done:
        break  # End interaction when the episode is done
