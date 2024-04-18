import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000, log_interval=5)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render("human")
    
    if terminated or truncated:
        obs, info = env.reset()