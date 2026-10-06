import time

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "PointMaze_UMaze-v3",
    render_mode="human",
    max_episode_steps=500,
    continuing_task=False,
)

observation, info = env.reset(seed=0)

for step in range(500):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    breakpoint()

    state = observation["observation"]
    print(
        f"step={step:03d} "
        f"position=({state[0]:+.2f}, {state[1]:+.2f}) "
        f"velocity=({state[2]:+.2f}, {state[3]:+.2f}) "
        f"action=({action[0]:+.2f}, {action[1]:+.2f})"
    )

    time.sleep(0.05)

    if terminated or truncated:
        observation, info = env.reset()

env.close()