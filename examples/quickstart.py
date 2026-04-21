"""Quickstart: create an environment, observe, and step through it."""

import glyphbench  # noqa: F401 — registers all environments
import gymnasium as gym

# Pick any environment from the 169 available
env = gym.make("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(seed=42)

print("=== Observation ===")
print(obs)
print()
print(f"Available actions: {env.unwrapped.action_spec.names}")
print(f"Action space size: {env.action_space.n}")
print()

# Take a few steps
for i in range(5):
    action = env.action_space.sample()
    action_name = env.unwrapped.action_spec.names[action]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i + 1}: action={action_name}, reward={reward:.1f}, done={terminated or truncated}")

env.close()
