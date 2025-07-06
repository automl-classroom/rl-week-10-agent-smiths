import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from carl.envs import CARLCartPole
from carl.context.context_space import NormalFloatContextFeature
from carl.context.sampler import ContextSampler

import random

# Create environment
context_distributions = [NormalFloatContextFeature("length", mu=0.5, sigma=1, upper=50, lower=0)]
context_sampler = ContextSampler(
    context_distributions=context_distributions,
    context_space=CARLCartPole.get_context_space(),
    seed=42,
)
contexts = context_sampler.sample_contexts(n_contexts=2)

contexts[0]['length'] = 0.5
contexts[1]['length'] = 1

print("Training contexts are:")
print(contexts)

for idx, context in enumerate(contexts.values()):

    curr_context = {idx: context}

    print(f"USING CONTEXT {context}")

    env = gym.make("carl/CARLCartPole-v0", render_mode="rgb_array", contexts=curr_context)
    env = FlattenObservation(env)

    best_reward = float('-inf')
    best_lr = 0
    best_var = 0

    for i in range(40):

        print(f"RS STEP {i}")

        random.seed(i)
        lr = 10 ** random.uniform(-6, -2)

        # Instantiate the agent
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=lr, seed=42)
        
        # Train the agent and display a progress bar
        model.learn(total_timesteps=int(1e4), progress_bar=True, log_interval=50)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_lr = lr
            best_var = std_reward

    print(f"BEST LR FOR CONTEXT {idx} = {best_lr} AT {best_reward} AND {best_var}")
