import os
import random
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from matplotlib import pyplot as plt

from exploration import DummyIntrinsicRewardModule, RNDNetwork, ICMNetwork
from env import MountainCarSparse

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class ReplayBuffer(object):
    """A simple ring-buffer replay buffer for off-policy Q-learning."""

    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def put(self, obs, action, extrinsic_reward, next_obs, truncated, terminated):
        """Stores a transition. Only the EXTRINSIC reward is stored — intrinsic
        rewards are recomputed on-the-fly at training time using the most
        up-to-date exploration module, instead of stale cached values."""
        self.buffer[self.ptr] = (obs, action, extrinsic_reward, next_obs, truncated, terminated)
        self.size = min(self.size + 1, self.capacity)
        self.ptr = (self.ptr + 1) % self.capacity

    def get(self, batch_size):
        return zip(*random.sample(self.buffer[: self.size], batch_size))

    def __len__(self):
        return self.size


class DQNNetwork(nn.Module):
    """Q-network: maps a state to one Q-value per discrete action."""

    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DQN:
    """DQN agent, optionally augmented with an intrinsic reward module
    (RND or ICM) to drive exploration in sparse-reward settings."""

    def __init__(
        self,
        env,
        replay_size=20000,
        batch_size=32,
        gamma=0.99,
        sync_after=5,
        lr=0.03,
        verbose=False,
        reward_module=None,
        render=False,
    ):
        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError("Continuous actions not implemented!")

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose
        self.render = render

        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        self.optim_dqn = optim.RMSprop(self.dqn_net.parameters(), lr=lr)

        if reward_module == "RND":
            self.intrinsic_reward_module = RNDNetwork(self.obs_dim, 128)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.predictor.parameters(), lr=lr,
            )
        elif reward_module == "ICM":
            self.intrinsic_reward_module = ICMNetwork(self.obs_dim, 256, self.act_dim)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.parameters(), lr=lr / 50.0
            )
        else:
            # No-op module: calculate_reward(...) always returns 0.0 -> vanilla DQN
            self.intrinsic_reward_module = DummyIntrinsicRewardModule()

    def learn(self, time_steps):
        eval_scores = []
        episode_length = 0
        episode_lengths = []

        obs, _ = self.env.reset()
        best_episode_len = float("inf")

        for timestep in range(1, time_steps + 1):
            if self.render and timestep % 15 == 0:
                self.env.render()

            epsilon = epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)

            next_obs, extrinsic_reward, terminated, truncated, _ = self.env.step(action)
            done = truncated or terminated

            self.replay_buffer.put(obs, action, extrinsic_reward, next_obs, truncated, terminated)
            obs = next_obs

            episode_length += 1
            if done:
                obs, _ = self.env.reset()
                if self.verbose and best_episode_len > episode_length and episode_length < 200:
                    best_episode_len = episode_length
                    print(f"[t={timestep}]: Solved after {best_episode_len} steps!")
                episode_lengths.append(episode_length)
                episode_length = 0

            if len(self.replay_buffer) > self.batch_size:
                obs_, actions, extrinsic_rewards, next_obs_, truncateds, terminateds = self.replay_buffer.get(
                    self.batch_size
                )
                obs_ = torch.stack([torch.Tensor(ob) for ob in obs_])
                next_obs_ = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs_])
                extrinsic_rewards = torch.Tensor(extrinsic_rewards)
                truncateds = torch.Tensor(truncateds)
                terminateds = torch.Tensor(terminateds)
                actions = torch.LongTensor(actions)

                if not isinstance(self.intrinsic_reward_module, DummyIntrinsicRewardModule):
                    # Recompute intrinsic rewards with the CURRENT module so the
                    # signal stays fresh instead of using stale buffered values
                    with torch.no_grad():
                        intrinsic_rewards_batch = self.intrinsic_reward_module.calculate_reward(
                            obs_, next_obs_, actions
                        )
                    total_rewards = extrinsic_rewards + intrinsic_rewards_batch
                else:
                    total_rewards = extrinsic_rewards

                dqn_loss = self.compute_msbe_loss(
                    obs_, actions, total_rewards, next_obs_, truncateds, terminateds
                )
                self.optim_dqn.zero_grad()
                dqn_loss.backward()
                self.optim_dqn.step()

                if not isinstance(self.intrinsic_reward_module, DummyIntrinsicRewardModule):
                    intrinsic_loss = self.intrinsic_reward_module.calculate_loss(
                        obs_, next_obs_, actions
                    )
                    self.optim_reward.zero_grad()
                    intrinsic_loss.backward()
                    self.optim_reward.step()

            if timestep % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

            if timestep % 1000 == 0 and len(episode_lengths) >= 7:
                score = self.evaluate(num_episodes=100)
                eval_scores.append(score)
                print(' ', timestep, score, end='')

        return eval_scores

    def evaluate(self, num_episodes=100):
        """Evaluates the current greedy policy for a number of episodes
        on a fresh copy of the environment."""
        env = deepcopy(self.env)
        lengths = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            t = 0
            while not done:
                action = self.predict(obs, epsilon=0.0)
                t += 1
                obs, _, terminated, truncated, _ = env.step(action)
                done = truncated or terminated
            lengths.append(t)
        return np.array(lengths).mean()

    def predict(self, state, epsilon=0.0):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.dqn_net.forward(state)
            action = q_value.argmax().item()
        else:
            action = random.randrange(self.act_dim)
        return action

    def compute_msbe_loss(self, obs, actions, rewards, next_obs, truncateds, terminateds):
        q_values = self.dqn_net(obs)
        next_q_values = self.dqn_target_net(next_obs)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        dones = truncateds + terminateds - truncateds * terminateds
        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)
        return F.mse_loss(q_values, expected_q_values)


def render_episodes(dqn, env_name="Agent", num_episodes=3):
    """Renders a few episodes with the learned policy for visual inspection."""
    print(f"\nRendering {num_episodes} episodes with the learned {env_name} policy...")
    env_render = MountainCarSparse(render_mode="human")

    for episode in range(num_episodes):
        obs, _ = env_render.reset()
        done = False
        episode_reward = 0
        step = 0
        print(f"Episode {episode + 1}:")
        while not done:
            action = dqn.predict(obs, epsilon=0)
            obs, reward, terminated, truncated, _ = env_render.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        print(f"  Steps: {step}, Total Reward: {episode_reward}")


def epsilon_by_timestep(timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000):
    """Linearly decays epsilon from epsilon_start to epsilon_final over frames_decay timesteps."""
    return max(
        epsilon_final,
        epsilon_start - (timestep / frames_decay) * (epsilon_start - epsilon_final),
    )


def test_policy_100(env, dqn):
    """Evaluates the greedy policy for 100 episodes (used for the final report)."""
    lengths = []
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        t = 0
        while not done:
            action = dqn.predict(obs, 0)
            t += 1
            obs, _, terminated, truncated, _ = env.step(action)
            done = truncated or terminated
        lengths.append(t)
    return np.array(lengths).mean()


def run_experiment(env_factory, reward_module, n_runs, timesteps, title, figure_name, log_name, render=False):
    """Runs `n_runs` independent training runs and plots mean +/- std of the
    evaluation curve, saving both the figure and the raw run data."""
    print(title)
    runs = []
    dqn = None
    for _ in range(n_runs):
        env = env_factory(render)
        dqn = DQN(env, verbose=True, reward_module=reward_module, render=render)
        scores = dqn.learn(timesteps)
        runs.append(scores)
        print('\n Mean over 100 eval episodes:', test_policy_100(env, dqn))

    min_len = min(len(r) for r in runs)
    arr = np.array([r[:min_len] for r in runs])
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    xs = np.arange(min_len)

    plt.figure()
    plt.plot(xs, mean)
    plt.fill_between(xs, mean - std, mean + std, alpha=0.3)
    plt.title(title)
    plt.xlabel('Time steps (x1000)')
    plt.ylabel('Episode length')
    plt.grid()
    plt.ylim((0, 210))
    plt.savefig(os.path.join(FIGURES_DIR, figure_name))
    plt.show()

    with open(os.path.join(LOGS_DIR, log_name), 'wb') as f:
        pickle.dump(runs, f)

    return dqn, runs


if __name__ == "__main__":
    RENDER = False
    TIMESTEPS = 50000

    # 1. Baseline: vanilla DQN on the original MountainCar (dense reward, -1/step)
    dqn, _ = run_experiment(
        env_factory=lambda render: gym.make('MountainCar-v0', render_mode="human" if render else None),
        reward_module=None,
        n_runs=3,
        timesteps=TIMESTEPS,
        title='Original MountainCar (dense reward)',
        figure_name='01_baseline_dense_reward.png',
        log_name='baseline_dense_reward.pkl',
        render=RENDER,
    )
    render_episodes(dqn, "Baseline DQN", 1)

    # 2. Vanilla DQN on the sparse MountainCar (epsilon-greedy only)
    dqn, _ = run_experiment(
        env_factory=lambda render: MountainCarSparse(render_mode="human" if render else None),
        reward_module=None,
        n_runs=3,
        timesteps=TIMESTEPS,
        title='Sparse MountainCar with epsilon-greedy exploration',
        figure_name='02_sparse_epsilon_greedy.png',
        log_name='sparse_epsilon_greedy.pkl',
        render=RENDER,
    )
    render_episodes(dqn, "Sparse DQN (epsilon-greedy)", 1)

    # 3. DQN + ICM on the sparse MountainCar
    dqn, _ = run_experiment(
        env_factory=lambda render: MountainCarSparse(render_mode="human" if render else None),
        reward_module="ICM",
        n_runs=3,
        timesteps=TIMESTEPS,
        title='Sparse MountainCar with ICM',
        figure_name='03_sparse_icm.png',
        log_name='sparse_icm.pkl',
        render=RENDER,
    )
    render_episodes(dqn, "Sparse DQN (ICM)", 1)

    # 4. DQN + RND on the sparse MountainCar
    dqn, _ = run_experiment(
        env_factory=lambda render: MountainCarSparse(render_mode="human" if render else None),
        reward_module="RND",
        n_runs=3,
        timesteps=TIMESTEPS,
        title='Sparse MountainCar with RND',
        figure_name='04_sparse_rnd.png',
        log_name='sparse_rnd.pkl',
        render=RENDER,
    )
    render_episodes(dqn, "Sparse DQN (RND)", 1)
