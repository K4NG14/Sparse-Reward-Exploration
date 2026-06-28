"""Hindsight Experience Replay (HER) adapted to the sparse-reward MountainCar.

Unlike the discrete bit-flipping environment HER is usually demonstrated on,
MountainCar has continuous states. Two adaptations were needed to make HER
work here:

1. A distance threshold (GOAL_THRESHOLD) replaces exact-match goal checking,
   since a continuous state will essentially never equal a goal exactly.
2. A unified reward scale (0 / 200) is used for both the real goal and the
   hindsight goals, matching the scale of the sparse environment so the
   Q-network doesn't have to deal with mismatched reward magnitudes.
"""

from env import MountainCarSparse
import random
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from collections import deque
import os

LR = 1e-3
GAMMA = 0.98
MAX_EPISODES = 5000
MEMORY_SIZE = 500_000
BATCH_SIZE = 128
USE_HER = True
K_FUTURE = 4

GOAL_THRESHOLD = 0.1

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
method_name = "HER" if USE_HER else "DQN"
file_name = os.path.join(RESULTS_DIR, f"{method_name.lower()}_training_log.csv")


class QNetwork(nn.Module):
    """Goal-conditioned Q-network: takes the state AND the goal as input,
    concatenated, so the same network can evaluate actions for any goal."""

    def __init__(self, state_dim, goal_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, states, goals):
        x = torch.cat([states, goals], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=int(capacity))

    def store(self, s, a, r, term, trunc, s_next, goal):
        self.memory.append((s, a, r, term, trunc, s_next, goal))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def reached_goal(state, goal, threshold=GOAL_THRESHOLD):
    """A continuous state counts as reaching the goal once it's within
    `threshold` of the goal position (exact equality is essentially
    impossible with continuous states)."""
    return abs(state[0] - goal[0]) < threshold


def hindsight_reward_and_done(s_next, hindsight_goal):
    """Recomputes reward and termination for a hindsight goal, using the
    same 0 / 200 scale as the real sparse environment."""
    if reached_goal(s_next, hindsight_goal):
        return 200.0, True
    return 0.0, False


class Agent:
    def __init__(self, state_dim, goal_dim, n_actions):
        self.n_actions = n_actions
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.model = QNetwork(state_dim, goal_dim, n_actions)
        self.target_model = QNetwork(state_dim, goal_dim, n_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.opt = Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.999

    def choose_action(self, state, goal):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        s_t = torch.FloatTensor(state).unsqueeze(0)
        g_t = torch.FloatTensor(goal).unsqueeze(0)
        with torch.no_grad():
            q = self.model(s_t, g_t)
        return q.argmax(dim=-1).item()

    def store(self, s, a, r, term, trunc, s_next, goal):
        self.memory.store(s, a, r, term, trunc, s_next, goal)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, terminateds, truncateds, next_states, goals = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        goals = torch.FloatTensor(np.array(goals))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor([float(t) for t in terminateds])

        q_current = self.model(states, goals).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_model(next_states, goals).max(dim=1)[0]
        targets = rewards + GAMMA * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft (Polyak) update of the target network
        for tp, lp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(0.05 * lp.data + 0.95 * tp.data)

        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)


def train():
    env = MountainCarSparse()

    fixed_goal = np.array([env.env.unwrapped.goal_position], dtype=np.float32)
    state_dim = env.observation_space.shape[0]
    goal_dim = 1
    n_actions = env.action_space.n

    agent = Agent(state_dim, goal_dim, n_actions)
    solved = 0
    loss = 0.0
    global_running_r = 0.0

    with open(file_name, "w") as f:
        f.write("episode,ep_reward,ep_running_reward,loss,epsilon,mem_size,solved_pct\n")

    for episode_num in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        # The real goal for the episode is always the environment's fixed goal
        goal = fixed_goal.copy()
        episode = []
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.choose_action(state, goal)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store the transition WITHOUT a goal attached yet
            episode.append((state.copy(), action, reward, terminated, truncated, next_state.copy()))

            episode_reward += reward
            done = terminated or truncated
            state = next_state

        if episode[-1][2] > 0:
            solved += 1

        for i, transition in enumerate(episode):
            s, a, r, term, trunc, s_next = transition

            r_normalized = 200.0 if reached_goal(s_next, goal) else 0.0
            term_normalized = r_normalized > 0
            agent.store(s, a, r_normalized, term_normalized, trunc, s_next, goal)

            # HER: replay this transition with k hindsight goals sampled
            # from states actually visited later in the same episode
            if USE_HER:
                future_indices = range(i + 1, len(episode))
                if len(future_indices) == 0:
                    continue

                sampled_futures = random.sample(
                    list(future_indices),
                    min(K_FUTURE, len(future_indices))
                )

                for future_idx in sampled_futures:
                    hindsight_goal = np.array([episode[future_idx][5][0]], dtype=np.float32)
                    h_reward, h_terminated = hindsight_reward_and_done(s_next, hindsight_goal)
                    agent.store(s, a, h_reward, h_terminated, trunc, s_next, hindsight_goal)

        losses = []
        for _ in range(len(episode)):
            l = agent.learn()
            if l > 0:
                losses.append(l)
        loss = np.mean(losses) if losses else 0.0

        agent.update_epsilon()

        if episode_num == 1:
            global_running_r = episode_reward
        else:
            global_running_r = 0.99 * global_running_r + 0.01 * episode_reward

        if episode_num % 100 == 0:
            solved_pct = 100 * solved / 100
            print(f"Ep:{episode_num}| "
                  f"Ep_r:{episode_reward:.3f}| "
                  f"Running_r:{global_running_r:.3f}| "
                  f"Loss:{loss:.5f}| "
                  f"Epsilon:{agent.epsilon:.3f}| "
                  f"Mem:{len(agent.memory)}| "
                  f"Solved:{solved_pct:.1f}%")
            with open(file_name, "a") as f:
                f.write(f"{episode_num},"
                        f"{episode_reward:.3f},"
                        f"{global_running_r:.3f},"
                        f"{loss:.3f},"
                        f"{agent.epsilon:.3f},"
                        f"{len(agent.memory)},"
                        f"{solved_pct:.2f}\n")
            solved = 0


if __name__ == "__main__":
    train()
