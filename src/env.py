import gymnasium as gym
from gymnasium import Wrapper


class MountainCarSparse(Wrapper):
    """Gymnasium wrapper for MountainCar with a sparse reward signal.

    Standard MountainCar gives -1 reward per step, which is dense enough
    to guide learning on its own. This wrapper strips that signal away to
    create a hard exploration problem:

    - Reward is 0 for every step
    - Reward is 200 only when the goal is reached
    - Episode is truncated after 200 steps

    Args:
        render_mode: Optional render mode ("human", "rgb_array", or None)
    """

    def __init__(self, render_mode=None):
        env = gym.make('MountainCar-v0', render_mode=render_mode)
        super().__init__(env)
        self.time_step = 0

    def step(self, action):
        self.time_step += 1
        obs, _, terminated, truncated, info = self.env.step(action)

        # Access the unwrapped env to read the raw physical state
        unwrapped_env = self.env.unwrapped

        goal_reached = bool(
            unwrapped_env.state[0] >= unwrapped_env.goal_position and
            unwrapped_env.state[1] >= unwrapped_env.goal_velocity
        )

        # Sparse reward: 0 everywhere, 200 only at the goal
        reward = 200.0 if goal_reached else 0.0

        if self.time_step >= 200:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.time_step = 0
        return obs, info
