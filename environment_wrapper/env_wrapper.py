from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces


class UnityEnv(gym.Env):
    def __init__(self, env_path, worker_id, no_graphics=False, seed=int(time.time()), log_folder=None):
        self.no_graphics = no_graphics
        self.env = UnityEnvironment(file_name=env_path, no_graphics=no_graphics, seed=seed, base_port=7000, worker_id=worker_id,
                                    log_folder=log_folder)

        self.env.reset()
        self.behaviour_name = (list(self.env.behavior_specs.keys()))[0]

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,2), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,6), dtype=np.float32)

    def reset(self, seed=None, options=None):

        decision_steps = None
        self.env.reset()

        while decision_steps is None or len(decision_steps.obs[0]) < 1:
            self.env.step()
            decision_steps, terminal_steps = self.env.get_steps(self.behaviour_name)

        state = decision_steps.obs[0]
        return state, {}

    def step(self, actions):
        actions = np.asarray(actions)
        actions = np.reshape(actions, self.action_space.shape)

        action_tuple = ActionTuple()
        action_tuple.add_continuous(actions)

        self.env.set_actions(self.behaviour_name, action_tuple)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behaviour_name)
        done = False
        if len(terminal_steps.interrupted) > 0:
            done = True
            state = terminal_steps.obs[0]
            reward = terminal_steps.reward[0]
        else:
            state = decision_steps.obs[0]
            reward = decision_steps.reward[0]

        return state, reward, False, done, {}

    def close(self):
        self.env.close()
