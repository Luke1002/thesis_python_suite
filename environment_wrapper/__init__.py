import gymnasium as gym
from gymnasium.envs.registration import register
from .env_wrapper import *

register(
    id="PelletFinder-v0",
    entry_point="environment_wrapper.env_wrapper:UnityEnv",
    kwargs={"env_path": None,
            "no_graphics": True,
            "worker_id": 0,
            "seed": int(time.time()),
            "log_folder": None}
)