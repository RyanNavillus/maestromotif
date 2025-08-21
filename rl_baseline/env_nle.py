import importlib
import os
import gym
import gymnasium
import numpy as np

import nle

from nle import nethack

from rl_baseline.obs_wrappers import RenderCharImagesWithNumpyWrapper, MessageWrapper, BlstatsWrapper, ModifierWrapper
from sample_factory.envs.env_registry import global_env_registry
from utils.forked_pdb import ForkedPdb

from syllabus.core import GymnasiumSyncWrapper, TaskWrapper
from syllabus.task_space import DiscreteTaskSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0


def load_player_class(policy_name, player_class):
    try:
        # Dynamically construct the module name
        module_name = f"rl_baseline.{policy_name}.execution"
        # Import the module
        module = importlib.import_module(module_name)
        # Get the Player class from the module
        PlayerClass = getattr(module, player_class)
        return PlayerClass
    except ModuleNotFoundError:
        print(f"Module for {policy_name} not found.")
        return None
    except AttributeError:
        print(f"Player class not found in {policy_name}.code.")
        return None

def _convert_space(space: gym.Space) -> gymnasium.Space:
    """Converts a gym space to a gymnasium space.

    Args:
        space: the space to convert

    Returns:
        The converted space
    """
    if isinstance(space, gymnasium.spaces.Discrete):
        return gym.spaces.Discrete(n=space.n)
    elif isinstance(space, gymnasium.spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gymnasium.spaces.MultiBinary):
        return gym.spaces.MultiBinary(n=space.n)
    elif isinstance(space, gymnasium.spaces.Tuple):
        return gym.spaces.Tuple(spaces=tuple(map(_convert_space, space.spaces)))
    elif isinstance(space, gymnasium.spaces.Dict):
        return gym.spaces.Dict(spaces={k: _convert_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, gymnasium.spaces.Sequence):
        return gym.spaces.Sequence(space=_convert_space(space.feature_space))
    elif isinstance(space, gymnasium.spaces.Graph):
        return gym.spaces.Graph(
            node_space=_convert_space(space.node_space),  # type: ignore
            edge_space=_convert_space(space.edge_space),  # type: ignore
        )
    elif isinstance(space, gymnasium.spaces.Text):
        return gym.spaces.Text(
            max_length=space.max_length,
            min_length=space.min_length,
            charset=space._char_str,
        )
    else:
        raise NotImplementedError(
            f"Cannot convert space of type {space}. Please upgrade your code to gymnasium."
        )


class GymConvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task_space = self.env.task_space
        self.observation_space = _convert_space(self.env.observation_space)
        self.action_space = _convert_space(self.env.action_space)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return obs, rew, term or trunc, info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs


class RootNLEWrapper(gym.Wrapper):
    """Some root-level additions to the NLE environment"""
    def __init__(self, env, num_skills=1, env_id=-1):
        super().__init__(env)
        
        bl_shape = 30 if 'Monk' in env.env.spec.name else 27
        bl_shape += num_skills
        task_space = {
            "tty_chars": gym.spaces.Box(0, 255, shape=(24, 80), dtype=np.uint8),
            "tty_colors": gym.spaces.Box(0, 31, shape=(24, 80), dtype=np.int8),
            "tty_cursor": gym.spaces.Box(0, 255, shape=(2,), dtype=np.uint8),
            "message": gym.spaces.Box(0, 255, shape=(256,), dtype=np.uint8),
            "glyphs": gym.spaces.Box(0, 5976, (21, 79), dtype=np.int16),
            "blstats": gym.spaces.Box(-2147483648, 2147483647, shape=(bl_shape,), dtype=np.int32),
        }

        self.env_id = np.array([env_id]).astype(np.int16)
        self.task_space = list(task_space.keys())

        self.observation_space = gym.spaces.Dict(task_space)

    def seed(self, *args):
        # Nethack does not allow seeding, so monkey-patch disable it here
        return

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = {k: obs[k] for k in obs if k in self.task_space}
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = {k: obs[k] for k in obs if k in self.task_space}
        return obs



class NethackSeedWrapper(TaskWrapper):
    """
    This wrapper allows you to change the task of an NLE environment.

    This wrapper was designed to meet two goals.
        1. Allow us to change the task of the NLE environment at the start of an episode
        2. Allow us to use the predefined NLE task definitions without copying/modifying their code.
           This makes it easier to integrate with other work on nethack tasks or curricula.

    Each task is defined as a subclass of the NLE, so you need to cast and reinitialize the
    environment to change its task. This wrapper manipulates the __class__ property to achieve this,
    but does so in a safe way. Specifically, we ensure that the instance variables needed for each
    task are available and reset at the start of the episode regardless of which task is active.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int = 0,
        num_seeds: int = 200,
    ):
        super().__init__(env)
        self.env = env
        self.task_space = DiscreteTaskSpace(num_seeds)

        # Task completion metrics
        self.episode_return = 0
        self.task = seed

        if seed is not None:
            self.seed(seed)

    def seed(self, seed):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        env.seed(core=seed, disp=seed)

    def _task_name(self, task):
        return task.__name__

    def reset(self, new_task=None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """
        self.episode_return = 0
        obs, info = super().reset(new_task=new_task, **kwargs)
        return self.observation(obs), info

    def change_task(self, new_task: int):
        """
        Change task by setting the seed.
        """
        # Ignore new task if mid episode
        self.task = new_task
        self.seed(new_task)

    def observation(self, observation):
        """
        Returns a modified observation.
        """
        return observation

    def _task_completion(self, obs, rew, term, trunc, info):
        return min(max(self.episode_return / 1000, 0.0), 1.0)

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, term, trunc, info = super().step(action)
        self.episode_return += rew
        return obs, rew, term, trunc, info


def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    root_env = cfg.root_env
    if env_config is not None and 'root_env' in env_config.keys():
        root_env = env_config['root_env']

    if env_config is not None and 'env_id' in env_config.keys():
        env_id = env_config['env_id']
    else:
        env_id = -1

    llm_reward = cfg.llm_reward if 'llm_reward' in cfg.keys() else 0.0
    penalty_mode = cfg.penalty_mode
    msg_max_keys = cfg.msg_max_keys

    base_actions = (
        [nethack.MiscAction.MORE] +
        list(nethack.CompassDirection) +
        list(nethack.CompassDirectionLonger) +
        list(nethack.MiscDirection) +
        [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
    )

    actions = base_actions + [
            nethack.Command.CAST,
            nethack.Command.ENHANCE,
            nethack.Command.PRAY,
            nethack.Command.QUAFF,
            nethack.Command.DROP,
        ]

    ttyrec = cfg.ttyrec if 'ttyrec' in cfg.keys() else cfg.save_ttyrec_every
    cfg.save_dir = f'{cfg.train_dir}/{cfg.experiment}/ttyrecs/A'

    env = RootNLEWrapper(gym.make(
            root_env,
            save_ttyrec_every=ttyrec,
            savedir=cfg.save_dir,
            observation_keys=[
                "tty_chars",
                "tty_colors", 
                "tty_cursor", 
                "blstats", 
                "message", 
                "glyphs", 
                "inv_strs", 
                "inv_letters", 
                "inv_oclasses"
                ],
            actions=tuple(actions),
            max_episode_steps=cfg.max_episode_steps,
            penalty_mode=penalty_mode,
            evaluation='none' not in cfg.eval_target,
            eval_target=cfg.eval_target,
            ),
            num_skills=cfg.num_skills,
            env_id=env_id,
        )

    env = RenderCharImagesWithNumpyWrapper(env, font_size=9, crop_size=cfg.crop_size, rescale_font_size=(6, 6))

    env = MessageWrapper(env, llm_reward=llm_reward, msg_max_keys=msg_max_keys)

    exploration_str = 'exploration'

    if cfg.eval_target != 'none':
        exp_folder = cfg.eval_target
    else:
        exp_folder = exploration_str

    seed = cfg.code_seed if cfg.code_seed != -1 else cfg.seed
    policy_name = f"meta_policies.{exp_folder}.seed{seed}"
    player_class = "NetHackPlayer"
    MetaPolicyClass = load_player_class(policy_name, player_class)

    env = ModifierWrapper(env, llm_reward=llm_reward, experiment=cfg.experiment, 
                          num_skills=cfg.num_skills, meta_policy_class=MetaPolicyClass)

    env = BlstatsWrapper(env, experiment=cfg.experiment, diff_h=cfg.diff_h, diffstats_size=cfg.diffstats_size)
    env = GymV21CompatibilityV0(env=env)
    env = NethackSeedWrapper(env, num_seeds=1)
    env = GymnasiumSyncWrapper(env, DiscreteTaskSpace(1), cfg.curriculum_components)

    env = GymConvWrapper(env)
    return env


global_env_registry().register_env(
    env_name_prefix='nle_',
    make_env_func=make_custom_env_func,
)
