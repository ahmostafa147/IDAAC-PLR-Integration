import gym
import torch
import numpy as np
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper
from procgen import ProcgenEnv


class PLRProcgenVecEnv:
    """64 individual ProcgenEnv(num_envs=1) with per-env PLR seed control."""

    def __init__(self, num_envs, env_name, level_sampler, distribution_mode='easy'):
        self.num_envs = num_envs
        self.env_name = env_name
        self.level_sampler = level_sampler
        self.distribution_mode = distribution_mode
        self.envs = [self._make_env(level_sampler.sample('sequential'))
                     for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def _make_env(self, seed):
        return ProcgenEnv(num_envs=1, env_name=self.env_name,
                          num_levels=1, start_level=seed,
                          distribution_mode=self.distribution_mode)

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return {k: np.concatenate([o[k] for o in obs]) for k in obs[0]}

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        all_obs, all_rew, all_done, all_info = [], [], [], []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self._actions[i:i+1])
            all_rew.append(rew)
            all_done.append(done)
            all_info.append(info[0])
            if done[0]:
                self.envs[i].close()
                self.envs[i] = self._make_env(self.level_sampler.sample())
                obs = self.envs[i].reset()
            all_obs.append(obs)
        return (
            {k: np.concatenate([o[k] for o in all_obs]) for k in all_obs[0]},
            np.concatenate(all_rew),
            np.concatenate(all_done),
            all_info,
        )

    def close(self):
        for env in self.envs:
            env.close()


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
        