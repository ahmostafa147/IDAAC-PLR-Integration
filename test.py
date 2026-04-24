import numpy as np
import torch

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)

from ppo_daac_idaac.envs import VecPyTorchProcgen


def evaluate(args, actor_critic, device):
    actor_critic.eval()

    num_eval_envs = getattr(args, 'num_eval_envs', 16)
    num_eval_episodes = getattr(args, 'num_eval_episodes', 10)

    # Sample Levels From the Full Distribution
    venv = ProcgenEnv(num_envs=num_eval_envs, env_name=args.env_name, \
        num_levels=0, start_level=0, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    eval_envs = VecPyTorchProcgen(venv, device)

    eval_episode_rewards = []
    all_values = []
    all_adv_preds = []
    obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_eval_episodes:
        with torch.no_grad():
            if args.algo == 'ppo':
                value, action, _, _ = actor_critic.act(obs)
                all_values.extend(value.squeeze(-1).tolist())
            else:
                adv, value, action, _, _ = actor_critic.act(obs)
                all_values.extend(value.squeeze(-1).tolist())
                all_adv_preds.extend(adv.squeeze(-1).tolist())

        obs, _, done, infos = eval_envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"\
        .format(len(eval_episode_rewards), \
        np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))

    stats = {
        'eval_episode_rewards': eval_episode_rewards,
        'eval_value_mean': float(np.mean(all_values)),
        'eval_value_std': float(np.std(all_values)),
    }
    if all_adv_preds:
        stats['eval_adv_pred_mean'] = float(np.mean(all_adv_preds))
        stats['eval_adv_pred_std'] = float(np.std(all_adv_preds))
    return stats

