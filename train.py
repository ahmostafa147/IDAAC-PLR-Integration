import os
import torch
import numpy as np
from collections import deque
from tqdm import trange, tqdm

import hyperparams as hps
from test import evaluate
from procgen import ProcgenEnv

from baselines import logger
from ppo_daac_idaac.wandb_utils import WandBLogger
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)

from ppo_daac_idaac import algo, utils
from ppo_daac_idaac.arguments import parser
from ppo_daac_idaac.model import PPOnet, IDAACnet, \
    LinearOrderClassifier, NonlinearOrderClassifier
from ppo_daac_idaac.storage import DAACRolloutStorage, \
    IDAACRolloutStorage, RolloutStorage
from ppo_daac_idaac.envs import VecPyTorchProcgen, PLRProcgenVecEnv
from ppo_daac_idaac.level_sampler import LevelSampler


def make_envs(args, device, level_sampler=None):
    """Create the vectorized env stack."""
    if args.use_plr:
        venv = PLRProcgenVecEnv(
            num_envs=args.num_processes, env_name=args.env_name,
            level_sampler=level_sampler,
            distribution_mode=args.distribution_mode)
    else:
        venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name,
            num_levels=args.num_levels, start_level=args.start_level,
            distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    return VecPyTorchProcgen(venv, device)


CHECKPOINT_INTERVAL = 250  # save every N updates


def _ckpt_path(save_dir: str, j: int) -> str:
    return os.path.join(save_dir, f"ckpt_{j:07d}.pt")


def _latest_checkpoint(save_dir: str):
    """Return (update_index, path) of the latest checkpoint, or None."""
    if not save_dir or not os.path.isdir(save_dir):
        return None
    ckpts = [f for f in os.listdir(save_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort()
    path = os.path.join(save_dir, ckpts[-1])
    j = int(ckpts[-1][len("ckpt_"):-len(".pt")])
    return j, path


def _save_checkpoint(j, actor_critic, agent, order_classifier, level_sampler,
                     episode_rewards, envs, save_dir, log_file):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "update": j,
        "actor_critic": actor_critic.state_dict(),
        "ob_rms": getattr(envs, "ob_rms", None),
        "episode_rewards": list(episode_rewards),
    }
    if order_classifier is not None:
        ckpt["order_classifier"] = order_classifier.state_dict()
    # save all optimizers
    if hasattr(agent, "optimizer"):
        ckpt["optimizer"] = agent.optimizer.state_dict()
    else:
        for attr in ("policy_optimizer", "value_optimizer", "clf_optimizer"):
            if hasattr(agent, attr):
                ckpt[attr] = getattr(agent, attr).state_dict()
    if level_sampler is not None:
        ckpt["level_sampler"] = {
            "seed_scores": level_sampler.seed_scores.copy(),
            "seed_staleness": level_sampler.seed_staleness.copy(),
            "unseen_seed_weights": level_sampler.unseen_seed_weights.copy(),
            "partial_seed_scores": level_sampler.partial_seed_scores.copy(),
            "partial_seed_steps": level_sampler.partial_seed_steps.copy(),
        }
    path = _ckpt_path(save_dir, j)
    torch.save(ckpt, path)
    # keep only the last 2 checkpoints to save disk
    existing = sorted(f for f in os.listdir(save_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    for old in existing[:-2]:
        try:
            os.remove(os.path.join(save_dir, old))
        except OSError:
            pass
    print(f"[ckpt] saved {path}")


def train(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.use_best_hps:
        args.value_epoch = hps.value_epoch[args.env_name]
        args.value_freq = hps.value_freq[args.env_name]
        args.adv_loss_coef = hps.adv_loss_coef[args.env_name]
        args.clf_hidden_size = hps.clf_hidden_size[args.env_name]
        args.order_loss_coef = hps.order_loss_coef[args.env_name]
        args.use_nonlinear_clf = args.env_name in hps.nonlin_envs
    print("\nArguments: ", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    log_file = '-{}-{}-s{}'.format(args.env_name, args.algo, args.seed)
    if args.use_plr:
        log_file += '-plr'
    logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file)

    wandb_name = args.wandb_name or f"{args.algo}-{args.env_name}-s{args.seed}{'-plr' if args.use_plr else ''}"
    wandb_logger = WandBLogger(
        project=args.wandb_project,
        run_name=wandb_name,
        config=vars(args),
        enabled=args.wandb_enabled,
        local_dir=args.log_dir,
    )

    # --- PLR setup ---
    level_sampler = None
    if args.use_plr:
        if args.level_replay_strategy == 'advantage_l1' and args.algo == 'ppo':
            raise ValueError(
                "--level_replay_strategy advantage_l1 requires an advantage head "
                "(use --algo idaac or daac); PPO has no advantage prediction.")
        level_sampler = LevelSampler(
            seeds=list(range(args.num_levels)),
            obs_space=None, action_space=None,
            num_actors=args.num_processes,
            strategy=args.level_replay_strategy,
            score_transform=args.level_replay_score_transform,
            temperature=args.level_replay_temperature,
            eps=args.level_replay_eps,
            rho=args.level_replay_rho,
            nu=args.level_replay_nu,
            alpha=args.level_replay_alpha,
            staleness_coef=args.staleness_coef,
            staleness_transform=args.staleness_transform,
            staleness_temperature=args.staleness_temperature,
        )

    envs = make_envs(args, device, level_sampler)
    obs_shape = envs.observation_space.shape

    # --- Model ---
    if args.algo == 'ppo':
        actor_critic = PPOnet(obs_shape, envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})
    else:
        actor_critic = IDAACnet(obs_shape, envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    if args.algo == 'idaac':
        if args.use_nonlinear_clf:
            order_classifier = NonlinearOrderClassifier(
                emb_size=args.hidden_size, hidden_size=args.clf_hidden_size)
        else:
            order_classifier = LinearOrderClassifier(emb_size=args.hidden_size)
        order_classifier.to(device)

    # --- Storage ---
    if args.algo == 'idaac':
        rollouts = IDAACRolloutStorage(args.num_steps, args.num_processes,
            obs_shape, envs.action_space)
    elif args.algo == 'daac':
        rollouts = DAACRolloutStorage(args.num_steps, args.num_processes,
            obs_shape, envs.action_space)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
            obs_shape, envs.action_space)

    # --- Agent ---
    if args.algo == 'idaac':
        agent = algo.IDAAC(actor_critic, order_classifier,
            args.clip_param, args.ppo_epoch, args.value_epoch, args.value_freq,
            args.num_mini_batch, args.value_loss_coef, args.adv_loss_coef,
            args.order_loss_coef, args.entropy_coef,
            lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    elif args.algo == 'daac':
        agent = algo.DAAC(actor_critic, args.clip_param, args.ppo_epoch,
            args.value_epoch, args.value_freq, args.num_mini_batch,
            args.value_loss_coef, args.adv_loss_coef, args.entropy_coef,
            lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    else:
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch,
            args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
            lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)

    # --- Checkpoint resume ---
    start_update = 0
    episode_rewards = deque(maxlen=10)
    nsteps = torch.zeros(args.num_processes)
    ckpt_info = _latest_checkpoint(args.save_dir) if args.save_dir else None
    if ckpt_info is not None:
        resume_j, ckpt_path = ckpt_info
        print(f"[ckpt] resuming from {ckpt_path} (update {resume_j})")
        ckpt = torch.load(ckpt_path, map_location=device)
        actor_critic.load_state_dict(ckpt["actor_critic"])
        if args.algo == "idaac" and "order_classifier" in ckpt:
            order_classifier.load_state_dict(ckpt["order_classifier"])
        if "optimizer" in ckpt and hasattr(agent, "optimizer"):
            agent.optimizer.load_state_dict(ckpt["optimizer"])
        for attr in ("policy_optimizer", "value_optimizer", "clf_optimizer"):
            if attr in ckpt and hasattr(agent, attr):
                getattr(agent, attr).load_state_dict(ckpt[attr])
        if level_sampler is not None and "level_sampler" in ckpt:
            ls = ckpt["level_sampler"]
            level_sampler.seed_scores[:] = ls["seed_scores"]
            level_sampler.seed_staleness[:] = ls["seed_staleness"]
            level_sampler.unseen_seed_weights[:] = ls["unseen_seed_weights"]
            level_sampler.partial_seed_scores[:] = ls["partial_seed_scores"]
            level_sampler.partial_seed_steps[:] = ls["partial_seed_steps"]
        episode_rewards.extend(ckpt.get("episode_rewards", []))
        # nsteps is not restored: envs.reset() below resets all episodes anyway
        start_update = resume_j + 1

    # --- Training loop ---
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    pbar = trange(start_update, num_updates, desc=f"train[{args.algo}|{args.env_name}]",
                  initial=start_update, total=num_updates)
    for j in pbar:
        actor_critic.train()

        for step in range(args.num_steps):
            with torch.no_grad():
                if args.algo == 'ppo':
                    value, action, action_log_prob, action_log_dist_t = \
                        actor_critic.act(rollouts.obs[step])
                else:
                    adv, value, action, action_log_prob, action_log_dist_t = \
                        actor_critic.act(rollouts.obs[step])

            obs, reward, done, infos = envs.step(action)

            if args.algo == 'idaac' or args.use_plr:
                levels = torch.LongTensor([info['level_seed'] for info in infos])
                if args.algo == 'idaac' and j == 0 and step == 0:
                    rollouts.levels[0].copy_(levels)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            nsteps += 1
            nsteps[done == True] = 0

            if args.algo == 'idaac':
                rollouts.insert(obs, action, action_log_prob, value,
                                reward, masks, adv, levels, nsteps)
            elif args.algo == 'daac':
                rollouts.insert(obs, action, action_log_prob, value,
                                reward, masks, adv)
            else:
                rollouts.insert(obs, action, action_log_prob, value,
                                reward, masks)

            # PLR needs level_seeds and action_log_dist in storage
            if args.use_plr:
                idx = (rollouts.step - 1) % args.num_steps
                rollouts.level_seeds[idx].copy_(levels.unsqueeze(-1))
                rollouts.action_log_dist[idx].copy_(action_log_dist_t)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # PLR: update level scores
        if level_sampler is not None:
            level_sampler.update_with_rollouts(rollouts)

        # Agent update
        if args.algo == 'idaac':
            rollouts.before_update()
            order_acc, order_loss, clf_loss, adv_loss, value_loss, \
                action_loss, dist_entropy = agent.update(rollouts)
        elif args.algo == 'daac':
            adv_loss, value_loss, action_loss, dist_entropy = agent.update(rollouts)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        if level_sampler is not None:
            level_sampler.after_update()

        # Periodic checkpoint + final model save
        is_last = (j == num_updates - 1)
        if args.save_dir and (j % CHECKPOINT_INTERVAL == 0 or is_last):
            _save_checkpoint(j, actor_critic, agent,
                             order_classifier if args.algo == 'idaac' else None,
                             level_sampler, episode_rewards, envs,
                             args.save_dir, log_file)
        if is_last and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save([actor_critic, getattr(envs, 'ob_rms', None)],
                os.path.join(args.save_dir, "agent{}.pt".format(log_file)))

        # Log
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            pbar.set_postfix(reward=f"{np.mean(episode_rewards):.1f}")
            tqdm.write("Update {}, step {}: mean/median reward {:.2f}/{:.2f}"
                .format(j, total_num_steps, np.mean(episode_rewards),
                        np.median(episode_rewards)))
            logger.logkv("train/total_num_steps", total_num_steps)
            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))
            # Value / advantage / return stats from the rollout
            v = rollouts.value_preds[:-1]
            r = rollouts.returns[:-1]
            adv_buffer = r - v
            logger.logkv("rollout/value_mean", v.mean().item())
            logger.logkv("rollout/value_std", v.std().item())
            logger.logkv("rollout/return_mean", r.mean().item())
            logger.logkv("rollout/return_std", r.std().item())
            logger.logkv("rollout/advantage_mean", adv_buffer.mean().item())
            logger.logkv("rollout/advantage_std", adv_buffer.std().item())
            logger.logkv("rollout/advantage_abs_mean", adv_buffer.abs().mean().item())
            logger.logkv("rollout/reward_mean", rollouts.rewards.mean().item())
            if args.algo in ('idaac', 'daac'):
                logger.logkv("rollout/adv_pred_mean", rollouts.adv_preds[:-1].mean().item())
                logger.logkv("rollout/adv_pred_std", rollouts.adv_preds[:-1].std().item())
            logger.logkv("losses/value_loss", value_loss)
            logger.logkv("losses/action_loss", action_loss)
            logger.logkv("losses/entropy", dist_entropy)
            if args.algo in ('idaac', 'daac'):
                logger.logkv("losses/adv_loss", adv_loss)
            if args.algo == 'idaac':
                logger.logkv("losses/order_loss", order_loss)
                logger.logkv("losses/clf_loss", clf_loss)
                logger.logkv("losses/order_acc", order_acc)
            eval_stats = evaluate(args, actor_critic, device)
            eval_episode_rewards = eval_stats['eval_episode_rewards']
            logger.logkv("test/mean_episode_reward", np.mean(eval_episode_rewards))
            logger.logkv("test/median_episode_reward", np.median(eval_episode_rewards))
            logger.logkv("test/value_mean", eval_stats['eval_value_mean'])
            logger.logkv("test/value_std", eval_stats['eval_value_std'])
            if 'eval_adv_pred_mean' in eval_stats:
                logger.logkv("test/adv_pred_mean", eval_stats['eval_adv_pred_mean'])
                logger.logkv("test/adv_pred_std", eval_stats['eval_adv_pred_std'])
            if level_sampler is not None:
                for _plr_k, _plr_v in level_sampler.get_stats().items():
                    logger.logkv(_plr_k, _plr_v)
            logger.dumpkvs()

            # Wandb logging
            wandb_metrics = {
                "train/total_num_steps": total_num_steps,
                "train/mean_episode_reward": np.mean(episode_rewards),
                "train/median_episode_reward": np.median(episode_rewards),
                "rollout/value_mean": v.mean().item(),
                "rollout/value_std": v.std().item(),
                "rollout/return_mean": r.mean().item(),
                "rollout/return_std": r.std().item(),
                "rollout/advantage_mean": adv_buffer.mean().item(),
                "rollout/advantage_std": adv_buffer.std().item(),
                "rollout/advantage_abs_mean": adv_buffer.abs().mean().item(),
                "rollout/reward_mean": rollouts.rewards.mean().item(),
                "losses/value_loss": value_loss,
                "losses/action_loss": action_loss,
                "losses/entropy": dist_entropy,
                "test/mean_episode_reward": np.mean(eval_episode_rewards),
                "test/median_episode_reward": np.median(eval_episode_rewards),
                "test/value_mean": eval_stats['eval_value_mean'],
                "test/value_std": eval_stats['eval_value_std'],
            }
            if args.algo in ('idaac', 'daac'):
                wandb_metrics["losses/adv_loss"] = adv_loss
                wandb_metrics["rollout/adv_pred_mean"] = rollouts.adv_preds[:-1].mean().item()
                wandb_metrics["rollout/adv_pred_std"] = rollouts.adv_preds[:-1].std().item()
            if args.algo == 'idaac':
                wandb_metrics["losses/order_loss"] = order_loss
                wandb_metrics["losses/clf_loss"] = clf_loss
                wandb_metrics["losses/order_acc"] = order_acc
            if 'eval_adv_pred_mean' in eval_stats:
                wandb_metrics["test/adv_pred_mean"] = eval_stats['eval_adv_pred_mean']
                wandb_metrics["test/adv_pred_std"] = eval_stats['eval_adv_pred_std']
            if level_sampler is not None:
                wandb_metrics.update(level_sampler.get_stats())
            wandb_logger.log(wandb_metrics, step=total_num_steps)

    wandb_logger.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
