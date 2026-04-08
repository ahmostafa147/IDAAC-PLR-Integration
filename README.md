# IDAAC + PLR

Combining [IDAAC](https://arxiv.org/abs/2106.04799) (Invariant Decoupled Advantage Actor-Critic) with [Prioritized Level Replay](https://arxiv.org/abs/2010.03934) for improved generalization on [Procgen](https://openai.com/research/procgen-benchmark). Built for CS 285.

IDAAC uses a decoupled architecture (separate policy and value networks) with an adversarial invariance objective to learn features that generalize across levels. PLR maintains a scoring system over training levels based on value prediction error and preferentially replays levels where the agent struggles. The two methods are orthogonal — IDAAC changes how you learn, PLR changes what you train on.

## Setup

Procgen only has x86_64 wheels, so training runs on [Modal](https://modal.com) (cloud GPU). Local machine just needs the Modal client.

```bash
python -m venv venv
source venv/bin/activate
pip install modal wandb
modal token new
wandb login
```

## Training

```bash
# IDAAC + PLR (the main experiment)
modal run scripts/modal_train.py -- \
  --algo idaac --env_name coinrun --use_plr \
  --num_processes 64 --num_steps 256 --num_env_steps 25000000 \
  --log_interval 5 --wandb_enabled

# IDAAC alone (baseline)
modal run scripts/modal_train.py -- \
  --algo idaac --env_name coinrun \
  --num_processes 64 --num_steps 256 --num_env_steps 25000000 \
  --log_interval 5 --wandb_enabled

# PPO + PLR (baseline)
modal run scripts/modal_train.py -- \
  --algo ppo --env_name coinrun --use_plr \
  --num_processes 64 --num_steps 256 --num_env_steps 25000000 \
  --log_interval 5 --wandb_enabled
```

Add `--detach` after `modal run` to keep the run alive after closing your terminal.

Logs and model checkpoints are saved to a Modal volume (`idaac-plr-volume`). Pull them with:

```bash
modal volume get idaac-plr-volume runs/logs/ ./local_logs/
modal volume get idaac-plr-volume runs/models/ ./local_models/
```

## Plotting

```bash
pip install pandas matplotlib
python scripts/plot.py local_logs
```

## Docker (alternative)

```bash
docker build -t idaac-plr .
docker run --gpus all idaac-plr --algo idaac --env_name coinrun --use_plr
```

## Repo structure

```
train.py                    # training loop
test.py                     # evaluation
hyperparams.py              # per-game best hyperparameters from IDAAC paper
ppo_daac_idaac/
  model.py                  # IDAACnet, PPOnet, ValueResNet, classifiers
  algo/                     # IDAAC, DAAC, PPO update logic
  storage.py                # rollout buffers
  envs.py                   # PLRProcgenVecEnv (per-env seed control), VecPyTorchProcgen
  level_sampler.py          # PLR scoring and sampling
  arguments.py              # all CLI args
  distributions.py          # categorical distribution wrapper
  utils.py                  # init helpers
  wandb_utils.py            # wandb logger
baselines/                  # vendored subset of openai/baselines (no TF dependency)
scripts/
  modal_train.py            # Modal GPU launcher
  test_training.py          # smoke test
  plot.py                   # visualization
```

## Key integration details

Procgen doesn't expose a `seed()` method — the only way to control which level plays is at environment creation via `ProcgenEnv(num_levels=1, start_level=seed)`. PLR needs per-environment seed control, so `PLRProcgenVecEnv` manages 64 individual environments and recreates them with PLR-chosen seeds when episodes end (~3ms overhead per reset).

PLR scores levels using `value_l1`: the mean absolute error between predicted values and computed returns. High error means the value network doesn't understand that level, so PLR replays it more. Scores are converted to sampling probabilities via rank transform with temperature 0.1.

## References

- Raparthy et al., "Generalization in Reinforcement Learning by Soft Data Augmentation" (IDAAC), 2021
- Jiang et al., "Prioritized Level Replay", 2021
