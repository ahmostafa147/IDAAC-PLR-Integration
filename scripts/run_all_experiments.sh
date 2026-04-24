#!/usr/bin/env bash
# Full benchmark launcher for IDAAC-PLR-Integration.
#
# All 16 Procgen games x 4 configs, paper-faithful hyperparameters
# (Table in IDAAC paper appendix: gamma=0.999, lambda=0.95, T=256,
# ppo_epoch=3, minibatches=8, entropy=0.01, clip=0.2, lr=5e-4,
# 64 envs, 25M steps, Adam). IDAAC runs additionally use per-game
# best hyperparameters from hyperparams.py (via --use_best_hps).
#
# Configs launched per game:
#   1) IDAAC alone                           (no PLR)
#   2) IDAAC + PLR with value_l1 scoring     (original PLR paper score)
#   3) IDAAC + PLR with advantage_l1 scoring (our contribution: uses IDAAC's adv head)
#   4) PPO   + PLR with value_l1 scoring     (baseline)
#
# 16 * 4 = 64 runs. Each is a detached Modal job on L4 (~$0.80/hr).
# Expected wall clock per run: IDAAC configs ~3-5 h, PPO config ~1.5-2 h.
# Total GPU-hours ~200, total cost ~$160, runs in parallel on Modal
# subject to your workspace's concurrency limit.
#
# Usage:
#   bash scripts/run_all_experiments.sh           # launch everything
#   # or: copy any single `modal run ...` line and paste it into a shell.
#
# To add more seeds, re-run the script with SEED=1 then SEED=2:
#   SEED=1 bash scripts/run_all_experiments.sh
#   SEED=2 bash scripts/run_all_experiments.sh

set -u

if ! command -v modal >/dev/null 2>&1; then
  echo "error: 'modal' not found in PATH. Activate the venv first:" >&2
  echo "    source venv/bin/activate && bash scripts/run_all_experiments.sh" >&2
  exit 1
fi

SEED="${SEED:-0}"
LAUNCHED=0

# --- Shared paper hyperparameters ---
# (defaults in arguments.py already match most of these; we pass them
#  explicitly so the command line is self-documenting and reproducible.)
COMMON_ARGS=(
  --num_processes 64
  --num_steps 256
  --num_env_steps 25000000
  --ppo_epoch 3
  --num_mini_batch 8
  --entropy_coef 0.01
  --clip_param 0.2
  --lr 5e-4
  --gamma 0.999
  --gae_lambda 0.95
  --seed "$SEED"
  --log_interval 10
  --wandb_enabled
)

launch() {
  local tag="$1"; shift
  echo "[launch] $tag"
  modal run --detach scripts/modal_train.py -- \
    --log_dir "runs/logs/$tag" \
    --save_dir "runs/models/$tag" \
    --wandb_name "$tag" \
    "${COMMON_ARGS[@]}" \
    "$@"
  LAUNCHED=$((LAUNCHED + 1))
}

# # ===================================================================
# # 1) IDAAC alone (no PLR)
# # ===================================================================
launch "idaac-bigfish-s$SEED"    --algo idaac --env_name bigfish    --use_best_hps
launch "idaac-bossfight-s$SEED"  --algo idaac --env_name bossfight  --use_best_hps
launch "idaac-caveflyer-s$SEED"  --algo idaac --env_name caveflyer  --use_best_hps
launch "idaac-chaser-s$SEED"     --algo idaac --env_name chaser     --use_best_hps
launch "idaac-climber-s$SEED"    --algo idaac --env_name climber    --use_best_hps
launch "idaac-coinrun-s$SEED"    --algo idaac --env_name coinrun    --use_best_hps
launch "idaac-dodgeball-s$SEED"  --algo idaac --env_name dodgeball  --use_best_hps
launch "idaac-fruitbot-s$SEED"   --algo idaac --env_name fruitbot   --use_best_hps
launch "idaac-heist-s$SEED"      --algo idaac --env_name heist      --use_best_hps
launch "idaac-jumper-s$SEED"     --algo idaac --env_name jumper     --use_best_hps
launch "idaac-leaper-s$SEED"     --algo idaac --env_name leaper     --use_best_hps
launch "idaac-maze-s$SEED"       --algo idaac --env_name maze       --use_best_hps
launch "idaac-miner-s$SEED"      --algo idaac --env_name miner      --use_best_hps
launch "idaac-ninja-s$SEED"      --algo idaac --env_name ninja      --use_best_hps
launch "idaac-plunder-s$SEED"    --algo idaac --env_name plunder    --use_best_hps
launch "idaac-starpilot-s$SEED"  --algo idaac --env_name starpilot  --use_best_hps

# # ===================================================================
# # 2) IDAAC + PLR (value_l1 scoring)
# # ===================================================================
launch "idaac-plr-value_l1-bigfish-s$SEED"   --algo idaac --env_name bigfish   --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-bossfight-s$SEED" --algo idaac --env_name bossfight --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-caveflyer-s$SEED" --algo idaac --env_name caveflyer --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-chaser-s$SEED"    --algo idaac --env_name chaser    --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-climber-s$SEED"   --algo idaac --env_name climber   --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-coinrun-s$SEED"   --algo idaac --env_name coinrun   --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-dodgeball-s$SEED" --algo idaac --env_name dodgeball --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-fruitbot-s$SEED"  --algo idaac --env_name fruitbot  --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-heist-s$SEED"     --algo idaac --env_name heist     --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-jumper-s$SEED"    --algo idaac --env_name jumper    --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-leaper-s$SEED"    --algo idaac --env_name leaper    --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-maze-s$SEED"      --algo idaac --env_name maze      --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-miner-s$SEED"     --algo idaac --env_name miner     --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-ninja-s$SEED"     --algo idaac --env_name ninja     --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-plunder-s$SEED"   --algo idaac --env_name plunder   --use_plr --level_replay_strategy value_l1 --use_best_hps
launch "idaac-plr-value_l1-starpilot-s$SEED" --algo idaac --env_name starpilot --use_plr --level_replay_strategy value_l1 --use_best_hps

# ===================================================================
# 3) IDAAC + PLR (advantage_l1 scoring; ours)
# ===================================================================
launch "idaac-plr-adv_l1-bigfish-s$SEED"   --algo idaac --env_name bigfish   --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-bossfight-s$SEED" --algo idaac --env_name bossfight --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-caveflyer-s$SEED" --algo idaac --env_name caveflyer --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-chaser-s$SEED"    --algo idaac --env_name chaser    --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-climber-s$SEED"   --algo idaac --env_name climber   --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-coinrun-s$SEED"   --algo idaac --env_name coinrun   --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-dodgeball-s$SEED" --algo idaac --env_name dodgeball --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-fruitbot-s$SEED"  --algo idaac --env_name fruitbot  --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-heist-s$SEED"     --algo idaac --env_name heist     --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-jumper-s$SEED"    --algo idaac --env_name jumper    --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-leaper-s$SEED"    --algo idaac --env_name leaper    --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-maze-s$SEED"      --algo idaac --env_name maze      --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-miner-s$SEED"     --algo idaac --env_name miner     --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-ninja-s$SEED"     --algo idaac --env_name ninja     --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-plunder-s$SEED"   --algo idaac --env_name plunder   --use_plr --level_replay_strategy advantage_l1 --use_best_hps
launch "idaac-plr-adv_l1-starpilot-s$SEED" --algo idaac --env_name starpilot --use_plr --level_replay_strategy advantage_l1 --use_best_hps

# # ===================================================================
# # 4) PPO + PLR (value_l1 scoring)
# # ===================================================================
launch "ppo-plr-value_l1-bigfish-s$SEED"   --algo ppo --env_name bigfish   --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-bossfight-s$SEED" --algo ppo --env_name bossfight --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-caveflyer-s$SEED" --algo ppo --env_name caveflyer --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-chaser-s$SEED"    --algo ppo --env_name chaser    --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-climber-s$SEED"   --algo ppo --env_name climber   --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-coinrun-s$SEED"   --algo ppo --env_name coinrun   --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-dodgeball-s$SEED" --algo ppo --env_name dodgeball --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-fruitbot-s$SEED"  --algo ppo --env_name fruitbot  --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-heist-s$SEED"     --algo ppo --env_name heist     --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-jumper-s$SEED"    --algo ppo --env_name jumper    --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-leaper-s$SEED"    --algo ppo --env_name leaper    --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-maze-s$SEED"      --algo ppo --env_name maze      --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-miner-s$SEED"     --algo ppo --env_name miner     --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-ninja-s$SEED"     --algo ppo --env_name ninja     --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-plunder-s$SEED"   --algo ppo --env_name plunder   --use_plr --level_replay_strategy value_l1
launch "ppo-plr-value_l1-starpilot-s$SEED" --algo ppo --env_name starpilot --use_plr --level_replay_strategy value_l1

echo
echo "[done] launched $LAUNCHED detached runs for seed=$SEED"
echo "       monitor: modal app list    |    wandb project: idaac-plr"
echo "       pull logs later: modal volume get idaac-plr-volume runs/logs ./local_logs"
