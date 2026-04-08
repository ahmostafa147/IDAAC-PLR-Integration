"""Plot training metrics from CSV log file."""
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot(csv_path):
    df = pd.read_csv(csv_path)
    steps = df['train/total_num_steps']

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('IDAAC + PLR Training', fontsize=14)

    # 1. Rewards
    ax = axes[0, 0]
    ax.plot(steps, df['train/mean_episode_reward'], label='train')
    ax.plot(steps, df['test/mean_episode_reward'], label='test')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward (train vs test)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Value function vs returns
    ax = axes[0, 1]
    ax.plot(steps, df['rollout/value_mean'], label='V(s) predicted')
    ax.plot(steps, df['rollout/return_mean'], label='Return (target)')
    ax.set_ylabel('Value')
    ax.set_title('Value Predictions vs Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Value prediction error (what PLR scores)
    ax = axes[0, 2]
    ax.plot(steps, df['rollout/advantage_abs_mean'], label='|advantage|')
    ax.set_ylabel('Mean |Advantage|')
    ax.set_title('Value Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Losses
    ax = axes[1, 0]
    ax.plot(steps, df['losses/value_loss'], label='value')
    ax.plot(steps, df['losses/adv_loss'], label='advantage')
    ax.set_ylabel('Loss')
    ax.set_title('Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Entropy
    ax = axes[1, 1]
    ax.plot(steps, df['losses/entropy'])
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.axhline(y=2.71, color='r', linestyle='--', alpha=0.5, label='max (uniform)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Order classifier
    ax = axes[1, 2]
    ax.plot(steps, df['losses/order_acc'], label='order accuracy')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='random chance')
    ax.set_ylabel('Accuracy')
    ax.set_title('Order Classifier (lower = more invariant)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. PLR scores
    ax = axes[2, 0]
    ax.plot(steps, df['plr/score_mean'], label='mean')
    ax.plot(steps, df['plr/score_max'], label='max')
    ax.plot(steps, df['plr/score_min'], label='min')
    ax.set_ylabel('Score')
    ax.set_title('PLR Level Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. PLR replay ratio + unique seeds
    ax = axes[2, 1]
    ax.plot(steps, df['plr/replay_ratio_last100'], label='replay ratio', color='tab:blue')
    ax.set_ylabel('Replay Ratio', color='tab:blue')
    ax.set_title('PLR Replay Ratio & Seed Diversity')
    ax2 = ax.twinx()
    ax2.plot(steps, df['plr/unique_seeds_last100'], label='unique seeds', color='tab:orange')
    ax2.set_ylabel('Unique Seeds (last 100)', color='tab:orange')
    ax.grid(True, alpha=0.3)

    # 9. Train vs test value (generalization gap)
    ax = axes[2, 2]
    ax.plot(steps, df['rollout/value_mean'], label='train V(s)')
    ax.plot(steps, df['test/value_mean'], label='test V(s)')
    ax.set_ylabel('Value')
    ax.set_title('Value Generalization (train vs test)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Env Steps')

    plt.tight_layout()
    out = csv_path.rsplit('.', 1)[0] + '_plots.png'
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()

if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'local_logs'
    plot(csv_path)
