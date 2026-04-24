"""Smoke test: run IDAAC+PLR for a few updates on Modal."""
import os
import subprocess
from pathlib import Path
import modal

APP_NAME = "idaac-plr-test"
PROJECT_DIR = "/root/project"

def load_gitignore_patterns():
    if not modal.is_local():
        return []
    root = Path(__file__).resolve().parents[1]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []
    patterns = []
    for line in gitignore_path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            patterns.append(f"**/{entry.rstrip('/')}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-dev", "libglib2.0-0")
    .pip_install(
        "torch>=1.12,<2.1", "numpy<2.0", "procgen==0.10.7",
        "gym==0.23.1", "tqdm", "wandb",
    )
    .add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())
)

app = modal.App(APP_NAME)

CONFIGS = [
    ("idaac-alone",     ["--algo", "idaac"]),
    ("idaac-plr",       ["--algo", "idaac", "--use_plr"]),
    ("idaac-plr-adv",   ["--algo", "idaac", "--use_plr",
                         "--level_replay_strategy", "advantage_l1"]),
    ("ppo-plr",         ["--algo", "ppo",   "--use_plr"]),
]

@app.function(image=image, timeout=1200, gpu="T4", cpu=4.0, memory=8192)
def smoke_test():
    for name, extra in CONFIGS:
        print(f"\n========== SMOKE: {name} ==========", flush=True)
        cmd = [
            "python", "-u", "train.py",
            "--env_name", "coinrun",
            "--num_processes", "8",
            "--num_steps", "64",
            "--num_env_steps", "8192",
            "--log_interval", "4",
            "--num_eval_envs", "16",
            "--num_eval_episodes", "2",
            "--log_dir", f"/tmp/logs/{name}",
            "--save_dir", "",
        ] + extra
        subprocess.run(cmd, cwd=PROJECT_DIR, check=True)
        print(f"========== OK: {name} ==========", flush=True)

@app.local_entrypoint()
def main():
    smoke_test.remote()
