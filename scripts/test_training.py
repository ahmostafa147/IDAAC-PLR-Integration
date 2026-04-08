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
        "gym==0.23.1",
    )
    .add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())
)

app = modal.App(APP_NAME)

@app.function(image=image, timeout=600, gpu="T4", cpu=4.0, memory=8192)
def smoke_test():
    cmd = [
        "python", "-u", "train.py",
        "--algo", "idaac",
        "--env_name", "coinrun",
        "--use_plr",
        "--num_processes", "4",
        "--num_steps", "32",
        "--num_env_steps", "2048",
        "--log_interval", "4",
        "--log_dir", "/tmp/logs",
        "--save_dir", "",
    ]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)

@app.local_entrypoint()
def main():
    smoke_test.remote()
