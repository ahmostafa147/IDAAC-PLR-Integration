from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "idaac-plr"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "A10G"
DEFAULT_CPU = 16.0
DEFAULT_MEMORY_MB = 32768
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24  # 24h
DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS = 300

volume = modal.Volume.from_name("idaac-plr-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    if not modal.is_local():
        return []
    root = Path(__file__).resolve().parents[1]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []
    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


def _to_volume_path(path_value: str) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return path_value
    return str(Path(VOLUME_PATH) / p)


def _rewrite_path_flag(args: list[str], flag: str, *, default: str | None = None) -> list[str]:
    out = list(args)
    found = False
    i = 0
    while i < len(out):
        if out[i] == flag:
            found = True
            if i + 1 >= len(out):
                raise ValueError(f"Missing value for {flag}")
            out[i + 1] = _to_volume_path(out[i + 1])
            i += 2
            continue
        if out[i].startswith(f"{flag}="):
            found = True
            key, value = out[i].split("=", 1)
            out[i] = f"{key}={_to_volume_path(value)}"
        i += 1
    if not found and default is not None:
        out.extend([flag, _to_volume_path(default)])
    return out


def _normalize_args(args: tuple[str, ...]) -> list[str]:
    normalized = list(args)
    normalized = _rewrite_path_flag(normalized, "--log_dir", default="runs/logs")
    normalized = _rewrite_path_flag(normalized, "--save_dir", default="runs/models")
    return normalized


def _run_subprocess_with_periodic_volume_commits(cmd: list[str]) -> None:
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR)
    returncode: int | None = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(timeout=DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        volume.commit()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


# --- Modal Image ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-dev", "libglib2.0-0")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu118 'torch>=1.12,<2.1'"
    )
    .pip_install(
        "numpy<2.0",
        "procgen==0.10.7",
        "gym==0.23.1",
        "wandb",
        "tqdm",
        "matplotlib",
        "pandas",
    )
)

if NETRC_PATH.is_file():
    image = image.add_local_file(NETRC_PATH, remote_path="/root/.netrc", copy=True)

image = image.add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())

app = modal.App(APP_NAME)

function_secrets = []
if os.environ.get("WANDB_API_KEY"):
    function_secrets.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def train_remote(*args: str) -> None:
    normalized = _normalize_args(args)
    cmd = ["python", "-u", "train.py", *normalized]
    _run_subprocess_with_periodic_volume_commits(cmd)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def eval_remote(*args: str) -> None:
    cmd = ["python", "-u", "test.py", *args]
    _run_subprocess_with_periodic_volume_commits(cmd)


@app.local_entrypoint()
def main(*args: str) -> None:
    """Default: forward args to train_remote."""
    train_remote.remote(*args)
