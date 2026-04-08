"""Quick Modal test: can we call seed() on a Procgen VecEnv?"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-dev", "libglib2.0-0")
    .pip_install("procgen==0.10.7", "gym==0.23.1", "numpy<2.0")
)

app = modal.App("test-procgen-seed")

@app.function(image=image, timeout=120)
def test_seed():
    from procgen import ProcgenEnv
    import numpy as np
    import time

    # Test: create env with specific seed, check info reports it
    env = ProcgenEnv(num_envs=1, env_name="coinrun", num_levels=1, start_level=42)
    obs = env.reset()
    _, _, _, infos = env.step(np.array([0]))
    print(f"start_level=42 -> info seed: {infos[0].get('level_seed')}")
    env.close()

    env = ProcgenEnv(num_envs=1, env_name="coinrun", num_levels=1, start_level=99)
    obs = env.reset()
    _, _, _, infos = env.step(np.array([0]))
    print(f"start_level=99 -> info seed: {infos[0].get('level_seed')}")
    env.close()

    # Test: how fast is env creation?
    t0 = time.time()
    for i in range(20):
        e = ProcgenEnv(num_envs=1, env_name="coinrun", num_levels=1, start_level=i)
        e.reset()
        e.close()
    elapsed = time.time() - t0
    print(f"20 env create/reset/close cycles: {elapsed:.3f}s ({elapsed/20*1000:.1f}ms each)")

    # Test: num_levels=200 and check seed distribution
    env = ProcgenEnv(num_envs=4, env_name="coinrun", num_levels=200, start_level=0)
    obs = env.reset()
    seen = set()
    for i in range(500):
        obs, rew, done, infos = env.step(np.zeros(4, dtype=np.int32))
        for info in infos:
            seen.add(info.get("level_seed"))
    print(f"num_levels=200, 500 steps: saw {len(seen)} unique seeds, range {min(seen)}-{max(seen)}")
    env.close()

@app.local_entrypoint()
def main():
    test_seed.remote()
