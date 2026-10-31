"""Check that MineDojo runs, then measure speed with the MP5 environment config.

Run inside the MineDojo image:
  xvfb-run -a python "Minedojo environment/validate_minedojo.py"
"""
import time

import minedojo
import numpy as np

# 1) Same idea as MineDojo's validate_install: create a task and take 20 steps.
t0 = time.time()
env = minedojo.make(task_id="harvest_milk", image_size=(288, 512), world_seed=123, seed=42)
print(f"[INFO] make: {time.time() - t0:.1f}s, prompt: {env.task_prompt}", flush=True)
t0 = time.time()
obs = env.reset()
print(f"[INFO] reset: {time.time() - t0:.1f}s", flush=True)
for _ in range(20):
    obs, r, done, info = env.step(env.action_space.no_op())
print("[INFO] rgb shape:", obs["rgb"].shape, "| pos:", obs["location_stats"]["pos"], flush=True)
env.close()
print("[INFO] Installation Success", flush=True)

# 2) Speed with the original MP5 settings (512x820, 11^3 voxels, 24x24 LiDAR rays).
v = 5
env = minedojo.make(
    task_id="harvest", target_names="diamond", target_quantities=100,
    image_size=(512, 820), seed=3, world_seed=123, specified_biome="forest",
    spawn_rate=1, break_speed_multiplier=100.0,
    spawn_range_low=(-10, -10, -10), spawn_range_high=(10, 10, 10),
    use_voxel=True, voxel_size=dict(xmin=-v, ymin=-v, zmin=-v, xmax=v, ymax=v, zmax=v),
    use_lidar=True,
    lidar_rays=[(np.pi * p / 180, np.pi * y / 180, 10)
                for p in np.arange(-60, 60, 5) for y in np.arange(-60, 60, 5)],
)
env.reset()
n = 200
t0 = time.time()
for _ in range(n):
    obs, r, done, info = env.step(env.action_space.no_op())
dt = time.time() - t0
print(f"[INFO] MP5 config: {n / dt:.1f} steps/s -> 12000 steps ~ {12000 / (n / dt) / 60:.1f} min", flush=True)
print("[INFO] voxels:", obs["voxels"]["block_name"].shape, "| lidar keys:", list(obs["rays"].keys())[:5], flush=True)
env.close()
