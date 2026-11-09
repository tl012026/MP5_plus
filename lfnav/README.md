# lfnav

LLM-free conditional goal navigation for the 16 creative tasks in MP5. No language model is called on the control path. MineLLM is only a perception baseline, and it is queried with fixed templates.

## Modules

```
rgb frame --> A perceive --> SceneState
TaskSpec + SceneState --> B match --> MatchResult
SceneState + MatchResult + visit grid + recent actions --> C act --> action chunk or waypoints
action chunk --> controller --> environment
stop --> judge (SR-b, SR-a)
```

| New module | Replaces in MP5 |
| --- | --- |
| A `lfnav/perception` | Percipient (MineLLM), without GPT-4 asking the questions |
| B `lfnav/matcher.py` | The part of the percipient prompt that decides the object is found |
| C0 / C1 / C2 | The scripted explore branch inside the find action |
| controller | `structured_actions.py` movement, reimplemented and limited to a 3x3x3 voxel crop |
| judge | A new automatic check. MP5's original success rule is SR-a |

C3 is `lfnav/policy/c3.py`. `act` raises `NotImplementedError`. The controller already accepts `(dyaw_deg, forward_blocks)` waypoints.

## Privileged information

- Wide lidar and ground-truth scene state are used by the expert and by the judge. Evaluation policies do not receive them, except GT perception when that backend is the experiment.
- The 3x3x3 voxel crop is a contact sensor for the controller only. C0, C1, and C2 do not read it.
- Position is odometry from the environment position field.

`near X` means X and the target are both inside the same camera frame. That is an approximation of the original sentence. SR-b distance is the nearest aligned-ray hit on the target at stop, so the target has to be in the camera frame. The fake env still has Euclidean `distance_to` for tests. SPL uses straight-line path length because Minecraft does not ship a geodesic: `SR-b * d / max(L, d)`.

Aligned rays need a camera model. `lfnav/gt/capture_rotations.py` turns in place (no walk, no sprint, skip water). `fit_intrinsics.py` fits `K` from the pure-rotation homography `x2 ~ K R K^{-1} x1`. HUD (hotbar, crosshair, hand) is masked; RANSAC drops movers. Copy the printed `fov_deg`/`fov_axis`, or a full `intrinsics` block, into `configs/lfnav/default.yaml`. `verify_alignment.py` treats median IoU above 0.5 as good enough. The one-pixel script `calibrate_camera.py` is still there.

## Run order

Login-node tests need no MineDojo and no GPU:

```bash
python3 -m unittest discover -s lfnav/tests
python3 lfnav/perception/train_perception.py --dry-run
python3 lfnav/policy/train_c2.py --dry-run
```

Real episodes follow `lfnav/slurm/README.md`. Data, features, checkpoints, and runs default to `/scratch/$USER/lfnav/` via `configs/lfnav/default.yaml`.

Each episode closes its environment, including when the episode raises. `reuse_env` defaults to false. Turn it on to keep one MineDojo process for consecutive episodes of the same task and difficulty and to start the next one with `reset`. In that mode `make()` uses the first episode's seed as `world_seed`. Later seeds stay on the result file and on agent-side randomness, but they do not build a new world.

## Matrix

`configs/lfnav/matrix.yaml` names the primary pairs: MineLLM+C0, DINOv2+C0, DINOv2+C1, DINOv2+C2, GT+C1, GT+C2. Perception options also include ViT-B/14 and MineCLIP. MineCLIP is not connected until a local checkout exists under the configured directory.

The original MP5 loop is not executed. `lfnav.eval.run.make_policy` rejects `mp5` and points at `eval/run_batch.py` for a later API run under the same specs, seeds, and judge.

## Limits

- Camera field of view is unconfirmed until `sbatch lfnav/slurm/calibrate_camera.sbatch` has been run. Time, brightness, and biome ids are filled by `lfnav/gt/calibrate_env_state.py`. Until `time_day_min` and `brightness_min_light` are set, those conditions are unknown rather than failed. See `lfnav/REPORT.md`.
- Action vectors in `configs/lfnav/actions.yaml` are confirmed against the harvest_milk probe.
- Tree labels match name fragments `log`, `leaves`, and `wood` until the probe shows the real block ids.
- A learned head without a checkpoint reports a token statistic, not a trained probability.
