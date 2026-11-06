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

`near X` means X and the target are both inside the same camera frame. That is an approximation of the original sentence. SPL uses straight-line distance because Minecraft does not ship a geodesic path: `SR-b * d / max(L, d)`.

## Run order

Login-node tests need no MineDojo and no GPU:

```bash
python3 -m unittest discover -s lfnav/tests
python3 lfnav/perception/train_perception.py --dry-run
python3 lfnav/policy/train_c2.py --dry-run
```

Real episodes follow `lfnav/slurm/README.md`. Data, features, checkpoints, and runs default to `/scratch/$USER/lfnav/` via `configs/lfnav/default.yaml`.

## Matrix

`configs/lfnav/matrix.yaml` names the primary pairs: MineLLM+C0, DINOv2+C0, DINOv2+C1, DINOv2+C2, GT+C1, GT+C2. Perception options also include ViT-B/14 and MineCLIP. MineCLIP is not connected until a local checkout exists under the configured directory.

The original MP5 loop is not executed. `lfnav.eval.run.make_policy` rejects `mp5` and points at `eval/run_batch.py` for a later API run under the same specs, seeds, and judge.

## Limits

- Camera field of view and several observation keys are unconfirmed until `probe_obs.py` and `calibrate_camera.py` have been run. See `lfnav/REPORT.md`.
- Action vectors in `configs/lfnav/actions.yaml` are a local contract (`confirmed: false`).
- Tree labels match name fragments `log`, `leaves`, and `wood` until the probe shows the real block ids.
- A learned head without a checkpoint reports a token statistic, not a trained probability.
