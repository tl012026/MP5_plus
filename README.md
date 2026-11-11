# MP5_plus

A fork of [MP5](https://arxiv.org/abs/2312.07472v2) (CVPR 2024, [project page](https://iranqin.github.io/MP5.github.io/)), a Minecraft agent built on [MineDojo](https://github.com/MineDojo/MineDojo).

Two tracks:

1. **Baseline.** The original MP5 decision loop, with YAML configs, per-episode logs, batch evaluation, and Slurm templates. `MP5_agent/agent/prompts/` is unchanged.
2. **LLM-free navigation.** [`lfnav/`](lfnav/README.md) solves the 16 creative tasks without a language model on the control path. Short crafting chains are not implemented.

## Baseline

Task files (`MP5_agent/agent/tasks/`) list pre-decomposed subgoals. The released code has no Parser and no shared Knowledge Memory. The file lists the final goal first; `run_agent.py` runs `task_list[::-1]`.

Each subgoal loops until success or 300 retries:

- **Plan.** A text LLM (default `gpt-4-0613`) returns a JSON workflow from the subgoal, inventory, ground/underground, up to two past plans, and the last failure note. Unparseable JSON is retried 5 times.
- **Execute.** A code check compares the inventory with the action's tool and materials. The performer then runs `find`, `move_to`, `mine`, `craft`, or `dig_down`. `fight`, `equip`, `apply`, `place`, and `dig_up` do not change the game.
- **`find`.** Above ground, every step sends the RGB frame to perception. Active mode uses a text LLM to ask the next question and MineLLM or `gpt-4-vision-preview` to answer it. Underground `find` uses voxels only. Hitting the exploration cap does not raise; the rest of the workflow still runs.
- **Check.** A text LLM (default `gpt-3.5-turbo-0613`) returns `success`, `feedback`, and `suggestion`. Failure goes back to planning.
- **Memory.** A successful workflow is stored for this episode only and retrieved by subgoal embedding.

MineDojo defaults (`configs/code_default.yaml`): `harvest` / `diamond`, 512×820, biome `forest`, break speed ×100, 6 dirt and 6 coal, `max_steps` 12000. `configs/strict.yaml` uses an empty inventory, break speed ×1, and a random biome. RGB is used only inside `find`. The 11³ voxel crop and 24×24 LiDAR are used only by the scripted mover.

An episode ends on success, 300 retries (`fail`), the step limit (`timeout` or `timeout_in_find`), death, or an exception. Process-task success is the final inventory item, checked in code. Creative-task `success` stays `null` in the original loop; the automatic judge is in `lfnav`.

| Role | Default | Config |
|---|---|---|
| Planner | `gpt-4-0613` | `models.planner` |
| Perception questions | `gpt-4-0613` | `models.percipient_question` |
| Vision answers | MineLLM or `gpt-4-vision-preview` | `percipient.*` |
| Completion check | `gpt-3.5-turbo-0613` | `models.patroller` |
| Memory embedding | `OpenAIEmbeddings()` | `agent/llm_factory.py` |

`--gpt_model_name` overrides the planner, patroller, and question generator only. Record the models actually served. MineLLM samples (`temperature=1.0`, `top_p=0.9`). Most API calls come from `find`. Exploration is a random walk. The original patroller retry path calls the LLM with the wrong arguments, so a second exception ends the episode as `error`.

```bash
python3 tools/validate_tasks.py && python3 tools/test_offline.py

export OPENAI_API_KEY=...
cd MP5_agent/agent
python3 run_agent.py --config ../../configs/code_default.yaml \
  --task tasks/process/craft_wooden_pickaxe.json --seed 10001 \
  --out_dir /scratch/$USER/mp5_runs/smoke/craft_wooden_pickaxe/10001 \
  --mllm_url http://127.0.0.1:25541/image --answer_method active --answer_model mllm

python3 eval/run_batch.py --config configs/code_default.yaml \
  --tasks MP5_agent/agent/tasks/process --seeds configs/seeds_example.txt \
  --out /scratch/$USER/mp5_runs/run1 --workers 1 \
  --mllm-url http://127.0.0.1:25541/image --skip-done
```

The key is read only from `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`). `--openai_key` is ignored. `run_episode` changes into `MP5_agent/agent` so prompt paths resolve. `--answer_model gpt-vision` skips the MineLLM server.

Each episode writes `OUT/<task>/<seed>.json` and `.events.jsonl`, plus `images/`, `video/` (if `save_frames`), `logs/`, and `memory/`. A run writes `run_meta.json` and `summary.csv`. Writes are atomic; `--skip-done` resumes.

| Tasks | Success in the original loop |
|---|---|
| `tasks/creative/` (`1-1` … `4-4`) | `null` (self-report and a ground-truth snapshot) |
| `tasks/process/` (pickaxes, ore, ingot, diamond) | final item in the inventory |

Item names follow `update_*_name` in `utils/common_utils.py`.

## LLM-free navigator

`lfnav` is Track A: conditional object navigation on those 16 creative tasks. Details, the task transcription, and the job order are in [lfnav/README.md](lfnav/README.md), [lfnav/tasks/README.md](lfnav/tasks/README.md), and [lfnav/slurm/README.md](lfnav/slurm/README.md).

A frame becomes a scene state, the matcher checks the task conditions, and a policy emits a short action chunk. The controller turns that into environment steps. MineLLM is only an HTTP baseline with fixed questions. The original agent is not started here; `make_policy("mp5")` points back at `eval/run_batch.py`.

Primary success (**SR-b**): the agent stops, the target is in view, distance is at most 3 blocks, and every condition holds. **SR-a** drops the distance check. A stop that fails SR-a is a false stop. SPL is `SR-b * d / max(L, d)` with straight-line `d`. Timeout is 3000 steps.

Policies: **C0** random walk, **C1** rules, **joint** A+C2 on frozen DINOv2. **C3** is unimplemented. The comparisons in `configs/lfnav/matrix.yaml` are MineLLM+C0, DINOv2+C0/C1/C2, and ground-truth+C1/C2.

```bash
python3 -m unittest discover -s lfnav/tests
python3 lfnav/policy/train_joint.py --dry-run
```

Those commands need no MineDojo and no GPU. Real collection, training, and eval are the Slurm scripts under `lfnav/slurm/`. Account and partition are blank; fill them locally. Data defaults to `/scratch/$USER/lfnav/`.

Before a MineDojo run: probe the observation (`lfnav/gt/probe_obs.py`), fit the camera (`lfnav/gt/calibrate_camera.py`), and set `actions.confirmed` only after the action vectors match. Until then, named actions on the real environment raise. Tree labels match the fragments `log`, `leaves`, and `wood` until the probe shows the real block names.

## Layout

```
configs/                 baseline YAML; configs/lfnav/ for the navigator
eval/                    run_batch.py, summarize.py
lfnav/                   LLM-free navigator, tests, and Slurm templates
slurm/                   baseline CPU / GPU templates (slurm/README.md)
Minedojo environment/    image setup and a MineDojo smoke test
MP5_agent/agent/         original loop, prompts, tasks/{creative,process}/
tools/                   validate_tasks.py, test_offline.py
```

Do not edit `MP5_agent/agent/structured_actions copy.py`. `LAMM/`, weights, `*.sif`, overlays, and key files stay out of git.

## External assets

- MineLLM code: [IranQin/LAMM](https://github.com/IranQin/LAMM), cloned as `LAMM/`.
- Weights, pinned in `models.lock.json`: [IranQin/MineLLM](https://huggingface.co/IranQin/MineLLM) (`Mine_52k_Vicuna_13b_v1.5_epoch_3_mlp2x_gelu/pytorch_model.pt`), [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5), and `mineclip_image_encoder_vit-B_196tokens.pth`.
- From `LAMM/src`, the server expects `../ckpt/<folder>/pytorch_model.pt`, `../model_zoo/vicuna_ckpt/13b_v1.5/`, `../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth`, and `../images/`.
- Start with `python mllm_api_mc_v1.5.py` in `LAMM/src`. It listens on port 25541. `POST /image` takes `file`, `text`, and `is_del`, and returns `{"answer": ...}`. fp16 needs about 27 GB. Do not use LAMM's `src/scripts/mllm_api_slurm.sh`.
- Offline perception data: [IranQin/MP5_dataset](https://huggingface.co/datasets/IranQin/MP5_dataset).

## Environment

Python ≥ 3.9, JDK 8, Xvfb, [MineDojo](https://docs.minedojo.org/sections/getting_started/install.html#prerequisites), and `MP5_agent/requirements.txt`. Image setup is in [`Minedojo environment/SETUP_MINEDOJO.md`](Minedojo%20environment/SETUP_MINEDOJO.md). The MineLLM server wants Python 3.10, PyTorch 1.13.1, `transformers==4.31.0`, `flask`, and `flask_cors`. Navigator training uses `lfnav/requirements-train.txt` (PyTorch 2.x) and is separate from the MineDojo image. Slurm account and partition stay in your local env, not in the repo.

## Known issues

- Patroller retry-on-exception uses the wrong call signature (original code).
- `openai.api_base` is process-wide.
- MineLLM flags in `slurm/eval_gpu_minellm.sbatch` are placeholders.
- `lfnav` camera field of view, several observation keys, and action vectors are still unconfirmed.

## Citation

```bibtex
@inproceedings{qin2024mp5,
  title={MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception},
  author={Qin, Yiran and Zhou, Enshen and Liu, Qichang and Yin, Zhenfei and Sheng, Lu and Zhang, Ruimao and Qiao, Yu and Shao, Jing},
  booktitle={CVPR}, pages={16307--16316}, year={2024}
}
```
