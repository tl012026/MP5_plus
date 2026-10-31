# MP5_plus

Eval/config fork of [MP5](https://arxiv.org/abs/2312.07472v2) (CVPR 2024, [project](https://iranqin.github.io/MP5.github.io/)). Minecraft agent on [MineDojo](https://github.com/MineDojo/MineDojo). **Decision loop and `prompts/` are unchanged.** This repo adds YAML config, per-episode outputs, batch eval, and Slurm templates.

Do not edit `MP5_agent/agent/structured_actions copy.py`. Do not install MineDojo/JDK on the login node; use `python3 tools/test_offline.py`.

## Pipeline

Not image+text → one answer. A **text subgoal list** drives a Minecraft loop. **RGB is used only in `find`.** Defaults: `configs/code_default.yaml`. Swap a model in that YAML or `llm_factory.py`.

**Inputs.** Instruction = task JSON (`task`, `quantity`, `material`, `tool`, `platform`, `tips`, `description`). Inventory from MineDojo. Screenshot only during `find`. Voxels/LiDAR only for movement (not sent to Planner). Past plans via embedding search.

JSON is run as `task_list[::-1]` (final goal first in the file). Example wooden pickaxe file: pickaxe → table → stick → planks → wood. Run: wood → … → pickaxe.

**Who does what** (`code_default`):

| Step | Model / mechanism | In → out |
|---|---|---|
| Planner | text LLM `gpt-4-0613` | subgoal + inventory + retrieved plans + feedback → JSON actions |
| Patroller prep | code, no LLM | inventory vs tool/materials → pass/fail |
| Patroller done? | text LLM `gpt-3.5-turbo-0613` | subgoal + inventory + Percipient notes → success + feedback |
| Percipient question | text LLM `gpt-4-0613` | subgoal + notes → next question (`status` 0/1/2) |
| Percipient vision | MineLLM HTTP or `gpt-4-vision-preview` | **image + question** → text |
| Memory store/search | `OpenAIEmbeddings()` + Chroma | embed `task_description`; top-2 similar workflows |
| Memory chat LLM | `gpt-3.5-turbo-0613` constructed, **never called** | — |
| Performer / actions | no LLM | dispatch; `env.step` |
| Process success | inventory count | file’s first item/qty present |
| Creative success | self-report | `success=null` |

`--gpt_model_name` overrides Planner, Patroller, Percipient-question only.

**Control flow**

```
subgoal (text) → Planner → workflow
                     │
          for each action:
            prep fail ──────────────► Planner (same subgoal, retry)
            prep ok → env.step ─────► timeout / death → END
                 │
                 find: walk, save RGB, Percipient
                   status 2 (found)  → next action (move_to / mine), no replan
                   0 / 3 (not found) → walk, new frame, Percipient again
                   explore cap       → leave find; rest of workflow still runs
                 craft/mine          → update inventory
                     │
          workflow done → Patroller “done?”
            no  → Planner (same subgoal, retry; 300 → END fail)
            yes → embed workflow in Chroma → next subgoal
```

Active perception (default): question LLM then VLM on the same frame until 2/0/3. Caption: one “describe this image”, then the question LLM. Planner never sees pixels.

Other: Planner parse fail retries 5 times then errors. `max_steps` → `timeout` (`timeout_in_find` if last action was `find`). `life <= 0` → `death`.

Batch: `tasks × seeds` → `OUT/<task>/<seed>.json` + `.events.jsonl` → `summary.csv`. `--skip-done` skips existing JSON.

## Layout and paths

```
configs/          code_default.yaml, strict.yaml, seeds_example.txt
eval/             run_batch.py, summarize.py
tools/            validate_tasks.py, test_offline.py
slurm/            CPU / GPU templates (see slurm/README.md)
MP5_agent/agent/  run_agent.py, modules, prompts/, tasks/{creative,process}/
```

Not in git: `LAMM/`, `*.sif`, overlay, `$HOME/.mp5_openai.env` (`chmod 600`, export `OPENAI_API_KEY`).

`run_episode` chdirs to `MP5_agent/agent` so prompts resolve. `OUT/<task>/<seed>/` holds `images/`, `video/` (`save_frames` only), `logs/`, `memory/`. Old `../images|video|logs` and shared `./memory/` are gone.

## Config and run

Key only from `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`). `--openai_key` is ignored.

- `code_default.yaml`: original literals (forest, break×100, 6 dirt + 6 coal, gpt-4 planner, gpt-3.5 patroller/memory). `max_steps=12000` is new; `save_frames=false`.
- `strict.yaml`: empty inventory, `break_speed=1`, random biome.

```bash
# login node
python3 tools/validate_tasks.py && python3 tools/test_offline.py

# one episode (compute node)
export OPENAI_API_KEY=...
cd MP5_agent/agent
python3 run_agent.py --config ../../../configs/code_default.yaml \
  --task tasks/process/craft_wooden_pickaxe.json --seed 10001 \
  --out_dir /scratch/$USER/mp5_runs/smoke/craft_wooden_pickaxe/10001 \
  --mllm_url http://127.0.0.1:25541/image --answer_method active --answer_model mllm

# batch (repo root)
python3 eval/run_batch.py --config configs/code_default.yaml \
  --tasks MP5_agent/agent/tasks/process --seeds configs/seeds_example.txt \
  --out /scratch/$USER/mp5_runs/run1 --workers 1 \
  --mllm-url http://127.0.0.1:25541/image --skip-done
```

Need Python ≥ 3.9, JDK 8, [MineDojo](https://docs.minedojo.org/sections/getting_started/install.html#prerequisites), `MP5_agent/requirements.txt` (legacy openai + langchain). `mllm` needs a live `mllm_url`. Slurm: Singularity + overlay + `xvfb-run -a`; fill image paths; account `<SLURM_ACCOUNT>`.

## Tasks and data

Fields as above. Names follow `update_*_name` in `utils/common_utils.py` (`wood`↔`log`, `diamond ore`→`diamond`).

| Dir | Role | Success |
|---|---|---|
| `tasks/creative/` | original scenes `1-1`…`4-4` | `success=null`, log snapshot |
| `tasks/process/` | wooden/stone/iron pickaxe, iron ore/ingot, diamond | inventory of file’s first item |

Official MineLLM data (not in this git): [MP5_dataset](https://huggingface.co/datasets/IranQin/MP5_dataset), [LAMM](https://github.com/IranQin/LAMM) → `datasets/2D_Instruct`, MineCLIP in `model_zoo/mineclip_ckpt`, weights [MineLLM](https://huggingface.co/IranQin/MineLLM) → `LAMM/ckpt`.

## Known leftovers

Patroller except-retry still passes `messages=` into the wrong signature (original). `openai.api_base` is process-wide. GPU sbatch MineLLM flags are placeholders.

```
@inproceedings{qin2024mp5,
  title={MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception},
  author={Qin, Yiran and Zhou, Enshen and Liu, Qichang and Yin, Zhenfei and Sheng, Lu and Zhang, Ruimao and Qiao, Yu and Shao, Jing},
  booktitle={CVPR}, pages={16307--16316}, year={2024}
}
```
