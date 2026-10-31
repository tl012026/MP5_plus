# MP5_plus

A fork of [MP5](https://arxiv.org/abs/2312.07472v2) (CVPR 2024, [project page](https://iranqin.github.io/MP5.github.io/)), a Minecraft agent built on [MineDojo](https://github.com/MineDojo/MineDojo).

This fork has two goals:

1. **Baseline.** Run the original MP5 decision loop unchanged, but with YAML configs, per-episode outputs, batch evaluation and Slurm templates, so results are reproducible.
2. **(?)Improvement.** Replace modules one at a time and measure the effect on a fixed task set. The planned direction is an **LLM-free** agent built from small trainable models (see [Planned direction](#planned-direction-llm-free-agent)).

The decision loop and everything in `MP5_agent/agent/prompts/` are unchanged from the original release.

---

## Pipeline (baseline, `configs/code_default.yaml`)

### 1. Inputs

- **Task file** (`tasks/creative/*.json` or `tasks/process/*.json`): a list of **pre-decomposed subgoals**. Each subgoal has the fields `task`, `quantity`, `material`, `tool`, `platform`, `tips`, `description`.
  - The original paper's Parser (automatic decomposition) and Knowledge Memory are **not** in the released code. Decomposition and `tips` are written by hand.
  - Order: the file lists the **final goal first**. `run_agent.py` executes `task_list[::-1]`, so the first subgoal executed is the last one in the file.
  - Example (wooden pickaxe): file order is pickaxe → crafting table → stick → planks → log; execution order is the reverse.
- **Observations from MineDojo**, used as follows:
  - Inventory: given to the Planner and the Patroller.
  - RGB frame: used **only inside the `find` action** (sent to the perception model). The Planner never sees pixels.
  - Voxels (11×11×11 around the agent) and LiDAR (24×24 rays, range 10): used **only by the scripted movement and mining code**. Never sent to any language model.
  - Position: used by the scripted code, and to tell the Planner "ground" or "underground".

### 2. Episode setup

- MineDojo `harvest` task (target `diamond`), image size 512×820, biome fixed to `forest`.
- Block breaking speed ×100.
- Starting inventory: 6 dirt and 6 coal.
- These values come from the original `run_agent.py`. `configs/strict.yaml` changes them to: empty inventory, breaking speed ×1, random biome.
- New in this fork: a step limit (`max_steps`, default 12000).

### 3. Per-subgoal loop

For each subgoal, repeat until it succeeds or the retry limit (300) is reached:

- **Step 3.1 – Plan (LLM).**
  - Model: text LLM (default `gpt-4-0613`).
  - Input: system prompt (`structured_action_system.txt`), the subgoal, "ground"/"underground", current inventory, up to 2 similar past plans retrieved from Memory, and the failure feedback from the previous attempt (if any).
  - Output: JSON `workflow`, a list of steps; each step has a repeat count `times` and a list of `actions` (`name` + `args`).
  - If the JSON cannot be parsed, it retries up to 5 times, then raises an error.
- **Step 3.2 – Execute each action.**
  - **Pre-check (code, no LLM):** the Patroller compares the inventory with the action's required tool/materials. If it fails, the feedback goes back to Step 3.1.
  - **Dispatch (code, no LLM):** the Performer maps each action to a scripted function that calls `env.step`:
    - `find(obj)`: explore until the object is seen (details in Step 3.3).
    - `move_to(obj)`: walk to the object using voxels/LiDAR.
    - `mine(obj, tool)`: break blocks; updates inventory.
    - `craft(obj, materials, platform)`: craft using inventory, crafting table or furnace; updates inventory.
    - `dig_down(y_level, tool)`: dig down to a target height.
    - `fight`, `equip`, `apply`: **pre-check only, no game action** (not implemented in the release).
    - `dig_up`: only sets the "underground" flag to false; no game action.
    - `place`: not implemented.
- **Step 3.3 – `find` and active perception.**
  - Above ground: loop {save the current RGB frame → ask the perception module "is the target (with all its conditions) visible?" → if not, move one step (forward; jump over obstacles; turn if blocked)}.
  - Perception runs **on every step** of this loop, so `find` is where most model calls happen.
  - Active perception (default `answer_method: active`), on one frame:
    - A text LLM (default `gpt-4-0613`) reads the subgoal and the answers collected so far, and either asks the next question or returns a status: `2` = target found, `0`/`3` = not found, `1` = keep asking.
    - A vision model answers each question about the frame: **MineLLM** (HTTP service, see [External assets](#external-assets)) or `gpt-4-vision-preview`.
    - Answers are stored as "environment notes" for the Patroller.
  - Caption mode (`answer_method: caption`): the vision model describes the whole frame once, then the text LLM decides.
  - Status `2` ends `find`, and the next action runs (no replanning).
  - Underground: `find` uses voxel detection only (no image, no perception model).
  - If the exploration cap is reached, `find` exits **without an error** and the rest of the workflow still runs (usually failing at the next `move_to`/`mine`).
- **Step 3.4 – Completion check (LLM).**
  - Model: text LLM (default `gpt-3.5-turbo-0613`).
  - Input: subgoal, inventory, environment notes from perception.
  - Output: JSON with `success`, `feedback`, `suggestion`.
  - Not done → back to Step 3.1 with the feedback (counts as one retry).
- **Step 3.5 – Memory.**
  - On success, the workflow is stored in a Chroma vector store, keyed by the embedding of the subgoal description (`OpenAIEmbeddings`).
  - Step 3.1 retrieves the top-2 most similar stored workflows.
  - In this fork, Memory is **per episode** (not shared across episodes), so it only contains subgoals completed earlier in the same episode.
  - A chat LLM (`gpt-3.5-turbo-0613`) is also created inside Memory but is never called.

### 4. Termination and scoring

- An episode ends when: all subgoals succeed; a subgoal hits 300 retries (`fail`); `max_steps` is exceeded (`timeout`, or `timeout_in_find` if the last action was `find`); health reaches 0 (`death`); or an exception occurs (`error`).
- **Process tasks:** success = the final item (first entry of the task file) is in the inventory with the required quantity. Checked by code, not by an LLM.
- **Creative tasks:** the agent's self-report is logged together with ground-truth signals (LiDAR hits, biome, time, weather) at that moment; `success` is `null` until an automatic judge is written.

### 5. Models used in the baseline

| Role | Default model | Can be changed in |
|---|---|---|
| Planner | `gpt-4-0613` | config `models.planner` |
| Perception question generator | `gpt-4-0613` | config `models.percipient_question` |
| Perception answerer (vision) | MineLLM (Vicuna-13B + MineCLIP) or `gpt-4-vision-preview` | config `percipient.*` |
| Completion check | `gpt-3.5-turbo-0613` | config `models.patroller` |
| Memory embedding | `OpenAIEmbeddings()` default | `agent/llm_factory.py` |

- `--gpt_model_name` overrides the Planner, Patroller and question-generator models only.
- Several of these OpenAI model names may no longer be available. Record the actual models used for every run.

### 6. Known behaviors that affect results

- **MineLLM is stochastic:** sampling is on (`temperature=1.0`, `top_p=0.9`), so the same question on the same image can get different answers.
- **Most API cost comes from `find`:** one perception round (2–3 LLM calls + 1 vision call) per step.
- **Exploration is a random walk** with obstacle handling; it has no map and no memory of visited places.
- **Patroller bug (original code):** its retry path on an exception calls the LLM with the wrong arguments, so a second exception ends the episode as `error`.

---

## (?)Planned direction: LLM-free agent

**Status: planned, not implemented.**

Goal: replace every LLM with a small trainable model or with code, on a narrower task set, and compare with the baseline on the same tasks and seeds. Expected benefits: no API cost, much lower latency, and no 13B model at test time.

### Task scope

- **Track A – conditional object navigation (first).** The 16 creative tasks, e.g. "find a pig near grass on the plains during the daytime". Only perception and navigation are needed; this is ObjectNav with semantic conditions.
- **Track B – short tech chain (second).** Log → wooden pickaxe → stone pickaxe, possibly extended to iron ore. Adds action planning and mining; searching for ore is the main bottleneck in the original paper.

### Module replacements

- **Parser / Planner →** code or a small classifier.
  - Option 1: symbolic planning on the recipe graph (the tech tree is deterministic).
  - Option 2: a small classifier that predicts the next action primitive from (subgoal, inventory, "target visible?"), trained by imitating the symbolic planner.
- **Perception (question generator + MineLLM) →** a multi-label classifier.
  - Frozen MineCLIP image encoder plus small heads.
  - Outputs: target visible, target direction, biome, day/night, weather, water nearby, etc.
  - No question generation is needed: all attributes are predicted every time.
- **Completion check →** code (inventory) plus the perception classifier for visual conditions.
- **`find` →** a goal-conditioned navigation policy (behavior cloning or diffusion policy), input: recent frames + goal embedding + spatial memory, output: movement action or short waypoint sequence.
- **Memory →** spatial memory instead of plan retrieval: explored area and locations where resources were seen.
- **Unchanged:** scripted `move_to`, `mine`, `craft`, `dig_down`.

### Training data

- **Perception labels:** generated automatically from MineDojo ground truth (LiDAR, voxels, location stats), the same way the MP5 dataset was built (rotate in place, one frame every 30°). The released MP5 dataset can also be reused.
- **Navigation demonstrations:** a privileged expert (known target position + A* on the voxel map) records (frames, actions). Targets are spawned near the agent to make collection efficient. DAgger-style rounds reduce distribution shift.
- Reinforcement learning is out of scope: the simulator is too slow for the compute budget.

### Evaluation

- Same harness, same seeds and step limit as the baseline.
- Metrics: success rate (with 95% CI), steps to find the target, path efficiency, wall-clock time, cost.
- Comparisons: baseline MP5 vs. LLM-free agent; ablations that replace only perception, only navigation, and both.

---

## Repository layout

```
configs/          code_default.yaml, strict.yaml, seeds_example.txt
eval/             run_batch.py (batch runner), summarize.py
tools/            validate_tasks.py, test_offline.py
slurm/            CPU / GPU job templates (see slurm/README.md)
MP5_agent/agent/  run_agent.py, planner.py, patroller.py, percipient.py,
                  performer.py, memory.py, structured_actions.py,
                  llm_factory.py, prompts/, tasks/{creative,process}/
```

- Do not edit `MP5_agent/agent/structured_actions copy.py` (old copy kept for reference).
- Not in git: `LAMM/`, model weights, `*.sif`, overlay files, API key files.

## Configuration

- `configs/code_default.yaml`: identical to the original hard-coded values (forest, breaking ×100, 6 dirt + 6 coal, GPT-4 planner, GPT-3.5 patroller), plus `max_steps=12000` and `save_frames=false`.
- `configs/strict.yaml`: empty inventory, breaking ×1, random biome.
- The API key is read only from `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`). `--openai_key` is ignored. Keep the key in a file with `chmod 600`; never commit it.

## Running

```bash
# Offline checks (no MineDojo, no API; works on a login node)
python3 tools/validate_tasks.py && python3 tools/test_offline.py

# One episode (compute node, inside the MineDojo environment)
export OPENAI_API_KEY=...
cd MP5_agent/agent
python3 run_agent.py --config ../../configs/code_default.yaml \
  --task tasks/process/craft_wooden_pickaxe.json --seed 10001 \
  --out_dir /scratch/$USER/mp5_runs/smoke/craft_wooden_pickaxe/10001 \
  --mllm_url http://127.0.0.1:25541/image --answer_method active --answer_model mllm

# Batch (from the repo root)
python3 eval/run_batch.py --config configs/code_default.yaml \
  --tasks MP5_agent/agent/tasks/process --seeds configs/seeds_example.txt \
  --out /scratch/$USER/mp5_runs/run1 --workers 1 \
  --mllm-url http://127.0.0.1:25541/image --skip-done
```

- `run_episode` changes directory to `MP5_agent/agent` so prompt paths resolve.
- Use `--answer_model gpt-vision` to run without a MineLLM server.

## Outputs

- Per episode: `OUT/<task>/<seed>.json` (result) and `OUT/<task>/<seed>.events.jsonl` (all model calls, actions, subgoals, timings).
- Per-episode working files in `OUT/<task>/<seed>/`: `images/`, `video/` (only if `save_frames`), `logs/`, `memory/`.
- Per run: `OUT/run_meta.json` (config, git commit, arguments) and `OUT/summary.csv` (one row per episode, with `failure_category`).
- Results are written atomically; `--skip-done` skips episodes that already have a result, so interrupted or requeued jobs resume.

## Tasks

| Directory | Content | Success |
|---|---|---|
| `tasks/creative/` | the paper's 16 context-dependent tasks (`1-1` … `4-4`) | `null` (self-report + ground-truth snapshot logged) |
| `tasks/process/` | wooden / stone / iron pickaxe, iron ore, iron ingot, diamond | final item in inventory |

- Item names follow the `update_*_name` helpers in `utils/common_utils.py` (e.g. `wood` ↔ `log`, `diamond ore` → `diamond`).

## External assets

- **LAMM** (MineLLM code): <https://github.com/IranQin/LAMM>, cloned into `LAMM/`.
- **Weights** (pinned revisions recorded in `models.lock.json`):
  - MineLLM delta weights (alignment MLP + LoRA, 0.17 GB): [IranQin/MineLLM](https://huggingface.co/IranQin/MineLLM). Baseline checkpoint: `Mine_52k_Vicuna_13b_v1.5_epoch_3_mlp2x_gelu/pytorch_model.pt`. Other folders are variants (CLIP encoder, caption-trained, LLaMA-2 bases).
  - Base LLM: [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) (26 GB).
  - MineCLIP image encoder: `mineclip_image_encoder_vit-B_196tokens.pth` (link in the original MP5 README).
- **Paths expected by the MineLLM server** (relative to `LAMM/src`):
  - `../ckpt/<checkpoint folder>/pytorch_model.pt`
  - `../model_zoo/vicuna_ckpt/13b_v1.5/`
  - `../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth`
  - `../images/` (temporary upload folder, must exist)
- **MineLLM server:** run `python mllm_api_mc_v1.5.py` from `LAMM/src`. It listens on port 25541; `POST /image` with form fields `file`, `text`, `is_del`; returns `{"answer": ...}`. fp16, about 27 GB GPU memory (use an A100 40 GB). `src/scripts/mllm_api_slurm.sh` in LAMM is incomplete; do not use it.
- **Dataset** (only for retraining or offline perception tests): [IranQin/MP5_dataset](https://huggingface.co/datasets/IranQin/MP5_dataset).

## Environment

- Agent: Python ≥ 3.9, JDK 8, Xvfb, [MineDojo](https://docs.minedojo.org/sections/getting_started/install.html#prerequisites), `MP5_agent/requirements.txt` (legacy `openai` and `langchain` APIs). On the cluster: the MineDojo Docker image as a Singularity image, plus an overlay for extra packages, run with `xvfb-run -a`.
- MineLLM server: Python 3.10, PyTorch 1.13.1 (CUDA 11.7), LAMM `requirements.txt` with `transformers==4.31.0`, plus `flask` and `flask_cors`.
- Slurm account: `<SLURM_ACCOUNT>`. Fill in image and overlay paths in `slurm/*.sbatch`.

## Known issues

- Patroller retry-on-exception uses the wrong call signature (original code).
- `openai.api_base` is set process-wide, so all OpenAI-compatible models in one process share one base URL.
- MineLLM flags in the GPU Slurm template are placeholders.
- `fight`, `equip`, `apply`, `dig_up`, `place` are not implemented as game actions.

## Citation

```bibtex
@inproceedings{qin2024mp5,
  title={MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception},
  author={Qin, Yiran and Zhou, Enshen and Liu, Qichang and Yin, Zhenfei and Sheng, Lu and Zhang, Ruimao and Qiao, Yu and Shao, Jing},
  booktitle={CVPR}, pages={16307--16316}, year={2024}
}
```