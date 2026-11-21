# MineLLM service (Percipient)

MineLLM is MP5's perception model: a Minecraft-tuned LLaVA-style model (Vicuna-13B-v1.5 + LoRA delta + MineCLIP image encoder) that answers questions about one game frame. We run the authors' server unchanged except for one import patch. We do not train it.

- Code: `LAMM/` is a git-ignored clone of [IranQin/LAMM](https://github.com/IranQin/LAMM) at `52303e6`, with `patches/lamm_openlamm_optional_lightllm.patch` applied.
- Weights: pinned with sha256 in `models.lock.json`. Real files are in `/scratch/wz2999/models/`, symlinked into `LAMM/ckpt/` and `LAMM/model_zoo/`.
- Python env: `/scratch/wz2999/env/minellm_env` (Python 3.10, torch 1.13.1+cu117, transformers 4.31.0; `requirements/minellm_env.freeze.txt`). Run it on the host, not inside the MineDojo container.
- Server: `LAMM/src/mllm_api_mc_v1.5.py`. Flask, single process.

## Endpoint

The server binds `0.0.0.0:25541`. The port is hard-coded, and the script takes no flags (`--port` and `--ckpt` do not exist). Run the agent in the same job and use:

```
http://127.0.0.1:25541/image
```

### Request

`POST /image`, `multipart/form-data`:

| Field | Value |
|---|---|
| `file` | The image (JPEG or PNG). It must have a non-empty filename. |
| `text` | The question, in English. |
| `is_del` | `1` deletes the uploaded copy after answering (use this). `0` keeps it in `LAMM/images/`. The value goes through `int()`, so `true` or `yes` crash the request with HTTP 500. |

### Response

`200 {"result": 1, "answer": "<text>"}`. Errors are `400 {"error": ...}` for a missing file, and HTML 500 for a bad `is_del` or a model crash. `GET /` returns 404 and `GET /image` returns 405.

```bash
curl -s -F file=@pigs_rain_day.jpg -F "text=Is there a pig in the image?" -F is_del=1 \
     http://127.0.0.1:25541/image
# {"answer":"The image shows a pig.","result":1}
# Job 35181 (A100-40GB) got this answer in 0.83 s with the same request sent from Python requests.
```

The framework client is `MineLLM` in `MP5_agent/agent/utils/percipient_mllm.py`. It sends `is_del=1` and returns `answer.strip()`. Pass the URL with `run_agent.py --mllm_url http://127.0.0.1:25541/image --answer_model mllm --answer_method active`. In active mode, an OpenRouter chat model (`percipient_question`) writes each question and MineLLM answers it.

## Starting and stopping

From an sbatch job on a GPU node:

```bash
source "$MP5_ROOT/slurm/minellm_server.sh"   # defaults: wz2999 env, logs in /scratch/wz2999/logs
trap minellm_stop EXIT
minellm_start      || exit 1
minellm_wait_ready || exit 1                 # sets MINELLM_URL, MINELLM_READY_SECONDS
# ... run the agent against "$MINELLM_URL" ...
```

To start it by hand:

```bash
export PATH=/scratch/wz2999/env/minellm_env/bin:$PATH
cd ~/MP5_plus/LAMM/src && python mllm_api_mc_v1.5.py
```

**Readiness.** The server is ready when a TCP connection to `127.0.0.1:25541` succeeds. Do not poll with `curl -sf` on `/` or `/image`: they return 404 and 405, so `-f` never succeeds. `minellm_wait_ready` uses a Python socket check, gives up after 30 minutes, and fails right away if the server process exits.

Look for `[!] Loading delta checkpoint: ../ckpt/.../pytorch_model.pt` in the log. If it prints `Loading vicuna checkpoint ... while ... not found!` instead, a symlink is broken and you are running plain Vicuna.

## Resources and timing (job 35181, A100-SXM4-40GB)

| | |
|---|---|
| Load time, launch to port open | **343 s**. Vicuna shards are read over NFS. |
| GPU memory | ~25.9 GB when ready, ~26.9 GB after queries (fp16). Needs a ≥32 GB card. |
| CPU RAM during load | Estimated ~52 GB peak (not measured): Vicuna loads in fp32 on the CPU, then moves to the GPU as fp16. Job 35181 used `--mem=80G` and loaded fine. |
| Answer latency | Mostly 0.8–2 s per question. The first request took ~9 s (warm-up). |
| Concurrency | One request at a time. Flask's dev server, one GPU, no batching. Use one agent per server. |

**Cluster rule.** A job whose GPU or CPU use stays low for about 20 minutes is cancelled automatically. Do not start the server in an interactive session and leave it waiting. Start it in the same sbatch job as the agent, wait for the port, run the agent immediately, and stop the server at the end. A separate "server job" that sits idle between agent runs will be killed. Also, the OOD VS Code session runs inside a Singularity container without `--nv`, so it cannot see the GPU at all.

## Answer quality (smoke test, job 35181)

The server samples with `temperature=1.0`, `top_p=0.9` (hard-coded in the script), so repeated questions can get different answers. The same image and the question "Is it daytime or nighttime?" asked 3 times gave "I cannot distinguish..." twice and "It is daytime" once.

| Image (ground truth) | Question | Answer |
|---|---|---|
| 3 pigs, oak trunk, rain, day | Is there a pig in the image? | The image shows a pig. ✅ |
| same | Could you describe this Minecraft image? | ...a grass, leaves, wood, grass block blocks. (misses the pigs) |
| same | Is it daytime or nighttime? | I cannot distinguish whether it's daytime or nighttime. |
| birch forest, day, no animals | Could you describe this Minecraft image? | ...a curious, chicken, pig, Minecraft. ❌ animals that are not there |
| same | Is there a tree in the image? | ...planting, and the grass block, grass, leaves, dirt, stone, wood. (no yes/no answer) |
| plains at night, no trees | Is it daytime or nighttime? | It is nighttime in the image. ✅ |
| same | Is there a tree in the image? | There is a tree in the image. ❌ |
| same | Could you describe this Minecraft image? | ...a birch forest biome. ❌ |

Takeaways for the agent:
- Treat answers as noisy. Yes/no answers about object presence can be wrong in both directions.
- Free-form descriptions can name animals or biomes that are not in the frame.
- Day/night answers are often "cannot distinguish".
- The model was trained on MineDojo frames. Our test images are 820×512 dataset frames and one author sample, not live agent frames, so live accuracy still needs checking.

Full results: `/scratch/wz2999/minellm_test/results_35181.md`. Test images and ground truth: `/scratch/wz2999/minellm_test/`.

## Testing without a GPU

`tools/fake_minellm_server.py` has the same interface: same fields, same status codes, a fixed answer. It uses only the standard library and logs each request as a JSON line. Use it to rehearse the agent pipeline on a CPU node (`slurm/integration_rehearsal_cpu.sbatch`).
