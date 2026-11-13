# lfnav: LLM-free conditional navigation for MP5

lfnav replaces the LLM components of MP5 with small trainable modules, on the 16 MP5 creative "find X with conditions" tasks. No language model runs on the control path. The main metric is **SR-b**: at the moment the agent stops, the target is visible, within 3 blocks, and every condition holds. MP5's own success rule (SR-a) is reported as a secondary metric.

Final question: can a navigation policy learned here transfer to robot simulation (AI2-THOR first)?

Detailed notes are in `lfnav/docs/`: `C0C1_BASELINE.md` (baseline fixes), `DEMOS.md` (expert data), `REPORT.md` (calibration). The shared evaluation protocol with the MP5 group is in `benchmark/creative16/`.

---

## 1. Pipeline

```
rgb frame ──► A: perceive ──► SceneState ──► B: match ──► MatchResult ──► C: act ──► action chunk ──► controller ──► MineDojo
                                              (task spec)                (memory)                                    │
                                                                                                    stop ──► judge (SR-b / SR-a)
```

| Module | What it does | Replaces in MP5 |
|---|---|---|
| **A: perception** (`lfnav/perception`) | Image → per-object presence, direction (left/center/right), distance (near ≤3 / mid ≤10 / far), plus biome, time, weather and brightness. Evaluation default is the A head of the joint model (frozen DINOv2). GT rays and MineLLM remain available as backends. | Percipient (MineLLM questioned by GPT) |
| **B: matcher** (`lfnav/matcher.py`) | Rule-based. Is the target found, are the conditions met, should the agent stop? Stops when the nearest target ray is within 3 blocks, the same test SR-b uses. | The "is it found" part of the percipient prompt |
| **C: policy** (`lfnav/policy`) | Chooses the next action chunk from the scene, the match result and memory. | The scripted explore branch of the find action |
| Controller (`lfnav/control`) | Executes actions. Jumps 1-block steps, avoids hazards, wades shallow water, detects stuck (displacement-based) and escapes. | `structured_actions.py` movement |

Policies, in build order:

1. **C0, straight walk**: walk forward and turn when blocked. Lower bound, close to MP5's blind exploration.
2. **C1, rules**: opening 360° scan, steer toward a visible target, approach at mid range in steps of at most 2, re-acquire a lost target, otherwise explore the least-visited of 8 sectors.
3. **Expert = C1 + 360° wide lidar**: used only to generate demonstrations (Section 3).
4. **Joint A+C2**: frozen DINOv2 features, one training run. The A head predicts `SceneState`; the C2 head predicts K = 4 actions from pooled features, A's output, task conditions, 8-sector counts, the last 4 actions, and `last_seen`. `run.py --policy joint` runs A then B then C2.
5. **C3, diffusion policy** (reserved): samples short action chunks or waypoints. The interface exists in `lfnav/policy/c3.py`, and the controller already accepts `(dyaw_deg, forward_blocks)` waypoints.

Order of work: GT perception + B + C0/C1 baselines → expert demos → extract DINOv2 features → joint A+C2 → C3 → transfer.

---

## 2. Data

### Tasks
16 MP5 creative tasks, rewritten as structured specs in `lfnav/tasks/specs/*.json` (original text in `MP5_agent/agent/tasks/creative/`). Each spec has a target, conditions (biome, time, weather, brightness, `near X`) and a scene (biome, start time, weather, what to spawn).

| Group | Tasks | Role |
|---|---|---|
| Animal target | 1-3, 1-4, 2-3, 2-4, 3-3, 3-4 | Main navigation tasks |
| Animal target + `near X` | 4-3 (cow near water), 4-4 (pig near grass) | Navigation plus a condition that requires search |
| Tree / grass target | 1-1, 1-2, 2-1, 2-2, 3-1, 3-2, 4-1, 4-2 | Target is everywhere; reported at easy only |

### Difficulty (`configs/lfnav/difficulty.yaml`)
| | Animal spawn distance | Distractor animals | Block target min. distance |
|---|---|---|---|
| easy | 4–8, any direction | 0 | — |
| medium | 8–16, any direction | 2 | 8 (usually `difficulty_unmet`) |
| hard | 16–32, behind the agent ±45° | 4 | 16 (usually `difficulty_unmet`) |

### Scenes
- Real Minecraft terrain from the world seed. `specified_biome` gives a single-biome world, checked with `check_biome_world`.
- lfnav summons animals itself after `reset()`, because MineDojo's `initial_mobs` reset hook does not actually spawn them. Commands go through `_send_chat` (whitelist bypass); `Invalid cmd` raises. If `location_stats.can_see_sky` is false, the world seed is remade (`seed*1000+k`, up to `max_reroll`). Exhausted remakes are `spawn_underground` and the episode is not run.
- **Sole setting (`configs/lfnav/main.yaml`)**: animals move; natural animals stay; vegetation is not cleared; `/difficulty peaceful` is the first spawn command; resistance is NBT on `/summon`. The only intended difference from stock MP5 is peaceful difficulty. Episode field `peaceful_sent` (old name `peaceful_ok`) means that command was sent, not that it worked. Whether hostiles were actually gone is `hostile_seen == 0`.
- Initial distance (and SPL) is the nearest same-species ray at the start.

### Amounts
| Set | Seeds | Size |
|---|---|---|
| Evaluation | `eval_seeds_baseline.txt`: 9, 19, 29, 39, 49 (all end in 9) | 16 × 3 × 5 = 240 episodes per policy |
| Expert demos | `train_seeds_demo.txt`: 44 seeds in 1–48, none ending in 9 | 8 animal tasks × 3 difficulties plus 8 block tasks at easy = 32 cells × 44 ≈ 1,400 episodes. Only successful episodes are used for training. |
| Joint features | same successful demos as training | Frozen DINOv2, one `.npy` per filtered episode under `$SCR/lfnav/features/<version>/` |

### Compared with MP5
| | lfnav | MP5 |
|---|---|---|
| Task text | Same 16 tasks | Same |
| Scene setup | Controlled spawn distance and direction by difficulty; reproducible by seed; **peaceful** | Random; relies on natural spawns; default difficulty |
| Success | SR-b (automatic, from GT rays) and SR-a | SR-a |
| `near X` | X and the target appear in the same frame | Decided by the LLM from the image |
| Episodes | Fixed seed lists; C0, C1 and C2 run on the same seeds | Not specified |

---

## 3. GT generation: the expert

The expert is C1 with ground-truth perception plus a **360° wide lidar** (horizontal ring, 5° steps, pitches 0°, −3° and −8°, 32-block range). Wide lidar is used only when aligned rays do **not** see the target (`match.satisfied.target` is false). If the target is in view but a `near` condition fails, C1 handles it.

- Each decision records the GT scene, the match result, memory (8-sector visits, recent actions), the expert's action chunk, the 224×364 frame, and `used_wide`.
- Demos are collected with `configs/lfnav/main.yaml` into `$SCR/lfnav/data/demos_main`. Failures store `record.json` and `steps.jsonl` only (no frames). Old `demos_dynamic` files are kept and may be used as C2 legacy data.
- The expert is never used as an evaluation policy, except in the explicit upper-bound run (`expert_eval`, requires `allow_privileged_eval`).

**Expert results (demos, `main.yaml`, 44 training seeds per cell, 4-3 excluded)**

| Subset | Episodes | Success rate | SPL | Median steps | `used_wide` share |
|---|---|---|---|---|---|
| Animal, easy | 264 | 0.95 | 0.73 | 40 (20–72) | 0.03 |
| Animal, medium | 264 | 0.90 | 0.82 | 100 (53–171) | 0.02 |
| Animal, hard | 264 | 0.74 | 0.75 | 263 (125–446) | 0.01 |
| Animal + `near X` (4-4, all difficulties) | 132 | 0.78 | 0.73 | 118 (56–589) | 0.03 |
| Tree / grass, easy | 352 | 0.97 | 0.60 | 0 (0–10) | 0.01 |

- Animal = 1-3, 1-4, 2-3, 2-4, 3-3, 3-4 (no `near X`). Tree / grass = 1-1, 1-2, 2-1, 2-2, 3-1, 3-2, 4-1, 4-2.
- Success rate is pooled over episodes. SPL and `used_wide` share are means of the per-task values. Median steps is the median of the per-task medians, with the range of per-task medians in parentheses.
- Training seeds, not the evaluation seeds. Per-cell numbers: `stats_demos_v2.md`.

---

## 4. Results

### 4.1 Main setting: C1 + GT (60 episodes, 2026-11-12)
Setting: `configs/lfnav/main.yaml` (targets move, tall grass kept, natural mobs kept, peaceful). Evaluation seeds 9/19/29/39/49, max 3000 steps. Full report: `result/lfnav_basline/report_c1_main.md`.

| Animal targets | C1 + GT |
|---|---|
| SR-b [95% CI] | **0.80** [0.68, 0.88] (48/60) |
| SR-b easy / medium / hard | **1.00 / 0.93 / 0.69** (10/10, 14/15, 24/35) |
| Median steps (successes) | 97 (easy 29, medium 48, hard 208) |
| SPL | not available (initial distance not logged, see report) |

- All 12 failures are timeouts after the target had been seen (`target_never_seen = 0`). C1's weakness is reaching or stopping at a moving target, not exploration.
- Hard is the only difficulty with room left. 2-4 is the outlier (hard 1/5).
- Covered: all 7 animal tasks at hard, plus representative easy/medium cells. Block tasks and the remaining animal easy/medium cells are not yet run.

### 4.2 Earlier static setting: C0 vs C1 (superseded)
Frozen targets, natural mobs removed, tall grass cleared, not peaceful. Not comparable with 4.1. Report: `result/lfnav_basline/report.md`.

| Animal targets | C0 | C1 |
|---|---|---|
| SR-b [95% CI] | 0.17 [0.12, 0.25] | 0.81 [0.73, 0.87] |
| SR-b easy / medium / hard | 0.30 / 0.15 / 0.07 | 0.85 / 0.80 / 0.78 |
| SPL | 0.06 | 0.63 |
| Median steps (successes) | 372 | 89 |

### 4.3 Main setting, to be filled
| Policy | SR-b (animal) | easy / medium / hard | SPL |
|---|---|---|---|
| C1 + GT | 0.80 | 1.00 / 0.93 / 0.69 | — |
| C2 (joint) | | | |
| C3 | | | |

---

## 5. Environment

Runs on HPC. Full guide: `Minedojo environment/` and the setup PDF.

| Part | Where | Used for |
|---|---|---|
| MineDojo image | `$SCR/env/minedojo.sif` (Singularity; Minecraft 1.11.2 via Malmo) | Every episode |
| Runtime | `$SCR/mp5_env.sh`. Must be sourced in every terminal: it sets `SCR`, `MP5`, `PYTHONPATH`, `GRADLE_USER_HOME` and the Slurm account. | Everything |
| Gradle cache | `$SCR/.gradle` plus a local MixinGradle repo. Each job copies it to node-local `/tmp` (`lfnav/env/gradle_home.sh`). | Minecraft build at launch |
| Training env | PyTorch 2.x (`~/lfnav_train_env.sh`) | Feature extract and joint A+C2 training |
| MineLLM (optional) | Separate conda env plus models on A100 | MineLLM perception baseline only |

Rules:
- Host-side scripts use only the Python standard library. Anything that needs numpy runs inside the image (`lfnav_sif_exec`).
- Code changes are made on the dev machine and pushed. the cluster only pulls and runs.
- Large outputs (runs, demos, features, checkpoints) stay in `/scratch/$USER/lfnav/`. Only small summaries go into the repo.

Quick checks (no MineDojo needed):
```bash
python3 -m unittest discover -s lfnav/tests
```
Job order and commands: `lfnav/slurm/README.md`.