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
| **A: perception** (`lfnav/perception`) | Image → per-object presence, direction (left/center/right), distance (near ≤3 / mid ≤10 / far), plus biome, time, weather and brightness. Backends: GT (rays), MineLLM with fixed templates, learned (frozen DINOv2 plus heads). | Percipient (MineLLM questioned by GPT) |
| **B: matcher** (`lfnav/matcher.py`) | Rule-based. Is the target found, are the conditions met, should the agent stop? Stops when the nearest target ray is within 3 blocks, the same test SR-b uses. | The "is it found" part of the percipient prompt |
| **C: policy** (`lfnav/policy`) | Chooses the next action chunk from the scene, the match result and memory. | The scripted explore branch of the find action |
| Controller (`lfnav/control`) | Executes actions. Jumps 1-block steps, avoids hazards, wades shallow water, detects stuck (displacement-based) and escapes. | `structured_actions.py` movement |

Policies, in build order:

1. **C0, straight walk**: walk forward and turn when blocked. Lower bound, close to MP5's blind exploration.
2. **C1, rules**: opening 360° scan, steer toward a visible target, approach at mid range in steps of at most 2, re-acquire a lost target, otherwise explore the least-visited of 8 sectors.
3. **Expert = C1 + 360° wide lidar**: used only to generate demonstrations (Section 3).
4. **C2, behaviour cloning**: an MLP trained on expert demos, predicting K = 4 action steps per decision. The first version reads the GT scene state (C2+GT); the next reads learned perception (C2+DINOv2).
5. **C3, diffusion policy** (reserved): samples short action chunks or waypoints. The interface exists in `lfnav/policy/c3.py`, and the controller already accepts `(dyaw_deg, forward_blocks)` waypoints.

Order of work: GT perception + B + C0/C1 baselines → expert demos → C2+GT → learned A (DINOv2) → C2+DINOv2 → C3 → transfer.

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
| Perception (planned) | training seeds | Pilot of 1,000 frames, then 30k–50k frames with patch labels |

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

**Expert results (dynamic demos)**: *to be filled from `lfnav/data/demo_stats.py`*

| Subset | Episodes | Success rate | SPL | Median steps | `used_wide` share |
|---|---|---|---|---|---|
| Animal, easy | | | | | |
| Animal, medium | | | | | |
| Animal, hard | | | | | |
| Animal + `near X` | | | | | |
| Tree / grass, easy | | | | | |

---

## 4. Results

### 4.1 Baseline: C0 vs C1, GT perception, static setting (460/480 episodes)
Full report: `result/lfnav_basline/report.md`.

| Animal targets | C0 | C1 |
|---|---|---|
| SR-b [95% CI] | 0.17 [0.12, 0.25] | **0.81** [0.73, 0.87] |
| SR-b easy / medium / hard | 0.30 / 0.15 / 0.07 | **0.85 / 0.80 / 0.78** |
| SPL | 0.06 | **0.63** |
| SR@500 | 0.09 | **0.72** |
| Median steps (successes) | 372 | **89** |
| Paired McNemar (115 pairs) | 2 episodes where only C0 succeeds | 74 episodes where only C1 succeeds; p = 7.7e-20 |

- C1 stays almost flat across difficulty; C0 collapses beyond 8 blocks.
- C1's remaining failures are mostly conditions (`near X`); exploration failures are rare (1 `never_seen`).
- 4-3 is nearly unsolvable: no water is placed near the cow in the desert yet.
- Tree and grass targets need no navigation, so C0 does well there (block, easy: 0.85 for C0 vs 0.91 for C1).

### 4.2 Dynamic setting (main), to be run
| Policy | SR-b (animal) | easy / medium / hard | SPL | SR@500 |
|---|---|---|---|---|
| C0 + GT | | | | |
| C1 + GT | | | | |
| Expert (upper bound) | | | | |
| C2 + GT | | | | |
| C2 + DINOv2 | | | | |
| C3 | | | | |

### 4.3 Robustness (planned)
- Static vs dynamic.
- Tall grass kept (evaluation only).
- C2 trained on static vs mixed data.
- C3 chunk-length ablation (H = 4, 8, 16, 32) with moving targets.

---

## 5. Environment

Runs on HPC. Full guide: `Minedojo environment/` and the setup PDF.

| Part | Where | Used for |
|---|---|---|
| MineDojo image | `$SCR/env/minedojo.sif` (Singularity; Minecraft 1.11.2 via Malmo) | Every episode |
| Runtime | `$SCR/mp5_env.sh`. Must be sourced in every terminal: it sets `SCR`, `MP5`, `PYTHONPATH`, `GRADLE_USER_HOME` and the Slurm account. | Everything |
| Gradle cache | `$SCR/.gradle` plus a local MixinGradle repo. Each job copies it to node-local `/tmp` (`lfnav/env/gradle_home.sh`). | Minecraft build at launch |
| Training env | PyTorch 2.x (`~/lfnav_train_env.sh`) | C2 and perception training |
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