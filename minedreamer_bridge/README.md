# minedreamer_bridge

MineDreamer ([Zhoues/MineDreamer](https://github.com/Zhoues/MineDreamer)) in MP5's **Performer** role. The planner, patroller, and memory are unchanged. Each workflow action becomes a MineDreamer text instruction. The Imaginator → Prompt Generator → STEVE-1 loop runs until the action's done-condition holds or its budget runs out. Every action writes one result file.

```
planner workflow action ──► instructions.py ──► text prompt + budget + done-condition
                                                      │
          every freq=25 steps: Imaginator(current frame, prompt) → goal frame
                               Prompt Generator(text, current, goal) → goal embedding
          every step:          STEVE-1(obs, embedding) → MineRL action → env.step
                                                      │
                     result file (schema.py) + check_result {success, feedback, suggestion} ──► MP5 planner
```

## Simulate before opening Minecraft

```bash
python3 -m minedreamer_bridge.simulate --out /scratch/$USER/mp5_runs/dreamer_sim/demo --save-frames
python3 -m minedreamer_bridge.simulate --action '{"name": "dig_down", "args": {"y_level": 50, "tool": null}}'
python3 -m unittest discover -s minedreamer_bridge/tests -t .
```

The simulator uses `fake_world.FakeMinecraft` (flat ground, trees, a numpy first-person picture, MineRL obs/action keys), `MockImaginator` (echoes the frame), `MockPromptGenerator` (hash), and `MockSteve1` (a pixel heuristic). A mock run checks plumbing and the file format only. It says nothing about MineDreamer's ability. `--imaginator-url http://HOST:25547/` sends the frames to a real Imaginator server instead.

## Action mapping

| MP5 action | prompt | in MineDreamer eval | done when |
|---|---|---|---|
| `mine(wood)` | chop a tree | yes | log +1 |
| `mine(dirt)` / `mine(grass)` | collect dirt / break grass | yes | item +1 |
| `mine(other)` | mine X | no | item +1 |
| `dig_down(y)` | dig down | yes | y ≤ target |
| `craft(planks)` | craft wooden planks | yes | planks +n |
| `find(X)` | find a X | no | perception says visible (checked every `freq` steps) |
| `move_to(X)` | go to the X | no | distance ≤ 3 blocks |
| other `craft` | — | — | `unsupported`: hand it to the scripted performer |
| `fight` / `equip` / `apply` / `dig_up` | — | — | `noop` (also a no-op in original MP5) |

`find` and `move_to` need someone to answer "is it visible / how far": in a real run that is MP5's percipient (or the lfnav matcher), passed as `perceive_fn` / `distance_fn`.

## Result file

`schema.py` documents and validates `mp5.minedreamer_performer.result/v1`: action, instruction, backends, outcome (`done | budget_exhausted | unsupported | noop | precheck_fail | error`), steps, imagination calls and latency, inventory before/after/delta in MP5 names, pose, done-check, and `check_result`. Each step goes to `<index>_<action>.trace.jsonl` as MineRL keys plus a MineDojo 8-vector view.

## Version split (MineDojo 1.11.2 vs MineRL 1.16.5)

STEVE-1/VPT were trained on 1.16 human gameplay, and MP5's scripted actions need MineDojo's voxels, lidar, and event-level craft. Neither one is ported. The MineDreamer track runs its world in 1.16. The translation happens at the edges:

| Piece | Where | How |
|---|---|---|
| item names | `mc116.MP5_TO_ID`, `ANY_OF`, `instructions.mp5_inventory` | `log` ↔ any `*_log`, `planks` ↔ any `*_planks`, `crafting table` ↔ `crafting_table` … |
| task setup | `mc116.task_setup(env_cfg, episode)` | biome → `preferred_spawn_biome`, start inventory → HumanSurvival `inventory`, time/weather/spawn → chat commands |
| crafting | `mc116.CommandCrafter` | recipe-checked `/clear` + `/give`, same semantics as MineDojo's instant event craft |
| Python stacks | `remote.py` | MP5 brain (py3.9, langchain 0.0.300) and the 1.16 body (MineRL + STEVE-1) run in separate processes and talk JSON over HTTP. Perception requests come back to the MP5 side. |

Not translatable (listed in `mc116.UNTRANSLATABLE`): break speed ×100 (the track uses vanilla speed), voxels/lidar, cross-version seeds, and simplified smelting fuel. The scripted baseline (1.11) and this track (1.16) therefore differ in simulator. A same-world ablation is STEVE-1 text-only vs MineDreamer, both in 1.16.

```bash
python3 -m minedreamer_bridge.simulate --demo pickaxe --remote   # whole wooden-pickaxe chain through the HTTP split
```

## Real run (not done yet)

STEVE-1 runs in **MineRL 1.0 / Minecraft 1.16.5**, not in MineDojo 1.11.2. A real run needs the MineDreamer MineRL env, VPT + STEVE-1 + CVAE weights, and an Imaginator server. The server loads the 7B MLLM in fp32 on the GPU, so it needs a GPU with at least 40 GB. `backends.load_real_stack()` wires MineDreamer's own loaders, but it has not been run here. MP5's planner and patroller would then read the MineRL inventory through `instructions.mp5_inventory`.
