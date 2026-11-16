# LLM baseline vs C2: perception interface and evaluation alignment

Two systems solve the same task family, "find X with conditions": the 16 MP5 creative tasks, transcribed in `lfnav/tasks/specs/`.

- **LLM system:** a large-model API that follows MP5's approach to decide what to do.
- **C2:** a trainable model only, with no large model.
- **Both use ground-truth perception for every frame.** Only decision-making is compared.

This note says what the LLM system must do for perception and how evaluation is aligned. All code references are repository-relative.

---

## 1. Recommended approach: implement the LLM system as an lfnav policy

The evaluation loop `lfnav/eval/run.py` already contains the scene setup, target spawning, GT perception, condition matching, controller, judge and logging. C0, C1 and C2 all run in it. Plugging the LLM system into the same loop gives identical conditions automatically.

- Subclass `Policy` in `lfnav/policy/base.py` and implement `act(policy_input: PolicyInput) -> PolicyOutput`.
- Register a name, for example `"llm"`, in `make_policy` in `lfnav/eval/run.py`.
- Run: `python -m lfnav.eval.run --config configs/lfnav/main.yaml --policy llm --perception gt --backend minedojo --difficulty <easy|medium|hard> --tasks lfnav/tasks/specs/<id>.json --seeds <seed file> --out <dir>`. For batch runs, use `lfnav/slurm/baseline_array.sbatch` with `POLICY=llm`.

Do not adapt the original MP5 loop (`MP5_agent/agent/run_agent.py`) for this comparison. Its scene setup, step counting and success rule all differ from lfnav; creative-task `success` is `null` there. Aligning it would mean rebuilding all of the above.

## 2. Perception: what the LLM system must do

### 2.1 Where the ground truth comes from (already implemented)
At every decision the loop builds a `PolicyInput` (`lfnav/types.py`) and passes it to the policy:

| field | source | content |
|---|---|---|
| `task` | `TaskSpec` | target and conditions (`near`, `biome`, `time`, `weather`, `brightness`) |
| `scene` | `GTPerception`, i.e. `scene_state_from_obs` in `lfnav/gt/scene_state.py` | per class: `present`, `direction` over left/center/right, `distance` over near/mid/far (near ≤ 3 blocks, mid ≤ 10); plus `biome`, `time`, `weather`, `brightness` |
| `match` | `Matcher.match` in `lfnav/matcher.py` | `status` (found / not_found), per-condition `satisfied`, `missing`, `target_direction`, `target_distance` bin, `stop` |
| `memory` | `lfnav/memory.py` | `sector8` (visit counts in 8 directions), `recent_actions`, `visits`, `pos`, `yaw` |
| `recent_actions` | loop | actions executed recently |

All of this comes from MineDojo rays and environment state. **It is the same ground truth that C1 and C2 use.**

### 2.2 What the LLM system has to do
1. **Serialize `PolicyInput` into a text prompt** for the model. Suggested content:
   - the task description;
   - whether the target is visible, its direction and distance bin;
   - whether each condition holds;
   - biome, time and weather;
   - `sector8` and the recent actions.
2. **Give the model no more information than C2 has, and preferably no less.** At each decision C2 receives:
   - target visibility, bearing, continuous distance, left/center/right direction and distance bin;
   - whether each condition holds;
   - from the state tracker S: scan progress, visits in 8 directions, displacement, and whether the agent is stuck.

   The LLM system should get equivalent information. If it only gets left/center/right and no bearing or continuous distance, state that in the report.
3. **Forbidden inputs:**
   - images;
   - MineLLM or any vision model;
   - the 360° wide lidar (`policy_sees_wide_lidar: false` in `main.yaml`);
   - true target coordinates or spawn offsets (`spawn_offsets`).

   C2 has none of these.

## 3. Actions and stopping

- **Actions:** `PolicyOutput.action_chunk` may only use action names from `configs/lfnav/actions.yaml`, for example `forward`, `turn_left_15`, `turn_right_45` and `backward`. Each decision executes at most 12 steps (`chunk_max_steps`), the same limit as C1 and C2.
- **Stopping:** decided by rule, **not by the LLM**. The agent stops when the target is visible, every condition holds and the distance is ≤ 3 blocks (`match.stop`), as for C1 (`lfnav/eval/run.py`, line 319). C2 also stops by a rule gate (visible, conditions met, distance), not a learned output. Both systems are therefore compared only on how they move.

## 4. Evaluation alignment

| item | value | where |
|---|---|---|
| Scene standard | `main.yaml`: targets move, tall grass kept, natural mobs kept, peaceful, entities not cleared | `configs/lfnav/main.yaml` |
| Tasks | 16 tasks, 4-3 excluded. Tree and grass tasks at easy only | `lfnav/tasks/specs/`. Array index to cell: `lfnav/eval/baseline_index.py` |
| Difficulty | easy / medium / hard | `configs/lfnav/difficulty.yaml` |
| Evaluation seeds | 9, 19, 29, 39, 49. Must not overlap with the training seeds | `configs/lfnav/eval_seeds_baseline.txt`; training seeds in `configs/lfnav/train_seeds_demo.txt` |
| Episode cap | 3000 steps | `judge.max_steps` |
| Primary metric | **SR-b**: the agent stops, the target is in view, distance ≤ 3 blocks, all conditions hold | `lfnav/gt/judge.py` |
| Secondary metrics | SR-a (no distance requirement), false stops, steps, SPL, failure category | per-episode `<out>/.../<seed>.json` |
| Code version | Both systems are evaluated at the same commit | record `git log -n 1` with the results |

The LLM system must also record, in `PolicyOutput.info`:
- model name and temperature;
- API calls per episode;
- input and output tokens;
- latency per call;
- cost.

These are where C2 is expected to win, so they must be measured.

## 5. Known limitations (both systems are affected; state them in the report)

- **Rays are blocked by tall grass.** Neither the ground truth nor the judge can see a target behind tall grass. Both systems are affected equally, so the comparison stays fair. In plains, however, "visible" differs from what a human sees in the image.
- **Ground truth is not pixel perception.** The conclusion is limited to decision-making under identical perception. It says nothing about a full system that starts from pixels.