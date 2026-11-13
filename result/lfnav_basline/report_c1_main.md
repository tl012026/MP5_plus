# C1 baseline under the main setting (2026-11-12)

## Setup
- Policy: C1 (rules) with ground-truth perception. The 360° wide lidar is off (`policy_sees_wide_lidar: false`).
- Scene: `configs/lfnav/main.yaml`. Targets move; tall grass and natural mobs are kept; peaceful difficulty. Targets are summoned after reset with the difficulty's distance and direction (hard: 16–32 blocks, behind the agent ±45°, 4 distractors).
- Seeds: `configs/lfnav/eval_seeds_baseline.txt` (9, 19, 29, 39, 49). Max 3000 steps.
- Metric: SR-b (target visible, within 3 blocks, all conditions met at the stop step), judged from GT rays. 95% CIs are Wilson intervals.
- Cells (12 × 5 = 60 episodes). Hard covers all 7 animal tasks (4-3 excluded). Easy and medium use representative cells, including the cells used for the first C2 checks.

## Results by difficulty
| Difficulty | SR-b | 95% CI | Median steps (successes) | Failures |
|---|---|---|---|---|
| easy | 1.00 (10/10) | [0.72, 1.00] | 29 | 0 |
| medium | 0.93 (14/15) | [0.70, 0.99] | 48 | 1 timeout |
| hard | 0.69 (24/35) | [0.52, 0.81] | 208 | 11 timeouts |
| **All** | **0.80 (48/60)** | **[0.68, 0.88]** | 97 | 12 timeouts |

## Results by cell
| Task | easy | medium | hard |
|---|---|---|---|
| 1-3 (cow) | 5/5 (30) | | 4/5 (122) |
| 1-4 (pig) | 5/5 (28) | | 4/5 (208) |
| 2-3 (cow, desert) | | 5/5 (35) | 4/5 (90) |
| 2-4 | | 4/5 (48) | **1/5** (103) |
| 3-3 | | | 4/5 (264) |
| 3-4 | | | 4/5 (308) |
| 4-4 (pig near grass) | | 5/5 (61) | 3/5 (1658) |

Cells show successes / episodes and, in brackets, the median steps of successful episodes.

## Failure analysis
- All 12 failures are timeouts at 3000 steps.
- In every failure the target had been seen at least once (`target_never_seen = false`). Exploration works. The failures happen after the target is found: C1 loses a moving target, is interrupted by terrain or stuck recovery, or never closes to within 3 blocks.
- Three of the 2-4 hard failures also had repeated stuck events (3–11 events, up to 112 stuck steps).
- 2-4 is the weakest task (hard 1/5, medium 4/5). The cause is not yet checked.
- 4-4 hard succeeds slowly (median 1658 steps), because the `near grass` condition needs extra search.

## Comparison with the earlier static setting
| | Static (2026-11-09) | Main (this report) |
|---|---|---|
| easy / medium / hard | 0.85 / 0.80 / 0.78 | 1.00 / 0.93 / 0.69 |

Not directly comparable. The settings differ in four ways that push in opposite directions: moving targets and tall grass make it harder; natural same-species mobs and targets wandering closer make it easier. The episode sets also differ (the static run covered all cells).

## Known issues
- `initial_target_distance` is logged as `> 32` for every episode, so SPL cannot be computed. Planned fix: initial distance = min(spawn offset, nearest same-species ray).
- `elapsed_s` values look like minutes, not seconds (about 11 for a 3000-step episode). To be checked.

## Not yet run
Block tasks (8 cells, easy) and the remaining animal easy/medium cells. They are expected to be near 1.00 and can run together with the final evaluation of the learned policies.

## Reproduce
Account and partition come from the local, untracked environment script.

    POLICY=c1 CFG=configs/lfnav/main.yaml ROOT_OUT=<output dir> \
      sbatch -c 4 --mem=16G --time=04:00:00 --export=ALL \
      --array=6,8,9,11,19,20,22,23,32,35,46,47%3 lfnav/slurm/baseline_array.sbatch

Array index = task index × 3 + difficulty (0 easy, 1 medium, 2 hard), with tasks in the order 1-1 … 4-4.