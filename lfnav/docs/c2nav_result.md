# C2 (c2nav) results

**Setting.** `configs/lfnav/main.yaml`: targets move, tall grass kept, natural mobs kept, peaceful. Evaluation seeds 9/19/29/39/49. Max 3000 steps. Metric: SR-b.

**System.** All methods compared here share the same frame-level perception: ground truth from MineDojo rays.
- **C2** = GT perception + S (state tracker, odometry) + B (rule matcher and stop gate) + C (learned MLP policy). Only C is learned.
- **C1** = the same GT perception and B, with a hand-written rule policy (the expert without wide lidar).
- An LLM-based planner with the same GT perception is the comparison partner. See `llm_baseline_alignment.md`.

## 1. C2 trained on easy demos only (r3 epoch 13)

Training data is `demos_main_v3`: easy cells only, 44 training seeds. Stop rule: target visible, conditions met, distance ≤ 2.8. Explore turns are sampled.

| cell | C2 SR-b | C1 SR-b | note |
|---|---|---|---|
| 1-3 easy | 5/5 | 5/5 | median steps C2 36 vs C1 29 over both easy cells |
| 1-4 easy | 5/5 | 5/5 | |
| 1-3 medium | 5/5 | 5/5 | not seen in training |
| 1-4 medium | 4/5 | 4/5 | different seeds fail |
| 1-3 hard | 3/5 | 4/5 | not seen in training |
| 1-4 hard | 3/5 | 4/5 | |
| **medium total** | **9/10** | **9/10** | |
| **hard total** | **6/10** | **8/10** | |

- **Every C2 failure is a timeout. C2 had no false stops**, so the rule stop gate holds at every difficulty.
- **Hard: C2 is slower, and the slowness is not in finding the target.** In 8 of 10 hard episodes, C2 first sees the target during the opening scan (by step 11). The time is lost after that, in two ways:
  - **Long approach:** for example 1-3 hard seed 9 spent 1248 decisions in approach mode before succeeding.
  - **Target lost, then long exploration that fails to re-find it:** about 700–950 explore decisions in 1-3 seeds 29 and 39, and in 1-4 seeds 9 and 49.
- **Likely cause:** hard targets start 16–32 blocks away, while easy demos start them 4–8 blocks away. Approaching a far target and re-acquiring a lost one are rare in the easy training data.

## 2. Next
- C1 on all 29 cells (in progress) and C2 on all 29 cells.
- Joint training on easy + medium demos (`demos_v4`, medium, max 500 steps). Hard stays as the unseen difficulty.
- Collect hard demos (max 1500 steps) if the joint model still lags C1 on hard.

---

## Appendix A. Learned perception (A) instead of GT (not part of the main system)

Tested on 1-3 / 1-4 easy, 10 episodes. Same C (r3 epoch 13).

| run | A | stop rule | SR-b |
|---|---|---|---|
| A-pred | A-only epoch 7 | single decision | 4/10 |
| A-pred + `stop_confirm: 2` | A-only epoch 7 | 2 consecutive decisions | 3/10 |

- Run-to-run noise with the same config is ±1–2 out of 10.
- Failure causes:
  - A false positives: horses taken for cows, and animals behind tall grass.
  - Distance underestimated over several frames.
  - Exploration timeouts.
- **Labels and judge cannot see through tall grass.** MineDojo rays stop at the first block. Entity rays also stop at blocks, so an animal behind tall grass is absent from the GT labels and from the judge.
  - MineDojo ray names are lowercase display names: `grass` is tallgrass, `plant` is double_plant, `flower` is a small flower, and `grass block` is the ground.
  - A sees these animals in the image, so its labels contradict the pixels. Hard-negative re-weighting did not converge.
- Offline A (validation, new easy data):
  - distance error median 0.55 blocks at GT ≤ 3;
  - frame-level target visibility at threshold 0.7: FPR 14%, recall 0.57.

## Unit tests
`lfnav/tests/test_c2nav*.py` and `lfnav/tests/test_occluder.py`. Run them inside the container: `timeout 120 python -m unittest discover -s lfnav/tests -p "test_*.py"`.