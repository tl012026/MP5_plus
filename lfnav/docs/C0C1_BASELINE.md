# C0/C1 baseline: problems found and fixes (2026-11-08)

Setup: C0 (random walk) and C1 (rule policy) with ground-truth perception on the 16 MP5 creative tasks, 3 difficulties, seeds 9/19/29/39/49, max 3000 steps. Main metric is SR-b. Smoke runs on real MineDojo exposed the issues below. They are listed in the order they were found.

## 1. Smoke test chose trivial tasks
- **Problem:** 1-1 (tree) and 1-4 (pig) start with the target in view, so they test nothing.
- **Fix:** smoke runs now use animal tasks with conditions at medium/hard (2-3, 3-4, 4-3, 4-4, plus 1-4 at all difficulties).

## 2. Target never appeared (animal tasks)
- **Evidence:** every animal task reported `initial_target_distance = "> 32"`; C0 and C1 timed out. A probe found no pig in any direction for three spawn boxes. A manual `/summon pig ~ ~1 ~3` after reset worked at once.
- **Cause:** MineDojo's reset hook sends `/summon` on the first tick after world creation, and the command is silently dropped. Pigs seen in earlier runs (camera alignment, the first 1-4 success) were natural spawns.
- **Fix (`d932973`):** lfnav summons animals itself after `reset()`:
  - wait a few real no-op ticks (invalid chat commands do not advance the tick);
  - sample a ring offset relative to the agent's actual yaw; hard puts the target behind the agent, ±45°;
  - send `/summon`.
  - The old `initial_mobs` path is kept behind `minedojo.spawn_via_reset_hook`.

## 3. Too many targets, or false spawn failures
- **Evidence:** ray-based spawn checks missed small, distant animals, so the retry loop summoned up to 3 targets. Hard desert cows were marked `spawn_failed` even though C0 later found one. Natural pigs, cows and sheep could also sit next to the agent, which made the difficulty setting meaningless.
- **Fix (`3e28f9c`, `d7b0097`):**
  - `/kill @e[type=!player]` before spawning;
  - summon each animal once, with no ray check and no retry;
  - drop from +2 with a short resistance effect;
  - freeze animals with `slowness` (gravity kept), so C0 and C1 face the same static target;
  - `initial_target_distance` comes from the spawn offset, and the ray estimate is logged separately;
  - `target_never_seen` is computed after the episode.

## 4. Rays blocked by plants
- **Evidence:** in plains, a pig (0.9 blocks tall) standing in tall grass (about 0.8 blocks) is hidden from rays cast from eye height. The ground-truth labels missed it.
- **Fix (`d7b0097`):** before spawning, clear `tallgrass` and `double_plant` within ±32 × ±4 × ±32 blocks, using chunked `/fill` because one command is capped at 32768 blocks. Tasks whose target or `near` condition is grass (1-2, 2-2, 3-2, 4-2, 4-4) are not cleared; this is logged as `vegetation_cleared`.

## 5. Exploration never looked behind
- **Evidence:** C1 failed every hard episode. Some had `target_never_seen = true`: the agent never once turned toward a target spawned behind it.
- **Cause:**
  - C1 explored only the three forward sectors (left, center, right) and never turned around.
  - It did no opening scan.
  - Ties went to `center` by name order, so it kept walking straight ahead.
- **Fix (`d7b0097`):**
  - opening 360° scan (24 × 15°) that stops early once the target is seen;
  - 8 visit sectors of 45°, including behind, with seeded random tie-breaking;
  - no visit counting while stuck.

## 6. Stop rule stricter than the success rule
- **Evidence:** C1 reached trees and grass but kept walking. The matcher stopped only when at least 50% of the target rays were within 3 blocks. A tree's leaves, or a field of grass, extend beyond 3 blocks, so that fraction was rarely met.
- **Fix (`d7b0097`):** the default `stop_rule: nearest` stops when the nearest target ray is within `judge.radius` (3 blocks). This is the same test SR-b uses. The old rule is kept as `fraction`.

## 7. Overshooting and losing close targets
- **Evidence:** frames show the pig found at mid range. The next action chunk walked past it, the pig dropped below the frame or behind the hand, and C1 went back to exploring.
- **Fix (`d7b0097`):**
  - at most 2 forward steps per decision at mid range;
  - remember `last_seen`;
  - if the target disappears within 20 steps, step back once and rescan.

## 8. Agent stuck spinning
- **Evidence:** C1 sat in a pit by a pond for about 2900 steps, alternating turns and forward moves. `stuck` never fired, because each single action "succeeded".
- **Cause:**
  - stuck detection looked at one action at a time;
  - all water counted as a hazard, so the pond side was blocked;
  - visit counts piled up on the agent's own cell, which drove endless turning.
- **Fix:**
  - **`d7b0097`:**
    - stuck means less than 1 block of displacement over 60 steps;
    - escape sequence: jump forward, back up and turn 90°, then turn 180° and jump forward;
    - shallow water can be crossed (deep water, lava, fire and cactus are still hazards).
  - **`f22bced`:**
    - stuck counts only when at least 50% of the actions in the window were forward moves, so C0 turning in place no longer triggers it;
    - episodes are cut exactly at `max_steps`.

## 9. Distant targets fell between rays
- **Evidence:** C1 needed 1075 steps for a pig 14 blocks away, and the opening scan never registered it. With one ray per patch, rays are about 4.3° apart, while a pig spans about 3.7° at 14 blocks and about 2° at 25 blocks. Ground truth was worse than the image, so C1 + GT was not a true upper bound.
- **Fix (`27784b3`):** `aligned_subrays: 3` casts 3×3 sub-rays per patch, about 1.4° apart:
  - a patch is labelled with every object any of its sub-rays hits;
  - its distance is the nearest hit among those sub-rays;
  - the output is still 16×26 patches;
  - `n = 1` reproduces the old behaviour.

## 10. Smaller fixes
- **`1588da1`:**
  - per-job Gradle cache copy on node-local disk;
  - when the target is out of view, direction and distance are `None` instead of a stale value;
  - evaluation frames every 10 steps.
- **Job script:** `mp5_env.sh` was sourced after `GRADLE_USER_HOME` was exported, which silently pointed jobs back at the shared cache. The order is now: source first, then export.
- **Spawn RNG:** seeded with `sha256(seed, task_id, index)`. Before this, every task had the same layout for a given seed.
- **Biome check:** `check_biome_world` confirmed that `specified_biome` worlds are a single biome (forest, so far), so `leave_biome` stays off.

## 11. Freeze, resistance, and vegetation commands were silently dropped
- **Evidence:** client logs showed `Invalid cmd` for `/effect` (slowness, resistance) and chunked `/fill` of tall grass. MineDojo's `execute_cmd` whitelist swallowed them; ticks did not advance and the world did not change. Earlier C0/C1 runs were therefore **not** frozen, **not** given resistance, and **not** vegetation-cleared, even when the config said so.
- **Fix:** `_send_chat` writes the command onto `action_space.no_op()["chat"]` and `step`s it. Any `Invalid cmd` now raises. Resistance is NBT on `/summon`. The main scene (`main.yaml`) does not freeze or clear vegetation; it does send `/difficulty peaceful`.

## Known limitations (kept, reported)
- **Tree and grass targets can't be made hard.** The target is everywhere, so medium and hard become `difficulty_unmet`. Report these tasks at easy only.
- **Biome, time and weather conditions are set at reset.** They hold for the whole episode, so the only condition that requires search is `near X`.
- **Grass-related tasks keep tall grass.** Those tasks (1-2, 2-2, 3-2, 4-2, 4-4) may still suffer occlusion.
- **SPL uses the straight-line start distance.** It ignores terrain detours.

## Status after fixes (partial full run)
| Task | easy | medium | hard |
|---|---|---|---|
| C1, 1-3 cow | 5/5 (30 steps) | 5/5 (65) | 2/3 (214) |
| C1, 1-4 pig | 5/5 (28) | 2/3 (670) | 5/5 (352) |

- Stuck steps make up about 1% of all steps.
- No episode exceeds 3000 steps.
- Open item: one C1 failure on 1-1 easy, where C0 succeeded 5/5.
