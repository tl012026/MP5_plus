# Baseline report: C0 vs C1, GT perception

Episodes: 460 (c0 235, c1 225)

### Main: animal targets (navigation required)

| group | policy | n | SR-b [95% CI] | SR-a | SPL (mean) | median steps (success) | SR@250 | SR@500 | SR@1000 | SR@3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| animal, all | c0 | 120 | 0.17 [0.12, 0.25] | 0.17 | 0.06 | 372 | 0.06 | 0.09 | 0.09 | 0.17 |
| animal, all | c1 | 115 | 0.81 [0.73, 0.87] | 0.81 | 0.63 | 89 | 0.59 | 0.72 | 0.77 | 0.81 |
| animal, easy | c0 | 40 | 0.30 [0.18, 0.45] | 0.30 | 0.12 | 302 | 0.12 | 0.20 | 0.20 | 0.30 |
| animal, easy | c1 | 40 | 0.85 [0.71, 0.93] | 0.85 | 0.71 | 31 | 0.82 | 0.82 | 0.85 | 0.85 |
| animal, medium | c0 | 40 | 0.15 [0.07, 0.29] | 0.15 | 0.06 | 741 | 0.05 | 0.07 | 0.07 | 0.15 |
| animal, medium | c1 | 35 | 0.80 [0.64, 0.90] | 0.80 | 0.63 | 67 | 0.60 | 0.71 | 0.77 | 0.80 |
| animal, hard | c0 | 40 | 0.07 [0.03, 0.20] | 0.07 | 0.01 | 2455 | 0.00 | 0.00 | 0.00 | 0.07 |
| animal, hard | c1 | 40 | 0.78 [0.62, 0.88] | 0.78 | 0.55 | 254 | 0.35 | 0.62 | 0.68 | 0.78 |
| animal, with near-X | c0 | 30 | 0.03 [0.01, 0.17] | 0.03 | 0.00 | 2804 | 0.00 | 0.00 | 0.00 | 0.03 |
| animal, with near-X | c1 | 25 | 0.36 [0.20, 0.55] | 0.36 | 0.26 | 56 | 0.28 | 0.28 | 0.28 | 0.36 |
| animal, no near-X | c0 | 90 | 0.22 [0.15, 0.32] | 0.22 | 0.08 | 360 | 0.08 | 0.12 | 0.12 | 0.22 |
| animal, no near-X | c1 | 90 | 0.93 [0.86, 0.97] | 0.93 | 0.73 | 96 | 0.68 | 0.84 | 0.90 | 0.93 |

### Block targets (tree / grass)

| group | policy | n | SR-b [95% CI] | SR-a | SPL (mean) | median steps (success) | SR@250 | SR@500 | SR@1000 | SR@3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| block, easy | c0 | 40 | 0.85 [0.71, 0.93] | 0.85 | 0.47 | 4 | 0.72 | 0.72 | 0.80 | 0.85 |
| block, easy | c1 | 35 | 0.91 [0.78, 0.97] | 0.91 | 0.49 | 2 | 0.91 | 0.91 | 0.91 | 0.91 |
| block, med/hard, met | c0 | 27 | 0.74 [0.55, 0.87] | 0.74 | 0.43 | 73 | 0.74 | 0.74 | 0.74 | 0.74 |
| block, med/hard, met | c1 | 25 | 0.56 [0.37, 0.73] | 0.56 | 0.28 | 151 | 0.56 | 0.56 | 0.56 | 0.56 |
| block, med/hard, unmet | c0 | 48 | 0.81 [0.68, 0.90] | 0.81 | 0.62 | 0 | 0.69 | 0.69 | 0.71 | 0.81 |
| block, med/hard, unmet | c1 | 50 | 1.00 [0.93, 1.00] | 1.00 | 0.56 | 0 | 0.96 | 0.96 | 0.96 | 1.00 |

### All valid episodes (excluding difficulty_unmet)

| group | policy | n | SR-b [95% CI] | SR-a | SPL (mean) | median steps (success) | SR@250 | SR@500 | SR@1000 | SR@3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| all valid | c0 | 187 | 0.40 [0.33, 0.47] | 0.40 | 0.20 | 51 | 0.30 | 0.32 | 0.34 | 0.40 |
| all valid | c1 | 175 | 0.79 [0.73, 0.85] | 0.79 | 0.55 | 65 | 0.65 | 0.74 | 0.77 | 0.79 |

### Per task (valid episodes): SR-b c0 / c1

| task | target | easy | medium | hard |
|---|---|---|---|---|
| 1-1 | tree | 5/5 / 4/5 | 3/3 / 2/3 | 2/2 / 2/2 |
| 1-2 | grass | 5/5 / 5/5 | 0/1 / 0/1 | 0/1 / 0/1 |
| 1-3 | cow | 2/5 / 5/5 | 1/5 / 5/5 | 1/5 / 3/5 |
| 1-4 | pig | 3/5 / 5/5 | 1/5 / 4/5 | 1/5 / 5/5 |
| 2-1 | tree | 5/5 / 4/5 | 3/3 / 2/3 | 2/2 / 2/2 |
| 2-2 | grass | 4/5 / 5/5 | 0/1 / 0/1 | 0/1 / 0/1 |
| 2-3 | cow | 2/5 / 4/5 | 2/5 / 5/5 | 0/5 / 5/5 |
| 2-4 | pig | 0/5 / 5/5 | 1/5 / 5/5 | 1/5 / 4/5 |
| 3-1 | tree | 5/5 / - | 3/3 / - | 2/2 / 2/2 |
| 3-2 | grass | 2/5 / 5/5 | - / 0/1 | 0/1 / 0/1 |
| 3-3 | cow | 2/5 / 5/5 | 0/5 / 5/5 | 0/5 / 5/5 |
| 3-4 | pig | 2/5 / 5/5 | 1/5 / 4/5 | 0/5 / 5/5 |
| 4-1 | tree | 5/5 / 4/5 | 3/3 / 2/3 | 2/2 / 2/2 |
| 4-2 | grass | 3/5 / 5/5 | 0/1 / 0/1 | 0/1 / 0/1 |
| 4-3 | cow | 0/5 / 0/5 | 0/5 / 0/5 | 0/5 / 1/5 |
| 4-4 | pig | 1/5 / 5/5 | 0/5 / - | 0/5 / 3/5 |

### Failure breakdown (valid episodes)

| policy | subset | n | success | false_stop | never_seen | seen_not_found | seen_condition_missing | found_not_reached | no_events |
|---|---|---|---|---|---|---|---|---|---|
| c0 | animal | 120 | 21 | 0 | 5 | 38 | 7 | 49 | 0 |
| c0 | block | 67 | 54 | 0 | 7 | 0 | 6 | 0 | 0 |
| c1 | animal | 115 | 93 | 0 | 1 | 0 | 14 | 7 | 0 |
| c1 | block | 60 | 46 | 0 | 8 | 0 | 0 | 6 | 0 |

### SR-b by initial target distance (animal targets)

| distance | c0 | c1 |
|---|---|---|
| 0-4 | - | - |
| 4-8 | 12/38 | 32/38 |
| 8-16 | 5/39 | 27/34 |
| 16-24 | 1/20 | 18/20 |
| 24-33 | 3/23 | 16/23 |

### Paired test C1 vs C0 (same task, difficulty, seed)

- animal: pairs 115, C1 only success = 74, C0 only success = 2, exact McNemar p = 7.7e-20
- all valid: pairs 174, C1 only success = 80, C0 only success = 8, exact McNemar p = 4.6e-16

### Health checks

- c0: difficulty_unmet 48/235, target_never_seen (animal) 5/120, stuck step share 0.090, episodes > 3000 steps 0
- c1: difficulty_unmet 50/225, target_never_seen (animal) 1/115, stuck step share 0.004, episodes > 3000 steps 0
