# B integration status

## Completed
- Fixed bounded Patroller retries and caller-relative task paths.
- Configured OpenRouter chat and embeddings.
- Added a process-task Parser with external structured knowledge, dependency/material validation, and at most one feedback repair.
- Parser knowledge currently covers the wood crafting chain; it is not the original paper's Knowledge Memory RAG.
- Original HPC working copy passed real chat, embeddings, and Chroma smoke tests.
- Original HPC working copy passed online Parser generation followed by live-model execution in a simulated game: 5 subgoals, 13 simulated actions, final wooden pickaxe obtained.
- After migration onto the team's reproduce_MP5 branch: 17 Parser tests, 5 Patroller tests, simulator self-test, and offline B pipeline tests passed.

## Test boundaries
- Offline pipeline tests use mocked model replies and vector storage.
- test_live_b.py uses real model/embedding calls but simulated game and perception.
- Real MineLLM, MineDojo execution, and real failure recovery have not yet been verified by B.
- API smoke tests and live-model tests were not rerun after migration.
- API keys must be supplied through environment variables, never committed.

## Next integration steps
1. A: provide the MineLLM startup command, reachable endpoint, and a successful image-question-response example.
2. C: provide the validated MineDojo environment/container, launch commands, observation format, and action test results.
3. Verify real screenshot -> MineLLM inference.
4. Run a real wooden-pickaxe episode with the existing pre-decomposed JSON.
5. Repeat with Parser-generated JSON.
6. Extend knowledge, task coverage, failure recovery, and independent success evaluation.

## Historical HPC evidence (not included in Git)
- Parser: logs/parser/20261114-232737-039097/
- Live B: logs/live_b/20261114-232941-444214/
