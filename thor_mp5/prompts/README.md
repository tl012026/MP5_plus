active_perception_system.txt — source MP5_agent/agent/prompts/active_perception_system.txt; decides whether to keep asking the perceptor; household agent, categories object/state/placement/nearby/room/lighting, island and table rule, household examples (status 2, then status 1 with a nearby query).
active_perception_query.txt — source MP5_agent/agent/prompts/active_perception_query.txt; fills the task and environment into the active-perception question; unchanged.
gpt-vision-active_perception_system.txt — source MP5_agent/agent/prompts/gpt-vision-active_perception_system.txt; answers one perception question from an image; household agent, HUD rule removed.
gpt-vision-caption_perception_system.txt — source MP5_agent/agent/prompts/gpt-vision-caption_perception_system.txt; captions one image; household agent, HUD rule removed.
check_caption_perception_system.txt — source MP5_agent/agent/prompts/check_caption_perception_system.txt; judges task completion from a caption; household agent, island and table rule.
check_caption_perception_query.txt — source MP5_agent/agent/prompts/check_caption_perception_query.txt; fills the task and caption into the check; unchanged.
single_shot_verify_system.txt — no source file; one image and the task, one JSON answer with thoughts and status 0, 1, or 2.
