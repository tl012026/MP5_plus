conda activate mc

# Keys: export OPENAI_API_KEY (and optional OPENAI_BASE_URL). Do not pass --openai_key.
# MineLLM: pass --mllm_url when a local or remote service is running.

python3 run_agent.py \
    --config ../../../configs/code_default.yaml \
    --gpt_model_name gpt-4-0125-preview \
    --answer_method active \
    --answer_model mllm \
    --task tasks/process/craft_wooden_pickaxe.json \
