conda activate mc

python3 run_agent.py \
    --mllm_url http://10.140.1.105:25541/image \
    --openai_key REDACTED_ORIGINAL_KEY \
    --gpt_model_name gpt-4-0125-preview \
    --answer_method active \
    --answer_model mllm \
    --task tasks/diamond_tools/diamond_ore.json \