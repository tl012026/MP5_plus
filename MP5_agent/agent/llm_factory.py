"""Create LLM clients from config. Does not change call style or prompt parameters."""

from __future__ import annotations

import os

from config import resolve_api_key, resolve_base_url


def apply_openai_endpoint(base_url):
    """Set the process-wide openai SDK base. Old langchain ChatOpenAI reads this."""
    import openai
    url = resolve_base_url(base_url)
    if url:
        openai.api_base = url
    return url


def create_chat_openai(model_cfg):
    """
    model_cfg: {model_name, base_url, temperature} from config.models.<role>
    Same ChatOpenAI constructor arguments as the original modules.
    """
    from langchain.chat_models import ChatOpenAI
    key = resolve_api_key()
    os.environ["OPENAI_API_KEY"] = key
    apply_openai_endpoint(model_cfg.get("base_url"))
    return ChatOpenAI(
        model_name=model_cfg["model_name"],
        temperature=model_cfg.get("temperature", 0),
    )


def get_openai_key():
    return resolve_api_key()
