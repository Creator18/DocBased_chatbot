# =============================================================================
# llm.py — Ollama Communication
# =============================================================================
# Thin wrapper around Ollama's /api/chat endpoint.
# Every module that needs the LLM calls through here — single point of
# control for model selection, timeouts, error handling, and future
# swaps (e.g. switching from Mistral to Llama 3).
#
# Extracted from: Notebook 2, Cells 4, 5, 6, 7 (scattered requests.post calls)

import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def chat(
    messages: list[dict],
    model: str = OLLAMA_MODEL,
    timeout: int = 60
) -> str:
    """
    Send a messages array to Ollama's /api/chat and return the response text.

    Args:
        messages: List of {"role": str, "content": str} dicts.
                  Follows Ollama's chat format (system/user/assistant).
        model:    Which Ollama model to use. Defaults to config.OLLAMA_MODEL.
        timeout:  Request timeout in seconds.

    Returns:
        The assistant's response text.

    Raises:
        requests.RequestException: If Ollama is unreachable or returns an error.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        },
        timeout=timeout
    )
    response.raise_for_status()

    return response.json()["message"]["content"]


def chat_with_system(
    system_prompt: str,
    user_prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = 60
) -> str:
    """
    Convenience wrapper for simple system + user prompt calls.
    Used by trigger detection, query reformulation, and summary generation
    where there's no conversation history — just a system instruction
    and a single user message.

    Args:
        system_prompt: The system instruction.
        user_prompt:   The user message.
        model:         Which Ollama model to use.
        timeout:       Request timeout in seconds.

    Returns:
        The assistant's response text.
    """
    return chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        timeout=timeout
    )


def check_connection() -> dict:
    """
    Verify Ollama is running and check available models.

    Returns:
        {
            "reachable": bool,
            "models":    list[str],
            "has_model": bool (whether OLLAMA_MODEL is available)
        }
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        return {
            "reachable": True,
            "models": models,
            "has_model": any(OLLAMA_MODEL in m for m in models)
        }
    except requests.ConnectionError:
        return {
            "reachable": False,
            "models": [],
            "has_model": False
        }