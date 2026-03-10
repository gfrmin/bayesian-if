"""Minimal Ollama HTTP client."""

from __future__ import annotations

import os

_DEFAULT_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_generate(
    prompt: str,
    model: str = "llama3.2",
    temperature: float = 0.3,
    base_url: str = _DEFAULT_BASE_URL,
) -> str:
    """Send a generate request to Ollama and return the response text."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("Install httpx: pip install bayesian-if[ollama]") from e

    resp = httpx.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def ollama_available(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Check whether Ollama is reachable."""
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False
