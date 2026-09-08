# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""Provedores de LLM suportados pela instância (OpenAI, Gemini, Anthropic).

Gemini e Anthropic expõem endpoints compatíveis com a API da OpenAI, por isso o
SDK ``openai`` é usado para os três, mudando apenas ``base_url``. A listagem de
modelos da Anthropic usa o endpoint nativo porque a camada compatível não a expõe.
"""

from typing import List

import requests
from openai import OpenAI

ANTHROPIC_VERSION = "2023-06-01"
REQUEST_TIMEOUT = 20

PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-flash-latest",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1/",
        "default_model": "claude-sonnet-4-5",
        "key_url": "https://console.anthropic.com/settings/keys",
    },
}

_OPENAI_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_NON_CHAT_MARKERS = (
    "embedding",
    "realtime",
    "audio",
    "tts",
    "transcribe",
    "whisper",
    "image",
    "imagen",
    "veo",
    "dall-e",
    "moderation",
    "search",
    "instruct",
    "live",
    "robotics",
    "computer-use",
    "aqa",
)


class LLMProviderError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _config(provider: str) -> dict:
    cfg = PROVIDERS.get((provider or "").lower())
    if not cfg:
        raise LLMProviderError(f"Unsupported provider: {provider}")
    return cfg


def _client(provider: str, api_key: str) -> OpenAI:
    cfg = _config(provider)
    if not api_key:
        raise LLMProviderError(f"Missing API key for provider: {cfg['name']}")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def _map_error(provider: str, exc: Exception) -> LLMProviderError:
    name = exc.__class__.__name__
    if name == "AuthenticationError" or getattr(exc, "status_code", None) in (401, 403):
        return LLMProviderError(f"Invalid API key for {provider}")
    if name == "RateLimitError" or getattr(exc, "status_code", None) == 429:
        return LLMProviderError(f"Rate limit exceeded for {provider}")
    return LLMProviderError(f"Error occurred while contacting {provider}: {name}")


def filter_chat_models(provider: str, model_ids: List[str]) -> List[str]:
    """Mantém só modelos de chat, remove prefixo ``models/`` (Gemini) e ordena."""
    provider = (provider or "").lower()
    out = set()
    for raw in model_ids:
        mid = (raw or "").strip()
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
        low = mid.lower()
        if not mid or any(m in low for m in _NON_CHAT_MARKERS):
            continue
        if provider == "gemini" and not low.startswith("gemini"):
            continue
        if provider == "openai" and not low.startswith(_OPENAI_CHAT_PREFIXES):
            continue
        if provider == "anthropic" and not low.startswith("claude"):
            continue
        out.add(mid)
    return sorted(out)


def list_models(provider: str, api_key: str) -> List[str]:
    cfg = _config(provider)
    if not api_key:
        raise LLMProviderError(f"Missing API key for provider: {cfg['name']}")
    provider = provider.lower()
    try:
        if provider == "anthropic":
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
                params={"limit": 100},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code in (401, 403):
                raise LLMProviderError(f"Invalid API key for {provider}")
            if resp.status_code != 200:
                raise LLMProviderError(f"Error occurred while contacting {provider}: HTTP {resp.status_code}")
            ids = [m.get("id", "") for m in resp.json().get("data", [])]
        else:
            client = _client(provider, api_key)
            ids = [m.id for m in client.models.list()]
    except LLMProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - qualquer falha do SDK/rede vira erro de provedor
        raise _map_error(provider, exc) from exc
    return filter_chat_models(provider, ids)


def chat(provider: str, api_key: str, model: str, text: str) -> str:
    provider = (provider or "").lower()
    client = _client(provider, api_key)
    try:
        completion = client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}])
    except Exception as exc:  # noqa: BLE001
        raise _map_error(provider, exc) from exc
    content = completion.choices[0].message.content if completion.choices else None
    if not content:
        raise LLMProviderError(f"Empty response from {provider}")
    return content
