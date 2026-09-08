# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plane.utils import llm_providers as lp


@pytest.mark.unit
def test_providers_have_required_fields():
    for key in ("openai", "gemini", "anthropic"):
        cfg = lp.PROVIDERS[key]
        assert cfg["name"] and cfg["default_model"] and cfg["key_url"]
    assert lp.PROVIDERS["openai"]["base_url"] is None
    assert lp.PROVIDERS["gemini"]["base_url"].startswith("https://generativelanguage.googleapis.com/")
    assert lp.PROVIDERS["anthropic"]["base_url"].startswith("https://api.anthropic.com/")


@pytest.mark.unit
def test_filter_chat_models_gemini_strips_prefix_and_drops_non_chat():
    ids = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-embedding-001",
        "models/imagen-4.0-generate-001",
        "models/gemini-2.5-flash-preview-tts",
        "models/veo-3.0-generate-001",
    ]
    assert lp.filter_chat_models("gemini", ids) == ["gemini-2.5-flash", "gemini-2.5-pro"]


@pytest.mark.unit
def test_filter_chat_models_openai_keeps_chat_only():
    ids = ["gpt-4o-mini", "text-embedding-3-small", "gpt-4o-realtime-preview", "o3-mini", "dall-e-3", "whisper-1"]
    assert lp.filter_chat_models("openai", ids) == ["gpt-4o-mini", "o3-mini"]


@pytest.mark.unit
def test_filter_chat_models_anthropic_keeps_claude():
    assert lp.filter_chat_models("anthropic", ["claude-sonnet-4-5", "other"]) == ["claude-sonnet-4-5"]


@pytest.mark.unit
def test_list_models_rejects_bad_input():
    with pytest.raises(lp.LLMProviderError):
        lp.list_models("nope", "k")
    with pytest.raises(lp.LLMProviderError):
        lp.list_models("gemini", "")


@pytest.mark.unit
def test_list_models_gemini_uses_openai_client_with_base_url():
    fake_client = MagicMock()
    fake_client.models.list.return_value = [
        SimpleNamespace(id="models/gemini-2.5-pro"),
        SimpleNamespace(id="models/gemini-2.5-flash"),
        SimpleNamespace(id="models/gemini-embedding-001"),
    ]
    with patch.object(lp, "OpenAI", return_value=fake_client) as ctor:
        models = lp.list_models("gemini", "key-1")
    ctor.assert_called_once_with(api_key="key-1", base_url=lp.PROVIDERS["gemini"]["base_url"])
    assert models == ["gemini-2.5-flash", "gemini-2.5-pro"]


@pytest.mark.unit
def test_list_models_anthropic_uses_native_endpoint():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": [{"id": "claude-sonnet-4-5"}, {"id": "claude-haiku-4-5"}]}
    with patch.object(lp.requests, "get", return_value=resp) as get:
        models = lp.list_models("anthropic", "key-2")
    assert models == ["claude-haiku-4-5", "claude-sonnet-4-5"]
    headers = get.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "key-2" and "anthropic-version" in headers


@pytest.mark.unit
def test_list_models_maps_authentication_error():
    class AuthenticationError(Exception):
        pass

    fake_client = MagicMock()
    fake_client.models.list.side_effect = AuthenticationError("bad key")
    with patch.object(lp, "OpenAI", return_value=fake_client):
        with pytest.raises(lp.LLMProviderError) as exc:
            lp.list_models("openai", "k")
    assert "Invalid API key" in exc.value.message


@pytest.mark.unit
def test_chat_returns_text_and_uses_model():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ola"))]
    )
    with patch.object(lp, "OpenAI", return_value=fake_client):
        assert lp.chat("gemini", "k", "gemini-2.5-flash", "oi") == "ola"
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["messages"] == [{"role": "user", "content": "oi"}]
