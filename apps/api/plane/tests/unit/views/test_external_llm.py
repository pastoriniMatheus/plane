# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from unittest.mock import patch

import pytest

from plane.app.views.external import base as ext
from plane.utils.llm_providers import LLMProviderError


def _cfg(values):
    """Simula get_configuration_value devolvendo (api_key, provider, model)."""
    return patch.object(ext, "get_configuration_value", return_value=values)


@pytest.mark.unit
def test_get_llm_config_uses_provider_default_model_when_empty():
    with _cfg(("key", "gemini", None)):
        assert ext.get_llm_config() == ("key", "gemini-flash-latest", "gemini")


@pytest.mark.unit
def test_get_llm_config_accepts_any_model_name():
    with _cfg(("key", "gemini", "gemini-9-ultra")):
        assert ext.get_llm_config() == ("key", "gemini-9-ultra", "gemini")


@pytest.mark.unit
def test_get_llm_config_rejects_unknown_provider_or_missing_key():
    with _cfg(("key", "llama", "x")), patch.object(ext, "log_exception"):
        assert ext.get_llm_config() == (None, None, None)
    with _cfg((None, "openai", "gpt-4o-mini")), patch.object(ext, "log_exception"):
        assert ext.get_llm_config() == (None, None, None)


@pytest.mark.unit
def test_get_llm_response_delegates_to_chat():
    with patch.object(ext, "chat", return_value="texto") as chat:
        assert ext.get_llm_response("Resuma", "corpo", "k", "gemini-flash-latest", "gemini") == ("texto", None)
    chat.assert_called_once_with("gemini", "k", "gemini-flash-latest", "Resuma\ncorpo")


@pytest.mark.unit
def test_get_llm_response_returns_error_message():
    with patch.object(ext, "chat", side_effect=LLMProviderError("Invalid API key for gemini")), patch.object(
        ext, "log_exception"
    ):
        assert ext.get_llm_response("t", "p", "k", "m", "gemini") == (None, "Invalid API key for gemini")
