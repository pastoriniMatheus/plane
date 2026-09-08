# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from unittest.mock import patch

import pytest
from rest_framework.test import APIRequestFactory
from rest_framework.throttling import AnonRateThrottle

from plane.license.api.permissions import InstanceAdminPermission
from plane.license.api.views import llm as llm_view
from plane.utils.llm_providers import LLMProviderError


def _post(body, permission_granted=True):
    request = APIRequestFactory().post("/api/instances/configurations/llm-models/", body, format="json")
    # O ambiente de teste unitário não tem Redis disponível. `BaseAPIView.initial()`
    # chama `check_throttles()` antes mesmo da checagem de permissão, o que exigiria
    # cache Redis via `AnonRateThrottle`. Como este é um teste unitário (sem infra),
    # o throttle é mockado da mesma forma que a permissão.
    with (
        patch.object(InstanceAdminPermission, "has_permission", return_value=permission_granted),
        patch.object(AnonRateThrottle, "allow_request", return_value=True),
    ):
        return llm_view.InstanceLLMModelsEndpoint.as_view()(request)


@pytest.mark.unit
def test_returns_models_for_given_key():
    with patch.object(llm_view, "list_models", return_value=["gemini-2.5-flash"]) as lm:
        resp = _post({"provider": "gemini", "api_key": "k1"})
    assert resp.status_code == 200
    assert resp.data == {"provider": "gemini", "models": ["gemini-2.5-flash"]}
    lm.assert_called_once_with("gemini", "k1")


@pytest.mark.unit
def test_falls_back_to_stored_key_when_blank():
    with (
        patch.object(llm_view, "get_configuration_value", return_value=("stored",)),
        patch.object(llm_view, "list_models", return_value=["gpt-4o-mini"]) as lm,
    ):
        resp = _post({"provider": "openai", "api_key": ""})
    assert resp.status_code == 200
    lm.assert_called_once_with("openai", "stored")


@pytest.mark.unit
def test_rejects_unknown_provider():
    resp = _post({"provider": "llama", "api_key": "k"})
    assert resp.status_code == 400
    assert "Unsupported provider" in resp.data["error"]


@pytest.mark.unit
def test_maps_provider_error_to_400():
    with patch.object(llm_view, "list_models", side_effect=LLMProviderError("Invalid API key for gemini")):
        resp = _post({"provider": "gemini", "api_key": "bad"})
    assert resp.status_code == 400
    assert resp.data == {"error": "Invalid API key for gemini"}


@pytest.mark.unit
def test_requires_key_when_nothing_stored():
    with patch.object(llm_view, "get_configuration_value", return_value=(None,)):
        resp = _post({"provider": "gemini", "api_key": ""})
    assert resp.status_code == 400
    assert "API key" in resp.data["error"]


@pytest.mark.unit
def test_rejects_non_admin_caller_without_contacting_provider():
    with patch.object(llm_view, "list_models") as lm:
        resp = _post({"provider": "gemini", "api_key": "k1"}, permission_granted=False)
    assert resp.status_code in (401, 403)
    assert not lm.called
