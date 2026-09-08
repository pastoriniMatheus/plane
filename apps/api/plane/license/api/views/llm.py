# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

# Python imports
import os

# Third party imports
from rest_framework import status
from rest_framework.response import Response

# Module imports
from plane.license.api.permissions import InstanceAdminPermission
from plane.license.utils.instance_value import get_configuration_value
from plane.utils.llm_providers import PROVIDERS, LLMProviderError, list_models

from .base import BaseAPIView


class InstanceLLMModelsEndpoint(BaseAPIView):
    """Lista os modelos de chat disponíveis para um provedor/chave (God Mode)."""

    permission_classes = [InstanceAdminPermission]

    def post(self, request):
        provider = str(request.data.get("provider", "") or "").strip().lower()
        if provider not in PROVIDERS:
            return Response({"error": f"Unsupported provider: {provider}"}, status=status.HTTP_400_BAD_REQUEST)

        api_key = str(request.data.get("api_key", "") or "").strip()
        if not api_key:
            (api_key,) = get_configuration_value(
                [{"key": "LLM_API_KEY", "default": os.environ.get("LLM_API_KEY", None)}]
            )
        if not api_key:
            return Response(
                {"error": f"API key is required for {PROVIDERS[provider]['name']}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            models = list_models(provider, api_key)
        except LLMProviderError as e:
            return Response({"error": e.message}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"provider": provider, "models": models}, status=status.HTTP_200_OK)
