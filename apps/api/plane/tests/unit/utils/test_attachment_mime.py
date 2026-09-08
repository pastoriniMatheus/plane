# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""Regression: browsers send an empty MIME type for extensions they do not know.

Real rejections observed in production:
    Attachment rejected: name=teste type='' not in ATTACHMENT_MIME_TYPES
    Attachment rejected: name=teste.log type='' not in ATTACHMENT_MIME_TYPES

The endpoint refused them outright, so a plain .md or .log file could not be
attached. When the browser says nothing, derive the type from the file name and
let the existing allow-list decide.
"""

import pytest

from plane.utils.attachment_mime import resolve_mime_type


@pytest.mark.unit
def test_browser_supplied_type_always_wins():
    assert resolve_mime_type("notes.md", "text/plain") == "text/plain"
    assert resolve_mime_type("photo.png", "image/png") == "image/png"


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,expected",
    [
        ("notes.md", "text/markdown"),
        ("NOTES.MD", "text/markdown"),
        ("notes.markdown", "text/markdown"),
        ("notes.txt", "text/plain"),
        ("server.log", "text/plain"),
        ("app.conf", "text/plain"),
        ("settings.ini", "text/plain"),
        ("pyproject.toml", "text/plain"),
        ("compose.yml", "application/x-yaml"),
        ("compose.yaml", "application/x-yaml"),
        ("dump.sql", "application/x-sql"),
        ("deploy.sh", "text/x-sh"),
    ],
)
def test_infers_from_extension_when_browser_sends_nothing(name, expected):
    assert resolve_mime_type(name, "") == expected


@pytest.mark.unit
def test_unknown_extension_stays_unknown():
    """No guess means the allow-list still refuses it - the control is kept."""
    assert resolve_mime_type("teste", "") == ""
    assert resolve_mime_type("firmware.xyzzy", "") == ""


@pytest.mark.unit
def test_handles_missing_arguments():
    assert resolve_mime_type(None, None) == ""
    assert resolve_mime_type("", "") == ""
