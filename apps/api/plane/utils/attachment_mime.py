# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""Resolve the MIME type of an upload when the browser does not supply one.

Browsers derive a file's type from its extension and send an empty string when
the extension is unknown to them, which is common for developer files such as
.md, .log or .yml on Linux. The upload endpoints refuse an empty type outright,
so those files could not be attached at all.

Guessing here only fills the gap; the resolved type is still checked against
``settings.ATTACHMENT_MIME_TYPES``, so the allow-list remains the control.
"""

import mimetypes
import os

# Extensions Python's mimetypes database does not map, or maps to a type the
# allow-list does not carry. Values must exist in ATTACHMENT_MIME_TYPES (or in
# ATTACHMENT_EXTRA_MIME_TYPES) to be accepted.
_EXTRA_EXTENSIONS = {
    ".log": "text/plain",
    ".env": "text/plain",
    ".conf": "text/plain",
    ".cfg": "text/plain",
    ".ini": "text/plain",
    ".toml": "text/plain",
    ".properties": "text/plain",
    ".yml": "application/x-yaml",
    ".yaml": "application/x-yaml",
    ".sql": "application/x-sql",
    ".sh": "text/x-sh",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


def resolve_mime_type(name, browser_type):
    """Return the MIME type to validate for an upload.

    The browser's value is authoritative whenever it sent one. Otherwise the
    type is derived from the file name's extension, and an empty string is
    returned when nothing can be derived, leaving the caller to refuse it.
    """
    if browser_type:
        return str(browser_type)

    if not name:
        return ""

    extension = os.path.splitext(str(name))[1].lower()
    if not extension:
        return ""

    guessed = _EXTRA_EXTENSIONS.get(extension) or mimetypes.guess_type("file" + extension)[0]
    return guessed or ""
