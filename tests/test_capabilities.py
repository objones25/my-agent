"""Contract tests for `my_agent.capabilities`.

The allowlist is defined by subtraction and the permission rules travel through
deepagents' private `_permissions`. Both are assumptions about a library we do
not control, so both are stated here as well as at import time.
"""

from __future__ import annotations

from typing import get_args

from deepagents import FilesystemPermission, FsToolName

from my_agent.capabilities import (
    DEFAULT_FILESYSTEM_TOOLS,
    SHELL_TOOL_NAME,
    least_privilege_filesystem,
)


def test_allowlist_is_every_filesystem_tool_except_the_shell() -> None:
    """If deepagents adds a filesystem tool, the import-time check forces a
    deliberate decision instead of granting it by default."""
    assert set(DEFAULT_FILESYSTEM_TOOLS) == set(get_args(FsToolName)) - {SHELL_TOOL_NAME}
    assert SHELL_TOOL_NAME not in DEFAULT_FILESYSTEM_TOOLS


def test_least_privilege_middleware_withholds_the_shell_tool() -> None:
    """The allowlist is only a request until the built middleware is read back."""
    middleware = least_privilege_filesystem(None)

    granted = {getattr(t, "name", None) for t in middleware.tools}

    assert granted != set()
    assert SHELL_TOOL_NAME not in granted


def test_permissions_reach_the_filesystem_middleware(deny_secrets: FilesystemPermission) -> None:
    """`permissions` only takes effect through FilesystemMiddleware's private
    `_permissions`. build_agent installs that middleware, so it must forward
    them or every rule is silently lost."""
    middleware = least_privilege_filesystem([deny_secrets])

    assert middleware._permissions == [deny_secrets]


def test_no_permissions_still_produces_a_usable_middleware() -> None:
    """`None` normalises to an empty rule list, not a missing attribute."""
    assert least_privilege_filesystem(None)._permissions == []
