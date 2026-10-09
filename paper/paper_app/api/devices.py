"""Device selection for the HTTP layer: the domain rule plus a pydantic wrapper.

The rule itself lives in :mod:`paper_app.domain.devices`, shared with the CLI and the
MCP server so three surfaces cannot drift into three different answers. All this
adds is the request-body form of it.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator

from paper_app.domain.devices import (
    ACCEPTED_DEVICES,
    DEVICES,
    INDEXED_DEVICES,
    DeviceError,
    parse_device,
)

DeviceField = Annotated[str | None, BeforeValidator(parse_device)]
"""Request-body form: an invalid value is a validation error, i.e. a 422.

``BeforeValidator`` rather than a pydantic pattern, because the rule is the
domain's and duplicating it as a regex is how two implementations start to
disagree.
"""


__all__ = [
    "ACCEPTED_DEVICES",
    "DEVICES",
    "INDEXED_DEVICES",
    "DeviceError",
    "DeviceField",
    "parse_device",
]
