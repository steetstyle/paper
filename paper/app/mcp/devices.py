"""Device selection for the MCP tools: the domain rule plus the tool help text.

The rule itself lives in :mod:`app.domain.devices`, shared with the CLI and the
HTTP layer. A device is a property of one call, not of the process: the provider
cache is keyed by ``(space, device)``, so asking for ``cuda`` on one bulk ingest
loads a second copy of the model onto the accelerator for that run and leaves
every other call on the configured device. That separation is the point — an
interactive query should not pin a model's worth of VRAM, but a batch job should
be allowed to borrow the card.
"""

from __future__ import annotations

from app.domain.devices import (
    ACCEPTED_DEVICES,
    DEVICES,
    INDEXED_DEVICES,
    DeviceError,
    parse_device,
)

DEVICE_HELP = (
    "Compute device for this call: cpu, cuda, cuda:1, mps or npu. Defaults to "
    "the configured device (cpu). Worth setting only for bulk work: it loads a "
    "second copy of the model onto the accelerator for this call, which is the "
    "trade being made, and a single short query is faster on CPU anyway."
)
"""One description, reused, so three tools cannot describe it three ways."""


__all__ = [
    "ACCEPTED_DEVICES",
    "DEVICES",
    "DEVICE_HELP",
    "INDEXED_DEVICES",
    "DeviceError",
    "parse_device",
]
