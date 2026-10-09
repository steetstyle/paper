"""What a device string may say, decided once.

Three surfaces accept a device — the CLI, the HTTP API and the MCP server — and a
caller who is told three different things will eventually send the string one of
them rejects. Worse, a rule that lives in three places drifts: the parity test in
``tests/test_mcp_device.py`` compares all three and failed the first time it ran,
on nothing but the wording of the accepted set.

So the rule is here, in the domain, and each surface is a thin adapter over it:
``paper_app/api/devices.py`` adds the pydantic wrapper, ``paper_app/mcp/devices.py`` the tool
help text, and ``paper_app/cli.py`` the Typer parameter.

The name is checked here rather than handed to torch, because torch's own
rejection of ``"cud"`` arrives only once the weights are being moved — after the
model is loaded, and on a real model with the accelerator already half occupied.
An error that lists the valid values costs nothing and says what to type instead.
"""

from __future__ import annotations

DEVICES: tuple[str, ...] = ("cpu", "cuda", "mps", "npu")
"""Accelerator names, each usable bare."""

INDEXED_DEVICES: tuple[str, ...] = ("cuda", "mps", "npu")
"""Of those, the ones that take an index. ``cpu`` has none: ``cpu:0`` is not a
device torch can be asked for, and accepting it would only defer the error."""

ACCEPTED_DEVICES = (
    f"{', '.join(DEVICES)}, or an index for cuda/mps/npu (cuda:1), or a bare index (1)"
)


class DeviceError(ValueError):
    """An unrecognised device. A ``ValueError`` so callers that already catch one
    for their own parsing keep working."""


def parse_device(value: object) -> str | None:
    """Normalise and validate a device, or raise :class:`DeviceError`.

    ``None`` means "no opinion" — the configured device, which is what every
    existing caller passes and therefore what keeps working unchanged.

    An empty string is refused rather than read as "no opinion": a caller that
    sends one has asked about a device, and quietly answering with the configured
    one hides the mistake behind a working call.

    Case and surrounding whitespace are ignored, a bare index is accepted as an
    index, and anything valid is passed through in lower case so ``cuda:1``
    reaches torch as typed.
    """
    if value is None:
        return None
    # bool before int: `device=True` is a wrong-shaped argument, not device 1.
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DeviceError(
            f"device must be a string, not {type(value).__name__}; choose {ACCEPTED_DEVICES}"
        )
    text = str(value).strip().lower()
    if not text:
        raise DeviceError(f"device is empty; choose {ACCEPTED_DEVICES}")
    head, separator, index = text.partition(":")
    ok = (head in DEVICES or head.isdigit()) if not separator else (
        head in INDEXED_DEVICES and index.isdigit()
    )
    if not ok:
        raise DeviceError(f"unknown device {str(value)!r}; choose {ACCEPTED_DEVICES}")
    return text


__all__ = [
    "ACCEPTED_DEVICES",
    "DEVICES",
    "INDEXED_DEVICES",
    "DeviceError",
    "parse_device",
]
