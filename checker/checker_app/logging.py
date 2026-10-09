"""Console/JSON logging with a run id, shared with the paper-app conventions."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def new_id() -> str:
    return uuid.uuid4().hex


def bind_run_id(value: str | None = None) -> str:
    rid = value or new_id()
    _run_id.set(rid)
    return rid


def current_run_id() -> str | None:
    return _run_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if _run_id.get():
            payload["run_id"] = _run_id.get()
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        color = self.COLORS.get(record.levelname, "")
        rid = _run_id.get() or ""
        suffix = f" run={rid[:8]}" if rid else ""
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        extra = " " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        head = f"{color}{ts} {record.levelname:<7}{self.RESET} {record.name:<26}{suffix}"
        line = f"{head} {record.getMessage()}{extra}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class SafeLoggerAdapter(logging.LoggerAdapter):
    """Drop reserved keys from ``extra`` instead of raising ``KeyError``."""

    def process(  # type: ignore[override]
        self, msg: Any, kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        extra = kwargs.get("extra")
        if not isinstance(extra, dict):
            return msg, kwargs
        safe = {k: v for k, v in extra.items() if k not in _RESERVED}
        if len(safe) != len(extra):
            kwargs["extra"] = safe
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    return SafeLoggerAdapter(logging.getLogger(name), {})
