"""Structured logging setup."""

from __future__ import annotations

import logging as stdlib_logging
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure stdlib logging and structlog consistently."""

    stdlib_logging.basicConfig(
        level=getattr(stdlib_logging, level.upper(), stdlib_logging.INFO),
        format="%(message)s",
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(stdlib_logging, level.upper(), stdlib_logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
