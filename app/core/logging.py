"""
Structured logging setup using structlog.
Provides consistent, JSON-formatted logging for production use.
"""

import logging
import sys
from typing import Any

import structlog


def setup_logging(debug: bool = False) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        debug: Enable debug level logging if True.
    """
    # Set log level based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog processors
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Use different renderer based on debug mode
    if debug:
        # Pretty console output for development
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    else:
        # JSON output for production
        processors.append(
            structlog.processors.JSONRenderer()
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__ of the module).
        
    Returns:
        Structured logger instance.
    """
    return structlog.get_logger(name)
