"""Logging helpers for Play Buni Platform."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import structlog

from .config import settings


def configure_logging() -> None:
    """Set up standard and structured logging."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if settings.environment == "production"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a configured logger."""
    return structlog.get_logger(name)


def create_signal_logger() -> structlog.stdlib.BoundLogger:
    """Dedicated logger for trading signals."""
    return get_logger("signal")


def log_signal_event(
    logger: structlog.stdlib.BoundLogger,
    event_type: str,
    signal_data: Dict[str, Any],
    **kwargs: Any,
) -> None:
    """Log signal-related events with structured data."""
    logger.info(
        "signal_event",
        event_type=event_type,
        token_symbol=signal_data.get("token_symbol"),
        signal_type=signal_data.get("signal_type"),
        confidence=signal_data.get("confidence"),
        **kwargs,
    )


def log_nft_verification(
    logger: structlog.stdlib.BoundLogger,
    wallet_address: str,
    verification_result: bool,
    **kwargs: Any,
) -> None:
    """Log NFT verification events."""
    logger.info(
        "nft_verification",
        wallet_address=wallet_address,
        verified=verification_result,
        **kwargs,
    )


def log_trade_execution(
    logger: structlog.stdlib.BoundLogger,
    trade_data: Dict[str, Any],
    **kwargs: Any,
) -> None:
    """Log trade execution events."""
    logger.info(
        "trade_execution",
        token_symbol=trade_data.get("token_symbol"),
        trade_type=trade_data.get("trade_type"),
        amount=trade_data.get("amount"),
        fee_collected=trade_data.get("fee_collected"),
        **kwargs,
    )
