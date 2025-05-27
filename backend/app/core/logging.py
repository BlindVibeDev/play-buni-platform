import structlog
import logging
import sys
from typing import Any, Dict
from .config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.environment == "production" 
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


def log_signal_event(
    logger: structlog.stdlib.BoundLogger,
    event_type: str,
    signal_data: Dict[str, Any],
    **kwargs
) -> None:
    """Log signal-related events with structured data."""
    logger.info(
        "signal_event",
        event_type=event_type,
        token_symbol=signal_data.get("token_symbol"),
        signal_type=signal_data.get("signal_type"),
        confidence=signal_data.get("confidence"),
        **kwargs
    )


def log_nft_verification(
    logger: structlog.stdlib.BoundLogger,
    wallet_address: str,
    verification_result: bool,
    **kwargs
) -> None:
    """Log NFT verification events."""
    logger.info(
        "nft_verification",
        wallet_address=wallet_address,
        verified=verification_result,
        **kwargs
    )


def log_trade_execution(
    logger: structlog.stdlib.BoundLogger,
    trade_data: Dict[str, Any],
    **kwargs
) -> None:
    """Log trade execution events."""
    logger.info(
        "trade_execution",
        token_symbol=trade_data.get("token_symbol"),
        trade_type=trade_data.get("trade_type"),
        amount=trade_data.get("amount"),
        fee_collected=trade_data.get("fee_collected"),
        **kwargs
    ) 