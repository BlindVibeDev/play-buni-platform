#!/usr/bin/env python3
"""Simple deployment tests for Play Buni Platform."""

import logging

from app.core.config import settings
from app.core.logging import configure_logging
from app.services.jupiter_service import JupiterQuoteRequest

configure_logging()
logger = logging.getLogger(__name__)


def test_configuration_loads() -> None:
    """Ensure required settings values are present."""
    required = ["app_name", "secret_key", "database_url", "redis_url"]
    for key in required:
        assert getattr(settings, key)


def test_jupiter_quote_request() -> None:
    """Jupiter quote request dataclass works with sample data."""
    qr = JupiterQuoteRequest(
        inputMint="So11111111111111111111111111111111111111112",
        outputMint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        amount=1_000_000_000,
        slippageBps=50,
    )
    assert qr.amount == 1_000_000_000


if __name__ == "__main__":
    print("Run 'pytest' to execute tests.")
