"""
Solana Actions/Blinks Service for Play Buni Platform

This service generates Solana Actions that enable users to execute trades
directly from signals with embedded fee collection. Integrates with Jupiter
API for optimal swap routing and automatic 1% fee collection to treasury.

Features:
- Solana Actions generation with proper metadata
- Jupiter API integration for swap execution
- Embedded fee collection (1% to treasury)
- Blink URL generation for social sharing
- Transaction monitoring and tracking
- Security validation and parameter checking
"""

import asyncio
import aiohttp
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac

from solana.publickey import PublicKey
from solana.transaction import Transaction
from solana.system_program import transfer, TransferParams
from solders.instruction import Instruction
from solders.message import Message
from solders.transaction import VersionedTransaction

from app.core.config import settings
from app.core.logging import get_logger
from app.core.cache import cache_manager
from app.core.database import get_db_session
from app.models.signals import Signal
from app.models.blinks import Blink, Trade

logger = get_logger(__name__)


class ActionType(Enum):
    """Types of Solana Actions"""

    SWAP = "swap"
    BUY = "buy"
    SELL = "sell"
    LIMIT_ORDER = "limit_order"


class BlinkStatus(Enum):
    """Status of generated Blinks"""

    ACTIVE = "active"
    EXPIRED = "expired"
    DISABLED = "disabled"
    EXECUTED = "executed"


@dataclass
class ActionMetadata:
    """Metadata for Solana Actions"""

    title: str
    description: str
    icon: str
    label: str
    disabled: bool = False
    error: Optional[str] = None


@dataclass
class ActionParameter:
    """Parameter for Solana Actions"""

    name: str
    label: str
    required: bool = True
    type: str = "text"
    pattern: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class SolanaAction:
    """Complete Solana Action specification"""

    type: str
    title: str
    icon: str
    description: str
    label: str
    links: Dict[str, Any]
    disabled: bool = False
    error: Optional[str] = None


@dataclass
class SwapQuote:
    """Jupiter swap quote information"""

    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    out_amount_with_slippage: int
    swap_mode: str
    slippage_bps: int
    platform_fee_bps: int
    price_impact_pct: float
    route_plan: List[Dict[str, Any]]
    context_slot: int
    time_taken: float


@dataclass
class TradeExecution:
    """Trade execution result"""

    transaction_id: str
    status: str
    input_amount: float
    output_amount: float
    fee_amount: float
    price_impact: float
    executed_at: datetime
    user_wallet: str
    signal_id: Optional[int]
    blink_id: Optional[int]


class SolanaActionsService:
    """
    Solana Actions/Blinks Service

    Generates executable Solana Actions with embedded fee collection
    and integrates with Jupiter API for optimal swap routing.
    """

    def __init__(self):
        self.jupiter_api_url = "https://quote-api.jup.ag/v6"
        self.treasury_wallet = settings.treasury_wallet_address
        self.platform_fee_bps = 100  # 1% fee (100 basis points)
        self.session: Optional[aiohttp.ClientSession] = None

        # Common token addresses
        self.token_addresses = {
            "SOL": "So11111111111111111111111111111111111111112",
            "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            "mSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
            "stSOL": "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "PlayBuni/1.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def validate_wallet_address(self, address: str) -> bool:
        """Validate Solana wallet address"""
        try:
            PublicKey(address)
            return True
        except Exception:
            return False

    def validate_token_address(self, address: str) -> bool:
        """Validate Solana token mint address"""
        try:
            PublicKey(address)
            return len(address) == 44  # Base58 encoded public key length
        except Exception:
            return False

    async def get_jupiter_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        user_wallet: Optional[str] = None,
    ) -> Optional[SwapQuote]:
        """Get swap quote from Jupiter API"""
        try:
            if not self.session:
                logger.error("HTTP session not initialized")
                return None

            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": str(slippage_bps),
                "platformFeeBps": str(self.platform_fee_bps),
                "maxAccounts": "64",
                "swapMode": "ExactIn",
            }

            if user_wallet:
                params["userPublicKey"] = user_wallet

            url = f"{self.jupiter_api_url}/quote"

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Jupiter API error: {response.status}")
                    return None

                data = await response.json()

                return SwapQuote(
                    input_mint=data["inputMint"],
                    output_mint=data["outputMint"],
                    in_amount=int(data["inAmount"]),
                    out_amount=int(data["outAmount"]),
                    out_amount_with_slippage=int(data["otherAmountThreshold"]),
                    swap_mode=data["swapMode"],
                    slippage_bps=int(data["slippageBps"]),
                    platform_fee_bps=int(data.get("platformFeeBps", 0)),
                    price_impact_pct=float(data.get("priceImpactPct", 0)),
                    route_plan=data.get("routePlan", []),
                    context_slot=int(data.get("contextSlot", 0)),
                    time_taken=float(data.get("timeTaken", 0)),
                )

        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return None

    async def get_jupiter_swap_transaction(
        self, quote: SwapQuote, user_wallet: str, priority_fee: int = 0
    ) -> Optional[str]:
        """Get swap transaction from Jupiter API"""
        try:
            if not self.session:
                logger.error("HTTP session not initialized")
                return None

            url = f"{self.jupiter_api_url}/swap"

            payload = {
                "quoteResponse": asdict(quote),
                "userPublicKey": user_wallet,
                "wrapAndUnwrapSol": True,
                "useSharedAccounts": True,
                "feeAccount": self.treasury_wallet,
                "prioritizationFeeLamports": priority_fee,
                "asLegacyTransaction": False,
                "useTokenLedger": False,
                "destinationTokenAccount": None,
                "dynamicComputeUnitLimit": True,
                "skipUserAccountsRpcCalls": False,
            }

            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Jupiter swap API error: {response.status}")
                    return None

                data = await response.json()
                return data.get("swapTransaction")

        except Exception as e:
            logger.error(f"Error getting Jupiter swap transaction: {e}")
            return None

    def generate_action_url(self, action_id: str, base_url: str) -> str:
        """Generate Solana Action URL"""
        return f"{base_url}/api/v1/blinks/{action_id}"

    def generate_blink_url(self, action_url: str) -> str:
        """Generate Blink URL for sharing"""
        encoded_url = base64.urlsafe_b64encode(action_url.encode()).decode().rstrip("=")
        return f"https://dial.to/?action=solana-action:{encoded_url}"

    async def create_swap_action(
        self,
        signal_id: int,
        input_token: str,
        output_token: str,
        suggested_amount: Optional[float] = None,
        max_slippage: float = 1.0,
    ) -> Optional[SolanaAction]:
        """Create a swap action from a trading signal"""
        try:
            # Get signal details
            async with get_db_session() as db:
                result = await db.execute(select(Signal).where(Signal.id == signal_id))
                signal = result.scalar_one_or_none()

                if not signal:
                    logger.error(f"Signal {signal_id} not found")
                    return None

            # Validate token addresses
            input_mint = self.token_addresses.get(input_token, input_token)
            output_mint = signal.mint_address

            if not self.validate_token_address(input_mint):
                logger.error(f"Invalid input token: {input_token}")
                return None

            if not self.validate_token_address(output_mint):
                logger.error(f"Invalid output token: {output_mint}")
                return None

            # Generate action metadata
            action_title = f"{signal.signal_type.upper()} {signal.symbol}"
            action_description = f"Execute {signal.signal_type} signal for {signal.symbol} with {signal.confidence_score:.0%} confidence"

            if signal.reasoning:
                action_description += f". {signal.reasoning[0]}"

            # Create action parameters
            parameters = [
                ActionParameter(
                    name="amount",
                    label=f"Amount ({input_token})",
                    required=True,
                    type="number",
                    min=0.001,
                    max=1000000,
                )
            ]

            if max_slippage > 0:
                parameters.append(
                    ActionParameter(
                        name="slippage",
                        label="Max Slippage (%)",
                        required=False,
                        type="number",
                        min=0.1,
                        max=10.0,
                    )
                )

            # Generate action links
            action_id = hashlib.md5(
                f"{signal_id}_{input_token}_{output_token}".encode()
            ).hexdigest()
            base_url = settings.base_url

            links = {
                "actions": [
                    {
                        "label": f"Swap {input_token} → {signal.symbol}",
                        "href": f"{base_url}/api/v1/blinks/{action_id}/execute",
                        "parameters": [param.__dict__ for param in parameters],
                    }
                ]
            }

            # Create Solana Action
            action = SolanaAction(
                type="action",
                title=action_title,
                icon=f"{base_url}/static/icons/{signal.symbol.lower()}.png",
                description=action_description,
                label=f"Trade {signal.symbol}",
                links=links,
                disabled=(
                    signal.expires_at < datetime.utcnow()
                    if signal.expires_at
                    else False
                ),
            )

            return action

        except Exception as e:
            logger.error(f"Error creating swap action: {e}")
            return None

    async def create_buy_action(
        self,
        signal_id: int,
        base_currency: str = "USDC",
        suggested_amounts: List[float] = None,
    ) -> Optional[SolanaAction]:
        """Create a buy action with preset amounts"""
        try:
            if suggested_amounts is None:
                suggested_amounts = [10, 25, 50, 100, 250]

            # Get signal details
            async with get_db_session() as db:
                result = await db.execute(select(Signal).where(Signal.id == signal_id))
                signal = result.scalar_one_or_none()

                if not signal:
                    return None

            # Generate action metadata
            action_title = f"Buy {signal.symbol}"
            action_description = f"Quick buy {signal.symbol} with preset amounts. Signal confidence: {signal.confidence_score:.0%}"

            # Create action links for different amounts
            action_id = hashlib.md5(
                f"buy_{signal_id}_{base_currency}".encode()
            ).hexdigest()
            base_url = settings.base_url

            actions = []
            for amount in suggested_amounts:
                actions.append(
                    {
                        "label": f"Buy ${amount} {signal.symbol}",
                        "href": f"{base_url}/api/v1/blinks/{action_id}/execute?amount={amount}&currency={base_currency}",
                    }
                )

            # Add custom amount option
            actions.append(
                {
                    "label": "Custom Amount",
                    "href": f"{base_url}/api/v1/blinks/{action_id}/execute",
                    "parameters": [
                        {
                            "name": "amount",
                            "label": f"Amount ({base_currency})",
                            "required": True,
                            "type": "number",
                            "min": 1,
                            "max": 10000,
                        }
                    ],
                }
            )

            links = {"actions": actions}

            action = SolanaAction(
                type="action",
                title=action_title,
                icon=f"{base_url}/static/icons/{signal.symbol.lower()}.png",
                description=action_description,
                label=f"Buy {signal.symbol}",
                links=links,
                disabled=(
                    signal.expires_at < datetime.utcnow()
                    if signal.expires_at
                    else False
                ),
            )

            return action

        except Exception as e:
            logger.error(f"Error creating buy action: {e}")
            return None

    async def execute_swap_action(
        self,
        action_id: str,
        user_wallet: str,
        amount: float,
        slippage: float = 0.5,
        priority_fee: int = 0,
    ) -> Optional[str]:
        """Execute a swap action and return transaction"""
        try:
            # Validate inputs
            if not self.validate_wallet_address(user_wallet):
                raise ValueError("Invalid wallet address")

            if amount <= 0:
                raise ValueError("Amount must be positive")

            if not (0.1 <= slippage <= 10.0):
                raise ValueError("Slippage must be between 0.1% and 10%")

            # Get action details from cache or database
            cache_key = f"action:{action_id}"
            action_data = await cache_manager.get_json(cache_key)

            if not action_data:
                logger.error(f"Action {action_id} not found")
                return None

            input_mint = action_data["input_mint"]
            output_mint = action_data["output_mint"]
            signal_id = action_data.get("signal_id")

            # Convert amount to lamports/smallest unit
            # Assuming input token has 6 decimals (USDC standard)
            amount_lamports = int(amount * 1_000_000)
            slippage_bps = int(slippage * 100)

            # Get Jupiter quote
            quote = await self.get_jupiter_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount_lamports,
                slippage_bps=slippage_bps,
                user_wallet=user_wallet,
            )

            if not quote:
                logger.error("Failed to get Jupiter quote")
                return None

            # Get swap transaction
            swap_transaction = await self.get_jupiter_swap_transaction(
                quote=quote, user_wallet=user_wallet, priority_fee=priority_fee
            )

            if not swap_transaction:
                logger.error("Failed to get swap transaction")
                return None

            # Store trade record
            await self.record_trade_execution(
                user_wallet=user_wallet,
                signal_id=signal_id,
                action_id=action_id,
                input_amount=amount,
                quote=quote,
                transaction_data=swap_transaction,
            )

            return swap_transaction

        except Exception as e:
            logger.error(f"Error executing swap action: {e}")
            return None

    async def record_trade_execution(
        self,
        user_wallet: str,
        signal_id: Optional[int],
        action_id: str,
        input_amount: float,
        quote: SwapQuote,
        transaction_data: str,
    ):
        """Record trade execution in database"""
        try:
            async with get_db_session() as db:
                # Calculate fee amount
                fee_amount = input_amount * (self.platform_fee_bps / 10000)
                output_amount = quote.out_amount / 1_000_000  # Convert from lamports

                trade_data = {
                    "user_wallet": user_wallet,
                    "signal_id": signal_id,
                    "blink_id": action_id,
                    "transaction_type": "swap",
                    "input_token": quote.input_mint,
                    "output_token": quote.output_mint,
                    "input_amount": input_amount,
                    "output_amount": output_amount,
                    "fee_amount": fee_amount,
                    "fee_percentage": self.platform_fee_bps / 100,
                    "slippage_bps": quote.slippage_bps,
                    "price_impact": quote.price_impact_pct,
                    "status": "pending",
                    "transaction_data": transaction_data,
                    "created_at": datetime.utcnow(),
                }

                result = await db.execute(
                    "INSERT INTO trades (user_wallet, signal_id, blink_id, transaction_type, "
                    "input_token, output_token, input_amount, output_amount, fee_amount, "
                    "fee_percentage, slippage_bps, price_impact, status, transaction_data, created_at) "
                    "VALUES (%(user_wallet)s, %(signal_id)s, %(blink_id)s, %(transaction_type)s, "
                    "%(input_token)s, %(output_token)s, %(input_amount)s, %(output_amount)s, "
                    "%(fee_amount)s, %(fee_percentage)s, %(slippage_bps)s, %(price_impact)s, "
                    "%(status)s, %(transaction_data)s, %(created_at)s) RETURNING id",
                    trade_data,
                )

                trade_id = result.fetchone()[0]
                await db.commit()

                logger.info(
                    f"Recorded trade execution {trade_id} for user {user_wallet}"
                )

        except Exception as e:
            logger.error(f"Error recording trade execution: {e}")

    async def save_blink_to_database(
        self, signal_id: int, action_id: str, action_data: SolanaAction, blink_url: str
    ) -> Optional[int]:
        """Save generated Blink to database"""
        try:
            async with get_db_session() as db:
                blink_data = {
                    "signal_id": signal_id,
                    "blink_id": action_id,
                    "title": action_data.title,
                    "description": action_data.description,
                    "icon_url": action_data.icon,
                    "action_url": action_data.links["actions"][0]["href"],
                    "blink_url": blink_url,
                    "action_data": asdict(action_data),
                    "status": BlinkStatus.ACTIVE.value,
                    "created_at": datetime.utcnow(),
                    "expires_at": datetime.utcnow() + timedelta(hours=24),
                }

                result = await db.execute(
                    "INSERT INTO blinks (signal_id, blink_id, title, description, icon_url, "
                    "action_url, blink_url, action_data, status, created_at, expires_at) "
                    "VALUES (%(signal_id)s, %(blink_id)s, %(title)s, %(description)s, %(icon_url)s, "
                    "%(action_url)s, %(blink_url)s, %(action_data)s, %(status)s, %(created_at)s, %(expires_at)s) "
                    "RETURNING id",
                    blink_data,
                )

                blink_id = result.fetchone()[0]
                await db.commit()

                logger.info(f"Saved Blink {blink_id} for signal {signal_id}")
                return blink_id

        except Exception as e:
            logger.error(f"Error saving Blink to database: {e}")
            return None

    async def generate_and_store_blink(
        self,
        signal_id: int,
        action_type: str = "swap",
        input_token: str = "USDC",
        suggested_amounts: Optional[List[float]] = None,
        max_slippage: float = 1.0,
    ) -> Optional[int]:
        """Generate a Blink and store it in the database.

        This helper creates the appropriate Solana action and persists the
        resulting Blink record without returning the action metadata.
        """

        async with self as service:
            if action_type.lower() == "buy":
                action = await service.create_buy_action(
                    signal_id=signal_id,
                    base_currency=input_token,
                    suggested_amounts=suggested_amounts,
                )
            else:
                # The swap action internally determines the output token from
                # the signal metadata, so we only need to provide a placeholder
                action = await service.create_swap_action(
                    signal_id=signal_id,
                    input_token=input_token,
                    output_token="placeholder",
                    max_slippage=max_slippage,
                )

            if not action:
                return None

            action_url = action.links["actions"][0]["href"]
            blink_url = service.generate_blink_url(action_url)
            action_id = action_url.split("/")[-2]

            return await service.save_blink_to_database(
                signal_id=signal_id,
                action_id=action_id,
                action_data=action,
                blink_url=blink_url,
            )

    async def get_treasury_balance(self) -> Dict[str, float]:
        """Get current treasury balance"""
        try:
            # This would integrate with Solana RPC to get actual balances
            # For now, we'll return a placeholder
            return {
                "SOL": 0.0,
                "USDC": 0.0,
                "total_fees_collected": 0.0,
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting treasury balance: {e}")
            return {}

    async def generate_action_signature(self, action_data: Dict[str, Any]) -> str:
        """Generate HMAC signature for action verification"""
        try:
            message = json.dumps(action_data, sort_keys=True)
            signature = hmac.new(
                settings.secret_key.encode(), message.encode(), hashlib.sha256
            ).hexdigest()
            return signature

        except Exception as e:
            logger.error(f"Error generating action signature: {e}")
            return ""

    async def verify_action_signature(
        self, action_data: Dict[str, Any], signature: str
    ) -> bool:
        """Verify action signature"""
        try:
            expected_signature = await self.generate_action_signature(action_data)
            return hmac.compare_digest(expected_signature, signature)

        except Exception as e:
            logger.error(f"Error verifying action signature: {e}")
            return False


# Global Solana Actions service instance
solana_actions_service = SolanaActionsService()
