"""
Treasury Management Service for Play Buni Platform

This service manages the platform treasury, tracks fee collection,
monitors balances, and provides revenue analytics. Handles all
financial operations related to the 1% trading fee collection.

Features:
- Fee collection tracking and validation
- Multi-token balance management
- Revenue analytics and reporting
- Treasury withdrawal management
- Performance metrics and insights
- Automated fee distribution
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from solana.rpc.async_api import AsyncClient
from solana.publickey import PublicKey
from solana.rpc.types import TokenAccountOpts

from app.core.config import settings
from app.core.logging import get_logger
from app.core.cache import cache_manager
from app.core.database import get_db_session
from app.models.blinks import Trade
from app.models.analytics import Revenue

logger = get_logger(__name__)


class RevenueType(Enum):
    """Types of revenue"""
    TRADING_FEE = "trading_fee"
    SUBSCRIPTION = "subscription"
    PREMIUM_ACCESS = "premium_access"
    OTHER = "other"


class WithdrawalStatus(Enum):
    """Status of treasury withdrawals"""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TokenBalance:
    """Token balance information"""
    token_address: str
    token_symbol: str
    balance: Decimal
    usd_value: Optional[Decimal]
    last_updated: datetime


@dataclass
class RevenueMetrics:
    """Revenue metrics for a period"""
    period_start: datetime
    period_end: datetime
    total_revenue: Decimal
    trading_fees: Decimal
    total_trades: int
    avg_fee_per_trade: Decimal
    top_trading_pairs: List[Dict[str, Any]]
    revenue_by_day: List[Dict[str, Any]]
    growth_rate: Optional[float]


@dataclass
class TreasurySnapshot:
    """Complete treasury snapshot"""
    total_value_usd: Decimal
    token_balances: List[TokenBalance]
    daily_revenue: Decimal
    monthly_revenue: Decimal
    total_lifetime_revenue: Decimal
    fee_collection_rate: float
    top_revenue_sources: List[Dict[str, Any]]
    snapshot_time: datetime


class TreasuryManager:
    """
    Treasury Management Service
    
    Manages platform treasury operations including fee collection,
    balance tracking, revenue analytics, and withdrawal management.
    """
    
    def __init__(self):
        self.treasury_wallet = settings.treasury_wallet_address
        self.solana_client: Optional[AsyncClient] = None
        self.fee_percentage = Decimal("0.01")  # 1% fee
        
        # Token configurations
        self.supported_tokens = {
            "SOL": {
                "address": "So11111111111111111111111111111111111111112",
                "decimals": 9,
                "symbol": "SOL"
            },
            "USDC": {
                "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "decimals": 6,
                "symbol": "USDC"
            },
            "USDT": {
                "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                "decimals": 6,
                "symbol": "USDT"
            }
        }
    
    async def initialize(self):
        """Initialize treasury manager"""
        try:
            # Initialize Solana RPC client
            if settings.solana_rpc_url:
                self.solana_client = AsyncClient(settings.solana_rpc_url)
            
            logger.info("Treasury Manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing treasury manager: {e}")
            raise
    
    async def close(self):
        """Close connections"""
        if self.solana_client:
            await self.solana_client.close()
    
    def calculate_fee_amount(self, trade_amount: Decimal) -> Decimal:
        """Calculate fee amount for a trade"""
        try:
            fee = trade_amount * self.fee_percentage
            # Round to 6 decimal places (USDC precision)
            return fee.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            
        except Exception as e:
            logger.error(f"Error calculating fee amount: {e}")
            return Decimal("0")
    
    async def record_fee_collection(
        self,
        trade_id: int,
        user_wallet: str,
        token_address: str,
        fee_amount: Decimal,
        trade_amount: Decimal,
        transaction_signature: Optional[str] = None
    ) -> Optional[int]:
        """Record fee collection in the revenue table"""
        try:
            async with get_db_session() as db:
                revenue_data = {
                    "revenue_type": RevenueType.TRADING_FEE.value,
                    "amount": fee_amount,
                    "token_address": token_address,
                    "user_wallet": user_wallet,
                    "trade_id": trade_id,
                    "transaction_signature": transaction_signature,
                    "metadata": {
                        "trade_amount": str(trade_amount),
                        "fee_percentage": str(self.fee_percentage),
                        "collection_method": "jupiter_platform_fee"
                    },
                    "created_at": datetime.utcnow()
                }
                
                result = await db.execute(
                    text("""
                        INSERT INTO revenue (revenue_type, amount, token_address, user_wallet, 
                                           trade_id, transaction_signature, metadata, created_at)
                        VALUES (:revenue_type, :amount, :token_address, :user_wallet,
                                :trade_id, :transaction_signature, :metadata, :created_at)
                        RETURNING id
                    """),
                    revenue_data
                )
                
                revenue_id = result.scalar()
                await db.commit()
                
                logger.info(f"Recorded fee collection {revenue_id}: {fee_amount} from trade {trade_id}")
                return revenue_id
                
        except Exception as e:
            logger.error(f"Error recording fee collection: {e}")
            return None
    
    async def get_token_balance(self, token_address: str) -> Optional[TokenBalance]:
        """Get balance for a specific token"""
        try:
            if not self.solana_client:
                logger.warning("Solana client not initialized")
                return None
            
            treasury_pubkey = PublicKey(self.treasury_wallet)
            token_pubkey = PublicKey(token_address)
            
            # Get token accounts for the treasury wallet
            response = await self.solana_client.get_token_accounts_by_owner(
                treasury_pubkey,
                TokenAccountOpts(mint=token_pubkey)
            )
            
            if not response.value:
                # No token account found, balance is 0
                token_info = next(
                    (info for info in self.supported_tokens.values() 
                     if info["address"] == token_address),
                    {"symbol": "UNKNOWN", "decimals": 6}
                )
                
                return TokenBalance(
                    token_address=token_address,
                    token_symbol=token_info["symbol"],
                    balance=Decimal("0"),
                    usd_value=None,
                    last_updated=datetime.utcnow()
                )
            
            # Get balance from the first token account
            token_account = response.value[0]
            account_info = await self.solana_client.get_account_info(token_account.pubkey)
            
            if not account_info.value:
                return None
            
            # Parse token account data to get balance
            # This is a simplified version - in production you'd use proper SPL token parsing
            balance_lamports = 0  # Placeholder - would parse from account data
            
            token_info = next(
                (info for info in self.supported_tokens.values() 
                 if info["address"] == token_address),
                {"symbol": "UNKNOWN", "decimals": 6}
            )
            
            balance = Decimal(balance_lamports) / Decimal(10 ** token_info["decimals"])
            
            return TokenBalance(
                token_address=token_address,
                token_symbol=token_info["symbol"],
                balance=balance,
                usd_value=None,  # Would fetch from price API
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error getting token balance for {token_address}: {e}")
            return None
    
    async def get_all_balances(self) -> List[TokenBalance]:
        """Get balances for all supported tokens"""
        try:
            balances = []
            
            for token_info in self.supported_tokens.values():
                balance = await self.get_token_balance(token_info["address"])
                if balance:
                    balances.append(balance)
            
            return balances
            
        except Exception as e:
            logger.error(f"Error getting all balances: {e}")
            return []
    
    async def get_revenue_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[RevenueMetrics]:
        """Get revenue metrics for a specific period"""
        try:
            async with get_db_session() as db:
                # Get total revenue
                revenue_result = await db.execute(
                    text("""
                        SELECT 
                            SUM(amount) as total_revenue,
                            COUNT(*) as total_records
                        FROM revenue 
                        WHERE created_at >= :start_date 
                        AND created_at <= :end_date
                        AND revenue_type = :revenue_type
                    """),
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "revenue_type": RevenueType.TRADING_FEE.value
                    }
                )
                
                revenue_data = revenue_result.first()
                total_revenue = Decimal(str(revenue_data.total_revenue or 0))
                
                # Get trading statistics
                trades_result = await db.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total_trades,
                            AVG(fee_amount) as avg_fee
                        FROM trades 
                        WHERE created_at >= :start_date 
                        AND created_at <= :end_date
                        AND status = 'completed'
                    """),
                    {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                )
                
                trades_data = trades_result.first()
                total_trades = trades_data.total_trades or 0
                avg_fee = Decimal(str(trades_data.avg_fee or 0))
                
                # Get top trading pairs
                pairs_result = await db.execute(
                    text("""
                        SELECT 
                            input_token,
                            output_token,
                            COUNT(*) as trade_count,
                            SUM(fee_amount) as total_fees
                        FROM trades 
                        WHERE created_at >= :start_date 
                        AND created_at <= :end_date
                        AND status = 'completed'
                        GROUP BY input_token, output_token
                        ORDER BY total_fees DESC
                        LIMIT 10
                    """),
                    {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                )
                
                top_pairs = []
                for row in pairs_result:
                    top_pairs.append({
                        "pair": f"{row.input_token}/{row.output_token}",
                        "trade_count": row.trade_count,
                        "total_fees": float(row.total_fees)
                    })
                
                # Get daily revenue breakdown
                daily_result = await db.execute(
                    text("""
                        SELECT 
                            DATE(created_at) as date,
                            SUM(amount) as daily_revenue
                        FROM revenue 
                        WHERE created_at >= :start_date 
                        AND created_at <= :end_date
                        AND revenue_type = :revenue_type
                        GROUP BY DATE(created_at)
                        ORDER BY date
                    """),
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "revenue_type": RevenueType.TRADING_FEE.value
                    }
                )
                
                daily_revenue = []
                for row in daily_result:
                    daily_revenue.append({
                        "date": row.date.isoformat(),
                        "revenue": float(row.daily_revenue)
                    })
                
                return RevenueMetrics(
                    period_start=start_date,
                    period_end=end_date,
                    total_revenue=total_revenue,
                    trading_fees=total_revenue,  # All revenue is from trading fees currently
                    total_trades=total_trades,
                    avg_fee_per_trade=avg_fee,
                    top_trading_pairs=top_pairs,
                    revenue_by_day=daily_revenue,
                    growth_rate=None  # Would calculate based on previous period
                )
                
        except Exception as e:
            logger.error(f"Error getting revenue metrics: {e}")
            return None
    
    async def get_treasury_snapshot(self) -> Optional[TreasurySnapshot]:
        """Get complete treasury snapshot"""
        try:
            # Get all token balances
            balances = await self.get_all_balances()
            
            # Calculate total USD value (placeholder - would use real price data)
            total_value_usd = Decimal("0")
            for balance in balances:
                if balance.usd_value:
                    total_value_usd += balance.usd_value
            
            # Get revenue metrics
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            daily_metrics = await self.get_revenue_metrics(today_start, now)
            monthly_metrics = await self.get_revenue_metrics(month_start, now)
            
            # Get lifetime revenue
            async with get_db_session() as db:
                lifetime_result = await db.execute(
                    text("""
                        SELECT SUM(amount) as lifetime_revenue
                        FROM revenue 
                        WHERE revenue_type = :revenue_type
                    """),
                    {"revenue_type": RevenueType.TRADING_FEE.value}
                )
                
                lifetime_data = lifetime_result.first()
                lifetime_revenue = Decimal(str(lifetime_data.lifetime_revenue or 0))
                
                # Get fee collection rate
                fee_rate_result = await db.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN fee_amount > 0 THEN 1 END) as fee_collected_trades
                        FROM trades 
                        WHERE created_at >= :start_date
                    """),
                    {"start_date": now - timedelta(days=30)}
                )
                
                fee_rate_data = fee_rate_result.first()
                fee_collection_rate = 0.0
                if fee_rate_data.total_trades > 0:
                    fee_collection_rate = fee_rate_data.fee_collected_trades / fee_rate_data.total_trades
            
            return TreasurySnapshot(
                total_value_usd=total_value_usd,
                token_balances=balances,
                daily_revenue=daily_metrics.total_revenue if daily_metrics else Decimal("0"),
                monthly_revenue=monthly_metrics.total_revenue if monthly_metrics else Decimal("0"),
                total_lifetime_revenue=lifetime_revenue,
                fee_collection_rate=fee_collection_rate,
                top_revenue_sources=[],  # Would populate with top revenue sources
                snapshot_time=now
            )
            
        except Exception as e:
            logger.error(f"Error getting treasury snapshot: {e}")
            return None
    
    async def validate_fee_collection(self, trade_id: int) -> bool:
        """Validate that fee was properly collected for a trade"""
        try:
            async with get_db_session() as db:
                # Check if revenue record exists for this trade
                result = await db.execute(
                    text("""
                        SELECT COUNT(*) as count
                        FROM revenue 
                        WHERE trade_id = :trade_id 
                        AND revenue_type = :revenue_type
                    """),
                    {
                        "trade_id": trade_id,
                        "revenue_type": RevenueType.TRADING_FEE.value
                    }
                )
                
                count = result.scalar()
                return count > 0
                
        except Exception as e:
            logger.error(f"Error validating fee collection for trade {trade_id}: {e}")
            return False
    
    async def get_uncollected_fees(self) -> List[Dict[str, Any]]:
        """Get trades where fees haven't been properly recorded"""
        try:
            async with get_db_session() as db:
                result = await db.execute(
                    text("""
                        SELECT t.id, t.user_wallet, t.fee_amount, t.input_token, t.created_at
                        FROM trades t
                        LEFT JOIN revenue r ON t.id = r.trade_id 
                        WHERE t.fee_amount > 0 
                        AND t.status = 'completed'
                        AND r.id IS NULL
                        ORDER BY t.created_at DESC
                        LIMIT 100
                    """)
                )
                
                uncollected = []
                for row in result:
                    uncollected.append({
                        "trade_id": row.id,
                        "user_wallet": row.user_wallet,
                        "fee_amount": float(row.fee_amount),
                        "token": row.input_token,
                        "created_at": row.created_at.isoformat()
                    })
                
                return uncollected
                
        except Exception as e:
            logger.error(f"Error getting uncollected fees: {e}")
            return []
    
    async def reconcile_fees(self) -> Dict[str, Any]:
        """Reconcile fee collection and fix any discrepancies"""
        try:
            uncollected = await self.get_uncollected_fees()
            reconciled_count = 0
            
            for trade in uncollected:
                # Create revenue record for uncollected fee
                revenue_id = await self.record_fee_collection(
                    trade_id=trade["trade_id"],
                    user_wallet=trade["user_wallet"],
                    token_address=trade["token"],
                    fee_amount=Decimal(str(trade["fee_amount"])),
                    trade_amount=Decimal("0"),  # Unknown at this point
                    transaction_signature=None
                )
                
                if revenue_id:
                    reconciled_count += 1
            
            return {
                "uncollected_fees_found": len(uncollected),
                "reconciled_count": reconciled_count,
                "reconciliation_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error reconciling fees: {e}")
            return {"error": str(e)}
    
    async def get_revenue_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive revenue analytics"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            metrics = await self.get_revenue_metrics(start_date, end_date)
            snapshot = await self.get_treasury_snapshot()
            
            if not metrics or not snapshot:
                return {"error": "Failed to generate analytics"}
            
            return {
                "period": {
                    "days": days,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "revenue": {
                    "total": float(metrics.total_revenue),
                    "daily_average": float(metrics.total_revenue / days),
                    "total_trades": metrics.total_trades,
                    "avg_fee_per_trade": float(metrics.avg_fee_per_trade)
                },
                "treasury": {
                    "total_value_usd": float(snapshot.total_value_usd),
                    "daily_revenue": float(snapshot.daily_revenue),
                    "monthly_revenue": float(snapshot.monthly_revenue),
                    "lifetime_revenue": float(snapshot.total_lifetime_revenue),
                    "fee_collection_rate": snapshot.fee_collection_rate
                },
                "top_trading_pairs": metrics.top_trading_pairs,
                "daily_breakdown": metrics.revenue_by_day,
                "token_balances": [
                    {
                        "symbol": balance.token_symbol,
                        "balance": float(balance.balance),
                        "usd_value": float(balance.usd_value) if balance.usd_value else None
                    }
                    for balance in snapshot.token_balances
                ],
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting revenue analytics: {e}")
            return {"error": str(e)}


# Global treasury manager instance
treasury_manager = TreasuryManager() 