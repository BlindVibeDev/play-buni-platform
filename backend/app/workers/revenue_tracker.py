"""
Revenue Tracking Worker
Background tasks for revenue tracking, fee collection, and treasury management
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from celery import shared_task

from app.workers.celery_app import celery_app
from app.core.config import settings
from app.core.logging import create_signal_logger
from app.services.treasury_manager import treasury_manager
from app.database import get_db_session
from app.models.revenue import Revenue, Transaction

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

@shared_task(bind=True, base=celery_app.Task)
def calculate_hourly_revenue(self):
    """
    Calculate and record hourly revenue from all sources
    Runs every hour via Celery Beat
    """
    try:
        logger.info("Starting hourly revenue calculation")
        
        result = asyncio.run(_calculate_hourly_revenue_async())
        
        logger.info(f"Hourly revenue calculation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Hourly revenue calculation failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _calculate_hourly_revenue_async():
    """Async implementation of hourly revenue calculation"""
    try:
        # Calculate time range for previous hour
        end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=1)
        
        revenue_data = {
            "period_start": start_time,
            "period_end": end_time,
            "blinks_fees": Decimal("0"),
            "trading_fees": Decimal("0"),
            "premium_subscriptions": Decimal("0"),
            "total_transactions": 0,
            "transaction_volume": Decimal("0"),
            "treasury_balance": Decimal("0")
        }
        
        # Calculate Blinks fees
        try:
            blinks_revenue = await _calculate_blinks_revenue(start_time, end_time)
            revenue_data.update(blinks_revenue)
        except Exception as e:
            logger.error(f"Error calculating Blinks revenue: {e}")
        
        # Calculate trading fees
        try:
            trading_revenue = await _calculate_trading_revenue(start_time, end_time)
            revenue_data.update(trading_revenue)
        except Exception as e:
            logger.error(f"Error calculating trading revenue: {e}")
        
        # Get current treasury balance
        try:
            treasury_balance = await treasury_manager.get_treasury_balance()
            revenue_data["treasury_balance"] = Decimal(str(treasury_balance))
        except Exception as e:
            logger.error(f"Error getting treasury balance: {e}")
        
        # Calculate total revenue
        total_revenue = (
            revenue_data["blinks_fees"] + 
            revenue_data["trading_fees"] + 
            revenue_data["premium_subscriptions"]
        )
        revenue_data["total_revenue"] = total_revenue
        
        # Store revenue record
        async with get_db_session() as db:
            revenue_record = Revenue(
                timestamp=end_time,
                period_type="hourly",
                blinks_fees=float(revenue_data["blinks_fees"]),
                trading_fees=float(revenue_data["trading_fees"]),
                premium_fees=float(revenue_data["premium_subscriptions"]),
                total_revenue=float(total_revenue),
                transaction_count=revenue_data["total_transactions"],
                transaction_volume=float(revenue_data["transaction_volume"]),
                treasury_balance=float(revenue_data["treasury_balance"])
            )
            db.add(revenue_record)
            await db.commit()
        
        # Log revenue summary
        signal_logger.info("hourly_revenue_calculated", extra={
            "period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "total_revenue": float(total_revenue),
            "blinks_fees": float(revenue_data["blinks_fees"]),
            "trading_fees": float(revenue_data["trading_fees"]),
            "transactions": revenue_data["total_transactions"],
            "treasury_balance": float(revenue_data["treasury_balance"])
        })
        
        return {
            "status": "success",
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "revenue": {
                "total": float(total_revenue),
                "blinks_fees": float(revenue_data["blinks_fees"]),
                "trading_fees": float(revenue_data["trading_fees"]),
                "transactions": revenue_data["total_transactions"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in hourly revenue calculation: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def process_transaction(self, transaction_data: Dict[str, Any]):
    """
    Process a single transaction and update revenue tracking
    """
    try:
        logger.info(f"Processing transaction: {transaction_data.get('signature', 'unknown')}")
        
        result = asyncio.run(_process_transaction_async(transaction_data))
        
        logger.info(f"Transaction processing completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Transaction processing failed: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _process_transaction_async(transaction_data: Dict[str, Any]):
    """Async implementation of transaction processing"""
    try:
        # Extract transaction details
        signature = transaction_data.get("signature")
        transaction_type = transaction_data.get("type", "unknown")
        amount = Decimal(str(transaction_data.get("amount", 0)))
        fee = Decimal(str(transaction_data.get("fee", 0)))
        user_id = transaction_data.get("user_id")
        token = transaction_data.get("token", "SOL")
        
        # Calculate platform fee (1% of transaction amount)
        platform_fee = amount * Decimal(str(settings.platform_fee_percentage / 100))
        
        # Store transaction record
        async with get_db_session() as db:
            transaction_record = Transaction(
                signature=signature,
                transaction_type=transaction_type,
                user_id=user_id,
                token=token,
                amount=float(amount),
                platform_fee=float(platform_fee),
                network_fee=float(fee),
                timestamp=datetime.now(timezone.utc),
                status="completed"
            )
            db.add(transaction_record)
            await db.commit()
        
        # Update treasury balance
        try:
            await treasury_manager.record_fee_collection(
                amount=float(platform_fee),
                transaction_signature=signature,
                source="blinks_transaction"
            )
        except Exception as e:
            logger.error(f"Error updating treasury balance: {e}")
        
        # Log transaction
        signal_logger.info("transaction_processed", extra={
            "signature": signature,
            "type": transaction_type,
            "amount": float(amount),
            "platform_fee": float(platform_fee),
            "token": token,
            "user_id": user_id
        })
        
        return {
            "status": "success",
            "signature": signature,
            "platform_fee": float(platform_fee),
            "amount": float(amount)
        }
        
    except Exception as e:
        logger.error(f"Error processing transaction: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def monitor_treasury_balance(self):
    """
    Monitor treasury balance and alert on significant changes
    Runs every 15 minutes
    """
    try:
        logger.info("Starting treasury balance monitoring")
        
        result = asyncio.run(_monitor_treasury_balance_async())
        
        logger.info(f"Treasury balance monitoring completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Treasury balance monitoring failed: {e}")
        self.retry(countdown=180 * (2 ** self.request.retries))

async def _monitor_treasury_balance_async():
    """Async implementation of treasury balance monitoring"""
    try:
        # Get current treasury balance
        current_balance = await treasury_manager.get_treasury_balance()
        
        # Get balance from 15 minutes ago for comparison
        comparison_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        previous_balance = await _get_historical_balance(comparison_time)
        
        # Calculate change
        balance_change = current_balance - previous_balance if previous_balance else 0
        change_percentage = (balance_change / previous_balance * 100) if previous_balance > 0 else 0
        
        # Check for significant changes (>5% or >100 SOL)
        significant_change = (
            abs(change_percentage) > 5.0 or 
            abs(balance_change) > 100.0
        )
        
        monitoring_data = {
            "current_balance": current_balance,
            "previous_balance": previous_balance,
            "balance_change": balance_change,
            "change_percentage": change_percentage,
            "significant_change": significant_change,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Alert on significant changes
        if significant_change:
            from app.workers.notification_sender import send_treasury_alert
            send_treasury_alert.delay({
                "type": "balance_change",
                "current_balance": current_balance,
                "change": balance_change,
                "change_percentage": change_percentage
            })
        
        # Log balance monitoring
        signal_logger.info("treasury_balance_monitored", extra={
            "current_balance": current_balance,
            "balance_change": balance_change,
            "change_percentage": change_percentage,
            "significant_change": significant_change
        })
        
        return {
            "status": "success",
            "balance_data": {
                "current": current_balance,
                "change": balance_change,
                "change_percentage": change_percentage,
                "significant_change": significant_change
            }
        }
        
    except Exception as e:
        logger.error(f"Error monitoring treasury balance: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def generate_revenue_report(self, period: str = "daily"):
    """
    Generate comprehensive revenue report
    """
    try:
        logger.info(f"Generating {period} revenue report")
        
        result = asyncio.run(_generate_revenue_report_async(period))
        
        logger.info(f"Revenue report generation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Revenue report generation failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _generate_revenue_report_async(period: str):
    """Async implementation of revenue report generation"""
    try:
        # Calculate date range based on period
        end_time = datetime.now(timezone.utc)
        if period == "daily":
            start_time = end_time - timedelta(days=1)
        elif period == "weekly":
            start_time = end_time - timedelta(weeks=1)
        elif period == "monthly":
            start_time = end_time - timedelta(days=30)
        else:
            raise ValueError(f"Invalid period: {period}")
        
        report_data = {
            "period": period,
            "start_time": start_time,
            "end_time": end_time,
            "summary": {},
            "breakdown": {},
            "trends": {},
            "top_performers": {}
        }
        
        # Calculate summary metrics
        async with get_db_session() as db:
            # This would involve complex revenue queries
            # For now, use placeholder data
            
            report_data["summary"] = {
                "total_revenue": 0.0,
                "blinks_revenue": 0.0,
                "trading_revenue": 0.0,
                "transaction_count": 0,
                "unique_users": 0,
                "avg_transaction_size": 0.0
            }
            
            report_data["breakdown"] = {
                "revenue_by_hour": [],
                "revenue_by_token": {},
                "revenue_by_user_tier": {}
            }
            
            report_data["trends"] = {
                "growth_rate": 0.0,
                "transaction_volume_trend": "stable",
                "user_adoption_trend": "growing"
            }
            
            report_data["top_performers"] = {
                "top_tokens": [],
                "top_users": [],
                "peak_hours": []
            }
        
        # Store report
        revenue_record = Revenue(
            timestamp=end_time,
            period_type=f"{period}_report",
            data=report_data
        )
        
        async with get_db_session() as db:
            db.add(revenue_record)
            await db.commit()
        
        # Log report generation
        signal_logger.info("revenue_report_generated", extra={
            "period": period,
            "total_revenue": report_data["summary"]["total_revenue"],
            "transaction_count": report_data["summary"]["transaction_count"]
        })
        
        return {
            "status": "success",
            "period": period,
            "report_summary": report_data["summary"]
        }
        
    except Exception as e:
        logger.error(f"Error generating revenue report: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def reconcile_treasury_transactions(self):
    """
    Reconcile treasury transactions with blockchain records
    Runs daily for data integrity
    """
    try:
        logger.info("Starting treasury transaction reconciliation")
        
        result = asyncio.run(_reconcile_treasury_transactions_async())
        
        logger.info(f"Treasury reconciliation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Treasury reconciliation failed: {e}")
        self.retry(countdown=600 * (2 ** self.request.retries))

async def _reconcile_treasury_transactions_async():
    """Async implementation of treasury transaction reconciliation"""
    try:
        reconciliation_results = {
            "transactions_checked": 0,
            "discrepancies_found": 0,
            "corrections_made": 0,
            "total_fees_verified": 0.0
        }
        
        # Get transactions from last 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        async with get_db_session() as db:
            # This would involve querying our transaction records
            # and comparing with blockchain data
            
            # For now, use placeholder reconciliation logic
            reconciliation_results["transactions_checked"] = 0
            reconciliation_results["discrepancies_found"] = 0
            reconciliation_results["corrections_made"] = 0
        
        # Log reconciliation results
        signal_logger.info("treasury_reconciliation", extra=reconciliation_results)
        
        return {
            "status": "success",
            "results": reconciliation_results
        }
        
    except Exception as e:
        logger.error(f"Error in treasury reconciliation: {e}")
        raise

# Helper functions
async def _calculate_blinks_revenue(start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Calculate revenue from Blinks transactions"""
    try:
        async with get_db_session() as db:
            # Query Blinks transactions in time range
            # This would be actual database queries in production
            
            return {
                "blinks_fees": Decimal("0"),
                "blinks_transactions": 0,
                "blinks_volume": Decimal("0")
            }
            
    except Exception as e:
        logger.error(f"Error calculating Blinks revenue: {e}")
        return {
            "blinks_fees": Decimal("0"),
            "blinks_transactions": 0,
            "blinks_volume": Decimal("0")
        }

async def _calculate_trading_revenue(start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Calculate revenue from trading fees"""
    try:
        async with get_db_session() as db:
            # Query trading transactions in time range
            # This would be actual database queries in production
            
            return {
                "trading_fees": Decimal("0"),
                "trading_transactions": 0,
                "trading_volume": Decimal("0")
            }
            
    except Exception as e:
        logger.error(f"Error calculating trading revenue: {e}")
        return {
            "trading_fees": Decimal("0"),
            "trading_transactions": 0,
            "trading_volume": Decimal("0")
        }

async def _get_historical_balance(timestamp: datetime) -> float:
    """Get treasury balance at a specific timestamp"""
    try:
        async with get_db_session() as db:
            # Query historical balance record closest to timestamp
            # This would be actual database query in production
            
            return 0.0  # Placeholder
            
    except Exception as e:
        logger.error(f"Error getting historical balance: {e}")
        return 0.0 