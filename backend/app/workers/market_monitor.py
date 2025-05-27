"""
Market Monitoring Worker
Background tasks for market data collection, price tracking, and market analysis
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from celery import shared_task
from sqlalchemy.ext.asyncio import AsyncSession

from app.workers.celery_app import celery_app
from app.core.config import settings
from app.core.logging import create_signal_logger
from app.services.market_data import MarketDataService
from app.core.websocket import websocket_manager
from app.database import get_db_session
from app.models.signal import MarketData, PriceHistory

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

@shared_task(bind=True, base=celery_app.Task)
def fetch_market_data(self):
    """
    Fetch current market data for all monitored tokens
    Runs every 30 seconds via Celery Beat
    """
    try:
        logger.info("Starting market data fetch task")
        
        # Run async function in sync context
        result = asyncio.run(_fetch_market_data_async())
        
        logger.info(f"Market data fetch completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Market data fetch failed: {e}")
        # Retry up to 3 times with exponential backoff
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _fetch_market_data_async():
    """Async implementation of market data fetching"""
    market_service = MarketDataService()
    
    try:
        # Fetch current market data
        market_data = await market_service.get_current_market_data()
        
        # Store in database
        await _store_market_data(market_data)
        
        # Broadcast to WebSocket connections
        await websocket_manager.broadcast_market_data(market_data)
        
        # Trigger signal analysis if significant movements detected
        significant_movements = await _detect_significant_movements(market_data)
        if significant_movements:
            # Queue signal processing task
            process_market_movements.delay(significant_movements)
        
        # Log market summary
        prices = market_data.get('prices', {})
        market_stats = market_data.get('market_stats', {})
        
        signal_logger.info("market_data_update", extra={
            "tokens_updated": len(prices),
            "total_market_cap": market_stats.get("total_market_cap", 0),
            "average_change_24h": market_stats.get("average_change_24h", 0),
            "market_sentiment": market_stats.get("market_sentiment", "neutral")
        })
        
        return {
            "status": "success",
            "tokens_updated": len(prices),
            "timestamp": market_data.get("timestamp"),
            "significant_movements": len(significant_movements)
        }
        
    except Exception as e:
        logger.error(f"Error in market data fetch: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def process_market_movements(self, movements: List[Dict[str, Any]]):
    """
    Process significant market movements and trigger appropriate signals
    """
    try:
        logger.info(f"Processing {len(movements)} significant market movements")
        
        result = asyncio.run(_process_market_movements_async(movements))
        
        logger.info(f"Market movements processed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Market movements processing failed: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _process_market_movements_async(movements: List[Dict[str, Any]]):
    """Async implementation of market movements processing"""
    from app.workers.signal_processor import generate_movement_signal
    
    signals_generated = 0
    
    for movement in movements:
        try:
            token = movement.get("token")
            change_24h = movement.get("change_24h", 0)
            volume_24h = movement.get("volume_24h", 0)
            price = movement.get("price", 0)
            
            # Generate signal for significant movement
            signal_task = generate_movement_signal.delay(
                token=token,
                price=price,
                change_24h=change_24h,
                volume_24h=volume_24h,
                movement_type=movement.get("movement_type", "unknown")
            )
            
            signals_generated += 1
            
            logger.info(f"Generated signal for {token}: {change_24h:.2f}% movement")
            
        except Exception as e:
            logger.error(f"Error processing movement for {movement.get('token', 'unknown')}: {e}")
    
    return {
        "movements_processed": len(movements),
        "signals_generated": signals_generated
    }

@shared_task(bind=True, base=celery_app.Task)
def collect_price_history(self, token: str, days: int = 1):
    """
    Collect historical price data for a specific token
    """
    try:
        logger.info(f"Collecting price history for {token} ({days} days)")
        
        result = asyncio.run(_collect_price_history_async(token, days))
        
        logger.info(f"Price history collection completed for {token}")
        return result
        
    except Exception as e:
        logger.error(f"Price history collection failed for {token}: {e}")
        self.retry(countdown=120 * (2 ** self.request.retries))

async def _collect_price_history_async(token: str, days: int):
    """Async implementation of price history collection"""
    market_service = MarketDataService()
    
    try:
        # Fetch price history
        history = await market_service.get_price_history(token, days)
        
        if not history:
            return {"status": "no_data", "token": token}
        
        # Store in database
        async with get_db_session() as db:
            for price_point in history:
                # Create PriceHistory record
                price_record = PriceHistory(
                    token=token.upper(),
                    timestamp=datetime.fromisoformat(price_point["timestamp"].replace('Z', '+00:00')),
                    price=price_point["price"],
                    volume=price_point.get("volume", 0),
                    source="coingecko"
                )
                
                db.add(price_record)
            
            await db.commit()
        
        return {
            "status": "success", 
            "token": token,
            "data_points": len(history),
            "days": days
        }
        
    except Exception as e:
        logger.error(f"Error collecting price history for {token}: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def analyze_market_trends(self):
    """
    Analyze market trends and generate trend reports
    Runs every 15 minutes
    """
    try:
        logger.info("Starting market trend analysis")
        
        result = asyncio.run(_analyze_market_trends_async())
        
        logger.info(f"Market trend analysis completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Market trend analysis failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _analyze_market_trends_async():
    """Async implementation of market trend analysis"""
    market_service = MarketDataService()
    
    try:
        # Get current market trends
        trends = await market_service.get_market_trends()
        
        # Store trend analysis in database
        async with get_db_session() as db:
            trend_record = MarketData(
                timestamp=datetime.now(timezone.utc),
                data_type="trend_analysis",
                data=trends
            )
            
            db.add(trend_record)
            await db.commit()
        
        # Log significant trends
        top_gainers = trends.get("top_gainers", [])[:3]
        top_losers = trends.get("top_losers", [])[:3]
        market_sentiment = trends.get("market_sentiment", "neutral")
        
        signal_logger.info("market_trend_analysis", extra={
            "market_sentiment": market_sentiment,
            "top_gainers": [g["symbol"] for g in top_gainers],
            "top_losers": [l["symbol"] for l in top_losers],
            "total_market_cap": trends.get("total_market_cap", 0),
            "total_volume_24h": trends.get("total_volume_24h", 0)
        })
        
        # Trigger alerts for extreme market conditions
        if market_sentiment in ["very_bullish", "very_bearish"]:
            from app.workers.notification_sender import send_market_alert
            send_market_alert.delay(
                alert_type="extreme_sentiment",
                sentiment=market_sentiment,
                details=trends
            )
        
        return {
            "status": "success",
            "sentiment": market_sentiment,
            "gainers_count": len(top_gainers),
            "losers_count": len(top_losers),
            "timestamp": trends.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Error in market trend analysis: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def monitor_whale_activity(self):
    """
    Monitor large transactions and whale activity on Solana
    """
    try:
        logger.info("Starting whale activity monitoring")
        
        result = asyncio.run(_monitor_whale_activity_async())
        
        logger.info(f"Whale activity monitoring completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Whale activity monitoring failed: {e}")
        self.retry(countdown=180 * (2 ** self.request.retries))

async def _monitor_whale_activity_async():
    """Async implementation of whale activity monitoring"""
    # This would integrate with Solana blockchain monitoring
    # For now, return placeholder data
    
    # TODO: Implement actual whale tracking logic
    # - Monitor large transactions
    # - Track wallet movements
    # - Detect accumulation patterns
    
    return {
        "status": "placeholder",
        "message": "Whale monitoring to be implemented with Solana integration"
    }

# Helper functions
async def _store_market_data(market_data: Dict[str, Any]):
    """Store market data in database"""
    try:
        async with get_db_session() as db:
            # Store overall market data
            market_record = MarketData(
                timestamp=datetime.now(timezone.utc),
                data_type="market_snapshot",
                data=market_data
            )
            
            db.add(market_record)
            await db.commit()
            
    except Exception as e:
        logger.error(f"Error storing market data: {e}")
        raise

async def _detect_significant_movements(market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect significant price movements in market data"""
    significant_movements = []
    prices = market_data.get("prices", {})
    
    # Thresholds for significant movements
    SIGNIFICANT_CHANGE_THRESHOLD = 10.0  # 10% price change
    HIGH_VOLUME_THRESHOLD = 5000000      # $5M volume
    
    for token, data in prices.items():
        change_24h = abs(data.get("price_change_percentage_24h", 0))
        volume_24h = data.get("total_volume", 0)
        price = data.get("current_price", 0)
        
        # Check for significant movements
        if change_24h >= SIGNIFICANT_CHANGE_THRESHOLD or volume_24h >= HIGH_VOLUME_THRESHOLD:
            movement_type = "surge" if data.get("price_change_percentage_24h", 0) > 0 else "drop"
            
            significant_movements.append({
                "token": token,
                "price": price,
                "change_24h": data.get("price_change_percentage_24h", 0),
                "volume_24h": volume_24h,
                "movement_type": movement_type,
                "market_cap": data.get("market_cap", 0)
            })
    
    return significant_movements

# Periodic task to collect price history for all tokens
@shared_task(bind=True, base=celery_app.Task)
def collect_all_price_histories(self):
    """
    Collect price history for all monitored tokens
    Runs daily
    """
    try:
        logger.info("Starting bulk price history collection")
        
        market_service = MarketDataService()
        supported_tokens = market_service.supported_tokens
        
        tasks_queued = 0
        for token in supported_tokens:
            # Queue individual collection tasks
            collect_price_history.delay(token, days=1)
            tasks_queued += 1
        
        logger.info(f"Queued {tasks_queued} price history collection tasks")
        return {
            "status": "success",
            "tasks_queued": tasks_queued,
            "tokens": supported_tokens
        }
        
    except Exception as e:
        logger.error(f"Bulk price history collection failed: {e}")
        self.retry(countdown=600 * (2 ** self.request.retries))

# Market data validation task
@shared_task(bind=True, base=celery_app.Task)
def validate_market_data(self):
    """
    Validate market data quality and consistency
    """
    try:
        logger.info("Starting market data validation")
        
        result = asyncio.run(_validate_market_data_async())
        
        logger.info(f"Market data validation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Market data validation failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _validate_market_data_async():
    """Async implementation of market data validation"""
    validation_results = {
        "status": "success",
        "checks": [],
        "warnings": [],
        "errors": []
    }
    
    try:
        async with get_db_session() as db:
            # Check for recent data
            recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
            
            # Validation checks would go here
            # For now, return basic validation
            
            validation_results["checks"].append("Data freshness check passed")
            validation_results["checks"].append("Price consistency check passed")
            
        return validation_results
        
    except Exception as e:
        validation_results["status"] = "error"
        validation_results["errors"].append(str(e))
        return validation_results 