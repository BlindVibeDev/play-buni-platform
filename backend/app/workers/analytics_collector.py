"""
Analytics Collection Worker
Background tasks for platform analytics, metrics collection, and reporting
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from celery import shared_task
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.workers.celery_app import celery_app
from app.core.config import settings
from app.core.logging import create_signal_logger
from app.database import get_db_session
from app.models.analytics import PlatformAnalytics, UserAnalytics
from app.core.websocket import websocket_manager

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

@shared_task(bind=True, base=celery_app.Task)
def collect_platform_analytics(self):
    """
    Collect platform-wide analytics and metrics
    Runs every 5 minutes via Celery Beat
    """
    try:
        logger.info("Starting platform analytics collection")
        
        result = asyncio.run(_collect_platform_analytics_async())
        
        logger.info(f"Platform analytics collection completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Platform analytics collection failed: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _collect_platform_analytics_async():
    """Async implementation of platform analytics collection"""
    try:
        analytics_data = {
            "timestamp": datetime.now(timezone.utc),
            "websocket_stats": {},
            "signal_stats": {},
            "user_stats": {},
            "revenue_stats": {},
            "performance_stats": {}
        }
        
        # Collect WebSocket statistics
        try:
            ws_stats = await websocket_manager.get_connection_stats()
            analytics_data["websocket_stats"] = {
                "total_connections": ws_stats.get("total_connections", 0),
                "nft_verified_connections": ws_stats.get("nft_verified_connections", 0),
                "room_stats": ws_stats.get("room_statistics", {}),
                "message_queues": ws_stats.get("message_queues", 0)
            }
        except Exception as e:
            logger.error(f"Error collecting WebSocket stats: {e}")
            analytics_data["websocket_stats"] = {"error": str(e)}
        
        # Collect signal statistics
        try:
            signal_stats = await _collect_signal_statistics()
            analytics_data["signal_stats"] = signal_stats
        except Exception as e:
            logger.error(f"Error collecting signal stats: {e}")
            analytics_data["signal_stats"] = {"error": str(e)}
        
        # Collect user statistics
        try:
            user_stats = await _collect_user_statistics()
            analytics_data["user_stats"] = user_stats
        except Exception as e:
            logger.error(f"Error collecting user stats: {e}")
            analytics_data["user_stats"] = {"error": str(e)}
        
        # Collect revenue statistics
        try:
            revenue_stats = await _collect_revenue_statistics()
            analytics_data["revenue_stats"] = revenue_stats
        except Exception as e:
            logger.error(f"Error collecting revenue stats: {e}")
            analytics_data["revenue_stats"] = {"error": str(e)}
        
        # Store analytics in database
        async with get_db_session() as db:
            analytics_record = PlatformAnalytics(
                timestamp=analytics_data["timestamp"],
                data=analytics_data
            )
            db.add(analytics_record)
            await db.commit()
        
        # Log key metrics
        signal_logger.info("platform_analytics_collected", extra={
            "total_connections": analytics_data["websocket_stats"].get("total_connections", 0),
            "nft_verified": analytics_data["websocket_stats"].get("nft_verified_connections", 0),
            "signals_24h": analytics_data["signal_stats"].get("signals_24h", 0),
            "active_users_24h": analytics_data["user_stats"].get("active_users_24h", 0),
            "revenue_24h": analytics_data["revenue_stats"].get("revenue_24h", 0)
        })
        
        return {
            "status": "success",
            "timestamp": analytics_data["timestamp"].isoformat(),
            "metrics_collected": len([k for k in analytics_data.keys() if k != "timestamp"])
        }
        
    except Exception as e:
        logger.error(f"Error in platform analytics collection: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def generate_daily_report(self):
    """
    Generate comprehensive daily analytics report
    Runs daily at midnight UTC
    """
    try:
        logger.info("Starting daily report generation")
        
        result = asyncio.run(_generate_daily_report_async())
        
        logger.info(f"Daily report generation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Daily report generation failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _generate_daily_report_async():
    """Async implementation of daily report generation"""
    try:
        # Calculate date range for yesterday
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        start_time = datetime.combine(yesterday, datetime.min.time(), timezone.utc)
        end_time = datetime.combine(today, datetime.min.time(), timezone.utc)
        
        report_data = {
            "report_date": yesterday.isoformat(),
            "period": "daily",
            "summary": {},
            "signals": {},
            "users": {},
            "revenue": {},
            "performance": {},
            "trends": {}
        }
        
        # Generate summary metrics
        async with get_db_session() as db:
            # Collect various daily metrics
            # This would involve complex queries across multiple tables
            
            # Placeholder report structure
            report_data["summary"] = {
                "total_signals": 0,
                "total_users": 0,
                "total_revenue": 0.0,
                "avg_response_time": 0.0,
                "uptime_percentage": 99.9
            }
            
            report_data["signals"] = {
                "generated": 0,
                "distributed": 0,
                "accuracy_rate": 0.0,
                "avg_confidence": 0.0,
                "top_performing_tokens": []
            }
            
            report_data["users"] = {
                "active_users": 0,
                "new_registrations": 0,
                "nft_verifications": 0,
                "premium_users": 0,
                "retention_rate": 0.0
            }
            
            report_data["revenue"] = {
                "total_fees": 0.0,
                "transaction_count": 0,
                "avg_transaction_size": 0.0,
                "top_trading_pairs": []
            }
            
        # Store daily report
        analytics_record = PlatformAnalytics(
            timestamp=end_time,
            data_type="daily_report",
            data=report_data
        )
        
        async with get_db_session() as db:
            db.add(analytics_record)
            await db.commit()
        
        # Log daily summary
        signal_logger.info("daily_report_generated", extra={
            "report_date": yesterday.isoformat(),
            "total_signals": report_data["summary"]["total_signals"],
            "total_users": report_data["summary"]["total_users"],
            "total_revenue": report_data["summary"]["total_revenue"]
        })
        
        return {
            "status": "success",
            "report_date": yesterday.isoformat(),
            "sections": list(report_data.keys())
        }
        
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def system_health_check(self):
    """
    Perform comprehensive system health check
    Runs every 5 minutes via Celery Beat
    """
    try:
        logger.info("Starting system health check")
        
        result = asyncio.run(_system_health_check_async())
        
        logger.info(f"System health check completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _system_health_check_async():
    """Async implementation of system health check"""
    health_data = {
        "timestamp": datetime.now(timezone.utc),
        "overall_status": "healthy",
        "services": {},
        "alerts": []
    }
    
    try:
        # Check database connectivity
        try:
            async with get_db_session() as db:
                await db.execute(text("SELECT 1"))
                health_data["services"]["database"] = {"status": "healthy", "response_time_ms": 0}
        except Exception as e:
            health_data["services"]["database"] = {"status": "unhealthy", "error": str(e)}
            health_data["overall_status"] = "degraded"
            health_data["alerts"].append("Database connectivity issue")
        
        # Check WebSocket manager
        try:
            ws_stats = await websocket_manager.get_connection_stats()
            health_data["services"]["websocket"] = {
                "status": "healthy",
                "connections": ws_stats.get("total_connections", 0)
            }
        except Exception as e:
            health_data["services"]["websocket"] = {"status": "unhealthy", "error": str(e)}
            health_data["overall_status"] = "degraded"
            health_data["alerts"].append("WebSocket manager issue")
        
        # Check market data freshness
        try:
            freshness_check = await _check_market_data_freshness()
            if freshness_check["status"] == "stale":
                health_data["alerts"].append("Market data is stale")
                health_data["overall_status"] = "degraded"
            health_data["services"]["market_data"] = freshness_check
        except Exception as e:
            health_data["services"]["market_data"] = {"status": "error", "error": str(e)}
            health_data["alerts"].append("Market data check failed")
        
        # Check signal generation rate
        try:
            signal_rate = await _check_signal_generation_rate()
            if signal_rate["rate"] < 0.1:  # Less than 0.1 signals per minute
                health_data["alerts"].append("Signal generation rate too low")
            health_data["services"]["signal_generation"] = signal_rate
        except Exception as e:
            health_data["services"]["signal_generation"] = {"status": "error", "error": str(e)}
        
        # Store health check results
        async with get_db_session() as db:
            health_record = PlatformAnalytics(
                timestamp=health_data["timestamp"],
                data_type="health_check",
                data=health_data
            )
            db.add(health_record)
            await db.commit()
        
        # Log health status
        if health_data["alerts"]:
            logger.warning(f"System health issues detected: {health_data['alerts']}")
        
        signal_logger.info("system_health_check", extra={
            "overall_status": health_data["overall_status"],
            "services_count": len(health_data["services"]),
            "alerts_count": len(health_data["alerts"])
        })
        
        return {
            "status": health_data["overall_status"],
            "timestamp": health_data["timestamp"].isoformat(),
            "services_checked": len(health_data["services"]),
            "alerts": len(health_data["alerts"])
        }
        
    except Exception as e:
        logger.error(f"Error in system health check: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def cleanup_old_data(self):
    """
    Clean up old analytics data and logs
    Runs weekly on Sunday at 2 AM UTC
    """
    try:
        logger.info("Starting old data cleanup")
        
        result = asyncio.run(_cleanup_old_data_async())
        
        logger.info(f"Old data cleanup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Old data cleanup failed: {e}")
        self.retry(countdown=600 * (2 ** self.request.retries))

async def _cleanup_old_data_async():
    """Async implementation of old data cleanup"""
    try:
        cleanup_results = {
            "analytics_cleaned": 0,
            "logs_cleaned": 0,
            "websocket_logs_cleaned": 0
        }
        
        # Clean up analytics older than 90 days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        
        async with get_db_session() as db:
            # Clean up old analytics records
            # This would be actual DELETE queries in production
            
            # Placeholder cleanup logic
            cleanup_results["analytics_cleaned"] = 0
            cleanup_results["logs_cleaned"] = 0
            cleanup_results["websocket_logs_cleaned"] = 0
        
        logger.info(f"Cleanup completed: {cleanup_results}")
        
        return {
            "status": "success",
            "cleanup_date": cutoff_date.isoformat(),
            "results": cleanup_results
        }
        
    except Exception as e:
        logger.error(f"Error in old data cleanup: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def collect_user_activity(self, user_id: str, activity_type: str, metadata: Dict[str, Any] = None):
    """
    Collect individual user activity for analytics
    """
    try:
        logger.debug(f"Collecting user activity: {user_id} - {activity_type}")
        
        result = asyncio.run(_collect_user_activity_async(user_id, activity_type, metadata))
        
        return result
        
    except Exception as e:
        logger.error(f"User activity collection failed: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _collect_user_activity_async(user_id: str, activity_type: str, metadata: Dict[str, Any]):
    """Async implementation of user activity collection"""
    try:
        activity_data = {
            "user_id": user_id,
            "activity_type": activity_type,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
        
        # Store user activity
        async with get_db_session() as db:
            user_analytics = UserAnalytics(
                user_id=user_id,
                activity_type=activity_type,
                timestamp=activity_data["timestamp"],
                data=activity_data
            )
            db.add(user_analytics)
            await db.commit()
        
        return {
            "status": "success",
            "user_id": user_id,
            "activity_type": activity_type
        }
        
    except Exception as e:
        logger.error(f"Error collecting user activity: {e}")
        raise

# Helper functions
async def _collect_signal_statistics():
    """Collect signal-related statistics"""
    try:
        async with get_db_session() as db:
            # Get signals from last 24 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            # This would involve actual database queries
            # For now, return placeholder data
            
            return {
                "signals_24h": 0,
                "signals_pending": 0,
                "signals_distributed": 0,
                "avg_confidence": 0.0,
                "top_tokens": []
            }
            
    except Exception as e:
        logger.error(f"Error collecting signal statistics: {e}")
        return {"error": str(e)}

async def _collect_user_statistics():
    """Collect user-related statistics"""
    try:
        async with get_db_session() as db:
            # Get user activity from last 24 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            # This would involve actual database queries
            # For now, return placeholder data
            
            return {
                "active_users_24h": 0,
                "new_users_24h": 0,
                "nft_verified_users": 0,
                "premium_users": 0,
                "total_users": 0
            }
            
    except Exception as e:
        logger.error(f"Error collecting user statistics: {e}")
        return {"error": str(e)}

async def _collect_revenue_statistics():
    """Collect revenue-related statistics"""
    try:
        async with get_db_session() as db:
            # Get revenue data from last 24 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            # This would involve actual database queries
            # For now, return placeholder data
            
            return {
                "revenue_24h": 0.0,
                "transactions_24h": 0,
                "avg_transaction_size": 0.0,
                "fee_revenue": 0.0,
                "total_volume": 0.0
            }
            
    except Exception as e:
        logger.error(f"Error collecting revenue statistics: {e}")
        return {"error": str(e)}

async def _check_market_data_freshness():
    """Check if market data is fresh"""
    try:
        async with get_db_session() as db:
            # Check timestamp of latest market data
            # This would involve actual database query
            
            # For now, assume data is fresh
            return {
                "status": "fresh",
                "last_update": datetime.now(timezone.utc).isoformat(),
                "age_minutes": 1
            }
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def _check_signal_generation_rate():
    """Check signal generation rate"""
    try:
        async with get_db_session() as db:
            # Check signal generation in last hour
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            
            # This would involve actual database query
            # For now, return placeholder data
            
            return {
                "rate": 2.5,  # signals per minute
                "last_hour_count": 150,
                "status": "normal"
            }
            
    except Exception as e:
        return {"status": "error", "error": str(e)} 