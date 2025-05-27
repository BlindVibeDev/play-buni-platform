"""
Signal Processing Worker
Background tasks for signal generation, analysis, and distribution
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
from app.services.signal_service import SignalService
from app.services.jupiter_service import jupiter_service, JupiterQuoteRequest
from app.core.websocket import websocket_manager
from app.database import get_db_session
from app.models.signals import Signal, SignalStatus, SignalType

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

@shared_task(bind=True, base=celery_app.Task)
def process_pending_signals(self):
    """
    Process all pending signals and distribute them
    Runs every minute via Celery Beat
    """
    try:
        logger.info("Starting pending signals processing")
        
        result = asyncio.run(_process_pending_signals_async())
        
        logger.info(f"Pending signals processing completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Pending signals processing failed: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _process_pending_signals_async():
    """Async implementation of pending signals processing"""
    signal_service = SignalService()
    
    try:
        # Get pending signals
        pending_signals = await signal_service.get_pending_signals()
        
        if not pending_signals:
            return {"status": "no_pending_signals", "processed": 0}
        
        processed_count = 0
        distributed_count = 0
        
        for signal in pending_signals:
            try:
                # Process the signal
                await _process_individual_signal(signal)
                processed_count += 1
                
                # Distribute via WebSocket
                if signal.get("status") == SignalStatus.ACTIVE:
                    await _distribute_signal_websocket(signal)
                    distributed_count += 1
                
                # Mark as processed
                await signal_service.update_signal_status(
                    signal["id"], 
                    SignalStatus.DISTRIBUTED
                )
                
            except Exception as e:
                logger.error(f"Error processing signal {signal.get('id', 'unknown')}: {e}")
        
        signal_logger.info("signals_batch_processed", extra={
            "total_pending": len(pending_signals),
            "processed": processed_count,
            "distributed": distributed_count
        })
        
        return {
            "status": "success",
            "total_pending": len(pending_signals),
            "processed": processed_count,
            "distributed": distributed_count
        }
        
    except Exception as e:
        logger.error(f"Error in pending signals processing: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def generate_movement_signal(self, token: str, price: float, change_24h: float, 
                           volume_24h: float, movement_type: str):
    """
    Generate a signal based on significant market movement
    """
    try:
        logger.info(f"Generating movement signal for {token}: {change_24h:.2f}% {movement_type}")
        
        result = asyncio.run(_generate_movement_signal_async(
            token, price, change_24h, volume_24h, movement_type
        ))
        
        logger.info(f"Movement signal generated for {token}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Movement signal generation failed for {token}: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _generate_movement_signal_async(token: str, price: float, change_24h: float,
                                        volume_24h: float, movement_type: str):
    """Async implementation of movement signal generation"""
    signal_service = SignalService()
    
    try:
        # Determine signal type and action
        if movement_type == "surge" and change_24h > 15:
            action = "BUY"
            signal_type = SignalType.MOMENTUM
            confidence = min(0.9, 0.6 + (change_24h - 15) / 100)
        elif movement_type == "drop" and change_24h < -15:
            action = "SELL"
            signal_type = SignalType.RISK_MANAGEMENT
            confidence = min(0.9, 0.6 + abs(change_24h + 15) / 100)
        else:
            action = "HOLD"
            signal_type = SignalType.TECHNICAL
            confidence = 0.5
        
        # Calculate target price and stop loss
        if action == "BUY":
            target_price = price * 1.20  # 20% upside target
            stop_loss = price * 0.90     # 10% downside protection
        elif action == "SELL":
            target_price = price * 0.80  # 20% downside target
            stop_loss = price * 1.10     # 10% upside stop
        else:
            target_price = price
            stop_loss = price * 0.95
        
        # Generate signal
        signal_data = {
            "token": token,
            "action": action,
            "signal_type": signal_type,
            "confidence": confidence,
            "current_price": price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "rationale": f"Significant {movement_type} detected: {change_24h:.2f}% change with ${volume_24h:,.0f} volume",
            "metadata": {
                "price_change_24h": change_24h,
                "volume_24h": volume_24h,
                "movement_type": movement_type,
                "generated_by": "movement_detector"
            }
        }
        
        # Create signal in database
        signal = await signal_service.create_signal(signal_data)
        
        # Log signal generation
        signal_logger.info("signal_generated", extra={
            "signal_id": signal["id"],
            "token": token,
            "action": action,
            "confidence": confidence,
            "trigger": "market_movement",
            "change_24h": change_24h,
            "volume_24h": volume_24h
        })
        
        # Queue for immediate distribution if high confidence
        if confidence > 0.8:
            distribute_signal.delay(signal["id"], priority="high")
        
        return {
            "status": "success",
            "signal_id": signal["id"],
            "token": token,
            "action": action,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error generating movement signal for {token}: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def generate_technical_signal(self, token: str, analysis_data: Dict[str, Any]):
    """
    Generate a signal based on technical analysis
    """
    try:
        logger.info(f"Generating technical signal for {token}")
        
        result = asyncio.run(_generate_technical_signal_async(token, analysis_data))
        
        logger.info(f"Technical signal generated for {token}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Technical signal generation failed for {token}: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _generate_technical_signal_async(token: str, analysis_data: Dict[str, Any]):
    """Async implementation of technical signal generation"""
    signal_service = SignalService()
    
    try:
        # Extract technical indicators
        rsi = analysis_data.get("rsi", 50)
        macd = analysis_data.get("macd", {})
        bollinger = analysis_data.get("bollinger_bands", {})
        volume_profile = analysis_data.get("volume_profile", {})
        
        # Determine signal based on technical analysis
        action = "HOLD"
        confidence = 0.5
        signal_type = SignalType.TECHNICAL
        
        # RSI-based signals
        if rsi < 30:  # Oversold
            action = "BUY"
            confidence += 0.2
        elif rsi > 70:  # Overbought
            action = "SELL"
            confidence += 0.2
        
        # MACD confirmation
        if macd.get("signal") == "bullish":
            if action == "BUY":
                confidence += 0.1
            elif action == "HOLD":
                action = "BUY"
                confidence += 0.15
        elif macd.get("signal") == "bearish":
            if action == "SELL":
                confidence += 0.1
            elif action == "HOLD":
                action = "SELL"
                confidence += 0.15
        
        # Volume confirmation
        if volume_profile.get("trend") == "increasing" and action == "BUY":
            confidence += 0.1
        elif volume_profile.get("trend") == "decreasing" and action == "SELL":
            confidence += 0.1
        
        # Cap confidence at 0.95
        confidence = min(0.95, confidence)
        
        # Only generate signal if confidence is above threshold
        if confidence < 0.6:
            return {
                "status": "low_confidence",
                "token": token,
                "confidence": confidence,
                "message": "Technical analysis confidence too low for signal generation"
            }
        
        # Create signal
        current_price = analysis_data.get("current_price", 0)
        
        if action == "BUY":
            target_price = current_price * 1.15
            stop_loss = current_price * 0.92
        elif action == "SELL":
            target_price = current_price * 0.85
            stop_loss = current_price * 1.08
        else:
            target_price = current_price
            stop_loss = current_price * 0.95
        
        signal_data = {
            "token": token,
            "action": action,
            "signal_type": signal_type,
            "confidence": confidence,
            "current_price": current_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "rationale": f"Technical analysis: RSI={rsi:.1f}, MACD={macd.get('signal', 'neutral')}, Volume={volume_profile.get('trend', 'stable')}",
            "metadata": {
                "rsi": rsi,
                "macd": macd,
                "bollinger_bands": bollinger,
                "volume_profile": volume_profile,
                "generated_by": "technical_analyzer"
            }
        }
        
        signal = await signal_service.create_signal(signal_data)
        
        signal_logger.info("technical_signal_generated", extra={
            "signal_id": signal["id"],
            "token": token,
            "action": action,
            "confidence": confidence,
            "rsi": rsi,
            "macd_signal": macd.get("signal", "neutral")
        })
        
        return {
            "status": "success",
            "signal_id": signal["id"],
            "token": token,
            "action": action,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error generating technical signal for {token}: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def distribute_signal(self, signal_id: str, priority: str = "normal"):
    """
    Distribute a specific signal to all appropriate channels
    """
    try:
        logger.info(f"Distributing signal {signal_id} with {priority} priority")
        
        result = asyncio.run(_distribute_signal_async(signal_id, priority))
        
        logger.info(f"Signal distribution completed for {signal_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Signal distribution failed for {signal_id}: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _distribute_signal_async(signal_id: str, priority: str):
    """Async implementation of signal distribution"""
    signal_service = SignalService()
    
    try:
        # Get signal details
        signal = await signal_service.get_signal_by_id(signal_id)
        if not signal:
            return {"status": "signal_not_found", "signal_id": signal_id}
        
        distribution_results = {
            "websocket": False,
            "database": False,
            "social": False
        }
        
        # Distribute via WebSocket (real-time)
        try:
            await _distribute_signal_websocket(signal)
            distribution_results["websocket"] = True
        except Exception as e:
            logger.error(f"WebSocket distribution failed for {signal_id}: {e}")
        
        # Update database status
        try:
            await signal_service.update_signal_status(signal_id, SignalStatus.DISTRIBUTED)
            await signal_service.mark_signal_broadcasted(signal_id)
            distribution_results["database"] = True
        except Exception as e:
            logger.error(f"Database update failed for {signal_id}: {e}")
        
        # Queue social media distribution (if enabled and not skipped)
        # This would be uncommented when social media is re-enabled
        # try:
        #     from app.workers.notification_sender import send_social_signal
        #     send_social_signal.delay(signal_id)
        #     distribution_results["social"] = True
        # except Exception as e:
        #     logger.error(f"Social media distribution failed for {signal_id}: {e}")
        
        signal_logger.info("signal_distributed", extra={
            "signal_id": signal_id,
            "token": signal.get("token"),
            "action": signal.get("action"),
            "confidence": signal.get("confidence"),
            "priority": priority,
            "websocket_success": distribution_results["websocket"],
            "database_success": distribution_results["database"]
        })
        
        return {
            "status": "success",
            "signal_id": signal_id,
            "distribution_results": distribution_results,
            "priority": priority
        }
        
    except Exception as e:
        logger.error(f"Error in signal distribution for {signal_id}: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def analyze_signal_performance(self):
    """
    Analyze the performance of historical signals
    Runs daily to track signal accuracy
    """
    try:
        logger.info("Starting signal performance analysis")
        
        result = asyncio.run(_analyze_signal_performance_async())
        
        logger.info(f"Signal performance analysis completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Signal performance analysis failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _analyze_signal_performance_async():
    """Async implementation of signal performance analysis"""
    signal_service = SignalService()
    
    try:
        # Get signals from last 24 hours for analysis
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        async with get_db_session() as db:
            # This would involve complex queries to analyze signal outcomes
            # For now, return placeholder analysis
            
            performance_metrics = {
                "total_signals": 0,
                "accurate_signals": 0,
                "accuracy_rate": 0.0,
                "avg_confidence": 0.0,
                "best_performing_type": "technical",
                "worst_performing_type": "momentum"
            }
            
            # TODO: Implement actual performance tracking
            # - Compare signal predictions with actual price movements
            # - Calculate accuracy rates by signal type
            # - Identify best performing conditions
            
            signal_logger.info("signal_performance_analysis", extra=performance_metrics)
            
            return {
                "status": "success",
                "metrics": performance_metrics,
                "analysis_period": "24h"
            }
            
    except Exception as e:
        logger.error(f"Error in signal performance analysis: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def cleanup_old_signals(self):
    """
    Clean up old signals and related data
    Runs weekly
    """
    try:
        logger.info("Starting old signals cleanup")
        
        result = asyncio.run(_cleanup_old_signals_async())
        
        logger.info(f"Old signals cleanup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Old signals cleanup failed: {e}")
        self.retry(countdown=600 * (2 ** self.request.retries))

async def _cleanup_old_signals_async():
    """Async implementation of old signals cleanup"""
    try:
        # Clean up signals older than 30 days
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        async with get_db_session() as db:
            # Delete old signals (this would be actual SQL in production)
            deleted_count = 0  # Placeholder
            
            # TODO: Implement actual cleanup logic
            # - Archive old signals to separate table
            # - Clean up associated metadata
            # - Maintain performance statistics
            
            return {
                "status": "success",
                "deleted_signals": deleted_count,
                "cutoff_date": cutoff_time.isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error in old signals cleanup: {e}")
        raise

# Helper functions
async def _process_individual_signal(signal: Dict[str, Any]):
    """Process and validate an individual signal"""
    try:
        # Validate signal data
        required_fields = ["token", "action", "confidence"]
        for field in required_fields:
            if field not in signal:
                raise ValueError(f"Missing required field: {field}")
        
        # Update signal metadata
        signal["processed_at"] = datetime.now(timezone.utc).isoformat()
        signal["status"] = SignalStatus.ACTIVE
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing individual signal: {e}")
        raise

async def _distribute_signal_websocket(signal: Dict[str, Any]):
    """Distribute signal via WebSocket to connected users"""
    try:
        # Determine if signal is premium only
        confidence = signal.get("confidence", 0)
        premium_only = confidence > 0.8  # High confidence signals are premium
        
        # Prepare signal data for WebSocket
        signal_data = {
            "signal_id": signal.get("id"),
            "token": signal.get("token"),
            "action": signal.get("action"),
            "confidence": confidence,
            "target_price": signal.get("target_price"),
            "stop_loss": signal.get("stop_loss"),
            "timestamp": signal.get("created_at", datetime.now(timezone.utc).isoformat()),
            "premium": premium_only,
            "rationale": signal.get("rationale", "")
        }
        
        # Broadcast via WebSocket manager
        await websocket_manager.broadcast_signal(signal_data, premium_only=premium_only)
        
        logger.info(f"Signal {signal.get('id')} distributed via WebSocket (premium: {premium_only})")
        
    except Exception as e:
        logger.error(f"Error distributing signal via WebSocket: {e}")
        raise 