"""
Signal Service
Central service for managing trading signals and their lifecycle
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from app.core.config import settings
from app.core.logging import create_signal_logger
from app.database import get_db_session
from app.models.signals import Signal, SignalStatus, SignalType

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

class SignalService:
    """
    Service for managing trading signals
    """
    
    def __init__(self):
        self.active_signals = {}
        self.signal_cache = {}
    
    async def create_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new trading signal"""
        try:
            # Create signal record
            signal = Signal(
                token=signal_data["token"],
                action=signal_data["action"],
                signal_type=signal_data.get("signal_type", SignalType.TECHNICAL),
                confidence=signal_data["confidence"],
                current_price=signal_data["current_price"],
                target_price=signal_data.get("target_price"),
                stop_loss=signal_data.get("stop_loss"),
                rationale=signal_data.get("rationale", ""),
                metadata=signal_data.get("metadata", {}),
                status=SignalStatus.PENDING,
                created_at=datetime.now(timezone.utc)
            )
            
            async with get_db_session() as db:
                db.add(signal)
                await db.commit()
                await db.refresh(signal)
            
            # Convert to dict for return
            signal_dict = {
                "id": str(signal.id),
                "token": signal.token,
                "action": signal.action,
                "signal_type": signal.signal_type.value if signal.signal_type else "technical",
                "confidence": signal.confidence,
                "current_price": signal.current_price,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "rationale": signal.rationale,
                "metadata": signal.metadata,
                "status": signal.status.value if signal.status else "pending",
                "created_at": signal.created_at.isoformat()
            }
            
            # Cache the signal
            self.active_signals[signal_dict["id"]] = signal_dict
            
            return signal_dict
            
        except Exception as e:
            logger.error(f"Error creating signal: {e}")
            raise
    
    async def get_signal_by_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get a signal by ID"""
        try:
            # Check cache first
            if signal_id in self.active_signals:
                return self.active_signals[signal_id]
            
            # Query database
            async with get_db_session() as db:
                signal = await db.get(Signal, signal_id)
                if signal:
                    signal_dict = {
                        "id": str(signal.id),
                        "token": signal.token,
                        "action": signal.action,
                        "signal_type": signal.signal_type.value if signal.signal_type else "technical",
                        "confidence": signal.confidence,
                        "current_price": signal.current_price,
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss,
                        "rationale": signal.rationale,
                        "metadata": signal.metadata,
                        "status": signal.status.value if signal.status else "pending",
                        "created_at": signal.created_at.isoformat()
                    }
                    
                    # Cache it
                    self.active_signals[signal_id] = signal_dict
                    return signal_dict
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting signal {signal_id}: {e}")
            return None
    
    async def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Get all pending signals"""
        try:
            async with get_db_session() as db:
                # This would be an actual query in production
                # For now, return signals from cache that are pending
                pending_signals = [
                    signal for signal in self.active_signals.values()
                    if signal.get("status") == "pending"
                ]
                
                return pending_signals
                
        except Exception as e:
            logger.error(f"Error getting pending signals: {e}")
            return []
    
    async def update_signal_status(self, signal_id: str, status: SignalStatus) -> bool:
        """Update signal status"""
        try:
            async with get_db_session() as db:
                signal = await db.get(Signal, signal_id)
                if signal:
                    signal.status = status
                    signal.updated_at = datetime.now(timezone.utc)
                    await db.commit()
                    
                    # Update cache
                    if signal_id in self.active_signals:
                        self.active_signals[signal_id]["status"] = status.value
                    
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error updating signal status {signal_id}: {e}")
            return False
    
    async def mark_signal_broadcasted(self, signal_id: str) -> bool:
        """Mark signal as broadcasted"""
        try:
            async with get_db_session() as db:
                signal = await db.get(Signal, signal_id)
                if signal:
                    signal.broadcasted_at = datetime.now(timezone.utc)
                    await db.commit()
                    
                    # Update cache
                    if signal_id in self.active_signals:
                        self.active_signals[signal_id]["broadcasted_at"] = signal.broadcasted_at.isoformat()
                    
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error marking signal broadcasted {signal_id}: {e}")
            return False
    
    async def get_recent_signals(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent signals within specified hours"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            recent_signals = [
                signal for signal in self.active_signals.values()
                if datetime.fromisoformat(signal["created_at"]) > cutoff_time
            ]
            
            # Sort by creation time, newest first
            recent_signals.sort(key=lambda x: x["created_at"], reverse=True)
            
            return recent_signals
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    async def cleanup_old_signals(self, days: int = 30) -> int:
        """Clean up signals older than specified days"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Remove from cache
            old_signal_ids = [
                signal_id for signal_id, signal in self.active_signals.items()
                if datetime.fromisoformat(signal["created_at"]) < cutoff_time
            ]
            
            for signal_id in old_signal_ids:
                del self.active_signals[signal_id]
            
            # This would involve actual database cleanup in production
            return len(old_signal_ids)
            
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {e}")
            return 0
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal statistics"""
        try:
            total_signals = len(self.active_signals)
            pending_signals = len([s for s in self.active_signals.values() if s.get("status") == "pending"])
            active_signals = len([s for s in self.active_signals.values() if s.get("status") == "active"])
            
            return {
                "total_signals": total_signals,
                "pending_signals": pending_signals,
                "active_signals": active_signals,
                "cache_size": len(self.active_signals)
            }
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {
                "total_signals": 0,
                "pending_signals": 0,
                "active_signals": 0,
                "cache_size": 0
            } 