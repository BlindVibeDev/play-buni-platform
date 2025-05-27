"""
Notification Sender Worker
Background tasks for sending notifications, alerts, and communications
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from celery import shared_task

from app.workers.celery_app import celery_app
from app.core.config import settings
from app.core.logging import create_signal_logger
from app.core.websocket import websocket_manager
from app.database import get_db_session
from app.models.notifications import Notification, NotificationType

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

@shared_task(bind=True, base=celery_app.Task)
def send_notification(self, user_id: str, notification_type: str, content: Dict[str, Any]):
    """
    Send a notification to a specific user
    """
    try:
        logger.info(f"Sending {notification_type} notification to user {user_id}")
        
        result = asyncio.run(_send_notification_async(user_id, notification_type, content))
        
        logger.info(f"Notification sent successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Notification sending failed: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _send_notification_async(user_id: str, notification_type: str, content: Dict[str, Any]):
    """Async implementation of notification sending"""
    try:
        notification_data = {
            "user_id": user_id,
            "type": notification_type,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
            "channels": []
        }
        
        # Determine delivery channels based on notification type
        channels = _get_notification_channels(notification_type, user_id)
        
        delivery_results = {}
        
        # Send via WebSocket (real-time)
        if "websocket" in channels:
            try:
                await _send_websocket_notification(user_id, notification_data)
                delivery_results["websocket"] = "success"
                notification_data["channels"].append("websocket")
            except Exception as e:
                logger.error(f"WebSocket notification failed: {e}")
                delivery_results["websocket"] = f"failed: {e}"
        
        # Send via in-app notification
        if "in_app" in channels:
            try:
                await _send_in_app_notification(user_id, notification_data)
                delivery_results["in_app"] = "success"
                notification_data["channels"].append("in_app")
            except Exception as e:
                logger.error(f"In-app notification failed: {e}")
                delivery_results["in_app"] = f"failed: {e}"
        
        # Store notification record
        async with get_db_session() as db:
            notification_record = Notification(
                user_id=user_id,
                notification_type=notification_type,
                content=content,
                channels=notification_data["channels"],
                timestamp=notification_data["timestamp"],
                delivered=len(notification_data["channels"]) > 0
            )
            db.add(notification_record)
            await db.commit()
        
        # Log notification
        signal_logger.info("notification_sent", extra={
            "user_id": user_id,
            "type": notification_type,
            "channels": notification_data["channels"],
            "delivered": len(notification_data["channels"]) > 0
        })
        
        return {
            "status": "success",
            "user_id": user_id,
            "notification_type": notification_type,
            "channels_used": notification_data["channels"],
            "delivery_results": delivery_results
        }
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def send_market_alert(self, alert_type: str, sentiment: str, details: Dict[str, Any]):
    """
    Send market alert to all premium users
    """
    try:
        logger.info(f"Sending market alert: {alert_type} - {sentiment}")
        
        result = asyncio.run(_send_market_alert_async(alert_type, sentiment, details))
        
        logger.info(f"Market alert sent: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Market alert sending failed: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _send_market_alert_async(alert_type: str, sentiment: str, details: Dict[str, Any]):
    """Async implementation of market alert sending"""
    try:
        alert_data = {
            "type": "market_alert",
            "alert_type": alert_type,
            "sentiment": sentiment,
            "details": details,
            "timestamp": datetime.now(timezone.utc),
            "urgency": "high" if sentiment in ["very_bullish", "very_bearish"] else "medium"
        }
        
        # Send to all premium users via WebSocket
        await websocket_manager.broadcast_to_room("alerts", {
            "type": "market_alert",
            "data": alert_data,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Get premium users for in-app notifications
        premium_users = await _get_premium_users()
        
        notifications_sent = 0
        for user in premium_users:
            try:
                # Queue individual notification
                send_notification.delay(
                    user["id"], 
                    "market_alert", 
                    alert_data
                )
                notifications_sent += 1
            except Exception as e:
                logger.error(f"Error queuing market alert for user {user.get('id', 'unknown')}: {e}")
        
        # Log market alert
        signal_logger.info("market_alert_sent", extra={
            "alert_type": alert_type,
            "sentiment": sentiment,
            "urgency": alert_data["urgency"],
            "users_notified": notifications_sent
        })
        
        return {
            "status": "success",
            "alert_type": alert_type,
            "sentiment": sentiment,
            "users_notified": notifications_sent
        }
        
    except Exception as e:
        logger.error(f"Error sending market alert: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def send_treasury_alert(self, alert_data: Dict[str, Any]):
    """
    Send treasury-related alerts to administrators
    """
    try:
        logger.info(f"Sending treasury alert: {alert_data.get('type', 'unknown')}")
        
        result = asyncio.run(_send_treasury_alert_async(alert_data))
        
        logger.info(f"Treasury alert sent: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Treasury alert sending failed: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _send_treasury_alert_async(alert_data: Dict[str, Any]):
    """Async implementation of treasury alert sending"""
    try:
        # Get admin users
        admin_users = await _get_admin_users()
        
        treasury_alert = {
            "type": "treasury_alert",
            "alert_type": alert_data.get("type", "unknown"),
            "current_balance": alert_data.get("current_balance", 0),
            "change": alert_data.get("change", 0),
            "change_percentage": alert_data.get("change_percentage", 0),
            "timestamp": datetime.now(timezone.utc),
            "urgency": "high"
        }
        
        notifications_sent = 0
        for admin in admin_users:
            try:
                # Send high-priority notification to admin
                send_notification.delay(
                    admin["id"],
                    "treasury_alert",
                    treasury_alert
                )
                notifications_sent += 1
            except Exception as e:
                logger.error(f"Error queuing treasury alert for admin {admin.get('id', 'unknown')}: {e}")
        
        # Log treasury alert
        signal_logger.info("treasury_alert_sent", extra={
            "alert_type": alert_data.get("type"),
            "current_balance": alert_data.get("current_balance"),
            "change": alert_data.get("change"),
            "admins_notified": notifications_sent
        })
        
        return {
            "status": "success",
            "alert_type": alert_data.get("type"),
            "admins_notified": notifications_sent
        }
        
    except Exception as e:
        logger.error(f"Error sending treasury alert: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def send_signal_alert(self, signal_id: str, signal_data: Dict[str, Any]):
    """
    Send signal-related alerts and notifications
    """
    try:
        logger.info(f"Sending signal alert for signal {signal_id}")
        
        result = asyncio.run(_send_signal_alert_async(signal_id, signal_data))
        
        logger.info(f"Signal alert sent: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Signal alert sending failed: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _send_signal_alert_async(signal_id: str, signal_data: Dict[str, Any]):
    """Async implementation of signal alert sending"""
    try:
        confidence = signal_data.get("confidence", 0)
        token = signal_data.get("token", "")
        action = signal_data.get("action", "")
        
        # Determine alert urgency based on confidence
        urgency = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        alert_content = {
            "type": "signal_alert",
            "signal_id": signal_id,
            "token": token,
            "action": action,
            "confidence": confidence,
            "urgency": urgency,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Send to appropriate user groups based on signal type
        if confidence > 0.8:  # High confidence signals to premium users
            target_room = "premium_signals"
            users = await _get_premium_users()
        else:  # Lower confidence signals to all users
            target_room = "general"
            users = await _get_all_active_users()
        
        # Broadcast via WebSocket
        await websocket_manager.broadcast_to_room(target_room, {
            "type": "signal_alert",
            "data": alert_content,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Send individual notifications for high-confidence signals
        notifications_sent = 0
        if confidence > 0.8:
            for user in users:
                try:
                    send_notification.delay(
                        user["id"],
                        "high_confidence_signal",
                        alert_content
                    )
                    notifications_sent += 1
                except Exception as e:
                    logger.error(f"Error queuing signal alert for user {user.get('id', 'unknown')}: {e}")
        
        # Log signal alert
        signal_logger.info("signal_alert_sent", extra={
            "signal_id": signal_id,
            "token": token,
            "action": action,
            "confidence": confidence,
            "urgency": urgency,
            "users_notified": notifications_sent,
            "room": target_room
        })
        
        return {
            "status": "success",
            "signal_id": signal_id,
            "urgency": urgency,
            "users_notified": notifications_sent,
            "broadcast_room": target_room
        }
        
    except Exception as e:
        logger.error(f"Error sending signal alert: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def send_system_alert(self, alert_type: str, message: str, severity: str = "medium"):
    """
    Send system-wide alerts for maintenance, issues, etc.
    """
    try:
        logger.info(f"Sending system alert: {alert_type} - {severity}")
        
        result = asyncio.run(_send_system_alert_async(alert_type, message, severity))
        
        logger.info(f"System alert sent: {result}")
        return result
        
    except Exception as e:
        logger.error(f"System alert sending failed: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _send_system_alert_async(alert_type: str, message: str, severity: str):
    """Async implementation of system alert sending"""
    try:
        system_alert = {
            "type": "system_alert",
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Broadcast to all connected users
        await websocket_manager.broadcast_to_room("general", {
            "type": "system_alert",
            "data": system_alert,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Send to admins for critical alerts
        if severity == "critical":
            admin_users = await _get_admin_users()
            for admin in admin_users:
                try:
                    send_notification.delay(
                        admin["id"],
                        "critical_system_alert",
                        system_alert
                    )
                except Exception as e:
                    logger.error(f"Error queuing system alert for admin {admin.get('id', 'unknown')}: {e}")
        
        # Log system alert
        signal_logger.info("system_alert_sent", extra={
            "alert_type": alert_type,
            "severity": severity,
            "message": message
        })
        
        return {
            "status": "success",
            "alert_type": alert_type,
            "severity": severity,
            "broadcast_completed": True
        }
        
    except Exception as e:
        logger.error(f"Error sending system alert: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def cleanup_old_notifications(self):
    """
    Clean up old notifications
    Runs weekly
    """
    try:
        logger.info("Starting old notifications cleanup")
        
        result = asyncio.run(_cleanup_old_notifications_async())
        
        logger.info(f"Notifications cleanup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Notifications cleanup failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _cleanup_old_notifications_async():
    """Async implementation of old notifications cleanup"""
    try:
        # Clean up notifications older than 30 days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        cleanup_results = {
            "notifications_deleted": 0,
            "cutoff_date": cutoff_date.isoformat()
        }
        
        async with get_db_session() as db:
            # This would involve actual DELETE queries in production
            # For now, use placeholder cleanup logic
            cleanup_results["notifications_deleted"] = 0
        
        return {
            "status": "success",
            "results": cleanup_results
        }
        
    except Exception as e:
        logger.error(f"Error in notifications cleanup: {e}")
        raise

# Helper functions
def _get_notification_channels(notification_type: str, user_id: str) -> List[str]:
    """Determine appropriate notification channels based on type and user preferences"""
    # Default channels for different notification types
    channel_mapping = {
        "signal_alert": ["websocket", "in_app"],
        "high_confidence_signal": ["websocket", "in_app"],
        "market_alert": ["websocket", "in_app"],
        "treasury_alert": ["websocket", "in_app"],
        "system_alert": ["websocket"],
        "critical_system_alert": ["websocket", "in_app"],
        "nft_status_change": ["websocket", "in_app"],
        "premium_upgrade": ["websocket", "in_app"],
        "trade_execution": ["websocket", "in_app"]
    }
    
    return channel_mapping.get(notification_type, ["in_app"])

async def _send_websocket_notification(user_id: str, notification_data: Dict[str, Any]):
    """Send notification via WebSocket"""
    try:
        await websocket_manager.send_alert(user_id, notification_data)
        logger.debug(f"WebSocket notification sent to user {user_id}")
    except Exception as e:
        logger.error(f"Error sending WebSocket notification to {user_id}: {e}")
        raise

async def _send_in_app_notification(user_id: str, notification_data: Dict[str, Any]):
    """Send in-app notification (store in database)"""
    try:
        async with get_db_session() as db:
            # Store notification for in-app display
            notification_record = Notification(
                user_id=user_id,
                notification_type=notification_data["type"],
                content=notification_data["content"],
                channels=["in_app"],
                timestamp=notification_data["timestamp"],
                delivered=True,
                read=False
            )
            db.add(notification_record)
            await db.commit()
        
        logger.debug(f"In-app notification stored for user {user_id}")
    except Exception as e:
        logger.error(f"Error storing in-app notification for {user_id}: {e}")
        raise

async def _get_premium_users() -> List[Dict[str, Any]]:
    """Get all premium users"""
    try:
        async with get_db_session() as db:
            # This would be actual database query in production
            # For now, return placeholder data
            return []
    except Exception as e:
        logger.error(f"Error getting premium users: {e}")
        return []

async def _get_admin_users() -> List[Dict[str, Any]]:
    """Get all admin users"""
    try:
        async with get_db_session() as db:
            # This would be actual database query in production
            # For now, return placeholder data
            return []
    except Exception as e:
        logger.error(f"Error getting admin users: {e}")
        return []

async def _get_all_active_users() -> List[Dict[str, Any]]:
    """Get all active users"""
    try:
        async with get_db_session() as db:
            # This would be actual database query in production
            # For now, return placeholder data
            return []
    except Exception as e:
        logger.error(f"Error getting active users: {e}")
        return [] 