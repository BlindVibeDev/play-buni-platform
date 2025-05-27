"""
Signal Distribution Coordinator for Play Buni Platform

This service handles the distribution of generated signals to various channels
including social media, real-time WebSocket connections, and internal systems.

Features:
- Queue processing with priority handling
- Channel-specific distribution logic
- Rate limiting and retry mechanisms
- Performance tracking and analytics
- Real-time delivery to premium users
- Social media automation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_

from app.core.config import settings
from app.core.logging import get_logger
from app.core.cache import cache_manager
from app.core.database import get_db_session
from app.models.signals import Signal, SignalQueue
from app.models.users import User, NFTHolder
from app.services.signal_engine import DistributionChannel, SignalPriority

logger = get_logger(__name__)


class DistributionStatus(Enum):
    """Distribution status for signals"""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ChannelConfig:
    """Configuration for distribution channels"""
    name: str
    enabled: bool
    rate_limit_per_minute: int
    max_retries: int
    retry_delay_seconds: int
    priority_boost: bool  # Whether this channel gets priority signals first


@dataclass
class DistributionMetrics:
    """Metrics for signal distribution"""
    total_signals: int
    delivered_signals: int
    failed_signals: int
    pending_signals: int
    delivery_rate: float
    avg_delivery_time: float
    channel_metrics: Dict[str, Dict[str, Any]]


class WebSocketManager:
    """Manages WebSocket connections for real-time signal delivery"""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}  # user_id -> websocket
        self.user_channels: Dict[str, Set[str]] = {}  # user_id -> set of subscribed channels
    
    async def add_connection(self, user_id: str, websocket, channels: List[str]):
        """Add a WebSocket connection for a user"""
        self.connections[user_id] = websocket
        self.user_channels[user_id] = set(channels)
        logger.info(f"Added WebSocket connection for user {user_id}")
    
    async def remove_connection(self, user_id: str):
        """Remove a WebSocket connection"""
        if user_id in self.connections:
            del self.connections[user_id]
        if user_id in self.user_channels:
            del self.user_channels[user_id]
        logger.info(f"Removed WebSocket connection for user {user_id}")
    
    async def send_signal_to_user(self, user_id: str, signal_data: Dict[str, Any]) -> bool:
        """Send signal to a specific user via WebSocket"""
        if user_id not in self.connections:
            return False
        
        try:
            websocket = self.connections[user_id]
            await websocket.send_text(json.dumps(signal_data))
            return True
        except Exception as e:
            logger.error(f"Error sending signal to user {user_id}: {e}")
            await self.remove_connection(user_id)
            return False
    
    async def broadcast_to_channel(self, channel: str, signal_data: Dict[str, Any]) -> int:
        """Broadcast signal to all users subscribed to a channel"""
        delivered_count = 0
        
        for user_id, channels in self.user_channels.items():
            if channel in channels:
                if await self.send_signal_to_user(user_id, signal_data):
                    delivered_count += 1
        
        return delivered_count


class SocialMediaDistributor:
    """Handles distribution to social media platforms"""
    
    def __init__(self):
        self.twitter_enabled = bool(settings.twitter_api_key)
        self.discord_enabled = bool(settings.discord_bot_token)
        self.last_post_times = {}
        
        # Rate limits (posts per hour)
        self.rate_limits = {
            "twitter": 10,
            "discord": 20
        }
    
    async def format_signal_for_social(self, signal: Dict[str, Any]) -> str:
        """Format signal for social media posting"""
        try:
            symbol = signal.get("symbol", "UNKNOWN")
            signal_type = signal.get("signal_type", "").upper()
            entry_price = signal.get("entry_price", 0)
            target_price = signal.get("target_price")
            confidence = signal.get("confidence_score", 0) * 100
            reasoning = signal.get("reasoning", [])
            
            # Create emoji based on signal type
            emoji_map = {
                "STRONG_BUY": "🚀",
                "BUY": "📈",
                "SELL": "📉",
                "STRONG_SELL": "🔻",
                "HOLD": "⏸️"
            }
            
            emoji = emoji_map.get(signal_type, "📊")
            
            # Format the message
            message = f"{emoji} {signal_type} Signal: ${symbol}\n\n"
            message += f"💰 Entry: ${entry_price:.6f}\n"
            
            if target_price:
                message += f"🎯 Target: ${target_price:.6f}\n"
            
            message += f"📊 Confidence: {confidence:.0f}%\n\n"
            
            if reasoning:
                message += "📋 Analysis:\n"
                for reason in reasoning[:3]:  # Limit to 3 reasons
                    message += f"• {reason}\n"
            
            message += "\n#Solana #DeFi #TradingSignals #PlayBuni"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting signal for social media: {e}")
            return f"📊 Trading Signal: {signal.get('symbol', 'UNKNOWN')} - {signal.get('signal_type', 'UNKNOWN')}"
    
    async def check_rate_limit(self, platform: str) -> bool:
        """Check if we can post to a platform based on rate limits"""
        cache_key = f"social_posts:{platform}"
        
        # Get recent post count
        recent_posts = await cache_manager.get(cache_key)
        if not recent_posts:
            recent_posts = 0
        else:
            recent_posts = int(recent_posts)
        
        limit = self.rate_limits.get(platform, 5)
        
        if recent_posts >= limit:
            return False
        
        # Increment counter
        await cache_manager.set(cache_key, str(recent_posts + 1), expire=3600)
        return True
    
    async def post_to_twitter(self, message: str) -> bool:
        """Post signal to Twitter"""
        if not self.twitter_enabled:
            logger.warning("Twitter not configured")
            return False
        
        if not await self.check_rate_limit("twitter"):
            logger.warning("Twitter rate limit exceeded")
            return False
        
        try:
            # This would integrate with Twitter API
            # For now, we'll simulate the posting
            logger.info(f"Posted to Twitter: {message[:50]}...")
            
            # In production, you would use tweepy or similar:
            # import tweepy
            # auth = tweepy.OAuthHandler(settings.twitter_api_key, settings.twitter_api_secret)
            # auth.set_access_token(settings.twitter_access_token, settings.twitter_access_token_secret)
            # api = tweepy.API(auth)
            # api.update_status(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error posting to Twitter: {e}")
            return False
    
    async def post_to_discord(self, message: str) -> bool:
        """Post signal to Discord"""
        if not self.discord_enabled:
            logger.warning("Discord not configured")
            return False
        
        if not await self.check_rate_limit("discord"):
            logger.warning("Discord rate limit exceeded")
            return False
        
        try:
            # This would integrate with Discord API
            # For now, we'll simulate the posting
            logger.info(f"Posted to Discord: {message[:50]}...")
            
            # In production, you would use discord.py or webhook:
            # import discord
            # client = discord.Client()
            # channel = client.get_channel(int(settings.discord_channel_id))
            # await channel.send(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error posting to Discord: {e}")
            return False
    
    async def distribute_to_social(self, signal: Dict[str, Any]) -> Dict[str, bool]:
        """Distribute signal to all configured social media platforms"""
        message = await self.format_signal_for_social(signal)
        results = {}
        
        if self.twitter_enabled:
            results["twitter"] = await self.post_to_twitter(message)
        
        if self.discord_enabled:
            results["discord"] = await self.post_to_discord(message)
        
        return results


class SignalDistributionCoordinator:
    """
    Signal Distribution Coordinator
    
    Processes the signal queue and distributes signals to appropriate channels
    based on user access levels and channel configurations.
    """
    
    def __init__(self):
        self.is_running = False
        self.distribution_task = None
        self.websocket_manager = WebSocketManager()
        self.social_distributor = SocialMediaDistributor()
        
        # Channel configurations
        self.channel_configs = {
            DistributionChannel.PUBLIC_SOCIAL.value: ChannelConfig(
                name="Public Social Media",
                enabled=True,
                rate_limit_per_minute=2,  # 2 signals per hour for public
                max_retries=2,
                retry_delay_seconds=300,  # 5 minutes
                priority_boost=False
            ),
            DistributionChannel.PREMIUM_REALTIME.value: ChannelConfig(
                name="Premium Real-time",
                enabled=True,
                rate_limit_per_minute=60,  # No practical limit for premium
                max_retries=3,
                retry_delay_seconds=30,
                priority_boost=True
            ),
            DistributionChannel.VIP_EXCLUSIVE.value: ChannelConfig(
                name="VIP Exclusive",
                enabled=True,
                rate_limit_per_minute=100,
                max_retries=5,
                retry_delay_seconds=10,
                priority_boost=True
            ),
            DistributionChannel.ADMIN_INTERNAL.value: ChannelConfig(
                name="Admin Internal",
                enabled=True,
                rate_limit_per_minute=1000,
                max_retries=1,
                retry_delay_seconds=5,
                priority_boost=True
            )
        }
        
        self.distribution_metrics = DistributionMetrics(
            total_signals=0,
            delivered_signals=0,
            failed_signals=0,
            pending_signals=0,
            delivery_rate=0.0,
            avg_delivery_time=0.0,
            channel_metrics={}
        )
    
    async def initialize(self):
        """Initialize the distribution coordinator"""
        try:
            logger.info("Initializing Signal Distribution Coordinator...")
            
            # Initialize channel metrics
            for channel in self.channel_configs:
                self.distribution_metrics.channel_metrics[channel] = {
                    "delivered": 0,
                    "failed": 0,
                    "avg_delivery_time": 0.0,
                    "last_delivery": None
                }
            
            logger.info("Signal Distribution Coordinator initialized")
            
        except Exception as e:
            logger.error(f"Error initializing distribution coordinator: {e}")
            raise
    
    async def get_pending_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending signals from the queue"""
        try:
            async with get_db_session() as db:
                # Get pending signals ordered by priority and scheduled time
                result = await db.execute(
                    select(SignalQueue, Signal)
                    .join(Signal, SignalQueue.signal_id == Signal.id)
                    .where(
                        and_(
                            SignalQueue.status == DistributionStatus.PENDING.value,
                            SignalQueue.scheduled_for <= datetime.utcnow(),
                            SignalQueue.retry_count < SignalQueue.max_retries
                        )
                    )
                    .order_by(SignalQueue.priority.asc(), SignalQueue.scheduled_for.asc())
                    .limit(limit)
                )
                
                signals = []
                for queue_item, signal in result:
                    signals.append({
                        "queue_id": queue_item.id,
                        "signal_id": signal.id,
                        "channel": queue_item.channel,
                        "priority": queue_item.priority,
                        "retry_count": queue_item.retry_count,
                        "max_retries": queue_item.max_retries,
                        "signal_data": {
                            "symbol": signal.symbol,
                            "mint_address": signal.mint_address,
                            "signal_type": signal.signal_type,
                            "signal_strength": signal.signal_strength,
                            "entry_price": float(signal.entry_price),
                            "target_price": float(signal.target_price) if signal.target_price else None,
                            "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
                            "confidence_score": float(signal.confidence_score),
                            "risk_score": float(signal.risk_score),
                            "timeframe": signal.timeframe,
                            "reasoning": signal.reasoning,
                            "metadata": signal.metadata,
                            "generated_at": signal.generated_at.isoformat(),
                            "expires_at": signal.expires_at.isoformat() if signal.expires_at else None
                        }
                    })
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting pending signals: {e}")
            return []
    
    async def update_queue_status(
        self, 
        queue_id: int, 
        status: DistributionStatus, 
        error_message: Optional[str] = None
    ):
        """Update the status of a queue item"""
        try:
            async with get_db_session() as db:
                update_data = {
                    "status": status.value,
                    "updated_at": datetime.utcnow()
                }
                
                if status == DistributionStatus.DELIVERED:
                    update_data["delivered_at"] = datetime.utcnow()
                elif status == DistributionStatus.FAILED:
                    update_data["error_message"] = error_message
                    # Increment retry count for failed deliveries
                    await db.execute(
                        update(SignalQueue)
                        .where(SignalQueue.id == queue_id)
                        .values(retry_count=SignalQueue.retry_count + 1)
                    )
                
                await db.execute(
                    update(SignalQueue)
                    .where(SignalQueue.id == queue_id)
                    .values(**update_data)
                )
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error updating queue status: {e}")
    
    async def get_eligible_users_for_channel(self, channel: str) -> List[str]:
        """Get list of user IDs eligible for a specific channel"""
        try:
            async with get_db_session() as db:
                if channel == DistributionChannel.PUBLIC_SOCIAL.value:
                    # Public signals don't target specific users
                    return []
                
                elif channel == DistributionChannel.PREMIUM_REALTIME.value:
                    # Premium users (NFT holders)
                    result = await db.execute(
                        select(User.id)
                        .join(NFTHolder, User.id == NFTHolder.user_id)
                        .where(NFTHolder.nft_count > 0)
                    )
                    return [str(user_id) for user_id, in result]
                
                elif channel == DistributionChannel.VIP_EXCLUSIVE.value:
                    # VIP users (high NFT count)
                    result = await db.execute(
                        select(User.id)
                        .join(NFTHolder, User.id == NFTHolder.user_id)
                        .where(NFTHolder.nft_count >= settings.vip_nft_threshold)
                    )
                    return [str(user_id) for user_id, in result]
                
                elif channel == DistributionChannel.ADMIN_INTERNAL.value:
                    # Admin users
                    result = await db.execute(
                        select(User.id)
                        .where(User.wallet_address.in_(settings.admin_wallets_list))
                    )
                    return [str(user_id) for user_id, in result]
                
                return []
                
        except Exception as e:
            logger.error(f"Error getting eligible users for channel {channel}: {e}")
            return []
    
    async def distribute_to_websocket(
        self, 
        channel: str, 
        signal_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Distribute signal via WebSocket to eligible users"""
        try:
            eligible_users = await self.get_eligible_users_for_channel(channel)
            
            if not eligible_users:
                # For public channels, we don't send via WebSocket
                if channel == DistributionChannel.PUBLIC_SOCIAL.value:
                    return True, None
                else:
                    return False, "No eligible users found"
            
            # Format signal for WebSocket delivery
            ws_message = {
                "type": "trading_signal",
                "channel": channel,
                "timestamp": datetime.utcnow().isoformat(),
                "data": signal_data
            }
            
            delivered_count = 0
            total_users = len(eligible_users)
            
            for user_id in eligible_users:
                if await self.websocket_manager.send_signal_to_user(user_id, ws_message):
                    delivered_count += 1
            
            if delivered_count > 0:
                logger.info(f"Delivered signal to {delivered_count}/{total_users} users via WebSocket")
                return True, None
            else:
                return False, "No active WebSocket connections"
                
        except Exception as e:
            logger.error(f"Error distributing to WebSocket: {e}")
            return False, str(e)
    
    async def distribute_to_social_media(
        self, 
        signal_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Distribute signal to social media platforms"""
        try:
            results = await self.social_distributor.distribute_to_social(signal_data)
            
            # Check if at least one platform succeeded
            success = any(results.values())
            
            if success:
                successful_platforms = [platform for platform, result in results.items() if result]
                logger.info(f"Posted signal to social media: {', '.join(successful_platforms)}")
                return True, None
            else:
                failed_platforms = [platform for platform, result in results.items() if not result]
                error_msg = f"Failed to post to: {', '.join(failed_platforms)}"
                return False, error_msg
                
        except Exception as e:
            logger.error(f"Error distributing to social media: {e}")
            return False, str(e)
    
    async def distribute_signal(self, queue_item: Dict[str, Any]) -> bool:
        """Distribute a single signal to its target channel"""
        queue_id = queue_item["queue_id"]
        channel = queue_item["channel"]
        signal_data = queue_item["signal_data"]
        
        try:
            # Mark as processing
            await self.update_queue_status(queue_id, DistributionStatus.PROCESSING)
            
            start_time = datetime.utcnow()
            success = False
            error_message = None
            
            # Route to appropriate distribution method
            if channel == DistributionChannel.PUBLIC_SOCIAL.value:
                success, error_message = await self.distribute_to_social_media(signal_data)
            
            elif channel in [
                DistributionChannel.PREMIUM_REALTIME.value,
                DistributionChannel.VIP_EXCLUSIVE.value,
                DistributionChannel.ADMIN_INTERNAL.value
            ]:
                success, error_message = await self.distribute_to_websocket(channel, signal_data)
            
            else:
                error_message = f"Unknown distribution channel: {channel}"
            
            # Update status based on result
            if success:
                await self.update_queue_status(queue_id, DistributionStatus.DELIVERED)
                
                # Update metrics
                delivery_time = (datetime.utcnow() - start_time).total_seconds()
                self.distribution_metrics.delivered_signals += 1
                
                if channel in self.distribution_metrics.channel_metrics:
                    channel_metrics = self.distribution_metrics.channel_metrics[channel]
                    channel_metrics["delivered"] += 1
                    channel_metrics["last_delivery"] = datetime.utcnow().isoformat()
                    
                    # Update average delivery time
                    current_avg = channel_metrics["avg_delivery_time"]
                    delivered_count = channel_metrics["delivered"]
                    channel_metrics["avg_delivery_time"] = (
                        (current_avg * (delivered_count - 1) + delivery_time) / delivered_count
                    )
                
                logger.info(f"Successfully distributed signal {queue_item['signal_id']} to {channel}")
                return True
                
            else:
                await self.update_queue_status(queue_id, DistributionStatus.FAILED, error_message)
                
                # Update metrics
                self.distribution_metrics.failed_signals += 1
                
                if channel in self.distribution_metrics.channel_metrics:
                    self.distribution_metrics.channel_metrics[channel]["failed"] += 1
                
                logger.warning(f"Failed to distribute signal {queue_item['signal_id']} to {channel}: {error_message}")
                return False
                
        except Exception as e:
            error_message = str(e)
            await self.update_queue_status(queue_id, DistributionStatus.FAILED, error_message)
            
            logger.error(f"Error distributing signal {queue_item['signal_id']}: {e}")
            return False
    
    async def process_distribution_batch(self, signals: List[Dict[str, Any]]):
        """Process a batch of signals for distribution"""
        if not signals:
            return
        
        logger.info(f"Processing distribution batch of {len(signals)} signals")
        
        # Group signals by channel for efficient processing
        channel_groups = {}
        for signal in signals:
            channel = signal["channel"]
            if channel not in channel_groups:
                channel_groups[channel] = []
            channel_groups[channel].append(signal)
        
        # Process each channel group
        for channel, channel_signals in channel_groups.items():
            config = self.channel_configs.get(channel)
            if not config or not config.enabled:
                logger.warning(f"Channel {channel} is disabled, skipping signals")
                continue
            
            # Apply rate limiting
            rate_limit_key = f"distribution_rate:{channel}"
            current_count = await cache_manager.get(rate_limit_key)
            current_count = int(current_count) if current_count else 0
            
            available_slots = config.rate_limit_per_minute - current_count
            if available_slots <= 0:
                logger.warning(f"Rate limit exceeded for channel {channel}, deferring signals")
                continue
            
            # Process signals up to rate limit
            signals_to_process = channel_signals[:available_slots]
            
            for signal in signals_to_process:
                try:
                    await self.distribute_signal(signal)
                    
                    # Update rate limit counter
                    await cache_manager.increment(rate_limit_key)
                    await cache_manager.expire(rate_limit_key, 60)  # 1 minute TTL
                    
                    # Small delay between distributions
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing signal {signal['signal_id']}: {e}")
                    continue
    
    async def run_distribution_cycle(self):
        """Run one complete distribution cycle"""
        try:
            # Get pending signals
            pending_signals = await self.get_pending_signals(limit=100)
            
            if not pending_signals:
                return
            
            # Update total signals metric
            self.distribution_metrics.total_signals += len(pending_signals)
            self.distribution_metrics.pending_signals = len(pending_signals)
            
            # Process signals in batches
            await self.process_distribution_batch(pending_signals)
            
            # Update delivery rate
            total = self.distribution_metrics.total_signals
            delivered = self.distribution_metrics.delivered_signals
            
            if total > 0:
                self.distribution_metrics.delivery_rate = delivered / total
            
            logger.debug(f"Distribution cycle completed. Delivery rate: {self.distribution_metrics.delivery_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Error in distribution cycle: {e}")
    
    async def start_distribution_loop(self):
        """Start the continuous distribution loop"""
        self.is_running = True
        logger.info("Starting signal distribution loop...")
        
        while self.is_running:
            try:
                await self.run_distribution_cycle()
                
                # Wait before next cycle (shorter interval for real-time delivery)
                await asyncio.sleep(10)  # 10 seconds
                
            except Exception as e:
                logger.error(f"Error in distribution loop: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retrying
    
    async def stop_distribution_loop(self):
        """Stop the distribution loop"""
        self.is_running = False
        if self.distribution_task:
            self.distribution_task.cancel()
        logger.info("Signal distribution loop stopped")
    
    async def get_distribution_metrics(self) -> DistributionMetrics:
        """Get current distribution metrics"""
        return self.distribution_metrics
    
    async def add_websocket_connection(self, user_id: str, websocket, access_level: str):
        """Add a WebSocket connection for real-time signal delivery"""
        # Determine channels based on access level
        channels = []
        
        if access_level in ["premium", "vip", "admin"]:
            channels.append(DistributionChannel.PREMIUM_REALTIME.value)
        
        if access_level in ["vip", "admin"]:
            channels.append(DistributionChannel.VIP_EXCLUSIVE.value)
        
        if access_level == "admin":
            channels.append(DistributionChannel.ADMIN_INTERNAL.value)
        
        await self.websocket_manager.add_connection(user_id, websocket, channels)
    
    async def remove_websocket_connection(self, user_id: str):
        """Remove a WebSocket connection"""
        await self.websocket_manager.remove_connection(user_id)


# Global signal distribution coordinator instance
signal_distribution_coordinator = SignalDistributionCoordinator() 