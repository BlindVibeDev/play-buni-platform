"""
Background Worker for WebSocket Streaming
Handles continuous market data streaming and real-time signal delivery
"""
import asyncio
import logging
from typing import Any, Dict
from datetime import datetime, timezone

from app.core.websocket import websocket_manager
from app.services.market_data import MarketDataService
from app.services.signal_service import SignalService
from app.core.config import settings

logger = logging.getLogger(__name__)

class WebSocketStreamingWorker:
    """Background worker for real-time data streaming"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.signal_service = SignalService()
        self.is_running = False
        self.tasks: Dict[str, asyncio.Task] = {}
    
    async def start_streaming(self):
        """Start all streaming tasks"""
        if self.is_running:
            logger.warning("Streaming worker already running")
            return
        
        self.is_running = True
        logger.info("Starting WebSocket streaming worker...")
        
        # Start individual streaming tasks
        self.tasks = {
            "market_data": asyncio.create_task(self._stream_market_data()),
            "connection_cleanup": asyncio.create_task(self._cleanup_connections()),
            "heartbeat": asyncio.create_task(self._send_heartbeat()),
            "signal_notifications": asyncio.create_task(self._stream_signal_notifications())
        }
        
        # Wait for any task to complete (shouldn't happen in normal operation)
        try:
            done, pending = await asyncio.wait(
                self.tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # If any task completed, log it and cancel others
            for task in done:
                logger.error(f"Streaming task completed unexpectedly: {task}")
            
            for task in pending:
                task.cancel()
                
        except Exception as e:
            logger.error(f"Error in streaming worker: {e}")
        finally:
            self.is_running = False
    
    async def stop_streaming(self):
        """Stop all streaming tasks"""
        if not self.is_running:
            return
        
        logger.info("Stopping WebSocket streaming worker...")
        self.is_running = False
        
        # Cancel all tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                logger.info(f"Cancelling {task_name} task")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.tasks.clear()
    
    async def _stream_market_data(self):
        """Continuously stream market data to premium users"""
        logger.info("Starting market data streaming...")
        
        while self.is_running:
            try:
                # Fetch latest market data
                market_data = await self.market_service.get_current_market_data()
                
                # Broadcast to market data room (premium users only)
                await websocket_manager.broadcast_market_data(market_data)
                
                # Log streaming activity
                logger.debug(f"Streamed market data to {len(websocket_manager.rooms.get('market_data', set()))} premium users")
                
                # Wait for next update (30 seconds)
                await asyncio.sleep(settings.market_data_update_interval)
                
            except Exception as e:
                logger.error(f"Error streaming market data: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _stream_signal_notifications(self):
        """Stream real-time signal notifications"""
        logger.info("Starting signal notification streaming...")
        
        while self.is_running:
            try:
                # Check for new signals to broadcast
                recent_signals = await self.signal_service.get_pending_broadcasts()
                
                for signal in recent_signals:
                    # Determine if signal is premium only
                    is_premium = signal.get("premium_only", False)
                    
                    # Prepare signal data for broadcast
                    signal_data = {
                        "signal_id": signal.get("id"),
                        "token": signal.get("token"),
                        "action": signal.get("action"),
                        "confidence": signal.get("confidence"),
                        "target_price": signal.get("target_price"),
                        "stop_loss": signal.get("stop_loss"),
                        "timestamp": signal.get("created_at"),
                        "premium": is_premium
                    }
                    
                    # Broadcast signal
                    await websocket_manager.broadcast_signal(signal_data, premium_only=is_premium)
                    
                    # Mark signal as broadcasted
                    await self.signal_service.mark_signal_broadcasted(signal.get("id"))
                    
                    logger.info(f"Broadcasted signal {signal.get('id')} for {signal.get('token')}")
                
                # Check every 10 seconds for new signals
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error streaming signal notifications: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_connections(self):
        """Periodically clean up stale WebSocket connections"""
        logger.info("Starting connection cleanup task...")
        
        while self.is_running:
            try:
                # Clean up stale connections (older than 30 minutes)
                await websocket_manager.cleanup_stale_connections(timeout_minutes=30)
                
                # Log connection statistics
                stats = await websocket_manager.get_connection_stats()
                logger.info(f"Active connections: {stats.get('total_connections', 0)}, "
                           f"NFT verified: {stats.get('nft_verified_connections', 0)}")
                
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _send_heartbeat(self):
        """Send periodic heartbeat to all connections"""
        logger.info("Starting heartbeat task...")
        
        while self.is_running:
            try:
                # Send heartbeat to all active connections
                heartbeat_data = {
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_status": "active",
                    "connected_users": len(websocket_manager.active_connections)
                }
                
                # Broadcast to general room (all users)
                await websocket_manager.broadcast_to_room("general", {
                    "type": "heartbeat",
                    "data": heartbeat_data,
                    "timestamp": datetime.now(timezone.utc)
                })
                
                # Send heartbeat every 2 minutes
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(120)

class PremiumFeatureStreamer:
    """Enhanced streaming features for premium NFT holders"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.alert_queue = asyncio.Queue()
    
    async def stream_personalized_alerts(self, user_id: str, wallet_address: str):
        """Stream personalized alerts to premium user"""
        try:
            # Check user's portfolio and preferences
            portfolio = await self._get_user_portfolio(wallet_address)
            
            # Generate personalized alerts based on holdings
            alerts = await self._generate_portfolio_alerts(portfolio)
            
            for alert in alerts:
                await websocket_manager.send_alert(user_id, alert)
            
        except Exception as e:
            logger.error(f"Error streaming personalized alerts for {user_id}: {e}")
    
    async def stream_advanced_analytics(self, connection_id: str):
        """Stream advanced market analytics to premium connection"""
        try:
            # Get advanced market analytics
            analytics = await self._calculate_advanced_analytics()
            
            # Send to specific premium connection
            await websocket_manager.send_personal_message(
                connection_id,
                {
                    "type": "advanced_analytics",
                    "data": analytics,
                    "timestamp": datetime.now(timezone.utc)
                }
            )
            
        except Exception as e:
            logger.error(f"Error streaming advanced analytics: {e}")
    
    async def _get_user_portfolio(self, wallet_address: str) -> Dict[str, Any]:
        """Get user's token portfolio from Solana blockchain"""
        # This would integrate with Solana RPC to get token holdings
        # For now, return mock data
        return {
            "tokens": ["SOL", "RAY", "ORCA"],
            "balances": {
                "SOL": 10.5,
                "RAY": 1500.0,
                "ORCA": 250.0
            }
        }
    
    async def _generate_portfolio_alerts(self, portfolio: Dict[str, Any]) -> list:
        """Generate alerts based on user's portfolio"""
        alerts = []
        
        for token in portfolio.get("tokens", []):
            # Get current price and check for significant movements
            try:
                price_data = await self.market_service.get_token_price(token)
                if price_data:
                    change_24h = price_data.get("price_change_percentage_24h", 0)
                    
                    if abs(change_24h) > 5:  # 5% threshold
                        alerts.append({
                            "type": "portfolio_movement",
                            "token": token,
                            "change_24h": change_24h,
                            "current_price": price_data.get("current_price", 0),
                            "message": f"{token} moved {change_24h:.2f}% in the last 24h"
                        })
            except Exception as e:
                logger.error(f"Error generating alert for {token}: {e}")
        
        return alerts
    
    async def _calculate_advanced_analytics(self) -> Dict[str, Any]:
        """Calculate advanced market analytics"""
        try:
            market_data = await self.market_service.get_current_market_data()
            
            # Calculate advanced metrics
            return {
                "market_volatility": self._calculate_volatility(market_data),
                "sector_performance": self._analyze_sector_performance(market_data),
                "momentum_indicators": self._calculate_momentum_indicators(market_data),
                "risk_metrics": self._calculate_risk_metrics(market_data)
            }
        except Exception as e:
            logger.error(f"Error calculating advanced analytics: {e}")
            return {}
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market volatility metrics"""
        prices = market_data.get("prices", {})
        
        # Calculate average volatility based on 24h changes
        changes = [abs(data.get("price_change_percentage_24h", 0)) for data in prices.values()]
        avg_volatility = sum(changes) / len(changes) if changes else 0
        
        return {
            "average_volatility_24h": avg_volatility,
            "high_volatility_count": len([c for c in changes if c > 10]),
            "low_volatility_count": len([c for c in changes if c < 2])
        }
    
    def _analyze_sector_performance(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by sector"""
        # Group tokens by sector (simplified)
        defi_tokens = ["RAY", "ORCA", "MNGO", "SRM", "TULIP", "PORT", "SLND"]
        gaming_tokens = ["ATLAS", "POLIS", "NINJA"]
        media_tokens = ["MEDIA", "AUDIO"]
        
        prices = market_data.get("prices", {})
        
        sectors = {
            "defi": [prices.get(token, {}).get("price_change_percentage_24h", 0) for token in defi_tokens if token in prices],
            "gaming": [prices.get(token, {}).get("price_change_percentage_24h", 0) for token in gaming_tokens if token in prices],
            "media": [prices.get(token, {}).get("price_change_percentage_24h", 0) for token in media_tokens if token in prices]
        }
        
        sector_performance = {}
        for sector, changes in sectors.items():
            if changes:
                sector_performance[sector] = {
                    "average_change": sum(changes) / len(changes),
                    "best_performer": max(changes),
                    "worst_performer": min(changes)
                }
        
        return sector_performance
    
    def _calculate_momentum_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        prices = market_data.get("prices", {})
        
        # Simple momentum based on price changes
        strong_bullish = len([p for p in prices.values() if p.get("price_change_percentage_24h", 0) > 5])
        bullish = len([p for p in prices.values() if 0 < p.get("price_change_percentage_24h", 0) <= 5])
        bearish = len([p for p in prices.values() if -5 <= p.get("price_change_percentage_24h", 0) < 0])
        strong_bearish = len([p for p in prices.values() if p.get("price_change_percentage_24h", 0) < -5])
        
        return {
            "strong_bullish": strong_bullish,
            "bullish": bullish,
            "bearish": bearish,
            "strong_bearish": strong_bearish,
            "momentum_score": (strong_bullish * 2 + bullish - bearish - strong_bearish * 2)
        }
    
    def _calculate_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics"""
        prices = market_data.get("prices", {})
        
        # Calculate risk based on volatility and volume
        high_risk_tokens = []
        for symbol, data in prices.items():
            volatility = abs(data.get("price_change_percentage_24h", 0))
            volume = data.get("total_volume", 0)
            
            # High volatility + low volume = high risk
            if volatility > 15 and volume < 1000000:  # $1M volume threshold
                high_risk_tokens.append({
                    "symbol": symbol,
                    "volatility": volatility,
                    "volume": volume
                })
        
        return {
            "high_risk_tokens": high_risk_tokens,
            "market_risk_level": "high" if len(high_risk_tokens) > 5 else "medium" if len(high_risk_tokens) > 2 else "low"
        }

# Global instances
streaming_worker = WebSocketStreamingWorker()
premium_streamer = PremiumFeatureStreamer() 