"""
WebSocket Connection Manager for Real-time Signal Delivery
Handles premium user connections, signal broadcasting, and room management
"""
import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import verify_jwt_token
from app.services.nft_verification import NFTVerificationService

logger = logging.getLogger(__name__)

class ConnectionInfo(BaseModel):
    """WebSocket connection information"""
    websocket: WebSocket
    user_id: str
    wallet_address: str
    nft_verified: bool
    connected_at: datetime
    last_ping: datetime
    rooms: Set[str] = set()

class WebSocketMessage(BaseModel):
    """Standard WebSocket message format"""
    type: str
    data: Any
    timestamp: datetime
    room: Optional[str] = None

class WebSocketManager:
    """Manages WebSocket connections for real-time features"""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, ConnectionInfo] = {}
        # Room-based connections for broadcasting
        self.rooms: Dict[str, Set[str]] = {
            "premium_signals": set(),
            "market_data": set(),
            "general": set(),
            "alerts": set()
        }
        # Message queues for offline delivery
        self.message_queues: Dict[str, List[WebSocketMessage]] = {}
        self.nft_service = NFTVerificationService()
        
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        token: Optional[str] = None,
        wallet_address: Optional[str] = None
    ) -> str:
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        connection_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Verify NFT ownership if wallet provided
        nft_verified = False
        if wallet_address and token:
            try:
                # Verify JWT token first
                payload = verify_jwt_token(token)
                if payload and payload.get("wallet_address") == wallet_address:
                    # Verify NFT ownership
                    nft_verified = await self.nft_service.verify_nft_ownership(wallet_address)
            except Exception as e:
                logger.warning(f"NFT verification failed for {wallet_address}: {e}")
        
        # Create connection info
        connection_info = ConnectionInfo(
            websocket=websocket,
            user_id=user_id,
            wallet_address=wallet_address or "",
            nft_verified=nft_verified,
            connected_at=datetime.now(timezone.utc),
            last_ping=datetime.now(timezone.utc),
            rooms=set()
        )
        
        self.active_connections[connection_id] = connection_info
        
        # Add to appropriate rooms
        await self._assign_rooms(connection_id, nft_verified)
        
        # Send connection confirmation
        await self.send_personal_message(
            connection_id,
            WebSocketMessage(
                type="connection_established",
                data={
                    "connection_id": connection_id,
                    "nft_verified": nft_verified,
                    "rooms": list(connection_info.rooms),
                    "server_time": datetime.now(timezone.utc).isoformat()
                },
                timestamp=datetime.now(timezone.utc)
            )
        )
        
        # Deliver queued messages
        await self._deliver_queued_messages(user_id, connection_id)
        
        logger.info(f"WebSocket connected: {connection_id}, NFT verified: {nft_verified}")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Remove connection and clean up"""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            
            # Remove from all rooms
            for room in connection.rooms:
                if room in self.rooms:
                    self.rooms[room].discard(connection_id)
            
            # Remove connection
            del self.active_connections[connection_id]
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            try:
                await connection.websocket.send_text(
                    json.dumps(message.dict(), default=str)
                )
                return True
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                await self.disconnect(connection_id)
                return False
        return False
    
    async def broadcast_to_room(self, room: str, message: WebSocketMessage):
        """Broadcast message to all connections in a room"""
        if room not in self.rooms:
            logger.warning(f"Room {room} does not exist")
            return
        
        message.room = room
        disconnected = []
        
        for connection_id in self.rooms[room].copy():
            success = await self.send_personal_message(connection_id, message)
            if not success:
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.rooms[room].discard(connection_id)
        
        logger.info(f"Broadcast to room {room}: {len(self.rooms[room]) - len(disconnected)} recipients")
    
    async def broadcast_signal(self, signal_data: Dict[str, Any], premium_only: bool = False):
        """Broadcast trading signal to appropriate users"""
        message = WebSocketMessage(
            type="trading_signal",
            data=signal_data,
            timestamp=datetime.now(timezone.utc)
        )
        
        if premium_only:
            # Only to NFT verified users
            await self.broadcast_to_room("premium_signals", message)
        else:
            # To all connected users
            await self.broadcast_to_room("general", message)
    
    async def broadcast_market_data(self, market_data: Dict[str, Any]):
        """Broadcast live market data updates"""
        message = WebSocketMessage(
            type="market_data",
            data=market_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast_to_room("market_data", message)
    
    async def send_alert(self, user_id: str, alert_data: Dict[str, Any]):
        """Send alert to specific user"""
        message = WebSocketMessage(
            type="alert",
            data=alert_data,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Find user connections
        user_connections = [
            conn_id for conn_id, conn in self.active_connections.items()
            if conn.user_id == user_id
        ]
        
        if user_connections:
            for connection_id in user_connections:
                await self.send_personal_message(connection_id, message)
        else:
            # Queue for offline delivery
            if user_id not in self.message_queues:
                self.message_queues[user_id] = []
            self.message_queues[user_id].append(message)
    
    async def handle_ping(self, connection_id: str):
        """Handle ping from client"""
        if connection_id in self.active_connections:
            self.active_connections[connection_id].last_ping = datetime.now(timezone.utc)
            
            # Send pong response
            await self.send_personal_message(
                connection_id,
                WebSocketMessage(
                    type="pong",
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                    timestamp=datetime.now(timezone.utc)
                )
            )
    
    async def refresh_nft_status(self, connection_id: str):
        """Re-verify NFT status and update room assignments"""
        if connection_id not in self.active_connections:
            return False
        
        connection = self.active_connections[connection_id]
        if not connection.wallet_address:
            return False
        
        try:
            # Re-verify NFT ownership
            nft_verified = await self.nft_service.verify_nft_ownership(
                connection.wallet_address
            )
            
            old_status = connection.nft_verified
            connection.nft_verified = nft_verified
            
            # Update room assignments if status changed
            if old_status != nft_verified:
                await self._reassign_rooms(connection_id)
                
                # Notify client of status change
                await self.send_personal_message(
                    connection_id,
                    WebSocketMessage(
                        type="nft_status_updated",
                        data={
                            "nft_verified": nft_verified,
                            "rooms": list(connection.rooms)
                        },
                        timestamp=datetime.now(timezone.utc)
                    )
                )
            
            return nft_verified
            
        except Exception as e:
            logger.error(f"Failed to refresh NFT status for {connection_id}: {e}")
            return False
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        total_connections = len(self.active_connections)
        nft_verified_count = sum(
            1 for conn in self.active_connections.values() 
            if conn.nft_verified
        )
        
        room_stats = {
            room: len(connections) 
            for room, connections in self.rooms.items()
        }
        
        return {
            "total_connections": total_connections,
            "nft_verified_connections": nft_verified_count,
            "room_statistics": room_stats,
            "message_queues": len(self.message_queues),
            "server_time": datetime.now(timezone.utc).isoformat()
        }
    
    async def cleanup_stale_connections(self, timeout_minutes: int = 30):
        """Remove connections that haven't pinged recently"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (timeout_minutes * 60)
        stale_connections = []
        
        for connection_id, connection in self.active_connections.items():
            if connection.last_ping.timestamp() < cutoff_time:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            await self.disconnect(connection_id)
        
        logger.info(f"Cleaned up {len(stale_connections)} stale connections")
    
    async def _assign_rooms(self, connection_id: str, nft_verified: bool):
        """Assign connection to appropriate rooms"""
        connection = self.active_connections[connection_id]
        
        # All users get general updates
        self.rooms["general"].add(connection_id)
        connection.rooms.add("general")
        
        # NFT holders get premium features
        if nft_verified:
            self.rooms["premium_signals"].add(connection_id)
            self.rooms["market_data"].add(connection_id)
            self.rooms["alerts"].add(connection_id)
            connection.rooms.update(["premium_signals", "market_data", "alerts"])
    
    async def _reassign_rooms(self, connection_id: str):
        """Reassign rooms after NFT status change"""
        connection = self.active_connections[connection_id]
        
        # Remove from all rooms
        for room in connection.rooms.copy():
            self.rooms[room].discard(connection_id)
        connection.rooms.clear()
        
        # Reassign based on current NFT status
        await self._assign_rooms(connection_id, connection.nft_verified)
    
    async def _deliver_queued_messages(self, user_id: str, connection_id: str):
        """Deliver queued messages to newly connected user"""
        if user_id in self.message_queues:
            messages = self.message_queues[user_id]
            
            for message in messages:
                await self.send_personal_message(connection_id, message)
            
            # Clear delivered messages
            del self.message_queues[user_id]
            logger.info(f"Delivered {len(messages)} queued messages to {user_id}")

# Global WebSocket manager instance
websocket_manager = WebSocketManager() 