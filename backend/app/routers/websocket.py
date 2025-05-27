"""
WebSocket Router for Real-time Features
Handles WebSocket connections, signal streaming, and premium user features
"""
import asyncio
import json
import logging
from typing import Optional, Any
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException
from fastapi.responses import HTMLResponse

from app.core.websocket import websocket_manager, WebSocketMessage
from app.core.security import get_current_user
from app.services.signal_service import SignalService
from app.services.market_data import MarketDataService
from app.models.user import UserInDB

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(..., description="User ID for connection"),
    token: Optional[str] = Query(None, description="JWT token for authentication"),
    wallet_address: Optional[str] = Query(None, description="Solana wallet address for NFT verification")
):
    """
    WebSocket endpoint for real-time connections
    
    Query Parameters:
    - user_id: Unique identifier for the user
    - token: JWT authentication token (optional for anonymous users)
    - wallet_address: Solana wallet address for NFT verification
    
    Message Types:
    - connection_established: Sent when connection is established
    - trading_signal: Real-time trading signals
    - market_data: Live market data updates
    - alert: Personal alerts and notifications
    - nft_status_updated: NFT verification status changes
    - pong: Response to ping messages
    """
    connection_id = None
    try:
        # Establish connection
        connection_id = await websocket_manager.connect(
            websocket, user_id, token, wallet_address
        )
        
        # Main message loop
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_client_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"Client {connection_id} disconnected")
                break
            except json.JSONDecodeError:
                await websocket_manager.send_personal_message(
                    connection_id,
                    WebSocketMessage(
                        type="error",
                        data={"message": "Invalid JSON format"},
                        timestamp=datetime.now(timezone.utc)
                    )
                )
            except Exception as e:
                logger.error(f"Error handling message from {connection_id}: {e}")
                await websocket_manager.send_personal_message(
                    connection_id,
                    WebSocketMessage(
                        type="error",
                        data={"message": "Internal server error"},
                        timestamp=datetime.now(timezone.utc)
                    )
                )
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)

@router.websocket("/ws/premium")
async def premium_websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(..., description="User ID for connection"),
    token: str = Query(..., description="JWT token for authentication"),
    wallet_address: str = Query(..., description="Solana wallet address for NFT verification")
):
    """
    Premium WebSocket endpoint exclusively for NFT holders
    Provides enhanced real-time features and priority signal delivery
    """
    connection_id = None
    try:
        # Establish connection with mandatory NFT verification
        connection_id = await websocket_manager.connect(
            websocket, user_id, token, wallet_address
        )
        
        # Check if user is actually NFT verified
        if connection_id in websocket_manager.active_connections:
            connection = websocket_manager.active_connections[connection_id]
            if not connection.nft_verified:
                await websocket_manager.send_personal_message(
                    connection_id,
                    WebSocketMessage(
                        type="access_denied",
                        data={
                            "message": "Premium features require NFT ownership",
                            "required_nft": True
                        },
                        timestamp=datetime.now(timezone.utc)
                    )
                )
                await websocket.close(code=4003, reason="NFT verification required")
                return
        
        # Send premium welcome message
        await websocket_manager.send_personal_message(
            connection_id,
            WebSocketMessage(
                type="premium_connected",
                data={
                    "message": "Premium WebSocket connected",
                    "features": [
                        "Real-time signal delivery",
                        "Live market data streaming",
                        "Personal alerts",
                        "Priority support"
                    ]
                },
                timestamp=datetime.now(timezone.utc)
            )
        )
        
        # Main message loop
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_premium_client_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"Premium client {connection_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error in premium WebSocket: {e}")
                break
                
    except Exception as e:
        logger.error(f"Premium WebSocket connection error: {e}")
        
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)

@router.get("/ws/stats")
async def get_websocket_stats(current_user: UserInDB = Depends(get_current_user)):
    """Get WebSocket connection statistics (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await websocket_manager.get_connection_stats()

@router.post("/ws/broadcast/signal")
async def broadcast_signal(
    signal_data: dict,
    premium_only: bool = False,
    current_user: UserInDB = Depends(get_current_user)
):
    """Broadcast trading signal via WebSocket (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await websocket_manager.broadcast_signal(signal_data, premium_only)
    return {"status": "signal_broadcasted", "premium_only": premium_only}

@router.post("/ws/broadcast/market-data")
async def broadcast_market_data(
    market_data: dict,
    current_user: UserInDB = Depends(get_current_user)
):
    """Broadcast market data via WebSocket (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await websocket_manager.broadcast_market_data(market_data)
    return {"status": "market_data_broadcasted"}

@router.post("/ws/alert/{user_id}")
async def send_user_alert(
    user_id: str,
    alert_data: dict,
    current_user: UserInDB = Depends(get_current_user)
):
    """Send alert to specific user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await websocket_manager.send_alert(user_id, alert_data)
    return {"status": "alert_sent", "user_id": user_id}

@router.post("/ws/refresh-nft/{connection_id}")
async def refresh_nft_status(
    connection_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Force refresh NFT verification status for a connection"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    result = await websocket_manager.refresh_nft_status(connection_id)
    return {"status": "nft_status_refreshed", "verified": result}

@router.get("/ws/test")
async def websocket_test_page():
    """Test page for WebSocket connections"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Play Buni WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .connection-form { background: #f0f0f0; padding: 20px; margin: 20px 0; }
            .messages { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
            .message { margin: 5px 0; padding: 5px; background: #fff; border-left: 3px solid #007bff; }
            .input-group { margin: 10px 0; }
            .input-group label { display: block; margin-bottom: 5px; }
            .input-group input { width: 100%; padding: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .status { margin: 10px 0; padding: 10px; background: #d4edda; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Play Buni WebSocket Test</h1>
            
            <div class="connection-form">
                <h3>Connection Settings</h3>
                <div class="input-group">
                    <label>User ID:</label>
                    <input type="text" id="userId" value="test_user_001" />
                </div>
                <div class="input-group">
                    <label>JWT Token (optional):</label>
                    <input type="text" id="token" />
                </div>
                <div class="input-group">
                    <label>Wallet Address (optional):</label>
                    <input type="text" id="walletAddress" />
                </div>
                <button onclick="connect()">Connect WebSocket</button>
                <button onclick="connectPremium()">Connect Premium</button>
                <button onclick="disconnect()">Disconnect</button>
            </div>
            
            <div class="status" id="status">Not connected</div>
            
            <div class="input-group">
                <label>Send Message:</label>
                <input type="text" id="messageInput" placeholder="Enter JSON message" />
                <button onclick="sendMessage()">Send</button>
                <button onclick="sendPing()">Send Ping</button>
            </div>
            
            <h3>Messages</h3>
            <div class="messages" id="messages"></div>
        </div>

        <script>
            let ws = null;
            let isConnected = false;

            function updateStatus(message, isError = false) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.style.background = isError ? '#f8d7da' : '#d4edda';
                status.style.borderColor = isError ? '#f5c6cb' : '#c3e6cb';
            }

            function addMessage(message) {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';
                messageDiv.innerHTML = `<strong>${new Date().toLocaleTimeString()}</strong>: ${JSON.stringify(message, null, 2)}`;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }

            function connect() {
                if (isConnected) {
                    updateStatus('Already connected', true);
                    return;
                }

                const userId = document.getElementById('userId').value;
                const token = document.getElementById('token').value;
                const walletAddress = document.getElementById('walletAddress').value;
                
                let url = `ws://localhost:8000/ws?user_id=${userId}`;
                if (token) url += `&token=${token}`;
                if (walletAddress) url += `&wallet_address=${walletAddress}`;

                ws = new WebSocket(url);

                ws.onopen = function(event) {
                    isConnected = true;
                    updateStatus('Connected to WebSocket');
                };

                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    addMessage(message);
                };

                ws.onclose = function(event) {
                    isConnected = false;
                    updateStatus(`Disconnected: ${event.code} - ${event.reason}`, true);
                };

                ws.onerror = function(error) {
                    updateStatus('WebSocket error', true);
                    console.error('WebSocket error:', error);
                };
            }

            function connectPremium() {
                if (isConnected) {
                    updateStatus('Already connected', true);
                    return;
                }

                const userId = document.getElementById('userId').value;
                const token = document.getElementById('token').value;
                const walletAddress = document.getElementById('walletAddress').value;
                
                if (!token || !walletAddress) {
                    updateStatus('Premium connection requires token and wallet address', true);
                    return;
                }

                const url = `ws://localhost:8000/ws/premium?user_id=${userId}&token=${token}&wallet_address=${walletAddress}`;
                
                ws = new WebSocket(url);

                ws.onopen = function(event) {
                    isConnected = true;
                    updateStatus('Connected to Premium WebSocket');
                };

                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    addMessage(message);
                };

                ws.onclose = function(event) {
                    isConnected = false;
                    updateStatus(`Premium Disconnected: ${event.code} - ${event.reason}`, true);
                };

                ws.onerror = function(error) {
                    updateStatus('Premium WebSocket error', true);
                    console.error('WebSocket error:', error);
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                    isConnected = false;
                    updateStatus('Disconnected');
                }
            }

            function sendMessage() {
                if (!isConnected) {
                    updateStatus('Not connected', true);
                    return;
                }

                const messageInput = document.getElementById('messageInput');
                try {
                    const message = JSON.parse(messageInput.value);
                    ws.send(JSON.stringify(message));
                    messageInput.value = '';
                } catch (e) {
                    updateStatus('Invalid JSON message', true);
                }
            }

            function sendPing() {
                if (!isConnected) {
                    updateStatus('Not connected', true);
                    return;
                }

                ws.send(JSON.stringify({type: 'ping', timestamp: new Date().toISOString()}));
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

async def handle_client_message(connection_id: str, message: dict):
    """Handle incoming client messages"""
    message_type = message.get("type")
    
    if message_type == "ping":
        await websocket_manager.handle_ping(connection_id)
    
    elif message_type == "refresh_nft":
        await websocket_manager.refresh_nft_status(connection_id)
    
    elif message_type == "subscribe":
        # Handle room subscription requests
        room = message.get("room")
        if room and connection_id in websocket_manager.active_connections:
            connection = websocket_manager.active_connections[connection_id]
            
            # Check permissions for premium rooms
            if room in ["premium_signals", "market_data", "alerts"] and not connection.nft_verified:
                await websocket_manager.send_personal_message(
                    connection_id,
                    WebSocketMessage(
                        type="subscription_denied",
                        data={
                            "room": room,
                            "reason": "NFT verification required"
                        },
                        timestamp=datetime.now(timezone.utc)
                    )
                )
                return
            
            # Add to room
            if room in websocket_manager.rooms:
                websocket_manager.rooms[room].add(connection_id)
                connection.rooms.add(room)
                
                await websocket_manager.send_personal_message(
                    connection_id,
                    WebSocketMessage(
                        type="subscribed",
                        data={"room": room},
                        timestamp=datetime.now(timezone.utc)
                    )
                )
    
    elif message_type == "unsubscribe":
        # Handle room unsubscription
        room = message.get("room")
        if room and connection_id in websocket_manager.active_connections:
            connection = websocket_manager.active_connections[connection_id]
            
            if room in websocket_manager.rooms:
                websocket_manager.rooms[room].discard(connection_id)
                connection.rooms.discard(room)
                
                await websocket_manager.send_personal_message(
                    connection_id,
                    WebSocketMessage(
                        type="unsubscribed",
                        data={"room": room},
                        timestamp=datetime.now(timezone.utc)
                    )
                )
    
    else:
        logger.warning(f"Unknown message type from {connection_id}: {message_type}")

async def handle_premium_client_message(connection_id: str, message: dict):
    """Handle incoming premium client messages"""
    message_type = message.get("type")
    
    if message_type == "ping":
        await websocket_manager.handle_ping(connection_id)
    
    elif message_type == "get_recent_signals":
        # Fetch recent signals for premium user
        try:
            signal_service = SignalService()
            recent_signals = await signal_service.get_recent_signals(limit=10)
            
            await websocket_manager.send_personal_message(
                connection_id,
                WebSocketMessage(
                    type="recent_signals",
                    data={"signals": recent_signals},
                    timestamp=datetime.now(timezone.utc)
                )
            )
        except Exception as e:
            logger.error(f"Failed to fetch recent signals: {e}")
    
    elif message_type == "get_market_data":
        # Fetch current market data
        try:
            market_service = MarketDataService()
            market_data = await market_service.get_current_market_data()
            
            await websocket_manager.send_personal_message(
                connection_id,
                WebSocketMessage(
                    type="market_data",
                    data=market_data,
                    timestamp=datetime.now(timezone.utc)
                )
            )
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
    
    else:
        # Handle standard messages
        await handle_client_message(connection_id, message) 