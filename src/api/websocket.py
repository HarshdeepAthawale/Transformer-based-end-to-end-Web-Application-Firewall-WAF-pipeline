"""
WebSocket Server for Real-time Updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
from datetime import datetime
from loguru import logger


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, List[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = []
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to a specific connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message_type: str, data: dict):
        """Broadcast message to all connected clients"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        message_json = json.dumps(message)
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()

# Router
router = APIRouter()


@router.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive ping/pong messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                        websocket
                    )
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Function to broadcast updates (called by background services)
async def broadcast_update(message_type: str, data: dict):
    """Broadcast update to all connected clients"""
    await manager.broadcast(message_type, data)
