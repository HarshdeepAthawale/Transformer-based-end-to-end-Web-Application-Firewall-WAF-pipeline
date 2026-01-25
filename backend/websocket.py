"""
WebSocket Server for Real-time Updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Set
import json
import asyncio
from datetime import datetime
from loguru import logger


class ConnectionManager:
    """Manages WebSocket connections with subscription filtering"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self._periodic_task: asyncio.Task = None
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()  # All message types by default
        self.connection_metadata[websocket] = {
            "connected_at": datetime.utcnow().isoformat(),
            "last_ping": datetime.utcnow().isoformat()
        }
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send connection confirmation
        await self.send_personal_message(
            json.dumps({
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat()
            }),
            websocket
        )
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to a specific connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message_type: str, data: dict, filter_subscriptions: bool = True):
        """Broadcast message to all connected clients with subscription filtering"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        message_json = json.dumps(message)
        
        disconnected = []
        for connection in self.active_connections:
            try:
                # Check if connection is subscribed to this message type
                if filter_subscriptions:
                    subscriptions = self.subscriptions.get(connection, set())
                    # If no subscriptions, send all messages (backward compatibility)
                    # If subscriptions exist, only send if subscribed
                    if subscriptions and message_type not in subscriptions:
                        continue
                
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def subscribe(self, websocket: WebSocket, message_types: List[str]):
        """Subscribe connection to specific message types"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(message_types)
            logger.debug(f"Connection subscribed to: {message_types}")
    
    async def unsubscribe(self, websocket: WebSocket, message_types: List[str]):
        """Unsubscribe connection from specific message types"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].difference_update(message_types)
            logger.debug(f"Connection unsubscribed from: {message_types}")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    async def start_periodic_metrics(self, interval: int = 10):
        """Start periodic metrics broadcasting"""
        if self._periodic_task and not self._periodic_task.done():
            return
        
        async def periodic_broadcast():
            from backend.database import SessionLocal
            from backend.services.metrics_service import MetricsService
            
            while True:
                try:
                    await asyncio.sleep(interval)
                    
                    # Get latest metrics
                    db = SessionLocal()
                    try:
                        metrics_service = MetricsService(db)
                        metrics = metrics_service.get_realtime_metrics()
                        await self.broadcast("metrics", metrics, filter_subscriptions=True)
                    finally:
                        db.close()
                        
                except Exception as e:
                    logger.error(f"Error in periodic metrics broadcast: {e}")
                    await asyncio.sleep(interval)
        
        self._periodic_task = asyncio.create_task(periodic_broadcast())
        logger.info(f"Started periodic metrics broadcasting (interval: {interval}s)")
    
    async def stop_periodic_metrics(self):
        """Stop periodic metrics broadcasting"""
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped periodic metrics broadcasting")


# Global connection manager
manager = ConnectionManager()

# Router
router = APIRouter()


@router.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with subscription support"""
    await manager.connect(websocket)
    
    # Start periodic metrics if not already started
    if manager._periodic_task is None or manager._periodic_task.done():
        await manager.start_periodic_metrics(interval=10)
    
    try:
        while True:
            # Receive messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    # Update last ping time
                    if websocket in manager.connection_metadata:
                        manager.connection_metadata[websocket]["last_ping"] = datetime.utcnow().isoformat()
                    
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                        websocket
                    )
                
                elif msg_type == "subscribe":
                    # Subscribe to message types
                    message_types = message.get("message_types", [])
                    if message_types:
                        await manager.subscribe(websocket, message_types)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "subscribed",
                                "message_types": message_types,
                                "timestamp": datetime.utcnow().isoformat()
                            }),
                            websocket
                        )
                
                elif msg_type == "unsubscribe":
                    # Unsubscribe from message types
                    message_types = message.get("message_types", [])
                    if message_types:
                        await manager.unsubscribe(websocket, message_types)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "unsubscribed",
                                "message_types": message_types,
                                "timestamp": datetime.utcnow().isoformat()
                            }),
                            websocket
                        )
                
                elif msg_type == "get_subscriptions":
                    # Get current subscriptions
                    subscriptions = list(manager.subscriptions.get(websocket, set()))
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "subscriptions",
                            "message_types": subscriptions,
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        websocket
                    )
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from WebSocket: {data}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Function to broadcast updates (called by background services)
async def broadcast_update(message_type: str, data: dict):
    """Broadcast update to all connected clients"""
    await manager.broadcast(message_type, data)
