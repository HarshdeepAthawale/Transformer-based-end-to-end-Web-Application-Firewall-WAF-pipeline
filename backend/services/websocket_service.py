"""
WebSocket Broadcasting Service
"""
import asyncio
from typing import Dict, Any
from loguru import logger

# Global event loop for background thread broadcasting
_loop = None
_loop_thread = None


def get_or_create_loop():
    """Get or create event loop for background threads"""
    global _loop, _loop_thread
    
    if _loop is None or _loop.is_closed():
        import threading
        
        def run_loop():
            global _loop
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)
            _loop.run_forever()
        
        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()
        
        # Wait for loop to be ready
        import time
        while _loop is None:
            time.sleep(0.1)
    
    return _loop


def broadcast_update_sync(message_type: str, data: Dict[str, Any]):
    """Synchronously broadcast update from background thread"""
    try:
        from backend.websocket import manager
        
        loop = get_or_create_loop()
        
        # Schedule broadcast on event loop
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(message_type, data),
            loop
        )
    except Exception as e:
        logger.error(f"Failed to broadcast update: {e}")


def broadcast_update_async(message_type: str, data: Dict[str, Any]):
    """Asynchronously broadcast update (for async contexts)"""
    try:
        from backend.websocket import manager
        return manager.broadcast(message_type, data)
    except Exception as e:
        logger.error(f"Failed to broadcast update: {e}")
