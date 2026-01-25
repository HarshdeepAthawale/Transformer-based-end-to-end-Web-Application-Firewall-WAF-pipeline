#!/usr/bin/env python3
"""
Start WAF API Server
"""
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import config
from backend.main import app


if __name__ == "__main__":
    print(f"Starting WAF API Server on {config.API_HOST}:{config.API_PORT}")
    print(f"Database: {config.DATABASE_URL}")
    print(f"WebSocket: {'Enabled' if config.WEBSOCKET_ENABLED else 'Disabled'}")
    
    # Use import string for reload to work properly
    uvicorn.run(
        "backend.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=1,  # Use 1 worker for development (WebSocket requires single worker)
        reload=True,
        log_level="info"
    )
