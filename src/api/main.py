"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from loguru import logger

from src.api.config import config
from src.api.database import init_db, close_db
from src.api.routes import metrics, alerts, activities, charts, traffic, threats, security, analytics
from src.api.websocket import router as websocket_router

# Initialize logger
logger.add("logs/api_server.log", rotation="10 MB", retention="7 days")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting WAF API Server...")
    logger.info(f"Database URL: {config.DATABASE_URL}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Start background workers
    from src.api.tasks.scheduler import start_background_workers
    try:
        start_background_workers()
        logger.info("Background workers started")
    except Exception as e:
        logger.warning(f"Failed to start some background workers: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down WAF API Server...")
    close_db()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="WAF Dashboard API",
    description="API server for Transformer-based WAF Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit logging middleware
try:
    from src.api.middleware.audit_middleware import AuditMiddleware
    app.add_middleware(AuditMiddleware)
except ImportError:
    logger.warning("Audit middleware not available")


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "waf-api",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }


# Include routers
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(activities.router, prefix="/api/activities", tags=["activities"])
app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(traffic.router, prefix="/api/traffic", tags=["traffic"])
app.include_router(threats.router, prefix="/api/threats", tags=["threats"])
app.include_router(security.router, prefix="/api/security", tags=["security"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])

# Advanced feature routes
try:
    from src.api.routes import ip_management, geo_rules, bot_detection, threat_intel, security_rules, users, audit
    app.include_router(ip_management.router, prefix="/api/ip", tags=["ip-management"])
    app.include_router(geo_rules.router, prefix="/api/geo", tags=["geo-fencing"])
    app.include_router(bot_detection.router, prefix="/api/bots", tags=["bot-detection"])
    app.include_router(threat_intel.router, prefix="/api/threat-intel", tags=["threat-intelligence"])
    app.include_router(security_rules.router, prefix="/api/rules", tags=["security-rules"])
    app.include_router(users.router, prefix="/api/users", tags=["users"])
    app.include_router(audit.router, prefix="/api/audit", tags=["audit"])
except ImportError as e:
    logger.warning(f"Some advanced routes not available: {e}")

# WebSocket router
if config.WEBSOCKET_ENABLED:
    app.include_router(websocket_router, prefix="/ws", tags=["websocket"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=1,  # Use 1 worker for development
        reload=True
    )
