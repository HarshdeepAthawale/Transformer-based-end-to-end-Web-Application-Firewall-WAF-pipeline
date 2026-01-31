"""
FastAPI Application Entry Point
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from backend.config import config
from backend.database import init_db, close_db
from backend.routes import (
    metrics,
    alerts,
    activities,
    charts,
    traffic,
    threats,
    analytics,
)
from backend.routes import health, test, debug
from backend.websocket import router as websocket_router

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

    # Create WAF service via core factory and store in app state
    try:
        from backend.core.waf_factory import create_waf_service

        app.state.waf_service = create_waf_service()
        if app.state.waf_service:
            logger.info("WAF service created and stored in app.state")
        else:
            logger.warning("WAF service not available (model/vocab may be missing)")
    except Exception as e:
        logger.warning(f"Could not create WAF service: {e}")
        app.state.waf_service = None

    # Start background workers
    from backend.tasks.scheduler import start_background_workers

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
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting - after CORS, before WAF
try:
    from backend.middleware.rate_limit_middleware import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware)
except Exception as e:
    logger.warning(f"Rate limit middleware not available: {e}")

# WAF Middleware - Must be after CORS but before other middleware
try:
    from backend.middleware.waf_middleware import WAFMiddleware

    if config.WAF_ENABLED:
        # WAF service will be initialized lazily in middleware
        app.add_middleware(WAFMiddleware, waf_service=None)
        logger.info("✓ WAF middleware enabled")
    else:
        logger.info("WAF middleware disabled (WAF_ENABLED=false)")
except ImportError as e:
    logger.warning(f"WAF middleware not available: {e}")
except Exception as e:
    logger.error(f"Error setting up WAF middleware: {e}")

# Audit logging middleware
try:
    from backend.middleware.audit_middleware import AuditMiddleware

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
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "errors": exc.errors(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


# Health, test, debug routers (no prefix)
app.include_router(health.router)
app.include_router(test.router)
app.include_router(debug.router)

# Test target router - goes through WAF middleware (not in /api/ path)
try:
    from backend.routes import test_target

    app.include_router(test_target.router, tags=["test-target"])
    logger.info("✓ Registered routes: /test/*")
except ImportError as e:
    logger.warning(f"Test target routes not available: {e}")

# Include routers
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(activities.router, prefix="/api/activities", tags=["activities"])
app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(traffic.router, prefix="/api/traffic", tags=["traffic"])
app.include_router(threats.router, prefix="/api/threats", tags=["threats"])

app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])

# WAF service routes (includes /middleware-metrics)
try:
    from backend.routes import waf

    app.include_router(waf.router, prefix="/api/waf", tags=["waf"])
    logger.info("✓ Registered routes: /api/waf")
except ImportError as e:
    logger.warning(f"WAF routes not available: {e}")
except Exception as e:
    logger.error(f"Error registering WAF routes: {e}")

# Advanced feature routes - register individually for better error handling
advanced_routes = [
    ("ip_management", "/api/ip", "ip-management"),
    ("geo_rules", "/api/geo", "geo-fencing"),
    ("bot_detection", "/api/bots", "bot-detection"),
    ("threat_intel", "/api/threat-intel", "threat-intelligence"),
    ("security_rules", "/api/rules", "security-rules"),
    ("users", "/api/users", "users"),
    ("audit", "/api/audit", "audit"),
]

for module_name, prefix, tag in advanced_routes:
    try:
        module = __import__(f"backend.routes.{module_name}", fromlist=[module_name])
        router = getattr(module, "router")
        app.include_router(router, prefix=prefix, tags=[tag])
        logger.info(f"✓ Registered routes: {prefix}")
    except ImportError as e:
        logger.error(f"✗ Failed to import {module_name}: {e}")
    except AttributeError as e:
        logger.error(f"✗ {module_name} missing 'router': {e}")
    except Exception as e:
        logger.error(f"✗ Error registering {module_name}: {e}")

# WebSocket router
if config.WEBSOCKET_ENABLED:
    app.include_router(websocket_router, prefix="/ws", tags=["websocket"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=1,  # Use 1 worker for development
        reload=True,
    )
