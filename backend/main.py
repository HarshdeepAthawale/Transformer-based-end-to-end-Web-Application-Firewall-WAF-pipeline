"""
FastAPI Application Entry Point
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from loguru import logger

from backend.config import config, redact_database_url
from backend.database import init_db, close_db, get_db
from backend.routes import (
    metrics,
    alerts,
    activities,
    charts,
    traffic,
    threats,
    analytics,
    events,
    stats,
    firewall_ai,
    ddos,
    credential_leak,
    dashboard,
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

    # Require secrets and secure defaults in production
    if os.getenv("ENV", "").lower() == "production":
        secret = getattr(config, "JWT_SECRET", "") or ""
        if not (secret and secret.strip()):
            logger.critical(
                "JWT_SECRET is required in production. Set JWT_SECRET to a long random string in .env. "
                "See .env.example for details."
            )
            raise SystemExit(1)
        admin_pw = os.getenv("ADMIN_PASSWORD", "admin123")
        if admin_pw == "admin123":
            logger.critical(
                "ADMIN_PASSWORD must be changed from default in production. "
                "Set ADMIN_PASSWORD in .env or set SEED_ADMIN=false."
            )
            raise SystemExit(1)
        if os.getenv("SEED_ADMIN", "true").lower() != "false":
            logger.warning(
                "SEED_ADMIN is enabled in production. Set SEED_ADMIN=false in .env for production."
            )

    logger.info(f"Database URL: {redact_database_url(config.DATABASE_URL)}")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Sync IP blacklist to Redis for gateway enforcement (B2B scalable)
    try:
        from backend.database import SessionLocal
        from backend.services.ip_fencing import IPFencingService
        from backend.services.blacklist_sync import sync_full_blacklist, REDIS_REQUIRED_MSG

        db = SessionLocal()
        try:
            service = IPFencingService(db)
            entries = service.get_blacklist(limit=10000)
            count = sync_full_blacklist(entries)
            if count > 0:
                logger.info(f"Blacklist synced to Redis: {count} IPs")
        finally:
            db.close()
    except RuntimeError as e:
        if REDIS_REQUIRED_MSG in str(e):
            logger.error(str(e))
            raise
        raise
    except Exception as e:
        logger.debug(f"Blacklist sync on startup skipped: {e}")

    # Create WAF service via core factory and store in app state
    try:
        from backend.core.waf_factory import create_waf_service, is_model_available

        app.state.waf_service = create_waf_service()
        if app.state.waf_service:
            logger.info("WAF service created and stored in app.state")
        else:
            logger.warning("WAF service not available (model/vocab may be missing)")
            if config.WAF_ENABLED and os.getenv("WAF_REQUIRE_MODEL", "false").lower() == "true":
                model_ok = is_model_available()
                if not model_ok:
                    logger.critical(
                        "WAF_ENABLED=true and WAF_REQUIRE_MODEL=true but model is missing. "
                        "Ensure models/waf-distilbert exists with config.json, tokenizer.json, and model weights. "
                        "Exiting."
                    )
                    raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        logger.warning(f"Could not create WAF service: {e}")
        app.state.waf_service = None

    # Start background workers (thread mode by default; Celery mode if WAF_USE_CELERY=true)
    from backend.tasks.scheduler import start_background_workers

    use_celery = os.getenv("WAF_USE_CELERY", "false").lower() == "true"
    logger.info(f"Task queue mode: {'celery' if use_celery else 'threads'}")

    try:
        start_background_workers()
        logger.info("Background workers started")
    except Exception as e:
        logger.warning(f"Failed to start some background workers: {e}")

    # Start continuous learning scheduler (optional)
    learning_scheduler = None
    if os.getenv("LEARNING_ENABLED", "false").lower() == "true":
        log_path = getattr(config, "LOG_PATH", None) or os.getenv("LOG_PATH")
        if log_path:
            try:
                from backend.ml.learning.scheduler import LearningScheduler

                learning_scheduler = LearningScheduler(
                    log_path=log_path,
                    model_path=os.getenv("WAF_MODEL_PATH", "models/waf-distilbert"),
                    update_interval_hours=int(os.getenv("LEARNING_UPDATE_INTERVAL_HOURS", "24")),
                )
                learning_scheduler.start()
                logger.info("Continuous learning scheduler started")
            except Exception as e:
                logger.warning(f"Learning scheduler not started: {e}")
        else:
            logger.debug("LEARNING_ENABLED=true but LOG_PATH not set, skipping scheduler")

    yield

    # Shutdown
    if learning_scheduler is not None:
        try:
            learning_scheduler.stop()
        except Exception:
            pass
    logger.info("Shutting down WAF API Server...")
    from backend.tasks.scheduler import stop_background_workers
    try:
        stop_background_workers()
    except Exception as e:
        logger.warning(f"Error stopping background workers: {e}")
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
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Requested-With"],
)

# Rate limiting - after CORS, before WAF
try:
    from backend.middleware.rate_limit_middleware import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware)
except Exception as e:
    logger.warning(f"Rate limit middleware not available: {e}")

# Agent-specific rate limiting
try:
    from backend.middleware.agent_rate_limit import AgentRateLimitMiddleware

    app.add_middleware(AgentRateLimitMiddleware)
except Exception as e:
    logger.warning(f"Agent rate limit middleware not available: {e}")

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

# Usage limit enforcement (billing quota)
try:
    from backend.middleware.usage_limit_middleware import UsageLimitMiddleware

    app.add_middleware(UsageLimitMiddleware)
except ImportError:
    logger.warning("Usage limit middleware not available")


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
    content = {
        "success": False,
        "message": "Internal server error",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Never leak exception details in responses — they are already logged above
    return JSONResponse(status_code=500, content=content)


# Health, test, debug routers (no prefix)
app.include_router(health.router)
app.include_router(test.router)
app.include_router(debug.router)


# Prometheus metrics (Feature 10)
@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics(db=Depends(get_db)):
    """Prometheus exposition format. Data from security_events (last 60 min). Set PROMETHEUS_METRICS_ENABLED=false to disable."""
    if not getattr(config, "PROMETHEUS_METRICS_ENABLED", True):
        raise HTTPException(status_code=404, detail="Metrics disabled")
    from backend.prometheus_metrics import get_prometheus_text
    return get_prometheus_text(db, window_minutes=60)

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
app.include_router(events.router, prefix="/api/events", tags=["events"])
try:
    from backend.routes import scan
    app.include_router(scan.router, prefix="/api/scan", tags=["scan"])
    logger.info("✓ Registered routes: /api/scan")
except (ImportError, RuntimeError) as e:
    _scan_stub = APIRouter()
    _scan_detail = "Upload scan not available. Install python-multipart: pip install python-multipart and restart the backend."
    @_scan_stub.post("/upload")
    def _scan_stub_upload():
        raise HTTPException(status_code=503, detail=_scan_detail)
    app.include_router(_scan_stub, prefix="/api/scan", tags=["scan"])
    logger.warning(f"Scan route stub registered (install python-multipart for /api/scan/upload): {e}")
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(firewall_ai.router, prefix="/api/firewall-ai", tags=["firewall-ai"])
app.include_router(ddos.router, prefix="/api/ddos", tags=["ddos"])
app.include_router(credential_leak.router, prefix="/api/credential-leak", tags=["credential-leak"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

# Feature 9: WAF API (security rules, rate limit config)
try:
    from backend.routes import security_rules, rate_limits
    app.include_router(security_rules.router, prefix="/api/security-rules", tags=["security-rules"])
    app.include_router(rate_limits.router, prefix="/api/rate-limits", tags=["rate-limits"])
    logger.info("✓ Registered routes: /api/security-rules, /api/rate-limits")
except ImportError as e:
    logger.warning(f"WAF API routes not available: {e}")

# Billing and subscriptions (Razorpay)
try:
    from backend.routes import billing, billing_webhooks
    app.include_router(billing.router, prefix="/api/billing", tags=["billing"])
    app.include_router(billing_webhooks.router, prefix="/api/billing/webhooks", tags=["billing-webhooks"])
    logger.info("Registered routes: /api/billing, /api/billing/webhooks")
except ImportError as e:
    logger.warning(f"Billing routes not available: {e}")

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
# Bot management (score, verified bots, score bands) - /api/bot
try:
    from backend.routes import bot

    app.include_router(bot.router, prefix="/api/bot", tags=["bot-management"])
    logger.info("✓ Registered routes: /api/bot")
except ImportError as e:
    logger.error(f"✗ Failed to import bot routes: {e}")

advanced_routes = [
    ("settings", "/api/settings", "settings"),
    ("ip_management", "/api/ip", "ip-management"),
    ("geo_rules", "/api/geo", "geo-fencing"),
    ("bot_detection", "/api/bots", "bot-detection"),
    ("threat_intel", "/api/threat-intel", "threat-intelligence"),
    ("security_rules", "/api/rules", "security-rules"),
    ("users", "/api/users", "users"),
    ("audit", "/api/audit", "audit"),
    ("agent", "/api/agent", "ai-agent"),
]

for module_name, prefix, tag in advanced_routes:
    try:
        module = __import__(f"backend.routes.{module_name}", fromlist=[module_name])
        router = getattr(module, "router")
        app.include_router(router, prefix=prefix, tags=[tag])
        logger.info(f"✓ Registered routes: {prefix}")
    except ImportError as e:
        logger.error(f"✗ Failed to import {module_name}: {e}")
        if module_name == "users":
            logger.error("Install auth deps: pip install 'PyJWT>=2.8.0' then restart the backend.")
            _stub = APIRouter()
            _detail = "Users module not loaded. Install PyJWT: pip install 'PyJWT>=2.8.0' and restart the backend."

            @_stub.get("")
            @_stub.post("")
            def _users_stub_base():
                raise HTTPException(status_code=503, detail=_detail)

            @_stub.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
            def _users_stub_path(path: str):
                raise HTTPException(status_code=503, detail=_detail)

            app.include_router(_stub, prefix=prefix, tags=[tag])
            logger.info(f"✓ Registered stub route: {prefix} (returns 503)")
        elif module_name == "audit":
            logger.error("Install auth deps: pip install 'PyJWT>=2.8.0' then restart the backend.")
            _audit_stub = APIRouter()
            _audit_detail = "Audit logs not available. Install PyJWT: pip install 'PyJWT>=2.8.0' and restart the backend."

            @_audit_stub.get("/logs")
            def _audit_stub_logs():
                raise HTTPException(status_code=503, detail=_audit_detail)

            @_audit_stub.get("/logs/{log_id}")
            def _audit_stub_log(log_id: int):
                raise HTTPException(status_code=503, detail=_audit_detail)

            app.include_router(_audit_stub, prefix=prefix, tags=[tag])
            logger.info(f"✓ Registered stub route: {prefix} (returns 503)")
    except AttributeError as e:
        logger.error(f"✗ {module_name} missing 'router': {e}")
    except Exception as e:
        logger.error(f"✗ Error registering {module_name}: {e}")

# WebSocket router
if config.WEBSOCKET_ENABLED:
    app.include_router(websocket_router, prefix="/ws", tags=["websocket"])

# Phase 1 — Edge Network & Performance (cache, DNS, SSL)
try:
    from backend.routes import cache
    app.include_router(cache.router, prefix="/api/v1/cache", tags=["cache"])
    logger.info("✓ Registered routes: /api/v1/cache")
except ImportError as e:
    logger.warning(f"Cache routes not available: {e}")
try:
    from backend.routes import dns
    app.include_router(dns.router, prefix="/api/v1/dns", tags=["dns"])
    logger.info("✓ Registered routes: /api/v1/dns")
except ImportError as e:
    logger.warning(f"DNS routes not available: {e}")
try:
    from backend.routes import ssl
    app.include_router(ssl.router, prefix="/api/v1/ssl", tags=["ssl"])
    logger.info("✓ Registered routes: /api/v1/ssl")
except ImportError as e:
    logger.warning(f"SSL routes not available: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=1,  # Use 1 worker for development
        reload=True,
    )
