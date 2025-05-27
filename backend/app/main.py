"""
Play Buni Platform - Main FastAPI Application

NFT-gated AI trading signals platform on Solana with:
- Real-time signal generation and distribution
- NFT-based access control
- Solana Blinks integration with fee collection
- Social media automation
- Premium WebSocket features
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .core.config import settings
from .core.logging import configure_logging
from .core.security import verify_api_key
from .core.access_control import AccessControlMiddleware
from .core.cache import init_cache, close_cache
from .database import get_db_session, init_db
from .services.signal_engine import signal_generation_engine
from .services.signal_distributor import signal_distribution_coordinator
from .services.treasury_manager import treasury_manager
from .workers.websocket_streaming import streaming_worker

# Import routers
from .routers import (
    auth,
    signals, 
    blinks,
    social,
    premium,
    health,
    admin,
    nft_verification,
    websocket,
    jupiter_monitoring
)

# Configure structured logging
configure_logging()
logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        # Log request
        logger.info(
            "request_started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = asyncio.get_event_loop().time() - start_time
            
            # Log response
            logger.info(
                "request_completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = asyncio.get_event_loop().time() - start_time
            
            logger.error(
                "request_failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time,
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.client_requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        current_time = asyncio.get_event_loop().time()
        
        # Clean old requests (older than 1 minute)
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls_per_minute} requests per minute allowed"
                }
            )
        
        # Add current request
        self.client_requests[client_ip].append(current_time)
        
        return await call_next(request)


# Global application state
app_state = {
    "signal_generation_engine": None,
    "signal_distribution_coordinator": None,
    "startup_complete": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    
    logger.info("Starting Play Buni Platform...")
    
    try:
        # Initialize database
        logger.info("Initializing database connection...")
        await init_db()
        
        # Initialize cache
        logger.info("Initializing cache manager...")
        await init_cache()
        
        # Initialize core services
        logger.info("Initializing Signal Generation Engine...")
        await signal_generation_engine.initialize()
        app_state["signal_generation_engine"] = signal_generation_engine
        
        logger.info("Initializing Signal Distribution Coordinator...")
        await signal_distribution_coordinator.initialize()
        app_state["signal_distribution_coordinator"] = signal_distribution_coordinator
        
        logger.info("Initializing Treasury Manager...")
        await treasury_manager.initialize()
        app_state["treasury_manager"] = treasury_manager
        
        # Start background services
        if settings.environment != "testing":
            logger.info("Starting background services...")
            
            # Start signal generation loop
            asyncio.create_task(signal_generation_engine.start_generation_loop())
            
            # Start signal distribution loop
            asyncio.create_task(signal_distribution_coordinator.start_distribution_loop())
            
            # Start WebSocket streaming worker
            asyncio.create_task(streaming_worker.start_streaming())
        
        app_state["startup_complete"] = True
        logger.info("Play Buni Platform startup complete!")
        
        yield
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Play Buni Platform...")
        
        if app_state["signal_generation_engine"]:
            await app_state["signal_generation_engine"].stop_generation_loop()
        
        if app_state["signal_distribution_coordinator"]:
            await app_state["signal_distribution_coordinator"].stop_distribution_loop()
        
        if app_state["treasury_manager"]:
            await app_state["treasury_manager"].close()
        
        # Stop WebSocket streaming worker
        await streaming_worker.stop_streaming()
        
        # Close cache
        await close_cache()
        
        logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Play Buni Platform API",
    description="NFT-gated AI trading signals platform on Solana",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware (order matters!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    max_age=86400  # 24 hours
)

app.add_middleware(RequestLoggingMiddleware)

# Add access control middleware for NFT-gated features
app.add_middleware(AccessControlMiddleware)

if settings.enable_rate_limiting:
    app.add_middleware(RateLimitMiddleware, calls_per_minute=settings.rate_limit_per_minute)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with structured logging."""
    
    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    
    logger.warning(
        "validation_error",
        errors=exc.errors(),
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    
    logger.error(
        "unexpected_error",
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        method=request.method,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with platform information."""
    return {
        "name": "Play Buni Platform",
        "description": "NFT-gated AI trading signals platform on Solana",
        "version": "1.0.0",
        "status": "operational" if app_state["startup_complete"] else "starting",
        "docs": "/docs" if settings.debug else None
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    
    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {
            "api": "healthy",
            "database": "unknown",
            "signal_agent": "unknown",
            "nft_verifier": "unknown",
            "social_distributor": "unknown"
        }
    }
    
    try:
        # Check database connection
        async with get_db_session() as db:
            await db.execute("SELECT 1")
            health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check signal generation engine
    if app_state["signal_generation_engine"]:
        health_status["services"]["signal_generation_engine"] = "healthy" if app_state["signal_generation_engine"].is_running else "stopped"
    
    # Check signal distribution coordinator
    if app_state["signal_distribution_coordinator"]:
        health_status["services"]["signal_distribution_coordinator"] = "healthy" if app_state["signal_distribution_coordinator"].is_running else "stopped"
    
    return health_status


# API versioning
@app.get("/api/v1")
async def api_v1_info():
    """API v1 information."""
    return {
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/v1/auth",
            "signals": "/api/v1/signals",
            "blinks": "/api/v1/blinks",
            "social": "/api/v1/social",
            "premium": "/api/v1/premium",
            "health": "/api/v1/health",
            "admin": "/api/v1/admin",
            "nft": "/api/v1/nft"
        }
    }


# Include routers with API versioning
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    signals.router,
    prefix="/api/v1/signals",
    tags=["Trading Signals"]
)

app.include_router(
    blinks.router,
    prefix="/api/v1/blinks",
    tags=["Solana Blinks"]
)

app.include_router(
    social.router,
    prefix="/api/v1/social",
    tags=["Social Distribution"]
)

app.include_router(
    premium.router,
    prefix="/api/v1/premium",
    tags=["Premium Features"]
)

app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["Health & Monitoring"]
)

app.include_router(
    admin.router,
    prefix="/api/v1/admin",
    tags=["Administration"],
    dependencies=[verify_api_key] if settings.environment == "production" else []
)

app.include_router(
    nft_verification.router,
    tags=["NFT Verification"]
)

app.include_router(
    websocket.router,
    prefix="/api/v1",
    tags=["WebSocket & Real-time"]
)

app.include_router(
    jupiter_monitoring.router,
    prefix="/api/v1",
    tags=["Jupiter API Monitoring"]
)


# WebSocket endpoint for real-time features
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket, user_id: str):
    """WebSocket endpoint for real-time signal delivery."""
    try:
        await websocket.accept()
        
        # Determine user access level (simplified for now)
        # In production, you'd verify the user's JWT token and check their NFT holdings
        access_level = "premium"  # This would be determined from user's NFT holdings
        
        # Add connection to distribution coordinator
        if app_state["signal_distribution_coordinator"]:
            await app_state["signal_distribution_coordinator"].add_websocket_connection(
                user_id, websocket, access_level
            )
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client (ping, subscription updates, etc.)
                data = await websocket.receive_text()
                
                # Handle client messages
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
            except Exception as e:
                logger.debug(f"WebSocket receive error for user {user_id}: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        # Remove connection
        if app_state["signal_distribution_coordinator"]:
            await app_state["signal_distribution_coordinator"].remove_websocket_connection(user_id)


# Startup event for additional initialization
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    logger.info("FastAPI startup event triggered")


# Shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup tasks on shutdown."""
    logger.info("FastAPI shutdown event triggered")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_config=None,  # Use our custom logging
        access_log=False  # Handled by our middleware
    ) 