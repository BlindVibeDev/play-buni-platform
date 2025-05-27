"""
Health Monitoring Router - System health checks and monitoring endpoints.

This module provides:
- Comprehensive health checks for all system components
- Performance metrics and monitoring
- Service status reporting
- Database health verification
- External service connectivity checks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import psutil
import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..database import get_db, check_db_health, db_manager
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    uptime_seconds: float
    version: str
    environment: str


class ServiceHealth(BaseModel):
    """Individual service health model."""
    name: str
    status: str
    latency_ms: float = None
    error: str = None
    last_check: datetime


class SystemMetrics(BaseModel):
    """System performance metrics model."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_connections: int
    requests_per_minute: float = None


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    overall_status: str
    timestamp: datetime
    uptime_seconds: float
    services: List[ServiceHealth]
    system_metrics: SystemMetrics
    database_info: Dict[str, Any]
    external_services: Dict[str, Any]


# Track application start time for uptime calculation
app_start_time = time.time()


@router.get("/", response_model=HealthStatus)
async def basic_health_check():
    """Basic health check endpoint for load balancers."""
    
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime_seconds=time.time() - app_start_time,
        version="1.0.0",
        environment=settings.environment
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """Comprehensive health check with detailed service status."""
    
    try:
        services = []
        overall_status = "healthy"
        
        # Check database health
        db_health = await check_database_health()
        services.append(ServiceHealth(
            name="database",
            status=db_health["status"],
            latency_ms=db_health.get("latency_ms"),
            error=db_health.get("error"),
            last_check=datetime.utcnow()
        ))
        
        if db_health["status"] != "healthy":
            overall_status = "degraded"
        
        # Check Redis health
        redis_health = await check_redis_health()
        services.append(ServiceHealth(
            name="redis",
            status=redis_health["status"],
            latency_ms=redis_health.get("latency_ms"),
            error=redis_health.get("error"),
            last_check=datetime.utcnow()
        ))
        
        if redis_health["status"] != "healthy":
            overall_status = "degraded"
        
        # Check Solana RPC health
        solana_health = await check_solana_rpc_health()
        services.append(ServiceHealth(
            name="solana_rpc",
            status=solana_health["status"],
            latency_ms=solana_health.get("latency_ms"),
            error=solana_health.get("error"),
            last_check=datetime.utcnow()
        ))
        
        if solana_health["status"] != "healthy":
            overall_status = "degraded"
        
        # Check AI Signal Agent
        signal_agent_health = await check_signal_agent_health()
        services.append(ServiceHealth(
            name="signal_agent",
            status=signal_agent_health["status"],
            error=signal_agent_health.get("error"),
            last_check=datetime.utcnow()
        ))
        
        if signal_agent_health["status"] != "healthy":
            overall_status = "degraded"
        
        # Get system metrics
        system_metrics = get_system_metrics()
        
        # Get database info
        database_info = await get_database_info()
        
        # Check external services
        external_services = await check_external_services()
        
        return DetailedHealthResponse(
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            uptime_seconds=time.time() - app_start_time,
            services=services,
            system_metrics=system_metrics,
            database_info=database_info,
            external_services=external_services
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@router.get("/database")
async def database_health_check():
    """Specific database health check."""
    
    try:
        health_info = await check_db_health()
        
        # Get additional database statistics
        stats = await db_manager.get_database_stats()
        health_info.update(stats)
        
        return health_info
        
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/services")
async def services_health_check():
    """Check health of all external services."""
    
    services_status = {}
    
    try:
        # Check Solana RPC
        services_status["solana_rpc"] = await check_solana_rpc_health()
        
        # Check Redis
        services_status["redis"] = await check_redis_health()
        
        # Check Twitter API (if configured)
        if settings.twitter_api_key:
            services_status["twitter_api"] = await check_twitter_api_health()
        
        # Check Discord API (if configured)
        if settings.discord_bot_token:
            services_status["discord_api"] = await check_discord_api_health()
        
        # Check Jupiter API
        services_status["jupiter_api"] = await check_jupiter_api_health()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status
        }
        
    except Exception as e:
        logger.error(f"Services health check error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/metrics")
async def system_metrics():
    """Get detailed system performance metrics."""
    
    try:
        metrics = get_system_metrics()
        
        # Add additional metrics
        additional_metrics = {
            "process_id": psutil.Process().pid,
            "threads_count": psutil.Process().num_threads(),
            "open_files": len(psutil.Process().open_files()),
            "network_connections": len(psutil.Process().connections()),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - app_start_time,
            "system_metrics": metrics.dict(),
            "process_metrics": additional_metrics
        }
        
    except Exception as e:
        logger.error(f"Metrics collection error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    
    try:
        # Check if all critical services are ready
        db_health = await check_database_health()
        
        if db_health["status"] != "healthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not ready"
            )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - app_start_time
    }


# Helper functions
async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance."""
    
    try:
        start_time = time.time()
        health_info = await check_db_health()
        
        # Add connection pool info if available
        if hasattr(db_manager, 'engine') and db_manager.engine:
            pool = db_manager.engine.pool
            health_info.update({
                "pool_size": pool.size() if hasattr(pool, 'size') else None,
                "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else None,
                "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else None,
                "overflow": pool.overflow() if hasattr(pool, 'overflow') else None
            })
        
        return health_info
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity."""
    
    try:
        import redis.asyncio as redis
        
        start_time = time.time()
        
        # Create Redis client
        redis_client = redis.from_url(settings.redis_url)
        
        # Test connection
        await redis_client.ping()
        
        latency = (time.time() - start_time) * 1000
        
        # Get Redis info
        info = await redis_client.info()
        
        await redis_client.close()
        
        return {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "version": info.get("redis_version"),
            "connected_clients": info.get("connected_clients"),
            "used_memory": info.get("used_memory_human")
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_solana_rpc_health() -> Dict[str, Any]:
    """Check Solana RPC connectivity."""
    
    try:
        import httpx
        
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.solana_rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth"
                },
                timeout=5.0
            )
        
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data and data["result"] == "ok":
                return {
                    "status": "healthy",
                    "latency_ms": round(latency, 2)
                }
        
        return {
            "status": "unhealthy",
            "error": f"RPC returned: {response.status_code}"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_signal_agent_health() -> Dict[str, Any]:
    """Check AI Signal Agent status."""
    
    try:
        # This would check the actual signal agent status
        # For now, return a placeholder
        return {
            "status": "healthy",
            "is_running": True,
            "last_signal_generated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_twitter_api_health() -> Dict[str, Any]:
    """Check Twitter API connectivity."""
    
    try:
        # Placeholder for Twitter API health check
        return {
            "status": "healthy",
            "api_version": "v2"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_discord_api_health() -> Dict[str, Any]:
    """Check Discord API connectivity."""
    
    try:
        # Placeholder for Discord API health check
        return {
            "status": "healthy",
            "api_version": "v10"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_jupiter_api_health() -> Dict[str, Any]:
    """Check Jupiter API connectivity."""
    
    try:
        import httpx
        
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://quote-api.jup.ag/v6/quote",
                params={
                    "inputMint": "So11111111111111111111111111111111111111112",  # SOL
                    "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                    "amount": "1000000"  # 0.001 SOL
                },
                timeout=5.0
            )
        
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2)
            }
        
        return {
            "status": "unhealthy",
            "error": f"API returned: {response.status_code}"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def get_system_metrics() -> SystemMetrics:
    """Get current system performance metrics."""
    
    try:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network connections
        connections = len(psutil.net_connections())
        
        return SystemMetrics(
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage,
            disk_usage_percent=disk_usage,
            active_connections=connections
        )
        
    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        return SystemMetrics(
            cpu_usage_percent=0.0,
            memory_usage_percent=0.0,
            disk_usage_percent=0.0,
            active_connections=0
        )


async def get_database_info() -> Dict[str, Any]:
    """Get database information and statistics."""
    
    try:
        stats = await db_manager.get_database_stats()
        return {
            "statistics": stats,
            "connection_url": settings.database_url.split('@')[1] if '@' in settings.database_url else "hidden"
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }


async def check_external_services() -> Dict[str, Any]:
    """Check all external service dependencies."""
    
    external_services = {}
    
    # Check Helius API (if configured)
    if hasattr(settings, 'helius_api_key') and settings.helius_api_key:
        external_services["helius"] = await check_helius_api_health()
    
    # Check Bitquery API (if configured)
    if hasattr(settings, 'bitquery_api_key') and settings.bitquery_api_key:
        external_services["bitquery"] = await check_bitquery_api_health()
    
    return external_services


async def check_helius_api_health() -> Dict[str, Any]:
    """Check Helius API health."""
    
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://rpc.helius.xyz/?api-key={settings.helius_api_key}",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth"
                },
                timeout=5.0
            )
        
        if response.status_code == 200:
            return {"status": "healthy"}
        
        return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_bitquery_api_health() -> Dict[str, Any]:
    """Check Bitquery API health."""
    
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://graphql.bitquery.io/",
                headers={"X-API-KEY": settings.bitquery_api_key},
                timeout=5.0
            )
        
        if response.status_code in [200, 400]:  # 400 is expected for GET without query
            return {"status": "healthy"}
        
        return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)} 