#!/usr/bin/env python3
"""
Production Startup Script for Play Buni Platform
Handles database initialization, Celery workers, and FastAPI server startup for Railway deployment
"""
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from multiprocessing import Process

import uvicorn
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import configure_logging
from app.database import init_db

# Configure logging first
configure_logging()
logger = logging.getLogger(__name__)

# Global process tracking
processes = []

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    # Terminate all background processes
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

async def initialize_database():
    """Initialize database and run migrations"""
    try:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialization completed")
        
        # Run Alembic migrations
        logger.info("Running database migrations...")
        migration_result = subprocess.run([
            "alembic", "upgrade", "head"
        ], capture_output=True, text=True, cwd="/app")
        
        if migration_result.returncode == 0:
            logger.info("Database migrations completed successfully")
        else:
            logger.warning(f"Migration warning: {migration_result.stderr}")
            # Don't fail on migration warnings, as tables might already exist
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # In production, we might want to retry or handle this differently
        # For now, log the error but continue
        pass

def start_celery_worker():
    """Start Celery worker process"""
    try:
        logger.info("Starting Celery worker...")
        subprocess.run([
            "celery", "-A", "app.workers.celery_app:celery_app", "worker",
            "--loglevel=info",
            "--concurrency=2",
            "--max-tasks-per-child=1000"
        ], check=False)
    except Exception as e:
        logger.error(f"Celery worker failed: {e}")

def start_celery_beat():
    """Start Celery beat scheduler"""
    try:
        logger.info("Starting Celery beat scheduler...")
        subprocess.run([
            "celery", "-A", "app.workers.celery_app:celery_app", "beat",
            "--loglevel=info"
        ], check=False)
    except Exception as e:
        logger.error(f"Celery beat failed: {e}")

def start_fastapi_server():
    """Start FastAPI server"""
    try:
        logger.info(f"Starting FastAPI server on {settings.host}:{settings.port}")
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            workers=1,  # Use 1 worker for Railway to avoid memory issues
            log_level="info",
            access_log=True,
            loop="asyncio",
            # Production optimizations
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30
        )
    except Exception as e:
        logger.error(f"FastAPI server failed: {e}")
        raise

async def main():
    """Main startup function"""
    logger.info("Starting Play Buni Platform in production mode...")
    
    try:
        # Initialize database
        await initialize_database()
        
        # For Railway deployment, we'll run everything in the main process
        # to avoid memory and process management issues
        
        # Check if we should start background workers
        if os.getenv("DISABLE_WORKERS") != "true":
            # Start Celery worker in background
            worker_process = Process(target=start_celery_worker)
            worker_process.start()
            processes.append(worker_process)
            logger.info("Celery worker started")
            
            # Start Celery beat in background
            beat_process = Process(target=start_celery_beat)
            beat_process.start()
            processes.append(beat_process)
            logger.info("Celery beat started")
            
            # Give workers time to start
            time.sleep(5)
        else:
            logger.info("Background workers disabled via DISABLE_WORKERS=true")
        
        # Start FastAPI server (this blocks)
        start_fastapi_server()
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Clean up processes
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except:
                pass
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 