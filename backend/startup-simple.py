#!/usr/bin/env python3
"""
Simplified Startup Script for Railway Deployment
Minimal version that starts FastAPI without complex dependencies
"""
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Simple startup function"""
    try:
        import uvicorn
        
        # Get port from environment (Railway sets this)
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        
        logger.info(f"Starting Play Buni Platform on {host}:{port}")
        
        # Start the FastAPI server
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            workers=1,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Try to start with minimal config
        try:
            from fastapi import FastAPI
            app = FastAPI(title="Play Buni Platform")
            
            @app.get("/")
            def root():
                return {"message": "Play Buni Platform is starting up!"}
            
            @app.get("/health")
            def health():
                return {"status": "healthy", "message": "Basic health check"}
            
            uvicorn.run(app, host=host, port=port)
            
        except Exception as e2:
            logger.error(f"Minimal startup also failed: {e2}")
            raise

if __name__ == "__main__":
    main() 