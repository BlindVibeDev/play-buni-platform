"""
Minimal Play Buni Platform - FastAPI Application
Simplified version for initial Railway deployment
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI application
app = FastAPI(
    title="Play Buni Platform",
    description="NFT-gated AI trading signals platform on Solana",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Play Buni Platform",
        "description": "NFT-gated AI trading signals platform on Solana",
        "version": "1.0.0",
        "status": "operational",
        "message": "Welcome to Play Buni Platform! 🚀"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Play Buni Platform",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/api/v1")
async def api_info():
    """API information"""
    return {
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "api_info": "/api/v1",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 