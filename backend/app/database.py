"""
Database connection and session management for Supabase PostgreSQL.

This module provides:
- Async database connection management
- Session factory and dependency injection
- Database initialization and health checks
- Connection pooling and error handling
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from .core.config import settings

logger = logging.getLogger(__name__)

# Create declarative base for models
Base = declarative_base()

# Global engine and session factory
engine: AsyncEngine = None
async_session_factory: async_sessionmaker = None


def create_database_engine() -> AsyncEngine:
    """Create async database engine with proper configuration."""
    
    # Build connection URL
    database_url = settings.database_url
    
    # For async PostgreSQL, replace postgresql:// with postgresql+asyncpg://
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif not database_url.startswith("postgresql+asyncpg://"):
        database_url = f"postgresql+asyncpg://{database_url}"
    
    # Engine configuration
    engine_config = {
        "echo": settings.debug,  # Log SQL queries in debug mode
        "echo_pool": settings.debug,  # Log connection pool events
        "pool_pre_ping": True,  # Verify connections before use
        "pool_recycle": 3600,  # Recycle connections every hour
        "connect_args": {
            "server_settings": {
                "application_name": "play_buni_platform",
            }
        }
    }
    
    # Use NullPool for serverless environments
    if settings.environment in ["production", "staging"]:
        engine_config["poolclass"] = NullPool
    else:
        engine_config.update({
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
        })
    
    return create_async_engine(database_url, **engine_config)


async def init_db() -> None:
    """Initialize database connection and create tables if needed."""
    global engine, async_session_factory
    
    try:
        logger.info("Initializing database connection...")
        
        # Create engine
        engine = create_database_engine()
        
        # Create session factory
        async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        # Test connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        
        # Import models to ensure they're registered
        from . import models  # This will import all model modules
        
        # Create tables if they don't exist (for development)
        if settings.environment == "development":
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created/verified")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db() -> None:
    """Close database connections."""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with automatic cleanup.
    
    Usage:
        async with get_db_session() as db:
            # Use db session
            result = await db.execute(query)
    """
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage in FastAPI routes:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            # Use db session
    """
    async with get_db_session() as session:
        yield session


async def check_db_health() -> dict:
    """Check database health and return status."""
    health_info = {
        "status": "unknown",
        "connection": False,
        "latency_ms": None,
        "error": None
    }
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        async with get_db_session() as db:
            # Simple query to test connection
            result = await db.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            
            if row and row.test == 1:
                end_time = asyncio.get_event_loop().time()
                health_info.update({
                    "status": "healthy",
                    "connection": True,
                    "latency_ms": round((end_time - start_time) * 1000, 2)
                })
            else:
                health_info.update({
                    "status": "unhealthy",
                    "error": "Invalid query result"
                })
                
    except Exception as e:
        health_info.update({
            "status": "unhealthy",
            "error": str(e)
        })
    
    return health_info


async def execute_raw_sql(query: str, params: dict = None) -> list:
    """
    Execute raw SQL query and return results.
    
    Args:
        query: SQL query string
        params: Query parameters
        
    Returns:
        List of result rows
    """
    async with get_db_session() as db:
        if params:
            result = await db.execute(text(query), params)
        else:
            result = await db.execute(text(query))
        
        return result.fetchall()


class DatabaseManager:
    """Database manager for advanced operations."""
    
    def __init__(self):
        self.engine = engine
        self.session_factory = async_session_factory
    
    async def create_tables(self, tables: list = None):
        """Create specific tables or all tables."""
        async with self.engine.begin() as conn:
            if tables:
                # Create specific tables
                for table in tables:
                    await conn.run_sync(table.create, checkfirst=True)
            else:
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self, tables: list = None):
        """Drop specific tables or all tables."""
        async with self.engine.begin() as conn:
            if tables:
                # Drop specific tables
                for table in tables:
                    await conn.run_sync(table.drop, checkfirst=True)
            else:
                # Drop all tables
                await conn.run_sync(Base.metadata.drop_all)
    
    async def get_table_info(self, table_name: str) -> dict:
        """Get information about a specific table."""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = :table_name
        ORDER BY ordinal_position
        """
        
        async with get_db_session() as db:
            result = await db.execute(text(query), {"table_name": table_name})
            columns = result.fetchall()
            
            return {
                "table_name": table_name,
                "columns": [
                    {
                        "name": col.column_name,
                        "type": col.data_type,
                        "nullable": col.is_nullable == "YES",
                        "default": col.column_default
                    }
                    for col in columns
                ]
            }
    
    async def get_database_stats(self) -> dict:
        """Get database statistics."""
        queries = {
            "total_tables": """
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """,
            "database_size": """
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """,
            "active_connections": """
                SELECT COUNT(*) as count 
                FROM pg_stat_activity 
                WHERE state = 'active'
            """
        }
        
        stats = {}
        async with get_db_session() as db:
            for key, query in queries.items():
                try:
                    result = await db.execute(text(query))
                    row = result.fetchone()
                    stats[key] = row[0] if row else None
                except Exception as e:
                    stats[key] = f"Error: {str(e)}"
        
        return stats


# Create global database manager instance
db_manager = DatabaseManager()


# Migration utilities
async def run_migration(migration_sql: str, description: str = None):
    """Run a database migration."""
    logger.info(f"Running migration: {description or 'Unnamed migration'}")
    
    try:
        async with get_db_session() as db:
            await db.execute(text(migration_sql))
            logger.info("Migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


async def check_migration_status() -> dict:
    """Check if migrations table exists and get migration status."""
    try:
        async with get_db_session() as db:
            # Check if migrations table exists
            result = await db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'alembic_version'
                )
            """))
            
            has_migrations_table = result.scalar()
            
            if has_migrations_table:
                # Get current migration version
                result = await db.execute(text("SELECT version_num FROM alembic_version"))
                current_version = result.scalar()
                
                return {
                    "migrations_enabled": True,
                    "current_version": current_version
                }
            else:
                return {
                    "migrations_enabled": False,
                    "current_version": None
                }
                
    except Exception as e:
        return {
            "migrations_enabled": False,
            "error": str(e)
        } 