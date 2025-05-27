"""
Access Control Middleware for Play Buni Platform

This module provides access control functionality that integrates with
NFT verification to gate premium features and API endpoints.

Features:
- NFT-based access control
- Role-based permissions
- Rate limiting for different user tiers
- Premium feature gating
- Middleware for FastAPI routes
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from functools import wraps
from enum import Enum

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import jwt

from app.core.config import settings
from app.core.cache import cache_manager
from app.core.logging import get_logger
from app.services.nft_verification import verify_wallet_nft_holdings
from app.models.users import User, NFTHolder
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = get_logger(__name__)

security = HTTPBearer()


class AccessLevel(Enum):
    """User Access Levels"""
    FREE = "free"
    PREMIUM = "premium"
    VIP = "vip"
    ADMIN = "admin"


class PermissionType(Enum):
    """Permission Types"""
    VIEW_SIGNALS = "view_signals"
    UNLIMITED_SIGNALS = "unlimited_signals"
    CREATE_BLINKS = "create_blinks"
    REAL_TIME_DATA = "real_time_data"
    ADVANCED_ANALYTICS = "advanced_analytics"
    API_ACCESS = "api_access"
    ADMIN_PANEL = "admin_panel"


# Permission mappings for each access level
ACCESS_PERMISSIONS = {
    AccessLevel.FREE: {
        PermissionType.VIEW_SIGNALS,
    },
    AccessLevel.PREMIUM: {
        PermissionType.VIEW_SIGNALS,
        PermissionType.UNLIMITED_SIGNALS,
        PermissionType.CREATE_BLINKS,
        PermissionType.REAL_TIME_DATA,
        PermissionType.API_ACCESS,
    },
    AccessLevel.VIP: {
        PermissionType.VIEW_SIGNALS,
        PermissionType.UNLIMITED_SIGNALS,
        PermissionType.CREATE_BLINKS,
        PermissionType.REAL_TIME_DATA,
        PermissionType.ADVANCED_ANALYTICS,
        PermissionType.API_ACCESS,
    },
    AccessLevel.ADMIN: {
        PermissionType.VIEW_SIGNALS,
        PermissionType.UNLIMITED_SIGNALS,
        PermissionType.CREATE_BLINKS,
        PermissionType.REAL_TIME_DATA,
        PermissionType.ADVANCED_ANALYTICS,
        PermissionType.API_ACCESS,
        PermissionType.ADMIN_PANEL,
    }
}

# Rate limits per access level (requests per minute)
RATE_LIMITS = {
    AccessLevel.FREE: 10,
    AccessLevel.PREMIUM: 100,
    AccessLevel.VIP: 500,
    AccessLevel.ADMIN: 1000,
}

# Signal limits per hour
SIGNAL_LIMITS = {
    AccessLevel.FREE: 2,
    AccessLevel.PREMIUM: -1,  # Unlimited
    AccessLevel.VIP: -1,      # Unlimited
    AccessLevel.ADMIN: -1,    # Unlimited
}


class AccessControlError(Exception):
    """Access control related errors"""
    pass


class InsufficientPermissionsError(AccessControlError):
    """Raised when user lacks required permissions"""
    pass


class RateLimitExceededError(AccessControlError):
    """Raised when rate limit is exceeded"""
    pass


class NFTVerificationError(AccessControlError):
    """Raised when NFT verification fails"""
    pass


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        wallet_address: str = payload.get("sub")
        if wallet_address is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        
        # Get user from database
        result = await db.execute(
            select(User).where(User.wallet_address == wallet_address)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )
        
        return user
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )


async def get_user_access_level(user: User, db: AsyncSession) -> AccessLevel:
    """
    Determine user's access level based on NFT holdings
    
    Args:
        user: User object
        db: Database session
        
    Returns:
        User's access level
    """
    try:
        # Check if user is admin
        if user.wallet_address in settings.admin_wallets_list:
            return AccessLevel.ADMIN
        
        # Check cache first
        cache_key = f"access_level:{user.wallet_address}"
        cached_level = await cache_manager.get(cache_key)
        if cached_level:
            return AccessLevel(cached_level)
        
        # Check NFT holdings
        result = await db.execute(
            select(NFTHolder).where(NFTHolder.user_id == user.id)
        )
        nft_holder = result.scalar_one_or_none()
        
        if nft_holder:
            # Verify NFT holdings are still valid
            is_verified, details = await verify_wallet_nft_holdings(
                user.wallet_address,
                required_collection=settings.required_nft_collection,
                min_nft_count=1
            )
            
            if is_verified:
                # Determine access level based on NFT count
                nft_count = details.get("nft_count", 0)
                if nft_count >= settings.vip_nft_threshold:
                    access_level = AccessLevel.VIP
                else:
                    access_level = AccessLevel.PREMIUM
                
                # Update NFT holder record
                nft_holder.last_verified = datetime.utcnow()
                nft_holder.nft_count = nft_count
                await db.commit()
                
                # Cache for 10 minutes
                await cache_manager.set(cache_key, access_level.value, expire=600)
                return access_level
            else:
                # NFT verification failed, downgrade to free
                logger.warning(f"NFT verification failed for user {user.wallet_address}: {details}")
                await cache_manager.set(cache_key, AccessLevel.FREE.value, expire=300)
                return AccessLevel.FREE
        
        # No NFT holdings, free tier
        await cache_manager.set(cache_key, AccessLevel.FREE.value, expire=300)
        return AccessLevel.FREE
        
    except Exception as e:
        logger.error(f"Error determining access level for user {user.wallet_address}: {e}")
        return AccessLevel.FREE


async def check_permission(
    user: User,
    permission: PermissionType,
    db: AsyncSession
) -> bool:
    """
    Check if user has specific permission
    
    Args:
        user: User object
        permission: Permission to check
        db: Database session
        
    Returns:
        True if user has permission
    """
    access_level = await get_user_access_level(user, db)
    return permission in ACCESS_PERMISSIONS.get(access_level, set())


async def check_rate_limit(user: User, endpoint: str) -> bool:
    """
    Check if user has exceeded rate limit for endpoint
    
    Args:
        user: User object
        endpoint: API endpoint
        
    Returns:
        True if within rate limit
    """
    try:
        # Get user's access level from cache or default to free
        cache_key = f"access_level:{user.wallet_address}"
        cached_level = await cache_manager.get(cache_key)
        access_level = AccessLevel(cached_level) if cached_level else AccessLevel.FREE
        
        # Get rate limit for access level
        rate_limit = RATE_LIMITS.get(access_level, RATE_LIMITS[AccessLevel.FREE])
        
        # Check current usage
        usage_key = f"rate_limit:{user.wallet_address}:{endpoint}"
        current_usage = await cache_manager.get(usage_key)
        
        if current_usage is None:
            # First request in this minute
            await cache_manager.set(usage_key, "1", expire=60)
            return True
        
        usage_count = int(current_usage)
        if usage_count >= rate_limit:
            return False
        
        # Increment usage
        await cache_manager.set(usage_key, str(usage_count + 1), expire=60)
        return True
        
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        return True  # Allow on error


async def check_signal_limit(user: User) -> bool:
    """
    Check if user has exceeded hourly signal limit
    
    Args:
        user: User object
        
    Returns:
        True if within signal limit
    """
    try:
        # Get user's access level
        cache_key = f"access_level:{user.wallet_address}"
        cached_level = await cache_manager.get(cache_key)
        access_level = AccessLevel(cached_level) if cached_level else AccessLevel.FREE
        
        # Get signal limit for access level
        signal_limit = SIGNAL_LIMITS.get(access_level, SIGNAL_LIMITS[AccessLevel.FREE])
        
        # Unlimited signals for premium users
        if signal_limit == -1:
            return True
        
        # Check current usage
        usage_key = f"signal_limit:{user.wallet_address}"
        current_usage = await cache_manager.get(usage_key)
        
        if current_usage is None:
            # First signal in this hour
            await cache_manager.set(usage_key, "1", expire=3600)
            return True
        
        usage_count = int(current_usage)
        if usage_count >= signal_limit:
            return False
        
        # Increment usage
        await cache_manager.set(usage_key, str(usage_count + 1), expire=3600)
        return True
        
    except Exception as e:
        logger.error(f"Error checking signal limit: {e}")
        return True  # Allow on error


def require_permission(permission: PermissionType):
    """
    Decorator to require specific permission for endpoint access
    
    Args:
        permission: Required permission
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user and db from kwargs
            user = kwargs.get('current_user')
            db = kwargs.get('db')
            
            if not user or not db:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            # Check permission
            has_permission = await check_permission(user, permission, db)
            if not has_permission:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {permission.value}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_nft_access():
    """
    Decorator to require NFT ownership for endpoint access
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user and db from kwargs
            user = kwargs.get('current_user')
            db = kwargs.get('db')
            
            if not user or not db:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            # Check access level
            access_level = await get_user_access_level(user, db)
            if access_level == AccessLevel.FREE:
                raise HTTPException(
                    status_code=403,
                    detail="NFT ownership required for this feature"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class AccessControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware for access control and rate limiting
    """
    
    def __init__(self, app, protected_paths: Optional[Set[str]] = None):
        super().__init__(app)
        self.protected_paths = protected_paths or {
            "/api/v1/signals/premium",
            "/api/v1/blinks",
            "/api/v1/analytics",
            "/api/v1/admin"
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request through access control
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Skip access control for non-protected paths
        if not any(request.url.path.startswith(path) for path in self.protected_paths):
            return await call_next(request)
        
        try:
            # Extract authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=401,
                    detail="Authorization header required"
                )
            
            # Get user from token
            token = auth_header.split(" ")[1]
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            
            wallet_address = payload.get("sub")
            if not wallet_address:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token"
                )
            
            # Create mock user for rate limiting
            class MockUser:
                def __init__(self, wallet_address: str):
                    self.wallet_address = wallet_address
            
            user = MockUser(wallet_address)
            
            # Check rate limit
            endpoint = request.url.path
            within_limit = await check_rate_limit(user, endpoint)
            if not within_limit:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
            
            # Add user info to request state
            request.state.user_wallet = wallet_address
            
            return await call_next(request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Access control middleware error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )


# Dependency functions for FastAPI
async def get_current_user_with_permission(
    permission: PermissionType
) -> Callable:
    """
    Create dependency function that requires specific permission
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def dependency(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
    ) -> User:
        has_permission = await check_permission(current_user, permission, db)
        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission.value}"
            )
        return current_user
    
    return dependency


async def get_premium_user(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency that requires premium access (NFT ownership)
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        User with premium access
        
    Raises:
        HTTPException: If user doesn't have premium access
    """
    access_level = await get_user_access_level(current_user, db)
    if access_level == AccessLevel.FREE:
        raise HTTPException(
            status_code=403,
            detail="Premium access required. Please connect a wallet with NFTs."
        )
    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency that requires admin access
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        User with admin access
        
    Raises:
        HTTPException: If user doesn't have admin access
    """
    access_level = await get_user_access_level(current_user, db)
    if access_level != AccessLevel.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user


async def check_signal_access(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Check if user can access signals (rate limiting)
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User if access allowed
        
    Raises:
        HTTPException: If signal limit exceeded
    """
    within_limit = await check_signal_limit(current_user)
    if not within_limit:
        raise HTTPException(
            status_code=429,
            detail="Signal limit exceeded. Upgrade to premium for unlimited access."
        )
    return current_user


# Utility functions
async def get_user_permissions(user: User, db: AsyncSession) -> Set[PermissionType]:
    """
    Get all permissions for a user
    
    Args:
        user: User object
        db: Database session
        
    Returns:
        Set of user permissions
    """
    access_level = await get_user_access_level(user, db)
    return ACCESS_PERMISSIONS.get(access_level, set())


async def get_user_rate_limit(user: User) -> int:
    """
    Get rate limit for a user
    
    Args:
        user: User object
        
    Returns:
        Rate limit (requests per minute)
    """
    cache_key = f"access_level:{user.wallet_address}"
    cached_level = await cache_manager.get(cache_key)
    access_level = AccessLevel(cached_level) if cached_level else AccessLevel.FREE
    return RATE_LIMITS.get(access_level, RATE_LIMITS[AccessLevel.FREE])


async def get_user_signal_limit(user: User) -> int:
    """
    Get signal limit for a user
    
    Args:
        user: User object
        
    Returns:
        Signal limit (signals per hour, -1 for unlimited)
    """
    cache_key = f"access_level:{user.wallet_address}"
    cached_level = await cache_manager.get(cache_key)
    access_level = AccessLevel(cached_level) if cached_level else AccessLevel.FREE
    return SIGNAL_LIMITS.get(access_level, SIGNAL_LIMITS[AccessLevel.FREE]) 