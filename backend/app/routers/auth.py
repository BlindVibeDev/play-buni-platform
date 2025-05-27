"""
Authentication Router - NFT-based authentication and wallet management.

This module provides:
- Wallet connection and verification
- NFT ownership validation
- Session management
- Access token generation
- User registration and profile management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from ..database import get_db
from ..core.security import create_access_token, verify_token
from ..services.nft_verifier import NFTVerifier
from ..services.solana_client import SolanaClient
from ..models.users import User, NFTHolder, UserSession

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer(auto_error=False)


# Pydantic models for request/response
class WalletConnectionRequest(BaseModel):
    """Request model for wallet connection."""
    wallet_address: str = Field(..., description="Solana wallet address")
    signature: str = Field(..., description="Signed message for verification")
    message: str = Field(..., description="Original message that was signed")


class NFTVerificationRequest(BaseModel):
    """Request model for NFT verification."""
    wallet_address: str = Field(..., description="Solana wallet address")
    nft_mint_address: Optional[str] = Field(None, description="Specific NFT mint to verify")


class AuthResponse(BaseModel):
    """Response model for authentication."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user_id: str = Field(..., description="User ID")
    wallet_address: str = Field(..., description="Wallet address")
    has_nft_access: bool = Field(..., description="Whether user has NFT access")
    nft_count: int = Field(default=0, description="Number of NFTs owned")


class UserProfile(BaseModel):
    """User profile response model."""
    user_id: str
    wallet_address: str
    created_at: datetime
    last_login: Optional[datetime]
    has_nft_access: bool
    nft_count: int
    premium_features: Dict[str, bool]
    usage_stats: Dict[str, Any]


class SessionInfo(BaseModel):
    """Session information response model."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool
    last_activity: datetime


# Dependency to get current user from token
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token."""
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        # Verify token and extract payload
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get user from database
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


# Optional authentication dependency
async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None."""
    
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


@router.post("/connect-wallet", response_model=AuthResponse)
async def connect_wallet(
    request: WalletConnectionRequest,
    db: AsyncSession = Depends(get_db),
    http_request: Request = None
):
    """
    Connect wallet and authenticate user.
    
    This endpoint:
    1. Verifies the wallet signature
    2. Checks for existing user or creates new one
    3. Verifies NFT ownership
    4. Creates session and returns access token
    """
    
    try:
        # Initialize Solana client for signature verification
        solana_client = SolanaClient()
        
        # Verify wallet signature
        is_valid_signature = await solana_client.verify_signature(
            request.wallet_address,
            request.message,
            request.signature
        )
        
        if not is_valid_signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid wallet signature"
            )
        
        # Check if user exists
        result = await db.execute(
            select(User).where(User.wallet_address == request.wallet_address)
        )
        user = result.scalar_one_or_none()
        
        # Create new user if doesn't exist
        if not user:
            user = User(
                wallet_address=request.wallet_address,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            db.add(user)
            await db.flush()  # Get user ID
            
            logger.info(f"New user created: {user.id}")
        else:
            # Update last login
            user.last_login = datetime.utcnow()
        
        # Verify NFT ownership
        nft_verifier = NFTVerifier()
        nft_verification = await nft_verifier.verify_nft_ownership(
            request.wallet_address
        )
        
        has_nft_access = nft_verification["has_access"]
        nft_count = nft_verification["nft_count"]
        
        # Update or create NFT holder record
        if has_nft_access:
            result = await db.execute(
                select(NFTHolder).where(NFTHolder.wallet_address == request.wallet_address)
            )
            nft_holder = result.scalar_one_or_none()
            
            if not nft_holder:
                nft_holder = NFTHolder(
                    user_id=user.id,
                    wallet_address=request.wallet_address,
                    nft_count=nft_count,
                    verified_at=datetime.utcnow(),
                    last_verified=datetime.utcnow()
                )
                db.add(nft_holder)
            else:
                nft_holder.nft_count = nft_count
                nft_holder.last_verified = datetime.utcnow()
        
        # Create session
        session = UserSession(
            user_id=user.id,
            session_token=create_access_token({"sub": str(user.id)}),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=7),
            ip_address=http_request.client.host if http_request and http_request.client else None,
            user_agent=http_request.headers.get("user-agent") if http_request else None
        )
        db.add(session)
        
        await db.commit()
        
        # Create access token
        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "wallet": request.wallet_address,
                "nft_access": has_nft_access
            },
            expires_delta=timedelta(days=7)
        )
        
        logger.info(
            f"User authenticated: {user.id}, NFT access: {has_nft_access}, NFTs: {nft_count}"
        )
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=7 * 24 * 3600,  # 7 days in seconds
            user_id=str(user.id),
            wallet_address=request.wallet_address,
            has_nft_access=has_nft_access,
            nft_count=nft_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Wallet connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to connect wallet"
        )


@router.post("/verify-nft", response_model=Dict[str, Any])
async def verify_nft_ownership(
    request: NFTVerificationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify NFT ownership for a wallet.
    
    This endpoint checks if the wallet owns any NFTs from the collection
    and updates the user's access level accordingly.
    """
    
    try:
        # Verify the wallet belongs to the current user
        if current_user.wallet_address != request.wallet_address:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only verify NFTs for your own wallet"
            )
        
        # Verify NFT ownership
        nft_verifier = NFTVerifier()
        verification_result = await nft_verifier.verify_nft_ownership(
            request.wallet_address,
            specific_nft=request.nft_mint_address
        )
        
        # Update NFT holder record if access granted
        if verification_result["has_access"]:
            result = await db.execute(
                select(NFTHolder).where(NFTHolder.user_id == current_user.id)
            )
            nft_holder = result.scalar_one_or_none()
            
            if not nft_holder:
                nft_holder = NFTHolder(
                    user_id=current_user.id,
                    wallet_address=request.wallet_address,
                    nft_count=verification_result["nft_count"],
                    verified_at=datetime.utcnow(),
                    last_verified=datetime.utcnow()
                )
                db.add(nft_holder)
            else:
                nft_holder.nft_count = verification_result["nft_count"]
                nft_holder.last_verified = datetime.utcnow()
            
            await db.commit()
        
        return {
            "has_access": verification_result["has_access"],
            "nft_count": verification_result["nft_count"],
            "verified_nfts": verification_result.get("nfts", []),
            "verification_timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NFT verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify NFT ownership"
        )


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's profile information."""
    
    try:
        # Get NFT holder information
        result = await db.execute(
            select(NFTHolder).where(NFTHolder.user_id == current_user.id)
        )
        nft_holder = result.scalar_one_or_none()
        
        has_nft_access = nft_holder is not None
        nft_count = nft_holder.nft_count if nft_holder else 0
        
        # Get usage statistics (placeholder)
        usage_stats = {
            "signals_received": 0,
            "blinks_created": 0,
            "total_trades": 0,
            "last_activity": current_user.last_login.isoformat() if current_user.last_login else None
        }
        
        # Define premium features based on NFT access
        premium_features = {
            "unlimited_signals": has_nft_access,
            "real_time_alerts": has_nft_access,
            "advanced_analytics": has_nft_access,
            "priority_support": has_nft_access,
            "custom_blinks": has_nft_access
        }
        
        return UserProfile(
            user_id=str(current_user.id),
            wallet_address=current_user.wallet_address,
            created_at=current_user.created_at,
            last_login=current_user.last_login,
            has_nft_access=has_nft_access,
            nft_count=nft_count,
            premium_features=premium_features,
            usage_stats=usage_stats
        )
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.get("/sessions", response_model=list[SessionInfo])
async def get_user_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all active sessions for the current user."""
    
    try:
        result = await db.execute(
            select(UserSession)
            .where(UserSession.user_id == current_user.id)
            .where(UserSession.is_active == True)
            .order_by(UserSession.created_at.desc())
        )
        sessions = result.scalars().all()
        
        return [
            SessionInfo(
                session_id=str(session.id),
                user_id=str(session.user_id),
                created_at=session.created_at,
                expires_at=session.expires_at,
                is_active=session.is_active,
                last_activity=session.last_activity or session.created_at
            )
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Sessions retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Logout user and invalidate current session."""
    
    try:
        # Invalidate all active sessions for the user
        result = await db.execute(
            select(UserSession)
            .where(UserSession.user_id == current_user.id)
            .where(UserSession.is_active == True)
        )
        sessions = result.scalars().all()
        
        for session in sessions:
            session.is_active = False
            session.logged_out_at = datetime.utcnow()
        
        await db.commit()
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout"
        )


@router.post("/refresh-token", response_model=AuthResponse)
async def refresh_access_token(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token for authenticated user."""
    
    try:
        # Re-verify NFT ownership
        nft_verifier = NFTVerifier()
        nft_verification = await nft_verifier.verify_nft_ownership(
            current_user.wallet_address
        )
        
        has_nft_access = nft_verification["has_access"]
        nft_count = nft_verification["nft_count"]
        
        # Create new access token
        access_token = create_access_token(
            data={
                "sub": str(current_user.id),
                "wallet": current_user.wallet_address,
                "nft_access": has_nft_access
            },
            expires_delta=timedelta(days=7)
        )
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=7 * 24 * 3600,
            user_id=str(current_user.id),
            wallet_address=current_user.wallet_address,
            has_nft_access=has_nft_access,
            nft_count=nft_count
        )
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get basic information about the current authenticated user."""
    
    return {
        "user_id": str(current_user.id),
        "wallet_address": current_user.wallet_address,
        "created_at": current_user.created_at.isoformat(),
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    } 