"""
NFT Verification API Router for Play Buni Platform

This router provides API endpoints for NFT verification, access control,
and premium feature management.

Endpoints:
- POST /verify-wallet: Verify wallet NFT holdings
- GET /access-level: Get user's current access level
- POST /refresh-verification: Refresh NFT verification
- GET /nft-holdings: Get user's NFT holdings
- GET /permissions: Get user's permissions
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.core.access_control import (
    get_current_user,
    get_user_access_level,
    get_user_permissions,
    get_user_rate_limit,
    get_user_signal_limit,
    AccessLevel,
    PermissionType
)
from app.services.nft_verification import (
    verify_wallet_nft_holdings,
    nft_verification_service,
    VerificationResult,
    NFTMetadata
)
from app.models.users import User, NFTHolder
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/nft", tags=["NFT Verification"])


# Pydantic models for request/response
class WalletVerificationRequest(BaseModel):
    """Request model for wallet verification"""
    wallet_address: str = Field(..., description="Solana wallet address to verify")
    collection_address: Optional[str] = Field(None, description="Optional collection address requirement")
    min_nft_count: int = Field(1, description="Minimum number of NFTs required")


class NFTMetadataResponse(BaseModel):
    """Response model for NFT metadata"""
    mint: str
    name: str
    symbol: str
    uri: str
    description: Optional[str]
    image: Optional[str]
    animation_url: Optional[str]
    external_url: Optional[str]
    attributes: Optional[List[Dict[str, Any]]]
    creators: List[Dict[str, Any]]
    seller_fee_basis_points: int
    collection_verified: bool


class VerificationResponse(BaseModel):
    """Response model for verification results"""
    verified: bool
    access_level: str
    nft_count: int
    collection_nft_count: Optional[int] = None
    verified_nfts: List[str]
    collection: Optional[str] = None
    error: Optional[str] = None
    verified_at: datetime


class AccessLevelResponse(BaseModel):
    """Response model for access level information"""
    access_level: str
    permissions: List[str]
    rate_limit: int
    signal_limit: int
    nft_count: int
    last_verified: Optional[datetime]


class NFTHoldingsResponse(BaseModel):
    """Response model for NFT holdings"""
    total_nfts: int
    nfts: List[NFTMetadataResponse]
    collections: Dict[str, int]


class RefreshVerificationRequest(BaseModel):
    """Request model for refreshing verification"""
    force_refresh: bool = Field(False, description="Force refresh even if recently verified")


@router.post("/verify-wallet", response_model=VerificationResponse)
async def verify_wallet(
    request: WalletVerificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Verify wallet NFT holdings for platform access
    
    This endpoint verifies if a wallet holds the required NFTs for premium access.
    It can optionally check for specific collection membership.
    
    Args:
        request: Wallet verification request
        db: Database session
        
    Returns:
        Verification result with access level and NFT details
    """
    try:
        logger.info(f"Verifying wallet: {request.wallet_address}")
        
        # Verify NFT holdings
        is_verified, details = await verify_wallet_nft_holdings(
            request.wallet_address,
            required_collection=request.collection_address,
            min_nft_count=request.min_nft_count
        )
        
        # Determine access level
        if is_verified:
            nft_count = details.get("nft_count", 0)
            collection_nft_count = details.get("collection_nft_count")
            
            # Determine access level based on NFT count
            if nft_count >= 10:  # VIP threshold
                access_level = AccessLevel.VIP
            elif nft_count >= 1:  # Premium threshold
                access_level = AccessLevel.PREMIUM
            else:
                access_level = AccessLevel.FREE
        else:
            access_level = AccessLevel.FREE
        
        return VerificationResponse(
            verified=is_verified,
            access_level=access_level.value,
            nft_count=details.get("nft_count", 0),
            collection_nft_count=details.get("collection_nft_count"),
            verified_nfts=details.get("verified_nfts", []),
            collection=details.get("collection"),
            error=details.get("error"),
            verified_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error verifying wallet {request.wallet_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@router.get("/access-level", response_model=AccessLevelResponse)
async def get_access_level(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user's access level and permissions
    
    Returns detailed information about the user's current access level,
    permissions, rate limits, and NFT holdings.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Access level information
    """
    try:
        # Get access level
        access_level = await get_user_access_level(current_user, db)
        
        # Get permissions
        permissions = await get_user_permissions(current_user, db)
        permission_names = [perm.value for perm in permissions]
        
        # Get rate and signal limits
        rate_limit = await get_user_rate_limit(current_user)
        signal_limit = await get_user_signal_limit(current_user)
        
        # Get NFT holder information
        result = await db.execute(
            select(NFTHolder).where(NFTHolder.user_id == current_user.id)
        )
        nft_holder = result.scalar_one_or_none()
        
        nft_count = nft_holder.nft_count if nft_holder else 0
        last_verified = nft_holder.last_verified if nft_holder else None
        
        return AccessLevelResponse(
            access_level=access_level.value,
            permissions=permission_names,
            rate_limit=rate_limit,
            signal_limit=signal_limit,
            nft_count=nft_count,
            last_verified=last_verified
        )
        
    except Exception as e:
        logger.error(f"Error getting access level for user {current_user.wallet_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get access level: {str(e)}"
        )


@router.post("/refresh-verification", response_model=VerificationResponse)
async def refresh_verification(
    request: RefreshVerificationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh NFT verification for current user
    
    This endpoint re-verifies the user's NFT holdings and updates their
    access level accordingly. Useful when users acquire new NFTs.
    
    Args:
        request: Refresh verification request
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated verification result
    """
    try:
        logger.info(f"Refreshing verification for user: {current_user.wallet_address}")
        
        # Check if we need to force refresh
        if not request.force_refresh:
            # Check if recently verified (within last 5 minutes)
            result = await db.execute(
                select(NFTHolder).where(NFTHolder.user_id == current_user.id)
            )
            nft_holder = result.scalar_one_or_none()
            
            if nft_holder and nft_holder.last_verified:
                time_since_verification = datetime.utcnow() - nft_holder.last_verified
                if time_since_verification.total_seconds() < 300:  # 5 minutes
                    raise HTTPException(
                        status_code=429,
                        detail="Verification was recently performed. Use force_refresh=true to override."
                    )
        
        # Clear cache to force fresh verification
        from app.core.cache import cache_manager
        cache_key = f"access_level:{current_user.wallet_address}"
        await cache_manager.delete(cache_key)
        
        # Verify NFT holdings
        is_verified, details = await verify_wallet_nft_holdings(
            current_user.wallet_address,
            min_nft_count=1
        )
        
        # Update or create NFT holder record
        if is_verified:
            nft_count = details.get("nft_count", 0)
            
            result = await db.execute(
                select(NFTHolder).where(NFTHolder.user_id == current_user.id)
            )
            nft_holder = result.scalar_one_or_none()
            
            if nft_holder:
                # Update existing record
                nft_holder.nft_count = nft_count
                nft_holder.last_verified = datetime.utcnow()
                nft_holder.verified_nfts = details.get("verified_nfts", [])
            else:
                # Create new record
                nft_holder = NFTHolder(
                    user_id=current_user.id,
                    wallet_address=current_user.wallet_address,
                    nft_count=nft_count,
                    verified_at=datetime.utcnow(),
                    last_verified=datetime.utcnow(),
                    verified_nfts=details.get("verified_nfts", [])
                )
                db.add(nft_holder)
            
            await db.commit()
            
            # Determine access level
            if nft_count >= 10:
                access_level = AccessLevel.VIP
            else:
                access_level = AccessLevel.PREMIUM
        else:
            access_level = AccessLevel.FREE
        
        return VerificationResponse(
            verified=is_verified,
            access_level=access_level.value,
            nft_count=details.get("nft_count", 0),
            verified_nfts=details.get("verified_nfts", []),
            error=details.get("error"),
            verified_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing verification for user {current_user.wallet_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh verification: {str(e)}"
        )


@router.get("/holdings", response_model=NFTHoldingsResponse)
async def get_nft_holdings(
    current_user: User = Depends(get_current_user),
    limit: int = Query(50, description="Maximum number of NFTs to return", le=200),
    include_metadata: bool = Query(True, description="Include detailed NFT metadata")
):
    """
    Get user's NFT holdings with metadata
    
    Returns a list of NFTs owned by the current user, optionally with
    detailed metadata including images and attributes.
    
    Args:
        current_user: Current authenticated user
        limit: Maximum number of NFTs to return
        include_metadata: Whether to include detailed metadata
        
    Returns:
        NFT holdings information
    """
    try:
        logger.info(f"Getting NFT holdings for user: {current_user.wallet_address}")
        
        async with nft_verification_service as service:
            # Get NFT mint addresses
            nft_mints = await service.get_nfts_by_owner(
                current_user.wallet_address,
                limit=limit
            )
            
            nfts = []
            collections = {}
            
            if include_metadata and nft_mints:
                # Get detailed metadata for each NFT
                verification_results = await service.verify_multiple_nfts(
                    nft_mints[:limit],
                    current_user.wallet_address
                )
                
                for result in verification_results:
                    if result.is_verified and result.metadata:
                        metadata = result.metadata
                        
                        nft_response = NFTMetadataResponse(
                            mint=metadata.mint,
                            name=metadata.name,
                            symbol=metadata.symbol,
                            uri=metadata.uri,
                            description=metadata.description,
                            image=metadata.image,
                            animation_url=metadata.animation_url,
                            external_url=metadata.external_url,
                            attributes=metadata.attributes,
                            creators=metadata.creators,
                            seller_fee_basis_points=metadata.seller_fee_basis_points,
                            collection_verified=bool(metadata.collection)
                        )
                        
                        nfts.append(nft_response)
                        
                        # Track collections
                        if metadata.collection:
                            collection_key = metadata.collection.get("key", "Unknown")
                            collections[collection_key] = collections.get(collection_key, 0) + 1
            
            return NFTHoldingsResponse(
                total_nfts=len(nft_mints),
                nfts=nfts,
                collections=collections
            )
            
    except Exception as e:
        logger.error(f"Error getting NFT holdings for user {current_user.wallet_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get NFT holdings: {str(e)}"
        )


@router.get("/verify/{mint_address}")
async def verify_nft_ownership(
    mint_address: str,
    current_user: User = Depends(get_current_user)
):
    """
    Verify ownership of a specific NFT
    
    Checks if the current user owns a specific NFT by mint address.
    
    Args:
        mint_address: NFT mint address to verify
        current_user: Current authenticated user
        
    Returns:
        Verification result for the specific NFT
    """
    try:
        logger.info(f"Verifying NFT ownership: {mint_address} for user: {current_user.wallet_address}")
        
        async with nft_verification_service as service:
            result = await service.verify_nft_ownership(
                mint_address,
                current_user.wallet_address
            )
            
            return {
                "verified": result.is_verified,
                "mint": result.mint,
                "owner": result.owner,
                "status": result.status.value,
                "error": result.error_message,
                "verified_at": result.verified_at,
                "metadata": {
                    "name": result.metadata.name if result.metadata else None,
                    "symbol": result.metadata.symbol if result.metadata else None,
                    "image": result.metadata.image if result.metadata else None,
                    "description": result.metadata.description if result.metadata else None,
                } if result.metadata else None
            }
            
    except Exception as e:
        logger.error(f"Error verifying NFT ownership {mint_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify NFT ownership: {str(e)}"
        )


@router.get("/collection/{collection_address}/verify")
async def verify_collection_membership(
    collection_address: str,
    current_user: User = Depends(get_current_user),
    limit: int = Query(100, description="Maximum NFTs to check", le=200)
):
    """
    Verify user's membership in a specific collection
    
    Checks how many NFTs the user owns from a specific collection.
    
    Args:
        collection_address: Collection mint address
        current_user: Current authenticated user
        limit: Maximum NFTs to check
        
    Returns:
        Collection membership information
    """
    try:
        logger.info(f"Verifying collection membership: {collection_address} for user: {current_user.wallet_address}")
        
        async with nft_verification_service as service:
            # Get user's NFTs
            nft_mints = await service.get_nfts_by_owner(
                current_user.wallet_address,
                limit=limit
            )
            
            # Check collection membership for each NFT
            collection_nfts = []
            for mint in nft_mints:
                is_member = await service.verify_collection_membership(
                    mint,
                    collection_address
                )
                if is_member:
                    collection_nfts.append(mint)
            
            return {
                "collection": collection_address,
                "total_nfts_checked": len(nft_mints),
                "collection_nfts_owned": len(collection_nfts),
                "collection_nfts": collection_nfts,
                "is_member": len(collection_nfts) > 0,
                "verified_at": datetime.utcnow()
            }
            
    except Exception as e:
        logger.error(f"Error verifying collection membership {collection_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify collection membership: {str(e)}"
        )


@router.get("/permissions")
async def get_user_permissions_endpoint(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed permissions for current user
    
    Returns a comprehensive list of what the user can and cannot do
    based on their current access level.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Detailed permissions information
    """
    try:
        # Get access level and permissions
        access_level = await get_user_access_level(current_user, db)
        permissions = await get_user_permissions(current_user, db)
        
        # Get limits
        rate_limit = await get_user_rate_limit(current_user)
        signal_limit = await get_user_signal_limit(current_user)
        
        # Create detailed permission breakdown
        permission_details = {
            "access_level": access_level.value,
            "permissions": {
                "view_signals": PermissionType.VIEW_SIGNALS in permissions,
                "unlimited_signals": PermissionType.UNLIMITED_SIGNALS in permissions,
                "create_blinks": PermissionType.CREATE_BLINKS in permissions,
                "real_time_data": PermissionType.REAL_TIME_DATA in permissions,
                "advanced_analytics": PermissionType.ADVANCED_ANALYTICS in permissions,
                "api_access": PermissionType.API_ACCESS in permissions,
                "admin_panel": PermissionType.ADMIN_PANEL in permissions,
            },
            "limits": {
                "rate_limit_per_minute": rate_limit,
                "signals_per_hour": signal_limit if signal_limit != -1 else "unlimited",
            },
            "features": {
                "premium_signals": access_level != AccessLevel.FREE,
                "real_time_updates": PermissionType.REAL_TIME_DATA in permissions,
                "blink_creation": PermissionType.CREATE_BLINKS in permissions,
                "advanced_charts": PermissionType.ADVANCED_ANALYTICS in permissions,
                "api_endpoints": PermissionType.API_ACCESS in permissions,
            }
        }
        
        return permission_details
        
    except Exception as e:
        logger.error(f"Error getting permissions for user {current_user.wallet_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get permissions: {str(e)}"
        ) 