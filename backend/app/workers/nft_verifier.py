"""
NFT Verification Worker
Background tasks for NFT ownership verification and premium status management
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from celery import shared_task

from app.workers.celery_app import celery_app
from app.core.config import settings
from app.core.logging import create_signal_logger
from app.services.nft_verification import NFTVerificationService
from app.core.websocket import websocket_manager
from app.database import get_db_session
from app.models.premium import PremiumUser, get_premium_tier

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

@shared_task(bind=True, base=celery_app.Task)
def refresh_nft_status(self):
    """
    Refresh NFT verification status for all users
    Runs every 10 minutes via Celery Beat
    """
    try:
        logger.info("Starting NFT status refresh")
        
        result = asyncio.run(_refresh_nft_status_async())
        
        logger.info(f"NFT status refresh completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"NFT status refresh failed: {e}")
        self.retry(countdown=120 * (2 ** self.request.retries))

async def _refresh_nft_status_async():
    """Async implementation of NFT status refresh"""
    nft_service = NFTVerificationService()
    
    try:
        refresh_results = {
            "users_checked": 0,
            "status_changes": 0,
            "new_verifications": 0,
            "revoked_verifications": 0
        }
        
        # Get all premium users
        async with get_db_session() as db:
            # This would be an actual query in production
            # For now, use placeholder logic
            
            premium_users = []  # Would be fetched from database
            
            for user in premium_users:
                try:
                    wallet_address = user.get("wallet_address")
                    current_nft_count = user.get("nft_count", 0)
                    
                    # Verify current NFT ownership
                    verification_result = await nft_service.verify_nft_ownership(wallet_address)
                    new_nft_count = verification_result.get("nft_count", 0)
                    
                    refresh_results["users_checked"] += 1
                    
                    # Check for status changes
                    if new_nft_count != current_nft_count:
                        await _update_user_nft_status(
                            user.get("id"),
                            wallet_address,
                            new_nft_count
                        )
                        refresh_results["status_changes"] += 1
                        
                        if new_nft_count > 0 and current_nft_count == 0:
                            refresh_results["new_verifications"] += 1
                        elif new_nft_count == 0 and current_nft_count > 0:
                            refresh_results["revoked_verifications"] += 1
                    
                except Exception as e:
                    logger.error(f"Error refreshing NFT status for user {user.get('id', 'unknown')}: {e}")
        
        # Log refresh summary
        signal_logger.info("nft_status_refresh", extra={
            "users_checked": refresh_results["users_checked"],
            "status_changes": refresh_results["status_changes"],
            "new_verifications": refresh_results["new_verifications"],
            "revoked_verifications": refresh_results["revoked_verifications"]
        })
        
        return {
            "status": "success",
            "results": refresh_results
        }
        
    except Exception as e:
        logger.error(f"Error in NFT status refresh: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def verify_wallet_nfts(self, wallet_address: str, user_id: Optional[str] = None):
    """
    Verify NFT ownership for a specific wallet
    """
    try:
        logger.info(f"Verifying NFTs for wallet: {wallet_address}")
        
        result = asyncio.run(_verify_wallet_nfts_async(wallet_address, user_id))
        
        logger.info(f"NFT verification completed for {wallet_address}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"NFT verification failed for {wallet_address}: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

async def _verify_wallet_nfts_async(wallet_address: str, user_id: Optional[str]):
    """Async implementation of wallet NFT verification"""
    nft_service = NFTVerificationService()
    
    try:
        # Verify NFT ownership
        verification_result = await nft_service.verify_nft_ownership(wallet_address)
        
        nft_count = verification_result.get("nft_count", 0)
        nfts = verification_result.get("nfts", [])
        verified = verification_result.get("verified", False)
        
        # Determine premium tier
        premium_tier = get_premium_tier(nft_count)
        
        # Update user's premium status if user_id provided
        if user_id:
            await _update_user_premium_status(
                user_id, 
                wallet_address, 
                nft_count, 
                premium_tier,
                verified
            )
        
        # Update WebSocket connections for this wallet
        await _update_websocket_nft_status(wallet_address, verified, premium_tier)
        
        # Log verification result
        signal_logger.info("nft_verification", extra={
            "wallet_address": wallet_address,
            "user_id": user_id,
            "nft_count": nft_count,
            "verified": verified,
            "premium_tier": premium_tier.value if premium_tier else None
        })
        
        return {
            "status": "success",
            "wallet_address": wallet_address,
            "nft_count": nft_count,
            "verified": verified,
            "premium_tier": premium_tier.value if premium_tier else None,
            "nfts": nfts
        }
        
    except Exception as e:
        logger.error(f"Error verifying NFTs for {wallet_address}: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def monitor_nft_collection(self):
    """
    Monitor the NFT collection for new mints and transfers
    Runs every 5 minutes
    """
    try:
        logger.info("Starting NFT collection monitoring")
        
        result = asyncio.run(_monitor_nft_collection_async())
        
        logger.info(f"NFT collection monitoring completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"NFT collection monitoring failed: {e}")
        self.retry(countdown=180 * (2 ** self.request.retries))

async def _monitor_nft_collection_async():
    """Async implementation of NFT collection monitoring"""
    nft_service = NFTVerificationService()
    
    try:
        # Monitor collection activity
        monitoring_result = await nft_service.monitor_collection_activity()
        
        new_mints = monitoring_result.get("new_mints", [])
        recent_transfers = monitoring_result.get("recent_transfers", [])
        
        # Process new mints
        for mint in new_mints:
            try:
                mint_address = mint.get("mint_address")
                owner = mint.get("owner")
                
                if owner:
                    # Queue verification for new NFT owner
                    verify_wallet_nfts.delay(owner)
                
                signal_logger.info("nft_mint_detected", extra={
                    "mint_address": mint_address,
                    "owner": owner,
                    "timestamp": mint.get("timestamp")
                })
                
            except Exception as e:
                logger.error(f"Error processing NFT mint: {e}")
        
        # Process transfers that might affect verification status
        affected_wallets = set()
        for transfer in recent_transfers:
            try:
                from_wallet = transfer.get("from")
                to_wallet = transfer.get("to")
                
                if from_wallet:
                    affected_wallets.add(from_wallet)
                if to_wallet:
                    affected_wallets.add(to_wallet)
                
            except Exception as e:
                logger.error(f"Error processing NFT transfer: {e}")
        
        # Queue verification for affected wallets
        for wallet in affected_wallets:
            verify_wallet_nfts.delay(wallet)
        
        return {
            "status": "success",
            "new_mints": len(new_mints),
            "recent_transfers": len(recent_transfers),
            "affected_wallets": len(affected_wallets)
        }
        
    except Exception as e:
        logger.error(f"Error monitoring NFT collection: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def update_premium_features(self, user_id: str, premium_tier: str):
    """
    Update premium features based on NFT tier
    """
    try:
        logger.info(f"Updating premium features for user {user_id} to tier {premium_tier}")
        
        result = asyncio.run(_update_premium_features_async(user_id, premium_tier))
        
        logger.info(f"Premium features update completed for {user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Premium features update failed for {user_id}: {e}")
        self.retry(countdown=30 * (2 ** self.request.retries))

async def _update_premium_features_async(user_id: str, premium_tier: str):
    """Async implementation of premium features update"""
    try:
        from app.models.premium import PremiumTier, PREMIUM_TIER_CONFIG
        
        # Get feature access for tier
        tier_enum = PremiumTier(premium_tier)
        feature_access = PREMIUM_TIER_CONFIG.get(tier_enum)
        
        if not feature_access:
            raise ValueError(f"Invalid premium tier: {premium_tier}")
        
        # Update user's premium features in database
        async with get_db_session() as db:
            # This would be an actual update query in production
            # For now, use placeholder logic
            
            premium_user_data = {
                "premium_tier": premium_tier,
                "real_time_signals": feature_access.real_time_signals,
                "advanced_analytics": feature_access.advanced_analytics,
                "portfolio_tracking": feature_access.portfolio_tracking,
                "custom_alerts": feature_access.custom_alerts,
                "priority_support": feature_access.priority_support,
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Log feature update
            signal_logger.info("premium_features_updated", extra={
                "user_id": user_id,
                "premium_tier": premium_tier,
                "features": {
                    "real_time_signals": feature_access.real_time_signals,
                    "advanced_analytics": feature_access.advanced_analytics,
                    "portfolio_tracking": feature_access.portfolio_tracking,
                    "custom_alerts": feature_access.custom_alerts,
                    "priority_support": feature_access.priority_support
                }
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "premium_tier": premium_tier,
            "features_updated": premium_user_data
        }
        
    except Exception as e:
        logger.error(f"Error updating premium features for {user_id}: {e}")
        raise

@shared_task(bind=True, base=celery_app.Task)
def cleanup_inactive_verifications(self):
    """
    Clean up inactive or expired NFT verifications
    Runs daily
    """
    try:
        logger.info("Starting inactive NFT verifications cleanup")
        
        result = asyncio.run(_cleanup_inactive_verifications_async())
        
        logger.info(f"NFT verifications cleanup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"NFT verifications cleanup failed: {e}")
        self.retry(countdown=300 * (2 ** self.request.retries))

async def _cleanup_inactive_verifications_async():
    """Async implementation of inactive verifications cleanup"""
    try:
        # Define inactivity threshold (30 days)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        cleanup_results = {
            "inactive_users_found": 0,
            "verifications_revoked": 0,
            "users_demoted": 0
        }
        
        async with get_db_session() as db:
            # Find users who haven't been active recently
            # This would be actual database queries in production
            
            inactive_users = []  # Would be fetched from database
            
            for user in inactive_users:
                try:
                    user_id = user.get("id")
                    wallet_address = user.get("wallet_address")
                    
                    cleanup_results["inactive_users_found"] += 1
                    
                    # Re-verify NFT ownership for inactive users
                    verification_result = await verify_wallet_nfts.delay(wallet_address, user_id)
                    
                    # If verification fails, demote user
                    if not verification_result.get("verified", False):
                        await _demote_user_premium_status(user_id)
                        cleanup_results["verifications_revoked"] += 1
                        cleanup_results["users_demoted"] += 1
                
                except Exception as e:
                    logger.error(f"Error processing inactive user {user.get('id', 'unknown')}: {e}")
        
        return {
            "status": "success",
            "cutoff_date": cutoff_date.isoformat(),
            "results": cleanup_results
        }
        
    except Exception as e:
        logger.error(f"Error in inactive verifications cleanup: {e}")
        raise

# Helper functions
async def _update_user_nft_status(user_id: str, wallet_address: str, nft_count: int):
    """Update user's NFT status in database"""
    try:
        premium_tier = get_premium_tier(nft_count)
        verified = nft_count > 0
        
        async with get_db_session() as db:
            # This would be an actual database update in production
            # For now, use placeholder logic
            
            # Update premium user record
            user_data = {
                "nft_count": nft_count,
                "premium_tier": premium_tier.value if premium_tier else "basic",
                "verified_at": datetime.now(timezone.utc) if verified else None,
                "last_verification": datetime.now(timezone.utc)
            }
            
            # Queue premium features update
            if premium_tier:
                update_premium_features.delay(user_id, premium_tier.value)
        
        logger.info(f"Updated NFT status for user {user_id}: {nft_count} NFTs, tier {premium_tier}")
        
    except Exception as e:
        logger.error(f"Error updating NFT status for user {user_id}: {e}")
        raise

async def _update_user_premium_status(user_id: str, wallet_address: str, nft_count: int, 
                                     premium_tier, verified: bool):
    """Update user's premium status and features"""
    try:
        async with get_db_session() as db:
            # This would be actual database operations in production
            # For now, use placeholder logic
            
            premium_user_data = {
                "user_id": user_id,
                "wallet_address": wallet_address,
                "nft_count": nft_count,
                "premium_tier": premium_tier.value if premium_tier else "basic",
                "verified_at": datetime.now(timezone.utc) if verified else None,
                "last_verification": datetime.now(timezone.utc),
                "is_active": True
            }
        
        # Queue premium features update
        if premium_tier:
            update_premium_features.delay(user_id, premium_tier.value)
        
    except Exception as e:
        logger.error(f"Error updating premium status for user {user_id}: {e}")
        raise

async def _update_websocket_nft_status(wallet_address: str, verified: bool, premium_tier):
    """Update NFT verification status for active WebSocket connections"""
    try:
        # Find active connections for this wallet
        active_connections = [
            conn_id for conn_id, conn in websocket_manager.active_connections.items()
            if conn.wallet_address == wallet_address
        ]
        
        for connection_id in active_connections:
            await websocket_manager.refresh_nft_status(connection_id)
        
        logger.info(f"Updated NFT status for {len(active_connections)} WebSocket connections")
        
    except Exception as e:
        logger.error(f"Error updating WebSocket NFT status: {e}")

async def _demote_user_premium_status(user_id: str):
    """Demote user's premium status due to NFT verification failure"""
    try:
        async with get_db_session() as db:
            # This would be actual database update in production
            # For now, use placeholder logic
            
            demoted_data = {
                "nft_count": 0,
                "premium_tier": "basic",
                "verified_at": None,
                "is_active": False,
                "demoted_at": datetime.now(timezone.utc)
            }
            
        logger.info(f"Demoted user {user_id} to basic tier due to NFT verification failure")
        
    except Exception as e:
        logger.error(f"Error demoting user {user_id}: {e}")
        raise 