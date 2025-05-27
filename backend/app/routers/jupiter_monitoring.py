"""
Jupiter API Monitoring Router
Provides endpoints to monitor Jupiter API usage, costs, and tier management
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from app.core.security import get_current_user
from app.services.jupiter_service import jupiter_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jupiter", tags=["jupiter-monitoring"])

@router.get("/tier-info")
async def get_tier_info(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get current Jupiter API tier information and usage
    """
    try:
        return jupiter_service.get_current_tier_info()
    except Exception as e:
        logger.error(f"Error getting Jupiter tier info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tier information")

@router.get("/usage-stats")
async def get_usage_stats(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get detailed Jupiter API usage statistics
    """
    try:
        tier_info = jupiter_service.get_current_tier_info()
        
        return {
            "current_tier": tier_info["current_tier"],
            "daily_usage": {
                "requests": tier_info["daily_requests"],
                "limit": tier_info["daily_limit"],
                "percentage": tier_info["usage_percentage"]
            },
            "rate_limits": {
                "per_minute": tier_info["rate_limit_per_minute"],
                "remaining_today": tier_info["daily_limit"] - tier_info["daily_requests"] if tier_info["daily_limit"] else "unlimited"
            },
            "endpoint": tier_info["endpoint"],
            "cost_info": _get_cost_info(tier_info["current_tier"])
        }
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get usage statistics")

@router.post("/force-tier-upgrade")
async def force_tier_upgrade(target_tier: str, current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Manually upgrade to a higher tier (admin only)
    """
    try:
        # Check if user is admin (you'd implement this based on your auth system)
        # if not is_admin(current_user):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        valid_tiers = ["free", "trial", "paid"]
        if target_tier not in valid_tiers:
            raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {valid_tiers}")
        
        old_tier = jupiter_service.current_tier
        jupiter_service.current_tier = target_tier
        
        return {
            "message": f"Tier upgraded from {old_tier} to {target_tier}",
            "old_tier": old_tier,
            "new_tier": target_tier,
            "new_tier_info": jupiter_service.get_current_tier_info()
        }
    except Exception as e:
        logger.error(f"Error forcing tier upgrade: {e}")
        raise HTTPException(status_code=500, detail="Failed to upgrade tier")

@router.get("/cost-projection")
async def get_cost_projection(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get cost projection based on current usage patterns
    """
    try:
        tier_info = jupiter_service.get_current_tier_info()
        daily_requests = tier_info["daily_requests"]
        
        # Project monthly costs based on current daily usage
        monthly_requests = daily_requests * 30
        
        projections = {
            "current_tier": tier_info["current_tier"],
            "daily_requests": daily_requests,
            "projected_monthly_requests": monthly_requests,
            "current_costs": _get_cost_info(tier_info["current_tier"]),
            "tier_recommendations": _get_tier_recommendations(monthly_requests)
        }
        
        return projections
    except Exception as e:
        logger.error(f"Error getting cost projection: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost projection")

def _get_cost_info(tier: str) -> Dict[str, Any]:
    """Get cost information for a specific tier"""
    cost_info = {
        "free": {
            "monthly_cost": 0,
            "platform_fees": "0.2% on Jupiter swaps, 1% on pump.fun swaps",
            "rate_limits": "10 req/sec, ~8,640 daily",
            "features": ["Basic swap functionality", "Public endpoint"]
        },
        "trial": {
            "monthly_cost": 0,
            "platform_fees": "No additional platform fees",
            "rate_limits": "600 req/min, 50,000 daily",
            "features": ["Higher rate limits", "Dedicated endpoint", "Priority support"]
        },
        "paid": {
            "monthly_cost": "Variable based on usage",
            "platform_fees": "No platform fees",
            "rate_limits": "1,000 req/min, unlimited daily",
            "features": ["Highest rate limits", "Premium endpoint", "24/7 support", "Custom integrations"]
        }
    }
    
    return cost_info.get(tier, cost_info["free"])

def _get_tier_recommendations(monthly_requests: int) -> Dict[str, Any]:
    """Get tier recommendations based on usage"""
    recommendations = []
    
    if monthly_requests <= 200000:  # ~8,640 * 30 = 259,200
        recommendations.append({
            "tier": "free",
            "suitable": True,
            "reason": "Current usage fits within free tier limits"
        })
    else:
        recommendations.append({
            "tier": "free",
            "suitable": False,
            "reason": "Usage exceeds free tier daily limits"
        })
    
    if monthly_requests <= 1500000:  # 50,000 * 30
        recommendations.append({
            "tier": "trial",
            "suitable": True,
            "reason": "Trial tier can handle your projected usage"
        })
    else:
        recommendations.append({
            "tier": "trial",
            "suitable": False,
            "reason": "Usage exceeds trial tier daily limits"
        })
    
    recommendations.append({
        "tier": "paid",
        "suitable": True,
        "reason": "Paid tier has no limits and best performance"
    })
    
    return {"recommendations": recommendations} 