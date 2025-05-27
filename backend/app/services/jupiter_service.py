"""
Jupiter API Service with Free/Paid Tier Management
Automatically switches between free and paid endpoints based on platform metrics
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import aiohttp
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import create_signal_logger

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

class JupiterQuoteRequest(BaseModel):
    inputMint: str
    outputMint: str
    amount: int
    slippageBps: Optional[int] = 50
    onlyDirectRoutes: Optional[bool] = False
    asLegacyTransaction: Optional[bool] = False

class JupiterQuoteResponse(BaseModel):
    inputMint: str
    inAmount: str
    outputMint: str
    outAmount: str
    otherAmountThreshold: str
    swapMode: str
    slippageBps: int
    platformFee: Optional[Dict[str, Any]]
    priceImpactPct: str
    routePlan: List[Dict[str, Any]]
    contextSlot: int
    timeTaken: float

class JupiterService:
    """
    Jupiter API Service with intelligent tier management
    Starts with free tier and automatically upgrades based on usage/revenue
    """
    
    def __init__(self):
        self.current_tier = "free"  # free, trial, paid
        self.daily_requests = 0
        self.monthly_revenue = Decimal("0")
        self.last_reset = datetime.now(timezone.utc).date()
        
        # API endpoints for different tiers
        self.endpoints = {
            "free": "https://www.jupiterapi.com",
            "trial": settings.quicknode_metis_endpoint if hasattr(settings, 'quicknode_metis_endpoint') else None,
            "paid": settings.jupiter_paid_endpoint if hasattr(settings, 'jupiter_paid_endpoint') else None
        }
        
        # Rate limits per tier
        self.rate_limits = {
            "free": {"requests_per_minute": 60, "daily_limit": 8640},  # 10 req/sec = 600/min, conservative daily
            "trial": {"requests_per_minute": 600, "daily_limit": 50000},
            "paid": {"requests_per_minute": 1000, "daily_limit": None}
        }
        
        # Upgrade thresholds
        self.upgrade_thresholds = {
            "trial": {"daily_requests": 5000, "monthly_revenue": Decimal("100")},
            "paid": {"daily_requests": 30000, "monthly_revenue": Decimal("1000")}
        }
    
    async def get_quote(self, quote_request: JupiterQuoteRequest) -> JupiterQuoteResponse:
        """
        Get quote from appropriate Jupiter API tier
        """
        try:
            # Check if we need to reset daily counters
            await self._reset_daily_counters()
            
            # Check if we should upgrade tier
            await self._check_tier_upgrade()
            
            # Check rate limits
            if not await self._check_rate_limits():
                raise Exception("Rate limit exceeded for current tier")
            
            # Get quote from current tier
            quote_response = await self._make_quote_request(quote_request)
            
            # Track usage
            self.daily_requests += 1
            await self._log_usage()
            
            return quote_response
            
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            # Fallback to free tier if paid tier fails
            if self.current_tier != "free":
                logger.info("Falling back to free tier")
                return await self._make_quote_request(quote_request, force_tier="free")
            raise
    
    async def get_swap_transaction(self, quote_response: JupiterQuoteResponse, 
                                 user_public_key: str) -> Dict[str, Any]:
        """
        Get swap transaction from Jupiter API
        """
        try:
            endpoint = self.endpoints[self.current_tier]
            if not endpoint:
                endpoint = self.endpoints["free"]
            
            swap_url = f"{endpoint}/swap"
            
            payload = {
                "quoteResponse": quote_response.dict(),
                "userPublicKey": user_public_key,
                "prioritizationFeeLamports": "auto",
                "dynamicComputeUnitLimit": True,
                "dynamicSlippage": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(swap_url, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Jupiter swap API error: {response.status}")
                    
                    result = await response.json()
                    return result
                    
        except Exception as e:
            logger.error(f"Error getting swap transaction: {e}")
            raise
    
    async def _make_quote_request(self, quote_request: JupiterQuoteRequest, 
                                force_tier: Optional[str] = None) -> JupiterQuoteResponse:
        """
        Make quote request to Jupiter API
        """
        tier = force_tier or self.current_tier
        endpoint = self.endpoints[tier]
        
        if not endpoint:
            tier = "free"
            endpoint = self.endpoints["free"]
        
        quote_url = f"{endpoint}/quote"
        params = {
            "inputMint": quote_request.inputMint,
            "outputMint": quote_request.outputMint,
            "amount": quote_request.amount,
            "slippageBps": quote_request.slippageBps
        }
        
        if quote_request.onlyDirectRoutes:
            params["onlyDirectRoutes"] = "true"
        if quote_request.asLegacyTransaction:
            params["asLegacyTransaction"] = "true"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(quote_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Jupiter API error: {response.status} - {error_text}")
                
                result = await response.json()
                return JupiterQuoteResponse(**result)
    
    async def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            self.daily_requests = 0
            self.last_reset = today
            logger.info("Reset daily Jupiter API counters")
    
    async def _check_tier_upgrade(self):
        """Check if we should upgrade to higher tier based on usage/revenue"""
        try:
            # Get current monthly revenue (this would be from your revenue tracker)
            current_revenue = await self._get_monthly_revenue()
            
            # Check for upgrade to trial tier
            if (self.current_tier == "free" and 
                (self.daily_requests > self.upgrade_thresholds["trial"]["daily_requests"] or
                 current_revenue > self.upgrade_thresholds["trial"]["monthly_revenue"])):
                
                if self.endpoints["trial"]:
                    self.current_tier = "trial"
                    logger.info("Upgraded to Jupiter trial tier")
                    signal_logger.info("jupiter_tier_upgraded", extra={
                        "new_tier": "trial",
                        "daily_requests": self.daily_requests,
                        "monthly_revenue": float(current_revenue)
                    })
            
            # Check for upgrade to paid tier
            elif (self.current_tier in ["free", "trial"] and
                  (self.daily_requests > self.upgrade_thresholds["paid"]["daily_requests"] or
                   current_revenue > self.upgrade_thresholds["paid"]["monthly_revenue"])):
                
                if self.endpoints["paid"]:
                    self.current_tier = "paid"
                    logger.info("Upgraded to Jupiter paid tier")
                    signal_logger.info("jupiter_tier_upgraded", extra={
                        "new_tier": "paid",
                        "daily_requests": self.daily_requests,
                        "monthly_revenue": float(current_revenue)
                    })
                    
        except Exception as e:
            logger.error(f"Error checking tier upgrade: {e}")
    
    async def _check_rate_limits(self) -> bool:
        """Check if current usage is within rate limits"""
        limits = self.rate_limits[self.current_tier]
        
        # Check daily limit
        if limits["daily_limit"] and self.daily_requests >= limits["daily_limit"]:
            logger.warning(f"Daily rate limit exceeded for tier {self.current_tier}")
            return False
        
        return True
    
    async def _get_monthly_revenue(self) -> Decimal:
        """Get current monthly revenue from revenue tracker"""
        try:
            # This would integrate with your revenue tracking system
            from app.services.treasury_manager import treasury_manager
            return await treasury_manager.get_monthly_revenue()
        except Exception as e:
            logger.error(f"Error getting monthly revenue: {e}")
            return Decimal("0")
    
    async def _log_usage(self):
        """Log Jupiter API usage for monitoring"""
        signal_logger.info("jupiter_api_usage", extra={
            "tier": self.current_tier,
            "daily_requests": self.daily_requests,
            "endpoint": self.endpoints[self.current_tier]
        })
    
    def get_current_tier_info(self) -> Dict[str, Any]:
        """Get information about current tier and usage"""
        limits = self.rate_limits[self.current_tier]
        return {
            "current_tier": self.current_tier,
            "daily_requests": self.daily_requests,
            "daily_limit": limits["daily_limit"],
            "rate_limit_per_minute": limits["requests_per_minute"],
            "endpoint": self.endpoints[self.current_tier],
            "usage_percentage": (self.daily_requests / limits["daily_limit"] * 100) if limits["daily_limit"] else 0
        }

# Global Jupiter service instance
jupiter_service = JupiterService() 