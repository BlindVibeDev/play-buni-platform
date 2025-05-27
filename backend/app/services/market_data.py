"""
Market Data Service for Real-time Price Feeds and Market Intelligence
Provides live market data streaming for premium NFT holders
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from app.core.config import settings
from app.core.logging import create_signal_logger

logger = logging.getLogger(__name__)
signal_logger = create_signal_logger()

class MarketDataService:
    """Service for real-time market data and price feeds"""
    
    def __init__(self):
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.last_update: Optional[datetime] = None
        self.update_interval = 30  # seconds
        self.supported_tokens = [
            "SOL", "USDC", "RAY", "SRM", "ORCA", "MNGO", "STEP", "ROPE",
            "COPE", "MEDIA", "ATLAS", "POLIS", "SAMO", "NINJA", "SHDW",
            "DFL", "SLND", "PORT", "TULIP", "FIDA", "KIN", "MAPS", "AUDIO",
        ]
    
    async def get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for all monitored tokens"""
        try:
            # Fetch from CoinGecko
            data = await self._fetch_coingecko_data()
            
            # Add market statistics
            market_stats = await self._calculate_market_stats(data)
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prices": data,
                "market_stats": market_stats,
                "update_interval": self.update_interval,
                "source": "coingecko_primary"
            }
            
            # Cache the data
            self.price_cache = data
            self.last_update = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            # Return cached data if available
            if self.price_cache:
                return {
                    "timestamp": self.last_update.isoformat() if self.last_update else None,
                    "prices": self.price_cache,
                    "market_stats": {},
                    "cached": True,
                    "error": str(e)
                }
            raise
    
    async def get_token_price(self, token_symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price for a specific token"""
        try:
            if token_symbol.upper() not in self.supported_tokens:
                logger.warning(f"Token {token_symbol} not in supported list")
            
            # Check cache first
            if (self.price_cache and 
                token_symbol.upper() in self.price_cache and 
                self.last_update and 
                datetime.now(timezone.utc) - self.last_update < timedelta(seconds=self.update_interval)):
                return self.price_cache[token_symbol.upper()]
            
            # Fetch fresh data
            data = await self._fetch_single_token_price(token_symbol)
            return data
            
        except Exception as e:
            logger.error(f"Failed to get price for {token_symbol}: {e}")
            return None
    
    async def get_price_history(self, token_symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get price history for a token"""
        try:
            async with aiohttp.ClientSession() as session:
                # Map token symbol to CoinGecko ID
                token_id = self._get_coingecko_id(token_symbol)
                if not token_id:
                    return []
                
                url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
                params = {
                    "vs_currency": "usd",
                    "days": days,
                    "interval": "hourly" if days <= 30 else "daily"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format price history
                        history = []
                        for i, (timestamp, price) in enumerate(data.get("prices", [])):
                            history.append({
                                "timestamp": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat(),
                                "price": price,
                                "volume": data.get("total_volumes", [[0, 0]])[i][1] if i < len(data.get("total_volumes", [])) else 0
                            })
                        
                        return history
                    else:
                        logger.error(f"CoinGecko API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Failed to get price history for {token_symbol}: {e}")
            return []
    
    async def get_market_trends(self) -> Dict[str, Any]:
        """Get current market trends and statistics"""
        try:
            current_data = await self.get_current_market_data()
            prices = current_data.get("prices", {})
            
            # Calculate trends
            gainers = []
            losers = []
            
            for symbol, data in prices.items():
                change_24h = data.get("price_change_percentage_24h", 0)
                if change_24h > 0:
                    gainers.append({
                        "symbol": symbol,
                        "price": data.get("current_price", 0),
                        "change_24h": change_24h,
                        "volume_24h": data.get("total_volume", 0)
                    })
                elif change_24h < 0:
                    losers.append({
                        "symbol": symbol,
                        "price": data.get("current_price", 0),
                        "change_24h": change_24h,
                        "volume_24h": data.get("total_volume", 0)
                    })
            
            # Sort by change percentage
            gainers.sort(key=lambda x: x["change_24h"], reverse=True)
            losers.sort(key=lambda x: x["change_24h"])
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "top_gainers": gainers[:10],
                "top_losers": losers[:10],
                "market_sentiment": self._calculate_market_sentiment(prices),
                "total_market_cap": sum(data.get("market_cap", 0) for data in prices.values()),
                "total_volume_24h": sum(data.get("total_volume", 0) for data in prices.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to get market trends: {e}")
            return {}
    
    async def start_price_streaming(self, callback_func):
        """Start continuous price streaming for real-time updates"""
        logger.info("Starting market data streaming...")
        
        while True:
            try:
                # Fetch latest market data
                market_data = await self.get_current_market_data()
                
                # Call the callback function (usually WebSocket broadcast)
                if callback_func:
                    await callback_func(market_data)
                
                # Log significant price movements
                await self._log_significant_movements(market_data.get("prices", {}))
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in price streaming: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _fetch_coingecko_data(self) -> Dict[str, Dict[str, Any]]:
        """Fetch data from CoinGecko API"""
        async with aiohttp.ClientSession() as session:
            # Build token list for CoinGecko
            token_ids = [self._get_coingecko_id(token) for token in self.supported_tokens]
            token_ids = [tid for tid in token_ids if tid]  # Remove None values
            
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "ids": ",".join(token_ids),
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "1h,24h,7d"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Format data by symbol
                    formatted_data = {}
                    for coin in data:
                        symbol = coin.get("symbol", "").upper()
                        if symbol in self.supported_tokens:
                            formatted_data[symbol] = {
                                "current_price": coin.get("current_price", 0),
                                "market_cap": coin.get("market_cap", 0),
                                "total_volume": coin.get("total_volume", 0),
                                "price_change_percentage_1h": coin.get("price_change_percentage_1h_in_currency", 0),
                                "price_change_percentage_24h": coin.get("price_change_percentage_24h", 0),
                                "price_change_percentage_7d": coin.get("price_change_percentage_7d_in_currency", 0),
                                "last_updated": coin.get("last_updated"),
                                "high_24h": coin.get("high_24h", 0),
                                "low_24h": coin.get("low_24h", 0),
                                "circulating_supply": coin.get("circulating_supply", 0),
                                "max_supply": coin.get("max_supply", 0)
                            }
                    
                    return formatted_data
                else:
                    raise Exception(f"CoinGecko API error: {response.status}")
    
    async def _fetch_single_token_price(self, token_symbol: str) -> Dict[str, Any]:
        """Fetch price for a single token"""
        token_id = self._get_coingecko_id(token_symbol)
        if not token_id:
            raise ValueError(f"Unknown token: {token_symbol}")
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": token_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_market_cap": "true"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    token_data = data.get(token_id, {})
                    
                    return {
                        "current_price": token_data.get("usd", 0),
                        "market_cap": token_data.get("usd_market_cap", 0),
                        "total_volume": token_data.get("usd_24h_vol", 0),
                        "price_change_percentage_24h": token_data.get("usd_24h_change", 0),
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    raise Exception(f"Failed to fetch price for {token_symbol}")
    
    async def _calculate_market_stats(self, prices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall market statistics"""
        if not prices:
            return {}
        
        # Calculate market-wide metrics
        total_market_cap = sum(data.get("market_cap", 0) for data in prices.values())
        total_volume = sum(data.get("total_volume", 0) for data in prices.values())
        
        # Price changes
        changes_24h = [data.get("price_change_percentage_24h", 0) for data in prices.values()]
        avg_change_24h = sum(changes_24h) / len(changes_24h) if changes_24h else 0
        
        # Count gainers vs losers
        gainers = len([c for c in changes_24h if c > 0])
        losers = len([c for c in changes_24h if c < 0])
        
        return {
            "total_market_cap": total_market_cap,
            "total_volume_24h": total_volume,
            "average_change_24h": avg_change_24h,
            "gainers_count": gainers,
            "losers_count": losers,
            "neutral_count": len(changes_24h) - gainers - losers,
            "market_sentiment": "bullish" if avg_change_24h > 2 else "bearish" if avg_change_24h < -2 else "neutral"
        }
    
    def _calculate_market_sentiment(self, prices: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall market sentiment"""
        if not prices:
            return "neutral"
        
        changes = [data.get("price_change_percentage_24h", 0) for data in prices.values()]
        avg_change = sum(changes) / len(changes)
        
        if avg_change > 5:
            return "very_bullish"
        elif avg_change > 2:
            return "bullish"
        elif avg_change > -2:
            return "neutral"
        elif avg_change > -5:
            return "bearish"
        else:
            return "very_bearish"
    
    async def _log_significant_movements(self, prices: Dict[str, Dict[str, Any]]):
        """Log significant price movements"""
        for symbol, data in prices.items():
            change_24h = data.get("price_change_percentage_24h", 0)
            price = data.get("current_price", 0)
            
            # Log significant movements (>10% change)
            if abs(change_24h) > 10:
                movement_type = "surge" if change_24h > 0 else "drop"
                
                signal_logger.info("market_movement", extra={
                    "token": symbol,
                    "price": price,
                    "change_24h": change_24h,
                    "movement_type": movement_type,
                    "volume_24h": data.get("total_volume", 0),
                    "market_cap": data.get("market_cap", 0)
                })
    
    def _get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Map token symbol to CoinGecko ID"""
        symbol_map = {
            "SOL": "solana",
            "USDC": "usd-coin",
            "RAY": "raydium",
            "SRM": "serum",
            "ORCA": "orca",
            "MNGO": "mango-markets",
            "STEP": "step-finance",
            "ROPE": "rope-token",
            "COPE": "cope",
            "MEDIA": "media-network",
            "ATLAS": "star-atlas",
            "POLIS": "star-atlas-dao",
            "SAMO": "samoyedcoin",
            "NINJA": "ninja-protocol",
            "SHDW": "genesysgo-shadow",
            "DFL": "defi-land",
            "SLND": "solend",
            "PORT": "port-finance",
            "TULIP": "tulip-protocol",
            "FIDA": "bonfida",
            "KIN": "kin",
            "MAPS": "maps",
            "AUDIO": "audius"
        }
        
        return symbol_map.get(symbol.upper())

# Utility functions for market data integration
async def get_market_sentiment() -> str:
    """Get current market sentiment"""
    service = MarketDataService()
    trends = await service.get_market_trends()
    return trends.get("market_sentiment", "neutral")

async def get_trending_tokens(limit: int = 10) -> List[Dict[str, Any]]:
    """Get trending tokens by volume and price movement"""
    service = MarketDataService()
    market_data = await service.get_current_market_data()
    prices = market_data.get("prices", {})
    
    # Sort by combination of volume and price change
    trending = []
    for symbol, data in prices.items():
        score = (
            data.get("price_change_percentage_24h", 0) * 0.3 +
            (data.get("total_volume", 0) / 1000000) * 0.7  # Volume in millions
        )
        trending.append({
            "symbol": symbol,
            "score": score,
            "price": data.get("current_price", 0),
            "change_24h": data.get("price_change_percentage_24h", 0),
            "volume_24h": data.get("total_volume", 0)
        })
    
    trending.sort(key=lambda x: x["score"], reverse=True)
    return trending[:limit] 