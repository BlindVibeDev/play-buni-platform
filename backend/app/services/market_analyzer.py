"""
Market Analyzer - Technical analysis and market data processing service.

This module provides comprehensive market analysis capabilities including:
- Technical indicator calculations
- Price pattern recognition
- Volume analysis
- Support/resistance level detection
- Market structure analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """
    Advanced market analysis engine for technical indicator calculation
    and market structure analysis.
    """
    
    def __init__(self):
        self.indicator_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def calculate_technical_indicators(
        self, 
        token_address: str, 
        token_data: Dict
    ) -> Dict[str, float]:
        """
        Calculate comprehensive technical indicators for a token.
        
        Returns a dictionary of technical indicators including:
        - RSI, MACD, Bollinger Bands
        - Support/Resistance levels
        - Volume indicators
        - Momentum indicators
        """
        
        # Check cache first
        cache_key = f"{token_address}_{int(datetime.utcnow().timestamp() // self.cache_ttl)}"
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        
        try:
            # Get historical price data
            price_data = await self._get_price_history(token_address)
            
            if not price_data or len(price_data) < 50:
                logger.warning(f"Insufficient price data for {token_address}")
                return self._get_default_indicators()
            
            df = pd.DataFrame(price_data)
            
            indicators = {}
            
            # Price-based indicators
            indicators.update(self._calculate_rsi(df))
            indicators.update(self._calculate_macd(df))
            indicators.update(self._calculate_bollinger_bands(df))
            indicators.update(self._calculate_moving_averages(df))
            
            # Volume indicators
            indicators.update(self._calculate_volume_indicators(df))
            
            # Support/Resistance
            indicators.update(self._calculate_support_resistance(df))
            
            # Momentum indicators
            indicators.update(self._calculate_momentum_indicators(df))
            
            # Pattern recognition
            indicators.update(self._detect_patterns(df))
            
            # Cache results
            self.indicator_cache[cache_key] = indicators
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {token_address}: {e}")
            return self._get_default_indicators()
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calculate Relative Strength Index."""
        if len(df) < period + 1:
            return {'rsi': 50.0}
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            'rsi_oversold': float(rsi.iloc[-1] < 30) if not pd.isna(rsi.iloc[-1]) else 0.0,
            'rsi_overbought': float(rsi.iloc[-1] > 70) if not pd.isna(rsi.iloc[-1]) else 0.0
        }
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(df) < 26:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
        
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            'macd_signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            'macd_histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
            'macd_bullish': float(macd_line.iloc[-1] > signal_line.iloc[-1]) if len(macd_line) > 0 else 0.0
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(df) < period:
            return {'bb_upper': 0.0, 'bb_lower': 0.0, 'bb_middle': 0.0, 'bb_position': 0.5}
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = df['close'].iloc[-1]
        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        return {
            'bb_upper': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else current_price * 1.02,
            'bb_lower': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else current_price * 0.98,
            'bb_middle': float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else current_price,
            'bb_position': float(bb_position) if not pd.isna(bb_position) else 0.5,
            'bb_squeeze': float(std.iloc[-1] < std.rolling(window=10).mean().iloc[-1]) if len(std) > 10 else 0.0
        }
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various moving averages."""
        current_price = df['close'].iloc[-1]
        
        mas = {}
        for period in [5, 10, 20, 50, 200]:
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean().iloc[-1]
                mas[f'ma_{period}'] = float(ma) if not pd.isna(ma) else current_price
                mas[f'price_above_ma_{period}'] = float(current_price > ma) if not pd.isna(ma) else 0.5
            else:
                mas[f'ma_{period}'] = current_price
                mas[f'price_above_ma_{period}'] = 0.5
        
        # Golden cross / Death cross signals
        if len(df) >= 50:
            ma_20 = df['close'].rolling(window=20).mean()
            ma_50 = df['close'].rolling(window=50).mean()
            
            mas['golden_cross'] = float(
                ma_20.iloc[-1] > ma_50.iloc[-1] and ma_20.iloc[-2] <= ma_50.iloc[-2]
            ) if len(ma_20) > 1 and not (pd.isna(ma_20.iloc[-1]) or pd.isna(ma_50.iloc[-1])) else 0.0
            
            mas['death_cross'] = float(
                ma_20.iloc[-1] < ma_50.iloc[-1] and ma_20.iloc[-2] >= ma_50.iloc[-2]
            ) if len(ma_20) > 1 and not (pd.isna(ma_20.iloc[-1]) or pd.isna(ma_50.iloc[-1])) else 0.0
        else:
            mas['golden_cross'] = 0.0
            mas['death_cross'] = 0.0
        
        return mas
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators."""
        if 'volume' not in df.columns or len(df) < 10:
            return {
                'volume_ratio': 1.0,
                'volume_sma_ratio': 1.0,
                'volume_trend': 0.0,
                'price_volume_trend': 0.0
            }
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        
        # Volume ratio (current vs average)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume SMA ratio
        volume_sma_10 = df['volume'].rolling(window=10).mean().iloc[-1]
        volume_sma_20 = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_sma_ratio = volume_sma_10 / volume_sma_20 if volume_sma_20 > 0 else 1.0
        
        # Volume trend
        volume_trend = 1.0 if len(df) < 5 else np.polyfit(range(5), df['volume'].tail(5), 1)[0]
        
        # Price Volume Trend (PVT)
        if len(df) > 1:
            price_change = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
            pvt = (price_change * df['volume']).cumsum()
            price_volume_trend = float(pvt.iloc[-1]) if not pd.isna(pvt.iloc[-1]) else 0.0
        else:
            price_volume_trend = 0.0
        
        return {
            'volume_ratio': float(volume_ratio),
            'volume_sma_ratio': float(volume_sma_ratio),
            'volume_trend': float(volume_trend),
            'price_volume_trend': price_volume_trend
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        if len(df) < 20:
            current_price = df['close'].iloc[-1]
            return {
                'support': current_price * 0.97,
                'resistance': current_price * 1.03,
                'support_strength': 0.5,
                'resistance_strength': 0.5
            }
        
        # Use pivot points method
        highs = df['high'].rolling(window=5, center=True).max()
        lows = df['low'].rolling(window=5, center=True).min()
        
        # Find pivot highs and lows
        pivot_highs = df[df['high'] == highs]['high'].dropna()
        pivot_lows = df[df['low'] == lows]['low'].dropna()
        
        current_price = df['close'].iloc[-1]
        
        # Find nearest support and resistance
        resistance_levels = pivot_highs[pivot_highs > current_price]
        support_levels = pivot_lows[pivot_lows < current_price]
        
        resistance = resistance_levels.min() if len(resistance_levels) > 0 else current_price * 1.03
        support = support_levels.max() if len(support_levels) > 0 else current_price * 0.97
        
        # Calculate strength based on number of touches
        resistance_strength = len(pivot_highs[(pivot_highs >= resistance * 0.99) & (pivot_highs <= resistance * 1.01)]) / 10
        support_strength = len(pivot_lows[(pivot_lows >= support * 0.99) & (pivot_lows <= support * 1.01)]) / 10
        
        return {
            'support': float(support),
            'resistance': float(resistance),
            'support_strength': min(float(resistance_strength), 1.0),
            'resistance_strength': min(float(support_strength), 1.0)
        }
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators."""
        if len(df) < 14:
            return {
                'momentum': 0.0,
                'rate_of_change': 0.0,
                'stochastic_k': 50.0,
                'stochastic_d': 50.0
            }
        
        # Momentum
        momentum = df['close'].iloc[-1] - df['close'].iloc[-10] if len(df) >= 10 else 0.0
        
        # Rate of Change
        roc = ((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100) if len(df) >= 10 else 0.0
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        
        k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'momentum': float(momentum),
            'rate_of_change': float(roc),
            'stochastic_k': float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else 50.0,
            'stochastic_d': float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else 50.0
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect common chart patterns."""
        if len(df) < 20:
            return {
                'double_top': 0.0,
                'double_bottom': 0.0,
                'head_shoulders': 0.0,
                'triangle': 0.0,
                'breakout': 0.0
            }
        
        patterns = {
            'double_top': 0.0,
            'double_bottom': 0.0,
            'head_shoulders': 0.0,
            'triangle': 0.0,
            'breakout': 0.0
        }
        
        # Simple pattern detection logic
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        
        # Double top pattern (simplified)
        if len(recent_highs) >= 5:
            max_high = recent_highs.max()
            high_count = len(recent_highs[recent_highs >= max_high * 0.98])
            if high_count >= 2:
                patterns['double_top'] = min(high_count / 5, 1.0)
        
        # Double bottom pattern (simplified)
        if len(recent_lows) >= 5:
            min_low = recent_lows.min()
            low_count = len(recent_lows[recent_lows <= min_low * 1.02])
            if low_count >= 2:
                patterns['double_bottom'] = min(low_count / 5, 1.0)
        
        # Breakout detection
        if len(df) >= 20:
            recent_range = df['high'].tail(20).max() - df['low'].tail(20).min()
            current_price = df['close'].iloc[-1]
            range_high = df['high'].tail(20).max()
            range_low = df['low'].tail(20).min()
            
            if current_price > range_high * 0.99:
                patterns['breakout'] = 1.0
            elif current_price < range_low * 1.01:
                patterns['breakout'] = -1.0
        
        return patterns
    
    async def _get_price_history(self, token_address: str, limit: int = 200) -> List[Dict]:
        """
        Get historical price data for a token.
        This would typically connect to a data provider like Bitquery, Helius, or Jupiter.
        """
        # Placeholder implementation
        # In a real implementation, this would fetch from:
        # - Bitquery API for historical DEX trades
        # - Helius API for token data
        # - Jupiter API for price history
        # - Direct Solana RPC for on-chain data
        
        try:
            # Simulate API call delay
            await asyncio.sleep(0.1)
            
            # Return mock data structure
            # Real implementation would return actual OHLCV data
            return [
                {
                    'timestamp': datetime.utcnow() - timedelta(minutes=i),
                    'open': 100.0 + np.random.randn() * 5,
                    'high': 105.0 + np.random.randn() * 5,
                    'low': 95.0 + np.random.randn() * 5,
                    'close': 100.0 + np.random.randn() * 5,
                    'volume': 1000000 + np.random.randn() * 100000
                }
                for i in range(limit)
            ]
            
        except Exception as e:
            logger.error(f"Error fetching price history for {token_address}: {e}")
            return []
    
    def _get_default_indicators(self) -> Dict[str, float]:
        """Return default indicator values when calculation fails."""
        return {
            'rsi': 50.0,
            'rsi_oversold': 0.0,
            'rsi_overbought': 0.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'macd_bullish': 0.0,
            'bb_upper': 0.0,
            'bb_lower': 0.0,
            'bb_middle': 0.0,
            'bb_position': 0.5,
            'bb_squeeze': 0.0,
            'ma_5': 0.0,
            'ma_10': 0.0,
            'ma_20': 0.0,
            'ma_50': 0.0,
            'ma_200': 0.0,
            'price_above_ma_5': 0.5,
            'price_above_ma_10': 0.5,
            'price_above_ma_20': 0.5,
            'price_above_ma_50': 0.5,
            'price_above_ma_200': 0.5,
            'golden_cross': 0.0,
            'death_cross': 0.0,
            'volume_ratio': 1.0,
            'volume_sma_ratio': 1.0,
            'volume_trend': 0.0,
            'price_volume_trend': 0.0,
            'support': 0.0,
            'resistance': 0.0,
            'support_strength': 0.5,
            'resistance_strength': 0.5,
            'momentum': 0.0,
            'rate_of_change': 0.0,
            'stochastic_k': 50.0,
            'stochastic_d': 50.0,
            'double_top': 0.0,
            'double_bottom': 0.0,
            'head_shoulders': 0.0,
            'triangle': 0.0,
            'breakout': 0.0
        } 