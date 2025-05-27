"""
Technical Analysis Engine for Play Buni Platform

This service performs technical analysis on market data to generate
trading signals. It includes various technical indicators, pattern
recognition, and trend analysis algorithms.

Features:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Pattern recognition
- Volume analysis
- Support/resistance levels
- Trend analysis
- Signal strength calculation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

from app.core.logging import get_logger
from app.core.cache import cache_manager
from app.services.market_data import TokenMetrics, market_data_service

logger = get_logger(__name__)


class TrendDirection(Enum):
    """Trend direction indicators"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    bb_upper: Optional[float]
    bb_middle: Optional[float]
    bb_lower: Optional[float]
    bb_width: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    ema_12: Optional[float]
    ema_26: Optional[float]
    volume_sma: Optional[float]
    atr: Optional[float]
    stoch_k: Optional[float]
    stoch_d: Optional[float]
    williams_r: Optional[float]
    timestamp: datetime


@dataclass
class PatternAnalysis:
    """Pattern recognition results"""
    pattern_type: Optional[str]
    confidence: float
    breakout_probability: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    pattern_duration: Optional[int]
    volume_confirmation: bool


@dataclass
class SupportResistance:
    """Support and resistance levels"""
    support_levels: List[float]
    resistance_levels: List[float]
    current_level_type: str  # "support", "resistance", "between"
    distance_to_support: Optional[float]
    distance_to_resistance: Optional[float]
    strength_score: float


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    short_term_trend: TrendDirection
    medium_term_trend: TrendDirection
    long_term_trend: TrendDirection
    trend_strength: float
    trend_duration: Optional[int]
    reversal_probability: float
    momentum_score: float


@dataclass
class TechnicalAnalysisResult:
    """Complete technical analysis result"""
    symbol: str
    mint_address: str
    indicators: TechnicalIndicators
    pattern: PatternAnalysis
    support_resistance: SupportResistance
    trend: TrendAnalysis
    overall_signal: str  # "BUY", "SELL", "HOLD"
    signal_strength: SignalStrength
    confidence_score: float
    risk_score: float
    timestamp: datetime


class TechnicalAnalysisEngine:
    """
    Technical Analysis Engine
    
    Performs comprehensive technical analysis on token price data
    to generate trading signals and market insights.
    """
    
    def __init__(self):
        self.price_history_cache = {}
        self.analysis_cache = {}
    
    async def get_price_history(
        self, 
        mint_address: str, 
        symbol: str, 
        periods: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data for technical analysis
        
        Args:
            mint_address: Token mint address
            symbol: Token symbol
            periods: Number of periods to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"price_history:{mint_address}:{periods}"
        cached = await cache_manager.get_json(cache_key)
        
        if cached:
            return pd.DataFrame(cached)
        
        try:
            # For now, we'll simulate historical data based on current metrics
            # In production, you'd fetch actual historical data from your data provider
            async with market_data_service as service:
                current_metrics = await service.get_comprehensive_metrics(mint_address, symbol)
                
                if not current_metrics:
                    return None
                
                # Generate simulated historical data
                # This is a placeholder - replace with actual historical data fetching
                dates = pd.date_range(
                    end=datetime.now(), 
                    periods=periods, 
                    freq='5T'  # 5-minute intervals
                )
                
                # Simulate price movement with some volatility
                base_price = current_metrics.price_usd
                volatility = current_metrics.volatility / 100 if current_metrics.volatility else 0.05
                
                # Generate random walk with trend
                returns = np.random.normal(0, volatility, periods)
                prices = [base_price]
                
                for i in range(1, periods):
                    new_price = prices[-1] * (1 + returns[i])
                    prices.append(max(new_price, 0.0001))  # Prevent negative prices
                
                # Create OHLCV data
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices],
                    'close': prices,
                    'volume': [current_metrics.volume_24h / 288 * (1 + np.random.normal(0, 0.3)) for _ in range(periods)]
                })
                
                # Ensure high >= close >= low and high >= open >= low
                df['high'] = df[['open', 'close', 'high']].max(axis=1)
                df['low'] = df[['open', 'close', 'low']].min(axis=1)
                
                # Cache for 5 minutes
                await cache_manager.set_json(cache_key, df.to_dict('records'), expire=300)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting price history for {mint_address}: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None
    
    def calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return (
                float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
                float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
                float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
            )
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return None, None, None
    
    def calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: float = 2
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            width = (upper_band - lower_band) / sma * 100
            
            return (
                float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else None,
                float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else None,
                float(width.iloc[-1]) if not pd.isna(width.iloc[-1]) else None
            )
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return None, None, None, None
    
    def calculate_stochastic(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return (
                float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else None,
                float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else None
            )
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return None, None
    
    def calculate_williams_r(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> Optional[float]:
        """Calculate Williams %R"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None
            
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return None
    
    def calculate_atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> Optional[float]:
        """Calculate Average True Range"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    def detect_support_resistance(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        current_price: float
    ) -> SupportResistance:
        """Detect support and resistance levels"""
        try:
            # Find local minima and maxima
            window = 5
            support_levels = []
            resistance_levels = []
            
            # Find support levels (local minima)
            for i in range(window, len(low) - window):
                if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                    support_levels.append(float(low.iloc[i]))
            
            # Find resistance levels (local maxima)
            for i in range(window, len(high) - window):
                if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                    resistance_levels.append(float(high.iloc[i]))
            
            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))
            
            # Find closest levels
            supports_below = [s for s in support_levels if s < current_price]
            resistances_above = [r for r in resistance_levels if r > current_price]
            
            closest_support = max(supports_below) if supports_below else None
            closest_resistance = min(resistances_above) if resistances_above else None
            
            # Determine current level type
            if closest_support and closest_resistance:
                if abs(current_price - closest_support) < abs(current_price - closest_resistance):
                    level_type = "near_support"
                else:
                    level_type = "near_resistance"
            elif closest_support:
                level_type = "above_support"
            elif closest_resistance:
                level_type = "below_resistance"
            else:
                level_type = "between"
            
            # Calculate distances
            distance_to_support = ((current_price - closest_support) / current_price * 100) if closest_support else None
            distance_to_resistance = ((closest_resistance - current_price) / current_price * 100) if closest_resistance else None
            
            # Calculate strength score based on number of touches
            strength_score = min(len(support_levels) + len(resistance_levels), 10) / 10
            
            return SupportResistance(
                support_levels=support_levels[-5:],  # Keep last 5
                resistance_levels=resistance_levels[-5:],  # Keep last 5
                current_level_type=level_type,
                distance_to_support=distance_to_support,
                distance_to_resistance=distance_to_resistance,
                strength_score=strength_score
            )
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return SupportResistance(
                support_levels=[],
                resistance_levels=[],
                current_level_type="unknown",
                distance_to_support=None,
                distance_to_resistance=None,
                strength_score=0.0
            )
    
    def analyze_trend(self, prices: pd.Series) -> TrendAnalysis:
        """Analyze price trend"""
        try:
            # Calculate different timeframe trends
            short_sma = prices.rolling(window=10).mean()
            medium_sma = prices.rolling(window=30).mean()
            long_sma = prices.rolling(window=50).mean()
            
            current_price = prices.iloc[-1]
            
            # Determine trend directions
            short_trend = TrendDirection.BULLISH if current_price > short_sma.iloc[-1] else TrendDirection.BEARISH
            medium_trend = TrendDirection.BULLISH if current_price > medium_sma.iloc[-1] else TrendDirection.BEARISH
            long_trend = TrendDirection.BULLISH if current_price > long_sma.iloc[-1] else TrendDirection.BEARISH
            
            # Calculate trend strength
            price_change = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] * 100
            trend_strength = min(abs(price_change) / 10, 1.0)  # Normalize to 0-1
            
            # Calculate momentum
            momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] * 100
            momentum_score = max(0, min(momentum / 5 + 0.5, 1.0))  # Normalize to 0-1
            
            # Estimate reversal probability
            # Higher when price is far from moving averages
            deviation_from_sma = abs(current_price - short_sma.iloc[-1]) / short_sma.iloc[-1] * 100
            reversal_probability = min(deviation_from_sma / 20, 0.8)  # Max 80%
            
            return TrendAnalysis(
                short_term_trend=short_trend,
                medium_term_trend=medium_trend,
                long_term_trend=long_trend,
                trend_strength=trend_strength,
                trend_duration=None,  # Would need more historical data
                reversal_probability=reversal_probability,
                momentum_score=momentum_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return TrendAnalysis(
                short_term_trend=TrendDirection.UNKNOWN,
                medium_term_trend=TrendDirection.UNKNOWN,
                long_term_trend=TrendDirection.UNKNOWN,
                trend_strength=0.0,
                trend_duration=None,
                reversal_probability=0.5,
                momentum_score=0.5
            )
    
    def detect_patterns(self, df: pd.DataFrame) -> PatternAnalysis:
        """Detect chart patterns"""
        try:
            # Simple pattern detection
            # In production, you'd implement more sophisticated pattern recognition
            
            close = df['close']
            volume = df['volume']
            
            # Check for breakout pattern
            recent_high = close.rolling(window=20).max().iloc[-1]
            recent_low = close.rolling(window=20).min().iloc[-1]
            current_price = close.iloc[-1]
            
            # Volume confirmation
            avg_volume = volume.rolling(window=20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_confirmation = current_volume > avg_volume * 1.5
            
            # Simple breakout detection
            if current_price > recent_high * 0.98:
                pattern_type = "bullish_breakout"
                confidence = 0.7 if volume_confirmation else 0.4
                breakout_probability = 0.8
                target_price = current_price * 1.1
                stop_loss = recent_high * 0.95
            elif current_price < recent_low * 1.02:
                pattern_type = "bearish_breakdown"
                confidence = 0.7 if volume_confirmation else 0.4
                breakout_probability = 0.8
                target_price = current_price * 0.9
                stop_loss = recent_low * 1.05
            else:
                pattern_type = "consolidation"
                confidence = 0.5
                breakout_probability = 0.3
                target_price = None
                stop_loss = None
            
            return PatternAnalysis(
                pattern_type=pattern_type,
                confidence=confidence,
                breakout_probability=breakout_probability,
                target_price=target_price,
                stop_loss=stop_loss,
                pattern_duration=20,  # Assuming 20 periods
                volume_confirmation=volume_confirmation
            )
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return PatternAnalysis(
                pattern_type=None,
                confidence=0.0,
                breakout_probability=0.0,
                target_price=None,
                stop_loss=None,
                pattern_duration=None,
                volume_confirmation=False
            )
    
    async def analyze_token(self, mint_address: str, symbol: str) -> Optional[TechnicalAnalysisResult]:
        """
        Perform comprehensive technical analysis on a token
        
        Args:
            mint_address: Token mint address
            symbol: Token symbol
            
        Returns:
            Complete technical analysis result
        """
        try:
            # Check cache first
            cache_key = f"technical_analysis:{mint_address}"
            cached = await cache_manager.get_json(cache_key)
            if cached:
                return TechnicalAnalysisResult(**cached)
            
            # Get price history
            df = await self.get_price_history(mint_address, symbol)
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient price data for {symbol}")
                return None
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(close)
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bollinger_bands(close)
            stoch_k, stoch_d = self.calculate_stochastic(high, low, close)
            williams_r = self.calculate_williams_r(high, low, close)
            atr = self.calculate_atr(high, low, close)
            
            # Moving averages
            sma_20 = float(close.rolling(window=20).mean().iloc[-1])
            sma_50 = float(close.rolling(window=50).mean().iloc[-1])
            ema_12 = float(close.ewm(span=12).mean().iloc[-1])
            ema_26 = float(close.ewm(span=26).mean().iloc[-1])
            volume_sma = float(volume.rolling(window=20).mean().iloc[-1])
            
            indicators = TechnicalIndicators(
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_width=bb_width,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=ema_12,
                ema_26=ema_26,
                volume_sma=volume_sma,
                atr=atr,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                williams_r=williams_r,
                timestamp=datetime.now()
            )
            
            # Pattern analysis
            pattern = self.detect_patterns(df)
            
            # Support/resistance analysis
            current_price = float(close.iloc[-1])
            support_resistance = self.detect_support_resistance(high, low, close, current_price)
            
            # Trend analysis
            trend = self.analyze_trend(close)
            
            # Generate overall signal
            signal_score = 0
            confidence_factors = []
            
            # RSI signals
            if rsi:
                if rsi < 30:
                    signal_score += 2  # Oversold - bullish
                    confidence_factors.append("RSI oversold")
                elif rsi > 70:
                    signal_score -= 2  # Overbought - bearish
                    confidence_factors.append("RSI overbought")
            
            # MACD signals
            if macd and macd_signal:
                if macd > macd_signal:
                    signal_score += 1
                    confidence_factors.append("MACD bullish")
                else:
                    signal_score -= 1
                    confidence_factors.append("MACD bearish")
            
            # Bollinger Bands signals
            if bb_lower and bb_upper and current_price:
                if current_price < bb_lower:
                    signal_score += 1  # Below lower band - potential bounce
                    confidence_factors.append("Below BB lower band")
                elif current_price > bb_upper:
                    signal_score -= 1  # Above upper band - potential pullback
                    confidence_factors.append("Above BB upper band")
            
            # Trend alignment
            bullish_trends = sum([
                trend.short_term_trend == TrendDirection.BULLISH,
                trend.medium_term_trend == TrendDirection.BULLISH,
                trend.long_term_trend == TrendDirection.BULLISH
            ])
            
            if bullish_trends >= 2:
                signal_score += 2
                confidence_factors.append("Trend alignment bullish")
            elif bullish_trends == 0:
                signal_score -= 2
                confidence_factors.append("Trend alignment bearish")
            
            # Pattern signals
            if pattern.pattern_type == "bullish_breakout":
                signal_score += 2
                confidence_factors.append("Bullish breakout pattern")
            elif pattern.pattern_type == "bearish_breakdown":
                signal_score -= 2
                confidence_factors.append("Bearish breakdown pattern")
            
            # Determine overall signal
            if signal_score >= 3:
                overall_signal = "BUY"
                signal_strength = SignalStrength.STRONG if signal_score >= 5 else SignalStrength.MODERATE
            elif signal_score <= -3:
                overall_signal = "SELL"
                signal_strength = SignalStrength.STRONG if signal_score <= -5 else SignalStrength.MODERATE
            else:
                overall_signal = "HOLD"
                signal_strength = SignalStrength.WEAK
            
            # Calculate confidence score
            confidence_score = min(len(confidence_factors) / 5, 1.0)
            
            # Calculate risk score
            risk_factors = []
            if bb_width and bb_width > 10:
                risk_factors.append("High volatility")
            if trend.reversal_probability > 0.6:
                risk_factors.append("High reversal probability")
            if not pattern.volume_confirmation:
                risk_factors.append("Weak volume confirmation")
            
            risk_score = len(risk_factors) / 3
            
            result = TechnicalAnalysisResult(
                symbol=symbol,
                mint_address=mint_address,
                indicators=indicators,
                pattern=pattern,
                support_resistance=support_resistance,
                trend=trend,
                overall_signal=overall_signal,
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                risk_score=risk_score,
                timestamp=datetime.now()
            )
            
            # Cache for 2 minutes
            await cache_manager.set_json(cache_key, result.__dict__, expire=120)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing token {symbol}: {e}")
            return None


# Global technical analysis engine instance
technical_analysis_engine = TechnicalAnalysisEngine() 