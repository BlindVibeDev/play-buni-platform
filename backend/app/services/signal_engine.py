"""
Signal Generation & Distribution Engine for Play Buni Platform

This is the core engine that orchestrates the entire signal generation
and distribution process. It combines market data, technical analysis,
sentiment analysis, and risk management to generate high-quality
trading signals for different user tiers.

Features:
- Multi-factor signal scoring
- Queue management for different access levels
- Performance tracking and feedback loops
- Risk management and filtering
- Dual-channel distribution (social vs premium)
- Real-time signal generation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import random
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, and_

from app.core.config import settings
from app.core.logging import get_logger
from app.core.cache import cache_manager
from app.core.database import get_db_session
from app.services.market_data import market_data_service, TokenMetrics
from app.services.technical_analysis import technical_analysis_engine, TechnicalAnalysisResult, SignalStrength
from app.models.signals import Signal, SignalPerformance, SignalQueue
from app.models.users import User

logger = get_logger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalPriority(Enum):
    """Signal priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DistributionChannel(Enum):
    """Signal distribution channels"""
    PUBLIC_SOCIAL = "public_social"
    PREMIUM_REALTIME = "premium_realtime"
    VIP_EXCLUSIVE = "vip_exclusive"
    ADMIN_INTERNAL = "admin_internal"


@dataclass
class SignalScore:
    """Signal scoring breakdown"""
    technical_score: float
    market_score: float
    sentiment_score: float
    risk_score: float
    volume_score: float
    momentum_score: float
    total_score: float
    confidence_level: float


@dataclass
class GeneratedSignal:
    """Generated trading signal"""
    symbol: str
    mint_address: str
    signal_type: SignalType
    signal_strength: SignalStrength
    priority: SignalPriority
    score: SignalScore
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    timeframe: str
    reasoning: List[str]
    risk_level: str
    expected_duration: Optional[int]  # minutes
    distribution_channels: List[DistributionChannel]
    metadata: Dict[str, Any]
    generated_at: datetime


@dataclass
class QueuedSignal:
    """Signal in distribution queue"""
    signal: GeneratedSignal
    queue_priority: int
    scheduled_for: datetime
    target_channels: List[DistributionChannel]
    retry_count: int
    max_retries: int


class SignalGenerationEngine:
    """
    Signal Generation Engine
    
    Core engine that generates, scores, and queues trading signals
    based on comprehensive market analysis.
    """
    
    def __init__(self):
        self.is_running = False
        self.generation_task = None
        self.monitored_tokens = []
        self.signal_history = []
        self.performance_tracker = {}
        
        # Scoring weights
        self.scoring_weights = {
            "technical": 0.30,
            "market": 0.25,
            "sentiment": 0.15,
            "risk": 0.15,
            "volume": 0.10,
            "momentum": 0.05
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            "strong_buy": 0.8,
            "buy": 0.6,
            "hold": 0.4,
            "sell": 0.3,
            "strong_sell": 0.1
        }
    
    async def initialize(self):
        """Initialize the signal generation engine"""
        try:
            logger.info("Initializing Signal Generation Engine...")
            
            # Load monitored tokens
            await self.load_monitored_tokens()
            
            # Initialize performance tracking
            await self.initialize_performance_tracking()
            
            logger.info(f"Signal engine initialized with {len(self.monitored_tokens)} tokens")
            
        except Exception as e:
            logger.error(f"Error initializing signal engine: {e}")
            raise
    
    async def load_monitored_tokens(self):
        """Load list of tokens to monitor for signals"""
        try:
            # Check cache first
            cached_tokens = await cache_manager.get_json("monitored_tokens")
            if cached_tokens:
                self.monitored_tokens = cached_tokens
                return
            
            # Get trending tokens from market data service
            async with market_data_service as service:
                trending = await service.get_trending_tokens(limit=settings.max_tokens_to_monitor)
                
                # Add some popular Solana tokens
                popular_tokens = [
                    ("So11111111111111111111111111111111111111112", "SOL"),
                    ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "USDC"),
                    ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", "USDT"),
                    ("mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So", "mSOL"),
                    ("7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj", "stSOL"),
                ]
                
                # Combine trending and popular tokens
                all_tokens = []
                for mint in trending:
                    all_tokens.append((mint, "UNKNOWN"))
                
                all_tokens.extend(popular_tokens)
                
                # Remove duplicates and limit
                seen = set()
                unique_tokens = []
                for mint, symbol in all_tokens:
                    if mint not in seen:
                        seen.add(mint)
                        unique_tokens.append((mint, symbol))
                
                self.monitored_tokens = unique_tokens[:settings.max_tokens_to_monitor]
                
                # Cache for 1 hour
                await cache_manager.set_json("monitored_tokens", self.monitored_tokens, expire=3600)
                
        except Exception as e:
            logger.error(f"Error loading monitored tokens: {e}")
            # Fallback to popular tokens only
            self.monitored_tokens = [
                ("So11111111111111111111111111111111111111112", "SOL"),
                ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "USDC"),
                ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", "USDT"),
            ]
    
    async def initialize_performance_tracking(self):
        """Initialize performance tracking for signals"""
        try:
            async with get_db_session() as db:
                # Load recent signal performance
                result = await db.execute(
                    select(SignalPerformance)
                    .where(SignalPerformance.created_at >= datetime.utcnow() - timedelta(days=30))
                    .order_by(SignalPerformance.created_at.desc())
                    .limit(1000)
                )
                
                performances = result.scalars().all()
                
                # Calculate success rates by token
                for perf in performances:
                    if perf.mint_address not in self.performance_tracker:
                        self.performance_tracker[perf.mint_address] = {
                            "total_signals": 0,
                            "successful_signals": 0,
                            "success_rate": 0.5,
                            "avg_return": 0.0,
                            "last_updated": datetime.utcnow()
                        }
                    
                    tracker = self.performance_tracker[perf.mint_address]
                    tracker["total_signals"] += 1
                    
                    if perf.actual_return and perf.actual_return > 0:
                        tracker["successful_signals"] += 1
                    
                    if tracker["total_signals"] > 0:
                        tracker["success_rate"] = tracker["successful_signals"] / tracker["total_signals"]
                    
                    if perf.actual_return:
                        tracker["avg_return"] = (tracker["avg_return"] + perf.actual_return) / 2
                
                logger.info(f"Loaded performance data for {len(self.performance_tracker)} tokens")
                
        except Exception as e:
            logger.error(f"Error initializing performance tracking: {e}")
    
    def calculate_technical_score(self, analysis: TechnicalAnalysisResult) -> float:
        """Calculate technical analysis score (0-1)"""
        try:
            score = 0.0
            factors = 0
            
            # RSI scoring
            if analysis.indicators.rsi:
                rsi = analysis.indicators.rsi
                if 30 <= rsi <= 70:
                    score += 0.8  # Neutral zone
                elif rsi < 30:
                    score += 1.0  # Oversold - bullish
                elif rsi > 70:
                    score += 0.2  # Overbought - bearish
                factors += 1
            
            # MACD scoring
            if analysis.indicators.macd and analysis.indicators.macd_signal:
                if analysis.indicators.macd > analysis.indicators.macd_signal:
                    score += 0.8  # Bullish crossover
                else:
                    score += 0.3  # Bearish crossover
                factors += 1
            
            # Bollinger Bands scoring
            if (analysis.indicators.bb_lower and analysis.indicators.bb_upper and 
                analysis.indicators.bb_middle):
                
                current_price = analysis.indicators.bb_middle  # Approximate
                bb_position = 0.5  # Default middle
                
                if current_price < analysis.indicators.bb_lower:
                    bb_position = 0.0  # Below lower band
                    score += 0.9  # Potential bounce
                elif current_price > analysis.indicators.bb_upper:
                    bb_position = 1.0  # Above upper band
                    score += 0.2  # Potential pullback
                else:
                    score += 0.6  # Within bands
                factors += 1
            
            # Trend alignment scoring
            bullish_trends = sum([
                analysis.trend.short_term_trend.value == "bullish",
                analysis.trend.medium_term_trend.value == "bullish",
                analysis.trend.long_term_trend.value == "bullish"
            ])
            
            trend_score = bullish_trends / 3
            score += trend_score
            factors += 1
            
            # Pattern scoring
            if analysis.pattern.pattern_type:
                if "bullish" in analysis.pattern.pattern_type:
                    score += analysis.pattern.confidence
                elif "bearish" in analysis.pattern.pattern_type:
                    score += (1 - analysis.pattern.confidence)
                else:
                    score += 0.5
                factors += 1
            
            return score / max(factors, 1)
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5
    
    def calculate_market_score(self, metrics: TokenMetrics) -> float:
        """Calculate market conditions score (0-1)"""
        try:
            score = 0.0
            factors = 0
            
            # Volume scoring
            if metrics.volume_24h > 0:
                # Higher volume is generally better
                volume_score = min(metrics.volume_24h / 1000000, 1.0)  # Normalize to $1M
                score += volume_score
                factors += 1
            
            # Liquidity scoring
            if metrics.liquidity > 0:
                liquidity_score = min(metrics.liquidity / 500000, 1.0)  # Normalize to $500K
                score += liquidity_score
                factors += 1
            
            # Market cap scoring (if available)
            if metrics.market_cap:
                # Prefer tokens with reasonable market cap
                if 1000000 <= metrics.market_cap <= 1000000000:  # $1M - $1B
                    score += 0.8
                elif metrics.market_cap > 1000000000:  # > $1B
                    score += 0.6
                else:  # < $1M
                    score += 0.4
                factors += 1
            
            # Price change momentum
            if abs(metrics.price_change_24h) > 0:
                # Moderate positive momentum is good
                if 2 <= metrics.price_change_24h <= 15:
                    score += 0.9
                elif -5 <= metrics.price_change_24h < 2:
                    score += 0.6
                elif metrics.price_change_24h > 15:
                    score += 0.4  # Too volatile
                else:
                    score += 0.3  # Declining
                factors += 1
            
            # Transaction activity
            if metrics.transactions_24h > 0:
                tx_score = min(metrics.transactions_24h / 1000, 1.0)  # Normalize to 1000 tx
                score += tx_score
                factors += 1
            
            return score / max(factors, 1)
            
        except Exception as e:
            logger.error(f"Error calculating market score: {e}")
            return 0.5
    
    def calculate_sentiment_score(self, metrics: TokenMetrics) -> float:
        """Calculate sentiment score based on market data (0-1)"""
        try:
            score = 0.5  # Default neutral
            
            # Buy/sell pressure (if available from Birdeye)
            if metrics.buy_pressure is not None and metrics.sell_pressure is not None:
                pressure_ratio = metrics.buy_pressure / max(metrics.sell_pressure, 1)
                if pressure_ratio > 1.5:
                    score = 0.8  # Strong buying pressure
                elif pressure_ratio > 1.0:
                    score = 0.7  # Moderate buying pressure
                elif pressure_ratio < 0.5:
                    score = 0.2  # Strong selling pressure
                elif pressure_ratio < 1.0:
                    score = 0.3  # Moderate selling pressure
            
            # Holder count growth (if available)
            if metrics.holders:
                # More holders generally indicates growing interest
                if metrics.holders > 10000:
                    score += 0.1
                elif metrics.holders > 1000:
                    score += 0.05
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return 0.5
    
    def calculate_risk_score(self, metrics: TokenMetrics, analysis: TechnicalAnalysisResult) -> float:
        """Calculate risk score (0-1, lower is better)"""
        try:
            risk_factors = 0
            total_factors = 0
            
            # Volatility risk
            if metrics.volatility > 20:  # High volatility
                risk_factors += 1
            total_factors += 1
            
            # Liquidity risk
            if metrics.liquidity < 100000:  # Low liquidity
                risk_factors += 1
            total_factors += 1
            
            # Market cap risk
            if metrics.market_cap and metrics.market_cap < 1000000:  # Very small cap
                risk_factors += 1
            total_factors += 1
            
            # Technical risk
            if analysis.risk_score > 0.7:
                risk_factors += 1
            total_factors += 1
            
            # Reversal probability risk
            if analysis.trend.reversal_probability > 0.6:
                risk_factors += 1
            total_factors += 1
            
            return risk_factors / max(total_factors, 1)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def calculate_volume_score(self, metrics: TokenMetrics) -> float:
        """Calculate volume-based score (0-1)"""
        try:
            if metrics.volume_24h <= 0:
                return 0.0
            
            # Volume relative to market cap
            if metrics.market_cap and metrics.market_cap > 0:
                volume_ratio = metrics.volume_24h / metrics.market_cap
                if volume_ratio > 0.1:  # High turnover
                    return 0.9
                elif volume_ratio > 0.05:  # Moderate turnover
                    return 0.7
                elif volume_ratio > 0.01:  # Low turnover
                    return 0.5
                else:  # Very low turnover
                    return 0.3
            
            # Absolute volume scoring
            if metrics.volume_24h > 5000000:  # > $5M
                return 0.9
            elif metrics.volume_24h > 1000000:  # > $1M
                return 0.7
            elif metrics.volume_24h > 100000:  # > $100K
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Error calculating volume score: {e}")
            return 0.5
    
    def calculate_momentum_score(self, metrics: TokenMetrics, analysis: TechnicalAnalysisResult) -> float:
        """Calculate momentum score (0-1)"""
        try:
            score = 0.0
            factors = 0
            
            # Price momentum
            if abs(metrics.price_change_24h) > 0:
                if metrics.price_change_24h > 5:
                    score += 0.8
                elif metrics.price_change_24h > 0:
                    score += 0.6
                elif metrics.price_change_24h > -5:
                    score += 0.4
                else:
                    score += 0.2
                factors += 1
            
            # Technical momentum
            if analysis.trend.momentum_score:
                score += analysis.trend.momentum_score
                factors += 1
            
            # Trend strength
            if analysis.trend.trend_strength:
                score += analysis.trend.trend_strength
                factors += 1
            
            return score / max(factors, 1)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.5
    
    async def generate_signal_score(
        self, 
        metrics: TokenMetrics, 
        analysis: TechnicalAnalysisResult
    ) -> SignalScore:
        """Generate comprehensive signal score"""
        try:
            # Calculate individual scores
            technical_score = self.calculate_technical_score(analysis)
            market_score = self.calculate_market_score(metrics)
            sentiment_score = self.calculate_sentiment_score(metrics)
            risk_score = self.calculate_risk_score(metrics, analysis)
            volume_score = self.calculate_volume_score(metrics)
            momentum_score = self.calculate_momentum_score(metrics, analysis)
            
            # Calculate weighted total score
            total_score = (
                technical_score * self.scoring_weights["technical"] +
                market_score * self.scoring_weights["market"] +
                sentiment_score * self.scoring_weights["sentiment"] +
                (1 - risk_score) * self.scoring_weights["risk"] +  # Invert risk score
                volume_score * self.scoring_weights["volume"] +
                momentum_score * self.scoring_weights["momentum"]
            )
            
            # Calculate confidence level
            score_variance = np.var([
                technical_score, market_score, sentiment_score, 
                1 - risk_score, volume_score, momentum_score
            ])
            confidence_level = max(0.1, 1 - score_variance)  # Lower variance = higher confidence
            
            # Apply historical performance adjustment
            if metrics.mint_address in self.performance_tracker:
                perf = self.performance_tracker[metrics.mint_address]
                performance_multiplier = 0.8 + (perf["success_rate"] * 0.4)  # 0.8 to 1.2 range
                total_score *= performance_multiplier
                confidence_level *= (0.5 + perf["success_rate"] * 0.5)
            
            return SignalScore(
                technical_score=technical_score,
                market_score=market_score,
                sentiment_score=sentiment_score,
                risk_score=risk_score,
                volume_score=volume_score,
                momentum_score=momentum_score,
                total_score=min(total_score, 1.0),
                confidence_level=min(confidence_level, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Error generating signal score: {e}")
            return SignalScore(
                technical_score=0.5,
                market_score=0.5,
                sentiment_score=0.5,
                risk_score=0.5,
                volume_score=0.5,
                momentum_score=0.5,
                total_score=0.5,
                confidence_level=0.5
            )
    
    def determine_signal_type(self, score: SignalScore, analysis: TechnicalAnalysisResult) -> SignalType:
        """Determine signal type based on score and analysis"""
        total_score = score.total_score
        
        if total_score >= self.signal_thresholds["strong_buy"]:
            return SignalType.STRONG_BUY
        elif total_score >= self.signal_thresholds["buy"]:
            return SignalType.BUY
        elif total_score <= self.signal_thresholds["strong_sell"]:
            return SignalType.STRONG_SELL
        elif total_score <= self.signal_thresholds["sell"]:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def determine_priority(self, signal_type: SignalType, score: SignalScore) -> SignalPriority:
        """Determine signal priority"""
        if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            if score.confidence_level > 0.8:
                return SignalPriority.CRITICAL
            else:
                return SignalPriority.HIGH
        elif signal_type in [SignalType.BUY, SignalType.SELL]:
            if score.confidence_level > 0.7:
                return SignalPriority.HIGH
            else:
                return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW
    
    def determine_distribution_channels(
        self, 
        signal_type: SignalType, 
        priority: SignalPriority,
        score: SignalScore
    ) -> List[DistributionChannel]:
        """Determine which channels should receive this signal"""
        channels = []
        
        # High-quality signals go to all channels
        if priority in [SignalPriority.CRITICAL, SignalPriority.HIGH]:
            channels.extend([
                DistributionChannel.PREMIUM_REALTIME,
                DistributionChannel.VIP_EXCLUSIVE
            ])
            
            # Only best signals go to public social
            if score.confidence_level > 0.8 and score.total_score > 0.7:
                channels.append(DistributionChannel.PUBLIC_SOCIAL)
        
        # Medium priority signals go to premium only
        elif priority == SignalPriority.MEDIUM:
            channels.append(DistributionChannel.PREMIUM_REALTIME)
        
        # All signals go to admin for monitoring
        channels.append(DistributionChannel.ADMIN_INTERNAL)
        
        return channels
    
    def generate_reasoning(
        self, 
        metrics: TokenMetrics, 
        analysis: TechnicalAnalysisResult, 
        score: SignalScore
    ) -> List[str]:
        """Generate human-readable reasoning for the signal"""
        reasoning = []
        
        # Technical reasons
        if score.technical_score > 0.7:
            if analysis.indicators.rsi and analysis.indicators.rsi < 30:
                reasoning.append("RSI indicates oversold conditions")
            if analysis.indicators.macd and analysis.indicators.macd_signal:
                if analysis.indicators.macd > analysis.indicators.macd_signal:
                    reasoning.append("MACD showing bullish momentum")
            if analysis.pattern.pattern_type and "bullish" in analysis.pattern.pattern_type:
                reasoning.append(f"Detected {analysis.pattern.pattern_type} pattern")
        
        # Market reasons
        if score.market_score > 0.7:
            if metrics.volume_24h > 1000000:
                reasoning.append(f"Strong 24h volume: ${metrics.volume_24h:,.0f}")
            if metrics.price_change_24h > 5:
                reasoning.append(f"Positive momentum: +{metrics.price_change_24h:.1f}%")
        
        # Risk factors
        if score.risk_score > 0.6:
            reasoning.append("⚠️ Higher risk due to volatility")
        
        # Volume confirmation
        if score.volume_score > 0.7:
            reasoning.append("Volume confirms price action")
        
        if not reasoning:
            reasoning.append("Mixed signals - proceed with caution")
        
        return reasoning
    
    async def generate_signal(self, mint_address: str, symbol: str) -> Optional[GeneratedSignal]:
        """Generate a trading signal for a specific token"""
        try:
            # Get market data and technical analysis
            async with market_data_service as service:
                metrics = await service.get_comprehensive_metrics(mint_address, symbol)
                
            if not metrics:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            analysis = await technical_analysis_engine.analyze_token(mint_address, symbol)
            if not analysis:
                logger.warning(f"No technical analysis available for {symbol}")
                return None
            
            # Generate signal score
            score = await self.generate_signal_score(metrics, analysis)
            
            # Skip low-quality signals
            if score.total_score < 0.3 or score.confidence_level < 0.4:
                return None
            
            # Determine signal characteristics
            signal_type = self.determine_signal_type(score, analysis)
            priority = self.determine_priority(signal_type, score)
            channels = self.determine_distribution_channels(signal_type, priority, score)
            reasoning = self.generate_reasoning(metrics, analysis, score)
            
            # Calculate targets and stop loss
            entry_price = metrics.price_usd
            target_price = None
            stop_loss = None
            
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if analysis.pattern.target_price:
                    target_price = analysis.pattern.target_price
                else:
                    target_price = entry_price * (1 + 0.05 + score.total_score * 0.15)  # 5-20% target
                
                if analysis.pattern.stop_loss:
                    stop_loss = analysis.pattern.stop_loss
                else:
                    stop_loss = entry_price * (1 - 0.03 - score.risk_score * 0.07)  # 3-10% stop loss
            
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                target_price = entry_price * (1 - 0.05 - score.total_score * 0.15)  # 5-20% down target
                stop_loss = entry_price * (1 + 0.03 + score.risk_score * 0.07)  # 3-10% stop loss
            
            # Determine timeframe and duration
            if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                timeframe = "1-4 hours"
                expected_duration = 120  # 2 hours
            elif signal_type in [SignalType.BUY, SignalType.SELL]:
                timeframe = "4-24 hours"
                expected_duration = 480  # 8 hours
            else:
                timeframe = "24+ hours"
                expected_duration = 1440  # 24 hours
            
            # Risk level
            if score.risk_score < 0.3:
                risk_level = "Low"
            elif score.risk_score < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return GeneratedSignal(
                symbol=symbol,
                mint_address=mint_address,
                signal_type=signal_type,
                signal_strength=analysis.signal_strength,
                priority=priority,
                score=score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                timeframe=timeframe,
                reasoning=reasoning,
                risk_level=risk_level,
                expected_duration=expected_duration,
                distribution_channels=channels,
                metadata={
                    "technical_analysis": asdict(analysis),
                    "market_metrics": asdict(metrics),
                    "generation_version": "1.0"
                },
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def save_signal_to_database(self, signal: GeneratedSignal) -> Optional[int]:
        """Save generated signal to database"""
        try:
            async with get_db_session() as db:
                signal_data = {
                    "symbol": signal.symbol,
                    "mint_address": signal.mint_address,
                    "signal_type": signal.signal_type.value,
                    "signal_strength": signal.signal_strength.value,
                    "entry_price": signal.entry_price,
                    "target_price": signal.target_price,
                    "stop_loss": signal.stop_loss,
                    "confidence_score": signal.score.confidence_level,
                    "risk_score": signal.score.risk_score,
                    "timeframe": signal.timeframe,
                    "reasoning": signal.reasoning,
                    "metadata": signal.metadata,
                    "generated_at": signal.generated_at,
                    "expires_at": signal.generated_at + timedelta(minutes=signal.expected_duration or 480)
                }
                
                result = await db.execute(insert(Signal).values(**signal_data))
                await db.commit()
                
                signal_id = result.inserted_primary_key[0]
                logger.info(f"Saved signal {signal_id} for {signal.symbol}")
                return signal_id
                
        except Exception as e:
            logger.error(f"Error saving signal to database: {e}")
            return None
    
    async def queue_signal_for_distribution(self, signal: GeneratedSignal, signal_id: int):
        """Queue signal for distribution to appropriate channels"""
        try:
            async with get_db_session() as db:
                for channel in signal.distribution_channels:
                    # Calculate queue priority
                    priority_map = {
                        SignalPriority.CRITICAL: 1,
                        SignalPriority.HIGH: 2,
                        SignalPriority.MEDIUM: 3,
                        SignalPriority.LOW: 4
                    }
                    
                    queue_priority = priority_map.get(signal.priority, 4)
                    
                    # Schedule for immediate distribution for high priority
                    if signal.priority in [SignalPriority.CRITICAL, SignalPriority.HIGH]:
                        scheduled_for = datetime.utcnow()
                    else:
                        # Add small delay for medium/low priority
                        scheduled_for = datetime.utcnow() + timedelta(minutes=random.randint(1, 5))
                    
                    queue_data = {
                        "signal_id": signal_id,
                        "channel": channel.value,
                        "priority": queue_priority,
                        "scheduled_for": scheduled_for,
                        "status": "pending",
                        "retry_count": 0,
                        "max_retries": 3,
                        "created_at": datetime.utcnow()
                    }
                    
                    await db.execute(insert(SignalQueue).values(**queue_data))
                
                await db.commit()
                logger.info(f"Queued signal {signal_id} for {len(signal.distribution_channels)} channels")
                
        except Exception as e:
            logger.error(f"Error queuing signal for distribution: {e}")
    
    async def process_token_batch(self, token_batch: List[Tuple[str, str]]) -> List[GeneratedSignal]:
        """Process a batch of tokens for signal generation"""
        signals = []
        
        for mint_address, symbol in token_batch:
            try:
                signal = await self.generate_signal(mint_address, symbol)
                if signal:
                    # Save to database
                    signal_id = await self.save_signal_to_database(signal)
                    if signal_id:
                        # Queue for distribution
                        await self.queue_signal_for_distribution(signal, signal_id)
                        signals.append(signal)
                        
                        logger.info(
                            f"Generated {signal.signal_type.value} signal for {symbol} "
                            f"(score: {signal.score.total_score:.2f}, "
                            f"confidence: {signal.score.confidence_level:.2f})"
                        )
                
                # Small delay to avoid overwhelming APIs
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing token {symbol}: {e}")
                continue
        
        return signals
    
    async def run_signal_generation_cycle(self):
        """Run one complete signal generation cycle"""
        try:
            logger.info("Starting signal generation cycle...")
            
            # Refresh monitored tokens periodically
            if random.random() < 0.1:  # 10% chance
                await self.load_monitored_tokens()
            
            # Process tokens in batches to avoid overwhelming APIs
            batch_size = 10
            all_signals = []
            
            for i in range(0, len(self.monitored_tokens), batch_size):
                batch = self.monitored_tokens[i:i + batch_size]
                batch_signals = await self.process_token_batch(batch)
                all_signals.extend(batch_signals)
                
                # Delay between batches
                await asyncio.sleep(2)
            
            logger.info(f"Generated {len(all_signals)} signals in this cycle")
            
            # Update performance tracking
            await self.update_performance_tracking()
            
        except Exception as e:
            logger.error(f"Error in signal generation cycle: {e}")
    
    async def update_performance_tracking(self):
        """Update performance tracking based on recent signal outcomes"""
        try:
            # This would be implemented to track actual signal performance
            # For now, we'll just log that it's running
            logger.debug("Performance tracking update completed")
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    async def start_generation_loop(self):
        """Start the continuous signal generation loop"""
        self.is_running = True
        logger.info("Starting signal generation loop...")
        
        while self.is_running:
            try:
                await self.run_signal_generation_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(settings.signal_processing_interval)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def stop_generation_loop(self):
        """Stop the signal generation loop"""
        self.is_running = False
        if self.generation_task:
            self.generation_task.cancel()
        logger.info("Signal generation loop stopped")


# Global signal generation engine instance
signal_generation_engine = SignalGenerationEngine() 