"""
AI Signal Agent - Core signal generation and market analysis engine.

This module implements the sophisticated AI-driven signal generation system that:
1. Monitors 200+ Solana tokens continuously
2. Processes multi-dimensional market data
3. Generates high-confidence trading signals
4. Manages signal distribution and performance tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..models.signals import Signal, SignalPerformance, SignalQueue
from .market_analyzer import MarketAnalyzer
from .data_aggregator import DataAggregator
from .ml_engine import MLEngine
from .sentiment_analyzer import SentimentAnalyzer
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal confidence levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class MarketConditions:
    """Current market state assessment."""
    trend_direction: str  # bullish, bearish, sideways
    volatility_level: float  # 0-1 scale
    volume_profile: str  # high, normal, low
    sentiment_score: float  # -1 to 1
    liquidity_score: float  # 0-1 scale
    market_regime: str  # trending, ranging, volatile


@dataclass
class SignalContext:
    """Contextual information for signal generation."""
    token_address: str
    token_symbol: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    market_cap: float
    liquidity: float
    holder_count: int
    social_mentions: int
    developer_activity: float


@dataclass
class GeneratedSignal:
    """Complete signal with all metadata."""
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-1
    token_address: str
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    timeframe: str
    reasoning: List[str]
    technical_indicators: Dict[str, float]
    sentiment_factors: Dict[str, float]
    risk_metrics: Dict[str, float]
    market_conditions: MarketConditions
    timestamp: datetime
    expires_at: datetime


class AISignalAgent:
    """
    Core AI Signal Agent that orchestrates market analysis and signal generation.
    
    Architecture:
    - Multi-layered analysis combining technical, fundamental, and sentiment data
    - Machine learning models for pattern recognition and prediction
    - Risk-adjusted signal generation with confidence scoring
    - Real-time market monitoring and adaptive strategy adjustment
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.market_analyzer = MarketAnalyzer()
        self.data_aggregator = DataAggregator()
        self.ml_engine = MLEngine()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager()
        
        # Agent configuration
        self.monitored_tokens: List[str] = []
        self.analysis_interval = 60  # seconds
        self.signal_threshold = 0.7  # minimum confidence for signal generation
        self.max_signals_per_hour = 10
        self.is_running = False
        
        # Performance tracking
        self.signals_generated = 0
        self.successful_signals = 0
        self.total_pnl = 0.0
        
    async def initialize(self) -> None:
        """Initialize the signal agent and load configuration."""
        logger.info("Initializing AI Signal Agent...")
        
        # Load monitored tokens from database or configuration
        await self._load_monitored_tokens()
        
        # Initialize ML models
        await self.ml_engine.load_models()
        
        # Initialize data connections
        await self.data_aggregator.initialize()
        
        # Load historical performance data
        await self._load_performance_metrics()
        
        logger.info(f"Signal Agent initialized with {len(self.monitored_tokens)} tokens")
    
    async def start_monitoring(self) -> None:
        """Start the continuous market monitoring and signal generation loop."""
        if self.is_running:
            logger.warning("Signal agent is already running")
            return
            
        self.is_running = True
        logger.info("Starting AI Signal Agent monitoring...")
        
        try:
            while self.is_running:
                await self._monitoring_cycle()
                await asyncio.sleep(self.analysis_interval)
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.is_running = False
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop gracefully."""
        logger.info("Stopping AI Signal Agent...")
        self.is_running = False
    
    async def _monitoring_cycle(self) -> None:
        """Execute one complete monitoring and analysis cycle."""
        cycle_start = datetime.utcnow()
        
        try:
            # 1. Collect market data for all monitored tokens
            market_data = await self.data_aggregator.collect_market_data(
                self.monitored_tokens
            )
            
            # 2. Analyze market conditions
            market_conditions = await self._analyze_market_conditions(market_data)
            
            # 3. Generate signals for qualifying tokens
            signals = await self._generate_signals(market_data, market_conditions)
            
            # 4. Filter and rank signals by confidence
            filtered_signals = await self._filter_and_rank_signals(signals)
            
            # 5. Store signals and queue for distribution
            await self._store_and_queue_signals(filtered_signals)
            
            # 6. Update performance metrics
            await self._update_performance_metrics()
            
            cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
            logger.info(
                f"Monitoring cycle completed in {cycle_duration:.2f}s. "
                f"Generated {len(filtered_signals)} signals."
            )
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    async def _analyze_market_conditions(self, market_data: Dict) -> MarketConditions:
        """Analyze overall market conditions to inform signal generation."""
        
        # Aggregate market metrics
        total_volume = sum(token['volume_24h'] for token in market_data.values())
        avg_price_change = np.mean([token['price_change_24h'] for token in market_data.values()])
        volatility_scores = [token.get('volatility', 0) for token in market_data.values()]
        
        # Determine trend direction
        if avg_price_change > 2.0:
            trend_direction = "bullish"
        elif avg_price_change < -2.0:
            trend_direction = "bearish"
        else:
            trend_direction = "sideways"
        
        # Calculate volatility level
        volatility_level = np.mean(volatility_scores) if volatility_scores else 0.5
        
        # Assess volume profile
        if total_volume > settings.HIGH_VOLUME_THRESHOLD:
            volume_profile = "high"
        elif total_volume < settings.LOW_VOLUME_THRESHOLD:
            volume_profile = "low"
        else:
            volume_profile = "normal"
        
        # Get sentiment score
        sentiment_score = await self.sentiment_analyzer.get_market_sentiment()
        
        # Calculate liquidity score
        liquidity_scores = [token.get('liquidity_score', 0.5) for token in market_data.values()]
        liquidity_score = np.mean(liquidity_scores)
        
        # Determine market regime
        if volatility_level > 0.7:
            market_regime = "volatile"
        elif abs(avg_price_change) > 1.0:
            market_regime = "trending"
        else:
            market_regime = "ranging"
        
        return MarketConditions(
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            volume_profile=volume_profile,
            sentiment_score=sentiment_score,
            liquidity_score=liquidity_score,
            market_regime=market_regime
        )
    
    async def _generate_signals(
        self, 
        market_data: Dict, 
        market_conditions: MarketConditions
    ) -> List[GeneratedSignal]:
        """Generate trading signals for all monitored tokens."""
        signals = []
        
        for token_address, token_data in market_data.items():
            try:
                signal = await self._analyze_token_for_signal(
                    token_address, token_data, market_conditions
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing token {token_address}: {e}")
        
        return signals
    
    async def _analyze_token_for_signal(
        self,
        token_address: str,
        token_data: Dict,
        market_conditions: MarketConditions
    ) -> Optional[GeneratedSignal]:
        """Perform comprehensive analysis on a single token to generate signal."""
        
        # Create signal context
        context = SignalContext(
            token_address=token_address,
            token_symbol=token_data.get('symbol', 'UNKNOWN'),
            current_price=token_data['price'],
            price_change_24h=token_data['price_change_24h'],
            volume_24h=token_data['volume_24h'],
            market_cap=token_data.get('market_cap', 0),
            liquidity=token_data.get('liquidity', 0),
            holder_count=token_data.get('holder_count', 0),
            social_mentions=token_data.get('social_mentions', 0),
            developer_activity=token_data.get('developer_activity', 0)
        )
        
        # 1. Technical Analysis
        technical_indicators = await self.market_analyzer.calculate_technical_indicators(
            token_address, token_data
        )
        
        # 2. ML Model Predictions
        ml_predictions = await self.ml_engine.predict_price_movement(
            token_address, token_data, technical_indicators
        )
        
        # 3. Sentiment Analysis
        sentiment_factors = await self.sentiment_analyzer.analyze_token_sentiment(
            token_address, context.token_symbol
        )
        
        # 4. Risk Assessment
        risk_metrics = await self.risk_manager.assess_token_risk(
            token_address, token_data, market_conditions
        )
        
        # 5. Signal Generation Logic
        signal_decision = await self._make_signal_decision(
            context, technical_indicators, ml_predictions, 
            sentiment_factors, risk_metrics, market_conditions
        )
        
        if signal_decision['generate_signal']:
            return await self._create_signal(
                context, signal_decision, technical_indicators,
                sentiment_factors, risk_metrics, market_conditions
            )
        
        return None
    
    async def _make_signal_decision(
        self,
        context: SignalContext,
        technical_indicators: Dict[str, float],
        ml_predictions: Dict[str, float],
        sentiment_factors: Dict[str, float],
        risk_metrics: Dict[str, float],
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """
        Core signal decision logic combining all analysis factors.
        
        Decision Framework:
        1. Technical confluence (multiple indicators alignment)
        2. ML model confidence and prediction strength
        3. Sentiment momentum and social validation
        4. Risk-adjusted opportunity assessment
        5. Market condition compatibility
        """
        
        # Technical Analysis Score (0-1)
        technical_score = self._calculate_technical_score(technical_indicators)
        
        # ML Prediction Score (0-1)
        ml_score = ml_predictions.get('confidence', 0) * ml_predictions.get('strength', 0)
        
        # Sentiment Score (-1 to 1, normalized to 0-1)
        sentiment_score = (sentiment_factors.get('composite_score', 0) + 1) / 2
        
        # Risk Score (0-1, inverted so lower risk = higher score)
        risk_score = 1 - risk_metrics.get('total_risk', 1)
        
        # Market Condition Compatibility (0-1)
        market_compatibility = self._assess_market_compatibility(
            ml_predictions.get('direction', 'hold'), market_conditions
        )
        
        # Weighted composite score
        weights = {
            'technical': 0.25,
            'ml': 0.30,
            'sentiment': 0.20,
            'risk': 0.15,
            'market': 0.10
        }
        
        composite_score = (
            technical_score * weights['technical'] +
            ml_score * weights['ml'] +
            sentiment_score * weights['sentiment'] +
            risk_score * weights['risk'] +
            market_compatibility * weights['market']
        )
        
        # Signal generation decision
        generate_signal = composite_score >= self.signal_threshold
        
        # Determine signal type and strength
        signal_type = self._determine_signal_type(
            ml_predictions.get('direction', 'hold'),
            composite_score,
            technical_indicators
        )
        
        signal_strength = self._determine_signal_strength(composite_score)
        
        return {
            'generate_signal': generate_signal,
            'composite_score': composite_score,
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'component_scores': {
                'technical': technical_score,
                'ml': ml_score,
                'sentiment': sentiment_score,
                'risk': risk_score,
                'market': market_compatibility
            }
        }
    
    def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
        """Calculate technical analysis confluence score."""
        scores = []
        
        # RSI analysis
        rsi = indicators.get('rsi', 50)
        if rsi < 30:  # Oversold
            scores.append(0.8)
        elif rsi > 70:  # Overbought
            scores.append(0.2)
        else:
            scores.append(0.5)
        
        # MACD analysis
        macd_signal = indicators.get('macd_signal', 0)
        if macd_signal > 0:
            scores.append(0.7)
        elif macd_signal < 0:
            scores.append(0.3)
        else:
            scores.append(0.5)
        
        # Bollinger Bands
        bb_position = indicators.get('bb_position', 0.5)
        scores.append(bb_position)
        
        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            scores.append(0.8)
        elif volume_ratio < 0.5:
            scores.append(0.3)
        else:
            scores.append(0.5)
        
        return np.mean(scores)
    
    def _assess_market_compatibility(self, direction: str, conditions: MarketConditions) -> float:
        """Assess how well the signal direction fits current market conditions."""
        if direction == 'buy':
            if conditions.trend_direction == 'bullish':
                return 0.9
            elif conditions.trend_direction == 'sideways':
                return 0.6
            else:
                return 0.3
        elif direction == 'sell':
            if conditions.trend_direction == 'bearish':
                return 0.9
            elif conditions.trend_direction == 'sideways':
                return 0.6
            else:
                return 0.3
        else:
            return 0.5
    
    def _determine_signal_type(
        self, direction: str, score: float, indicators: Dict[str, float]
    ) -> SignalType:
        """Determine the specific signal type based on analysis."""
        if direction == 'buy':
            if score > 0.85:
                return SignalType.STRONG_BUY
            else:
                return SignalType.BUY
        elif direction == 'sell':
            if score > 0.85:
                return SignalType.STRONG_SELL
            else:
                return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _determine_signal_strength(self, score: float) -> SignalStrength:
        """Determine signal strength based on composite score."""
        if score >= 0.9:
            return SignalStrength.VERY_STRONG
        elif score >= 0.8:
            return SignalStrength.STRONG
        elif score >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    async def _create_signal(
        self,
        context: SignalContext,
        decision: Dict[str, Any],
        technical_indicators: Dict[str, float],
        sentiment_factors: Dict[str, float],
        risk_metrics: Dict[str, float],
        market_conditions: MarketConditions
    ) -> GeneratedSignal:
        """Create a complete signal object with all metadata."""
        
        # Calculate target and stop loss prices
        target_price, stop_loss = await self._calculate_price_targets(
            context.current_price,
            decision['signal_type'],
            risk_metrics,
            technical_indicators
        )
        
        # Generate reasoning
        reasoning = self._generate_signal_reasoning(
            decision, technical_indicators, sentiment_factors, market_conditions
        )
        
        # Determine timeframe
        timeframe = self._determine_timeframe(
            decision['signal_strength'], market_conditions.volatility_level
        )
        
        # Calculate expiration
        expires_at = datetime.utcnow() + timedelta(hours=self._get_expiration_hours(timeframe))
        
        return GeneratedSignal(
            signal_type=decision['signal_type'],
            strength=decision['signal_strength'],
            confidence=decision['composite_score'],
            token_address=context.token_address,
            entry_price=context.current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe=timeframe,
            reasoning=reasoning,
            technical_indicators=technical_indicators,
            sentiment_factors=sentiment_factors,
            risk_metrics=risk_metrics,
            market_conditions=market_conditions,
            timestamp=datetime.utcnow(),
            expires_at=expires_at
        )
    
    async def _calculate_price_targets(
        self,
        current_price: float,
        signal_type: SignalType,
        risk_metrics: Dict[str, float],
        technical_indicators: Dict[str, float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target price and stop loss based on risk management rules."""
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            # Calculate upside target
            resistance = technical_indicators.get('resistance', current_price * 1.05)
            target_price = min(resistance, current_price * 1.08)  # Max 8% target
            
            # Calculate stop loss
            support = technical_indicators.get('support', current_price * 0.97)
            stop_loss = max(support, current_price * 0.95)  # Max 5% loss
            
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            # Calculate downside target
            support = technical_indicators.get('support', current_price * 0.95)
            target_price = max(support, current_price * 0.92)  # Max 8% target
            
            # Calculate stop loss
            resistance = technical_indicators.get('resistance', current_price * 1.03)
            stop_loss = min(resistance, current_price * 1.05)  # Max 5% loss
            
        else:
            target_price = None
            stop_loss = None
        
        return target_price, stop_loss
    
    def _generate_signal_reasoning(
        self,
        decision: Dict[str, Any],
        technical_indicators: Dict[str, float],
        sentiment_factors: Dict[str, float],
        market_conditions: MarketConditions
    ) -> List[str]:
        """Generate human-readable reasoning for the signal."""
        reasoning = []
        
        # Technical factors
        if decision['component_scores']['technical'] > 0.7:
            reasoning.append("Strong technical confluence detected")
        
        # ML factors
        if decision['component_scores']['ml'] > 0.7:
            reasoning.append("AI models show high confidence prediction")
        
        # Sentiment factors
        if decision['component_scores']['sentiment'] > 0.7:
            reasoning.append("Positive sentiment momentum")
        elif decision['component_scores']['sentiment'] < 0.3:
            reasoning.append("Negative sentiment pressure")
        
        # Market conditions
        if market_conditions.trend_direction == 'bullish':
            reasoning.append("Favorable market trend")
        elif market_conditions.volume_profile == 'high':
            reasoning.append("High volume confirmation")
        
        # Risk factors
        if decision['component_scores']['risk'] > 0.7:
            reasoning.append("Low risk environment")
        
        return reasoning
    
    def _determine_timeframe(self, strength: SignalStrength, volatility: float) -> str:
        """Determine appropriate timeframe for the signal."""
        if strength == SignalStrength.VERY_STRONG:
            return "4h" if volatility > 0.7 else "1d"
        elif strength == SignalStrength.STRONG:
            return "1h" if volatility > 0.7 else "4h"
        else:
            return "15m" if volatility > 0.7 else "1h"
    
    def _get_expiration_hours(self, timeframe: str) -> int:
        """Get expiration hours based on timeframe."""
        timeframe_hours = {
            "15m": 1,
            "1h": 4,
            "4h": 12,
            "1d": 24
        }
        return timeframe_hours.get(timeframe, 4)
    
    async def _filter_and_rank_signals(self, signals: List[GeneratedSignal]) -> List[GeneratedSignal]:
        """Filter and rank signals by confidence and quality."""
        
        # Filter by minimum confidence
        filtered = [s for s in signals if s.confidence >= self.signal_threshold]
        
        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to max signals per hour
        return filtered[:self.max_signals_per_hour]
    
    async def _store_and_queue_signals(self, signals: List[GeneratedSignal]) -> None:
        """Store signals in database and queue for distribution."""
        for signal in signals:
            try:
                # Store in database
                db_signal = Signal(
                    token_address=signal.token_address,
                    signal_type=signal.signal_type.value,
                    strength=signal.strength.value,
                    confidence=signal.confidence,
                    entry_price=signal.entry_price,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    timeframe=signal.timeframe,
                    reasoning=signal.reasoning,
                    technical_indicators=signal.technical_indicators,
                    sentiment_factors=signal.sentiment_factors,
                    risk_metrics=signal.risk_metrics,
                    market_conditions=asdict(signal.market_conditions),
                    expires_at=signal.expires_at
                )
                
                self.db.add(db_signal)
                
                # Queue for distribution
                queue_entry = SignalQueue(
                    signal_id=db_signal.id,
                    priority=self._calculate_priority(signal),
                    distribution_channels=['premium', 'social'] if signal.confidence > 0.8 else ['social']
                )
                
                self.db.add(queue_entry)
                
                self.signals_generated += 1
                
            except Exception as e:
                logger.error(f"Error storing signal: {e}")
        
        await self.db.commit()
    
    def _calculate_priority(self, signal: GeneratedSignal) -> int:
        """Calculate distribution priority for signal."""
        if signal.strength == SignalStrength.VERY_STRONG:
            return 1
        elif signal.strength == SignalStrength.STRONG:
            return 2
        elif signal.strength == SignalStrength.MODERATE:
            return 3
        else:
            return 4
    
    async def _load_monitored_tokens(self) -> None:
        """Load the list of tokens to monitor from configuration or database."""
        # This would typically load from database or configuration
        # For now, using a sample list
        self.monitored_tokens = [
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
            # Add more token addresses...
        ]
    
    async def _load_performance_metrics(self) -> None:
        """Load historical performance metrics."""
        # Load from database
        pass
    
    async def _update_performance_metrics(self) -> None:
        """Update agent performance metrics."""
        # Calculate and update performance metrics
        pass
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "is_running": self.is_running,
            "monitored_tokens": len(self.monitored_tokens),
            "signals_generated": self.signals_generated,
            "success_rate": self.successful_signals / max(self.signals_generated, 1),
            "total_pnl": self.total_pnl,
            "last_cycle": datetime.utcnow().isoformat()
        } 