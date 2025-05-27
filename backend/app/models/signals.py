"""
Signal Models - Database models for trading signal management.

This module defines:
- Signal: Core trading signal information
- SignalPerformance: Signal performance tracking and analytics
- SignalQueue: Signal distribution queue management
- Pydantic schemas for API requests/responses
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from decimal import Decimal
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, JSON, ForeignKey, Numeric, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..database import Base


# Enums for signal types and statuses
class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    LONG = "long"
    SHORT = "short"


class SignalStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"


class DistributionChannel(str, Enum):
    PREMIUM = "premium"
    SOCIAL = "social"
    BOTH = "both"


# SQLAlchemy Models
class Signal(Base):
    """Core trading signal model."""
    
    __tablename__ = "signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Token information
    token_address = Column(String(44), nullable=False, index=True)
    token_symbol = Column(String(20), nullable=True, index=True)
    token_name = Column(String(100), nullable=True)
    
    # Signal details
    signal_type = Column(String(20), nullable=False, index=True)  # buy, sell, hold, long, short
    strength = Column(String(20), nullable=False, index=True)  # weak, moderate, strong, very_strong
    confidence = Column(Numeric(5, 3), nullable=False)  # 0.000 to 1.000
    
    # Price information
    entry_price = Column(Numeric(20, 8), nullable=False)
    target_price = Column(Numeric(20, 8), nullable=True)
    stop_loss = Column(Numeric(20, 8), nullable=True)
    current_price = Column(Numeric(20, 8), nullable=True)
    
    # Timing and duration
    timeframe = Column(String(10), nullable=False)  # 15m, 1h, 4h, 1d
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    triggered_at = Column(DateTime, nullable=True)
    
    # Signal analysis and reasoning
    reasoning = Column(JSON, default=list)  # List of analysis points
    technical_indicators = Column(JSON, default=dict)  # Technical analysis data
    market_conditions = Column(JSON, default=dict)  # Market context
    risk_assessment = Column(JSON, default=dict)  # Risk analysis
    
    # AI/ML model information
    model_version = Column(String(20), nullable=True)
    prediction_confidence = Column(Numeric(5, 3), nullable=True)
    feature_importance = Column(JSON, default=dict)
    
    # Distribution and access
    distribution_channel = Column(String(20), nullable=False, default="premium")
    is_public = Column(Boolean, default=False, nullable=False)
    premium_only = Column(Boolean, default=True, nullable=False)
    
    # Status and tracking
    status = Column(String(20), default="active", nullable=False, index=True)
    view_count = Column(Integer, default=0, nullable=False)
    interaction_count = Column(Integer, default=0, nullable=False)
    
    # Performance tracking
    max_price_reached = Column(Numeric(20, 8), nullable=True)
    min_price_reached = Column(Numeric(20, 8), nullable=True)
    current_pnl_percentage = Column(Numeric(8, 4), nullable=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    # Relationships
    performance = relationship("SignalPerformance", back_populates="signal", uselist=False)
    interactions = relationship("SignalInteraction", back_populates="signal")
    queue_entries = relationship("SignalQueue", back_populates="signal")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_signals_token_created', 'token_address', 'created_at'),
        Index('idx_signals_type_strength', 'signal_type', 'strength'),
        Index('idx_signals_status_expires', 'status', 'expires_at'),
        Index('idx_signals_distribution', 'distribution_channel', 'is_public'),
    )
    
    def __repr__(self):
        return f"<Signal(id={self.id}, token={self.token_symbol}, type={self.signal_type})>"


class SignalPerformance(Base):
    """Signal performance tracking and analytics model."""
    
    __tablename__ = "signal_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=False, unique=True)
    
    # Performance metrics
    entry_price = Column(Numeric(20, 8), nullable=False)
    exit_price = Column(Numeric(20, 8), nullable=True)
    highest_price = Column(Numeric(20, 8), nullable=True)
    lowest_price = Column(Numeric(20, 8), nullable=True)
    
    # PnL calculations
    pnl_percentage = Column(Numeric(8, 4), nullable=True)  # Profit/Loss percentage
    pnl_absolute = Column(Numeric(20, 8), nullable=True)  # Absolute PnL
    max_drawdown = Column(Numeric(8, 4), nullable=True)  # Maximum drawdown
    max_profit = Column(Numeric(8, 4), nullable=True)  # Maximum profit reached
    
    # Timing metrics
    duration_minutes = Column(Integer, nullable=True)  # Signal duration in minutes
    time_to_target = Column(Integer, nullable=True)  # Time to reach target (minutes)
    time_to_stop_loss = Column(Integer, nullable=True)  # Time to hit stop loss (minutes)
    
    # Outcome tracking
    outcome = Column(String(20), nullable=True)  # success, failure, partial, expired
    target_reached = Column(Boolean, default=False, nullable=False)
    stop_loss_hit = Column(Boolean, default=False, nullable=False)
    
    # Market context during signal
    market_volatility = Column(Numeric(8, 4), nullable=True)
    market_trend = Column(String(20), nullable=True)  # bullish, bearish, sideways
    volume_profile = Column(JSON, default=dict)
    
    # Performance scoring
    accuracy_score = Column(Numeric(5, 3), nullable=True)  # 0.000 to 1.000
    risk_adjusted_return = Column(Numeric(8, 4), nullable=True)
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    
    # Tracking timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Analysis and insights
    performance_notes = Column(Text, nullable=True)
    lessons_learned = Column(JSON, default=list)
    improvement_suggestions = Column(JSON, default=list)
    
    # Relationships
    signal = relationship("Signal", back_populates="performance")
    
    def __repr__(self):
        return f"<SignalPerformance(signal_id={self.signal_id}, pnl={self.pnl_percentage}%)>"


class SignalQueue(Base):
    """Signal distribution queue management model."""
    
    __tablename__ = "signal_queue"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=False)
    
    # Queue management
    queue_type = Column(String(20), nullable=False)  # premium, social, webhook
    priority = Column(Integer, default=5, nullable=False)  # 1-10, higher = more priority
    scheduled_at = Column(DateTime, nullable=False)
    
    # Processing status
    status = Column(String(20), default="pending", nullable=False)  # pending, processing, sent, failed
    processed_at = Column(DateTime, nullable=True)
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    
    # Distribution details
    distribution_channel = Column(String(50), nullable=False)  # twitter, discord, websocket, webhook
    target_audience = Column(String(20), nullable=True)  # free, premium, vip, all
    message_template = Column(String(50), nullable=True)
    
    # Result tracking
    success = Column(Boolean, nullable=True)
    error_message = Column(Text, nullable=True)
    response_data = Column(JSON, default=dict)
    
    # Engagement metrics
    views = Column(Integer, default=0, nullable=False)
    clicks = Column(Integer, default=0, nullable=False)
    shares = Column(Integer, default=0, nullable=False)
    conversions = Column(Integer, default=0, nullable=False)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    signal = relationship("Signal", back_populates="queue_entries")
    
    # Indexes for queue processing
    __table_args__ = (
        Index('idx_queue_status_scheduled', 'status', 'scheduled_at'),
        Index('idx_queue_type_priority', 'queue_type', 'priority'),
        Index('idx_queue_channel_audience', 'distribution_channel', 'target_audience'),
    )
    
    def __repr__(self):
        return f"<SignalQueue(signal_id={self.signal_id}, channel={self.distribution_channel})>"


# Pydantic Schemas for API
class SignalBase(BaseModel):
    """Base signal schema."""
    token_address: str = Field(..., min_length=32, max_length=44)
    token_symbol: Optional[str] = Field(None, max_length=20)
    token_name: Optional[str] = Field(None, max_length=100)
    signal_type: SignalType
    strength: SignalStrength
    confidence: float = Field(..., ge=0.0, le=1.0)
    entry_price: Decimal = Field(..., gt=0)
    target_price: Optional[Decimal] = Field(None, gt=0)
    stop_loss: Optional[Decimal] = Field(None, gt=0)
    timeframe: str = Field(..., regex="^(15m|1h|4h|1d)$")
    reasoning: List[str] = Field(default_factory=list)


class SignalCreate(SignalBase):
    """Schema for creating a new signal."""
    expires_at: datetime
    technical_indicators: Optional[Dict[str, Any]] = Field(default_factory=dict)
    market_conditions: Optional[Dict[str, Any]] = Field(default_factory=dict)
    risk_assessment: Optional[Dict[str, Any]] = Field(default_factory=dict)
    distribution_channel: DistributionChannel = DistributionChannel.PREMIUM
    premium_only: bool = True
    tags: List[str] = Field(default_factory=list)


class SignalUpdate(BaseModel):
    """Schema for updating signal information."""
    current_price: Optional[Decimal] = Field(None, gt=0)
    status: Optional[SignalStatus] = None
    triggered_at: Optional[datetime] = None
    max_price_reached: Optional[Decimal] = None
    min_price_reached: Optional[Decimal] = None
    current_pnl_percentage: Optional[Decimal] = None


class SignalResponse(SignalBase):
    """Schema for signal API responses."""
    id: str
    created_at: datetime
    expires_at: datetime
    triggered_at: Optional[datetime]
    status: SignalStatus
    current_price: Optional[Decimal]
    distribution_channel: DistributionChannel
    is_public: bool
    premium_only: bool
    view_count: int
    interaction_count: int
    current_pnl_percentage: Optional[Decimal]
    technical_indicators: Dict[str, Any]
    market_conditions: Dict[str, Any]
    tags: List[str]
    
    class Config:
        from_attributes = True


class SignalPerformanceBase(BaseModel):
    """Base signal performance schema."""
    entry_price: Decimal = Field(..., gt=0)
    exit_price: Optional[Decimal] = Field(None, gt=0)
    pnl_percentage: Optional[Decimal] = None
    outcome: Optional[str] = None


class SignalPerformanceCreate(SignalPerformanceBase):
    """Schema for creating signal performance record."""
    signal_id: str


class SignalPerformanceUpdate(BaseModel):
    """Schema for updating signal performance."""
    exit_price: Optional[Decimal] = None
    highest_price: Optional[Decimal] = None
    lowest_price: Optional[Decimal] = None
    pnl_percentage: Optional[Decimal] = None
    outcome: Optional[str] = None
    target_reached: Optional[bool] = None
    stop_loss_hit: Optional[bool] = None
    completed_at: Optional[datetime] = None


class SignalPerformanceResponse(SignalPerformanceBase):
    """Schema for signal performance API responses."""
    id: str
    signal_id: str
    highest_price: Optional[Decimal]
    lowest_price: Optional[Decimal]
    pnl_absolute: Optional[Decimal]
    max_drawdown: Optional[Decimal]
    max_profit: Optional[Decimal]
    duration_minutes: Optional[int]
    target_reached: bool
    stop_loss_hit: bool
    accuracy_score: Optional[Decimal]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class SignalQueueBase(BaseModel):
    """Base signal queue schema."""
    queue_type: str = Field(..., regex="^(premium|social|webhook)$")
    priority: int = Field(default=5, ge=1, le=10)
    distribution_channel: str
    target_audience: Optional[str] = None


class SignalQueueCreate(SignalQueueBase):
    """Schema for creating signal queue entry."""
    signal_id: str
    scheduled_at: datetime


class SignalQueueUpdate(BaseModel):
    """Schema for updating signal queue entry."""
    status: Optional[str] = None
    processed_at: Optional[datetime] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    views: Optional[int] = None
    clicks: Optional[int] = None
    shares: Optional[int] = None
    conversions: Optional[int] = None


class SignalQueueResponse(SignalQueueBase):
    """Schema for signal queue API responses."""
    id: str
    signal_id: str
    scheduled_at: datetime
    status: str
    processed_at: Optional[datetime]
    attempts: int
    success: Optional[bool]
    error_message: Optional[str]
    views: int
    clicks: int
    shares: int
    conversions: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Utility schemas
class SignalStats(BaseModel):
    """Signal statistics schema."""
    total_signals: int
    active_signals: int
    signals_today: int
    success_rate: float
    average_pnl: float
    best_performing_token: Optional[str]
    worst_performing_token: Optional[str]


class SignalAnalytics(BaseModel):
    """Signal analytics schema."""
    period_days: int
    total_signals: int
    successful_signals: int
    failed_signals: int
    success_rate: float
    average_return: float
    total_return: float
    max_return: float
    min_return: float
    volatility: float
    sharpe_ratio: float


class TrendingToken(BaseModel):
    """Trending token schema."""
    token_address: str
    token_symbol: str
    signal_count: int
    avg_confidence: float
    latest_signal: datetime
    trend_score: float
    price_change_24h: Optional[float] = None 