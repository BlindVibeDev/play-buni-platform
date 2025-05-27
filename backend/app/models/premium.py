"""
Premium Features Data Models
Models for premium dashboard, analytics, and real-time features for NFT holders
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship

from .base import Base

class PremiumTier(str, Enum):
    """Premium tiers based on NFT holdings"""
    BASIC = "basic"           # 1 NFT
    PREMIUM = "premium"       # 5+ NFTs
    VIP = "vip"              # 10+ NFTs
    ELITE = "elite"          # 25+ NFTs

class AnalyticsType(str, Enum):
    """Types of analytics data"""
    MARKET_OVERVIEW = "market_overview"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    RISK_METRICS = "risk_metrics"
    PERFORMANCE_REPORT = "performance_report"
    SECTOR_ANALYSIS = "sector_analysis"

class AlertType(str, Enum):
    """Types of premium alerts"""
    PRICE_MOVEMENT = "price_movement"
    PORTFOLIO_CHANGE = "portfolio_change"
    MARKET_SENTIMENT = "market_sentiment"
    TECHNICAL_INDICATOR = "technical_indicator"
    NEWS_ALERT = "news_alert"

# Database Models
class PremiumUser(Base):
    """Premium user profile with NFT verification status"""
    __tablename__ = "premium_users"
    
    id = Column(String, primary_key=True, index=True)
    wallet_address = Column(String, unique=True, index=True, nullable=False)
    nft_count = Column(Integer, default=0)
    premium_tier = Column(String, default=PremiumTier.BASIC)
    verified_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    last_verification = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    
    # Premium features access
    real_time_signals = Column(Boolean, default=True)
    advanced_analytics = Column(Boolean, default=False)
    portfolio_tracking = Column(Boolean, default=False)
    custom_alerts = Column(Boolean, default=False)
    priority_support = Column(Boolean, default=False)
    
    # Usage statistics
    signals_received = Column(Integer, default=0)
    trades_executed = Column(Integer, default=0)
    total_volume = Column(Float, default=0.0)
    last_active = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    
    # Relationships
    analytics_data = relationship("PremiumAnalytics", back_populates="user")
    custom_alerts = relationship("CustomAlert", back_populates="user")
    portfolio_snapshots = relationship("PortfolioSnapshot", back_populates="user")

class PremiumAnalytics(Base):
    """Analytics data for premium users"""
    __tablename__ = "premium_analytics"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("premium_users.id"), nullable=False)
    analytics_type = Column(String, nullable=False)
    data = Column(JSON, nullable=False)
    generated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("PremiumUser", back_populates="analytics_data")

class CustomAlert(Base):
    """Custom alerts for premium users"""
    __tablename__ = "custom_alerts"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("premium_users.id"), nullable=False)
    alert_type = Column(String, nullable=False)
    token = Column(String, nullable=True)
    condition = Column(JSON, nullable=False)  # Alert conditions
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    last_triggered = Column(DateTime(timezone=True), nullable=True)
    trigger_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("PremiumUser", back_populates="custom_alerts")

class PortfolioSnapshot(Base):
    """Portfolio snapshots for premium users"""
    __tablename__ = "portfolio_snapshots"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("premium_users.id"), nullable=False)
    wallet_address = Column(String, nullable=False)
    holdings = Column(JSON, nullable=False)  # Token holdings data
    total_value_usd = Column(Float, default=0.0)
    snapshot_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    
    # Performance metrics
    change_24h = Column(Float, default=0.0)
    change_7d = Column(Float, default=0.0)
    change_30d = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("PremiumUser", back_populates="portfolio_snapshots")

class WebSocketConnection(Base):
    """Track active WebSocket connections"""
    __tablename__ = "websocket_connections"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    wallet_address = Column(String, nullable=True)
    connection_id = Column(String, unique=True, nullable=False)
    nft_verified = Column(Boolean, default=False)
    premium_tier = Column(String, default=PremiumTier.BASIC)
    connected_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    last_ping = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    rooms = Column(JSON, default=list)  # Subscribed rooms
    is_active = Column(Boolean, default=True)

# Pydantic Models for API
class PremiumUserCreate(BaseModel):
    """Create premium user"""
    wallet_address: str
    nft_count: int = 0

class PremiumUserUpdate(BaseModel):
    """Update premium user"""
    nft_count: Optional[int] = None
    premium_tier: Optional[PremiumTier] = None
    is_active: Optional[bool] = None

class PremiumUserResponse(BaseModel):
    """Premium user response"""
    id: str
    wallet_address: str
    nft_count: int
    premium_tier: PremiumTier
    verified_at: datetime
    last_verification: datetime
    is_active: bool
    
    # Features access
    real_time_signals: bool
    advanced_analytics: bool
    portfolio_tracking: bool
    custom_alerts: bool
    priority_support: bool
    
    # Usage stats
    signals_received: int
    trades_executed: int
    total_volume: float
    last_active: datetime
    
    class Config:
        from_attributes = True

class MarketAnalytics(BaseModel):
    """Market analytics data structure"""
    timestamp: datetime
    market_volatility: Dict[str, float]
    sector_performance: Dict[str, Dict[str, float]]
    momentum_indicators: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    trending_tokens: List[Dict[str, Any]]
    
class PortfolioAnalytics(BaseModel):
    """Portfolio analytics data structure"""
    timestamp: datetime
    total_value_usd: float
    holdings: Dict[str, Dict[str, Any]]
    performance: Dict[str, float]
    allocation: Dict[str, float]
    risk_score: float
    recommendations: List[str]

class RealTimeSignal(BaseModel):
    """Real-time signal structure for WebSocket"""
    signal_id: str
    token: str
    action: str
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    timestamp: datetime
    premium: bool = True
    
class MarketDataUpdate(BaseModel):
    """Market data update structure for WebSocket"""
    timestamp: datetime
    prices: Dict[str, Dict[str, Any]]
    market_stats: Dict[str, Any]
    update_interval: int
    source: str

class CustomAlertCreate(BaseModel):
    """Create custom alert"""
    alert_type: AlertType
    token: Optional[str] = None
    condition: Dict[str, Any]

class CustomAlertResponse(BaseModel):
    """Custom alert response"""
    id: str
    alert_type: AlertType
    token: Optional[str]
    condition: Dict[str, Any]
    is_active: bool
    created_at: datetime
    last_triggered: Optional[datetime]
    trigger_count: int
    
    class Config:
        from_attributes = True

class PremiumDashboardData(BaseModel):
    """Complete premium dashboard data"""
    user: PremiumUserResponse
    market_analytics: MarketAnalytics
    portfolio_analytics: Optional[PortfolioAnalytics]
    recent_signals: List[RealTimeSignal]
    active_alerts: List[CustomAlertResponse]
    connection_status: Dict[str, Any]

class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str
    data: Any
    timestamp: datetime
    room: Optional[str] = None

class ConnectionStats(BaseModel):
    """WebSocket connection statistics"""
    total_connections: int
    nft_verified_connections: int
    room_statistics: Dict[str, int]
    message_queues: int
    server_time: str

class PremiumFeatureAccess(BaseModel):
    """Premium feature access matrix"""
    tier: PremiumTier
    real_time_signals: bool
    advanced_analytics: bool
    portfolio_tracking: bool
    custom_alerts: bool
    priority_support: bool
    max_alerts: int
    signal_frequency: str
    analytics_retention_days: int

# Premium tier configuration
PREMIUM_TIER_CONFIG = {
    PremiumTier.BASIC: PremiumFeatureAccess(
        tier=PremiumTier.BASIC,
        real_time_signals=True,
        advanced_analytics=False,
        portfolio_tracking=False,
        custom_alerts=False,
        priority_support=False,
        max_alerts=0,
        signal_frequency="standard",
        analytics_retention_days=7
    ),
    PremiumTier.PREMIUM: PremiumFeatureAccess(
        tier=PremiumTier.PREMIUM,
        real_time_signals=True,
        advanced_analytics=True,
        portfolio_tracking=True,
        custom_alerts=True,
        priority_support=False,
        max_alerts=5,
        signal_frequency="high",
        analytics_retention_days=30
    ),
    PremiumTier.VIP: PremiumFeatureAccess(
        tier=PremiumTier.VIP,
        real_time_signals=True,
        advanced_analytics=True,
        portfolio_tracking=True,
        custom_alerts=True,
        priority_support=True,
        max_alerts=15,
        signal_frequency="real_time",
        analytics_retention_days=90
    ),
    PremiumTier.ELITE: PremiumFeatureAccess(
        tier=PremiumTier.ELITE,
        real_time_signals=True,
        advanced_analytics=True,
        portfolio_tracking=True,
        custom_alerts=True,
        priority_support=True,
        max_alerts=50,
        signal_frequency="instant",
        analytics_retention_days=365
    )
}

def get_premium_tier(nft_count: int) -> PremiumTier:
    """Determine premium tier based on NFT count"""
    if nft_count >= 25:
        return PremiumTier.ELITE
    elif nft_count >= 10:
        return PremiumTier.VIP
    elif nft_count >= 5:
        return PremiumTier.PREMIUM
    else:
        return PremiumTier.BASIC

def get_feature_access(tier: PremiumTier) -> PremiumFeatureAccess:
    """Get feature access for premium tier"""
    return PREMIUM_TIER_CONFIG[tier] 