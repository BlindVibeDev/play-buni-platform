"""
Analytics and Social Models - Database models for platform analytics and social integration.

This module defines:
- PlatformAnalytics: Platform-wide analytics and metrics
- SocialPost: Social media post tracking
- UserFeedback: User feedback and rating system
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


# Enums for analytics and social
class AnalyticsType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    REAL_TIME = "real_time"


class SocialPlatform(str, Enum):
    TWITTER = "twitter"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    REDDIT = "reddit"


class PostStatus(str, Enum):
    SCHEDULED = "scheduled"
    POSTED = "posted"
    FAILED = "failed"
    DELETED = "deleted"


class FeedbackType(str, Enum):
    SIGNAL_RATING = "signal_rating"
    PLATFORM_FEEDBACK = "platform_feedback"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"


# SQLAlchemy Models
class PlatformAnalytics(Base):
    """Platform-wide analytics and metrics tracking model."""
    
    __tablename__ = "platform_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time period
    date = Column(DateTime, nullable=False, index=True)
    analytics_type = Column(String(20), nullable=False, index=True)  # daily, weekly, monthly, real_time
    
    # User metrics
    total_users = Column(Integer, default=0, nullable=False)
    active_users = Column(Integer, default=0, nullable=False)
    new_users = Column(Integer, default=0, nullable=False)
    nft_holders = Column(Integer, default=0, nullable=False)
    premium_users = Column(Integer, default=0, nullable=False)
    
    # Signal metrics
    signals_generated = Column(Integer, default=0, nullable=False)
    signals_distributed = Column(Integer, default=0, nullable=False)
    signals_viewed = Column(Integer, default=0, nullable=False)
    signals_acted_upon = Column(Integer, default=0, nullable=False)
    average_signal_confidence = Column(Numeric(5, 3), nullable=True)
    
    # Trading metrics
    total_trades = Column(Integer, default=0, nullable=False)
    successful_trades = Column(Integer, default=0, nullable=False)
    total_volume_usd = Column(Numeric(20, 8), default=0, nullable=False)
    total_fees_collected = Column(Numeric(20, 8), default=0, nullable=False)
    average_trade_size = Column(Numeric(20, 8), nullable=True)
    
    # Blink metrics
    blinks_created = Column(Integer, default=0, nullable=False)
    blinks_used = Column(Integer, default=0, nullable=False)
    blink_conversion_rate = Column(Numeric(5, 3), nullable=True)
    
    # Social metrics
    social_posts = Column(Integer, default=0, nullable=False)
    social_engagement = Column(Integer, default=0, nullable=False)
    social_reach = Column(Integer, default=0, nullable=False)
    social_clicks = Column(Integer, default=0, nullable=False)
    
    # Performance metrics
    signal_success_rate = Column(Numeric(5, 3), nullable=True)
    average_pnl = Column(Numeric(8, 4), nullable=True)
    user_retention_rate = Column(Numeric(5, 3), nullable=True)
    platform_uptime = Column(Numeric(5, 3), nullable=True)
    
    # Revenue metrics
    revenue_total = Column(Numeric(20, 8), default=0, nullable=False)
    revenue_trading_fees = Column(Numeric(20, 8), default=0, nullable=False)
    revenue_premium = Column(Numeric(20, 8), default=0, nullable=False)
    
    # Engagement metrics
    page_views = Column(Integer, default=0, nullable=False)
    session_duration_avg = Column(Integer, nullable=True)  # in seconds
    bounce_rate = Column(Numeric(5, 3), nullable=True)
    
    # Token metrics
    top_tokens = Column(JSON, default=list)  # List of top performing tokens
    token_distribution = Column(JSON, default=dict)  # Token usage distribution
    
    # Metadata
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes for analytics queries
    __table_args__ = (
        Index('idx_analytics_date_type', 'date', 'analytics_type'),
        Index('idx_analytics_type_created', 'analytics_type', 'created_at'),
    )
    
    def __repr__(self):
        return f"<PlatformAnalytics(date={self.date}, type={self.analytics_type})>"


class SocialPost(Base):
    """Social media post tracking and management model."""
    
    __tablename__ = "social_posts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    
    # Platform details
    platform = Column(String(20), nullable=False, index=True)  # twitter, discord, telegram, reddit
    platform_post_id = Column(String(100), nullable=True, index=True)
    platform_url = Column(String(255), nullable=True)
    
    # Post content
    content = Column(Text, nullable=False)
    media_urls = Column(JSON, default=list)  # Images, videos, etc.
    hashtags = Column(JSON, default=list)
    mentions = Column(JSON, default=list)
    
    # Scheduling and timing
    scheduled_at = Column(DateTime, nullable=True)
    posted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Status and tracking
    status = Column(String(20), default="scheduled", nullable=False, index=True)
    post_type = Column(String(20), nullable=False)  # signal, announcement, engagement, educational
    
    # Engagement metrics
    views = Column(Integer, default=0, nullable=False)
    likes = Column(Integer, default=0, nullable=False)
    shares = Column(Integer, default=0, nullable=False)
    comments = Column(Integer, default=0, nullable=False)
    clicks = Column(Integer, default=0, nullable=False)
    
    # Performance tracking
    engagement_rate = Column(Numeric(5, 3), nullable=True)
    reach = Column(Integer, default=0, nullable=False)
    impressions = Column(Integer, default=0, nullable=False)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Relationships
    signal = relationship("Signal")
    
    # Indexes for social media management
    __table_args__ = (
        Index('idx_social_platform_status', 'platform', 'status'),
        Index('idx_social_scheduled_posted', 'scheduled_at', 'posted_at'),
        Index('idx_social_signal_platform', 'signal_id', 'platform'),
    )
    
    def __repr__(self):
        return f"<SocialPost(id={self.id}, platform={self.platform}, status={self.status})>"


class UserFeedback(Base):
    """User feedback and rating system model."""
    
    __tablename__ = "user_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    
    # Feedback details
    feedback_type = Column(String(20), nullable=False, index=True)  # signal_rating, platform_feedback, feature_request, bug_report
    rating = Column(Integer, nullable=True)  # 1-5 star rating
    title = Column(String(200), nullable=True)
    content = Column(Text, nullable=False)
    
    # Categorization
    category = Column(String(50), nullable=True)  # accuracy, speed, ui_ux, features, etc.
    priority = Column(String(20), default="medium", nullable=False)  # low, medium, high, critical
    tags = Column(JSON, default=list)
    
    # Status tracking
    status = Column(String(20), default="open", nullable=False)  # open, in_progress, resolved, closed
    admin_response = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Metadata
    user_agent = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    device_info = Column(JSON, default=dict)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    signal = relationship("Signal")
    
    # Indexes for feedback management
    __table_args__ = (
        Index('idx_feedback_user_type', 'user_id', 'feedback_type'),
        Index('idx_feedback_status_priority', 'status', 'priority'),
        Index('idx_feedback_signal_rating', 'signal_id', 'rating'),
        Index('idx_feedback_created_type', 'created_at', 'feedback_type'),
    )
    
    def __repr__(self):
        return f"<UserFeedback(id={self.id}, type={self.feedback_type}, rating={self.rating})>"


class SystemEvent(Base):
    """System events and audit log model."""
    
    __tablename__ = "system_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # user_login, signal_generated, trade_executed, etc.
    event_category = Column(String(20), nullable=False, index=True)  # auth, trading, signals, admin, system
    severity = Column(String(20), default="info", nullable=False)  # debug, info, warning, error, critical
    
    # Event data
    description = Column(Text, nullable=False)
    details = Column(JSON, default=dict)
    
    # Context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes for event querying
    __table_args__ = (
        Index('idx_events_type_created', 'event_type', 'created_at'),
        Index('idx_events_category_severity', 'event_category', 'severity'),
        Index('idx_events_user_created', 'user_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SystemEvent(id={self.id}, type={self.event_type}, severity={self.severity})>"


# Pydantic Schemas for API
class PlatformAnalyticsBase(BaseModel):
    """Base platform analytics schema."""
    date: datetime
    analytics_type: AnalyticsType
    total_users: int = 0
    active_users: int = 0
    new_users: int = 0
    signals_generated: int = 0
    total_trades: int = 0
    total_volume_usd: Decimal = Decimal("0")


class PlatformAnalyticsCreate(PlatformAnalyticsBase):
    """Schema for creating platform analytics record."""
    pass


class PlatformAnalyticsResponse(PlatformAnalyticsBase):
    """Schema for platform analytics API responses."""
    id: str
    nft_holders: int
    premium_users: int
    signals_distributed: int
    signals_viewed: int
    signals_acted_upon: int
    successful_trades: int
    total_fees_collected: Decimal
    blinks_created: int
    blinks_used: int
    social_posts: int
    social_engagement: int
    signal_success_rate: Optional[Decimal]
    average_pnl: Optional[Decimal]
    revenue_total: Decimal
    top_tokens: List[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class SocialPostBase(BaseModel):
    """Base social post schema."""
    platform: SocialPlatform
    content: str = Field(..., max_length=2000)
    post_type: str = Field(..., regex="^(signal|announcement|engagement|educational)$")
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)


class SocialPostCreate(SocialPostBase):
    """Schema for creating social post."""
    signal_id: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    media_urls: List[str] = Field(default_factory=list)


class SocialPostUpdate(BaseModel):
    """Schema for updating social post."""
    status: Optional[PostStatus] = None
    platform_post_id: Optional[str] = None
    platform_url: Optional[str] = None
    posted_at: Optional[datetime] = None
    views: Optional[int] = None
    likes: Optional[int] = None
    shares: Optional[int] = None
    comments: Optional[int] = None
    clicks: Optional[int] = None
    error_message: Optional[str] = None


class SocialPostResponse(SocialPostBase):
    """Schema for social post API responses."""
    id: str
    signal_id: Optional[str]
    platform_post_id: Optional[str]
    platform_url: Optional[str]
    scheduled_at: Optional[datetime]
    posted_at: Optional[datetime]
    status: PostStatus
    views: int
    likes: int
    shares: int
    comments: int
    clicks: int
    engagement_rate: Optional[Decimal]
    reach: int
    impressions: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserFeedbackBase(BaseModel):
    """Base user feedback schema."""
    feedback_type: FeedbackType
    title: Optional[str] = Field(None, max_length=200)
    content: str = Field(..., max_length=5000)
    rating: Optional[int] = Field(None, ge=1, le=5)
    category: Optional[str] = None
    priority: str = Field(default="medium", regex="^(low|medium|high|critical)$")


class UserFeedbackCreate(UserFeedbackBase):
    """Schema for creating user feedback."""
    signal_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class UserFeedbackUpdate(BaseModel):
    """Schema for updating user feedback."""
    status: Optional[str] = Field(None, regex="^(open|in_progress|resolved|closed)$")
    admin_response: Optional[str] = None
    resolved_at: Optional[datetime] = None


class UserFeedbackResponse(UserFeedbackBase):
    """Schema for user feedback API responses."""
    id: str
    user_id: str
    signal_id: Optional[str]
    status: str
    admin_response: Optional[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class SystemEventBase(BaseModel):
    """Base system event schema."""
    event_type: str
    event_category: str = Field(..., regex="^(auth|trading|signals|admin|system)$")
    severity: str = Field(default="info", regex="^(debug|info|warning|error|critical)$")
    description: str


class SystemEventCreate(SystemEventBase):
    """Schema for creating system event."""
    user_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


class SystemEventResponse(SystemEventBase):
    """Schema for system event API responses."""
    id: str
    user_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Utility schemas
class AnalyticsSummary(BaseModel):
    """Analytics summary schema."""
    period: str
    total_users: int
    active_users: int
    signals_generated: int
    trades_executed: int
    revenue_generated: Decimal
    success_rate: float
    growth_rate: float


class SocialEngagement(BaseModel):
    """Social engagement metrics schema."""
    platform: str
    total_posts: int
    total_engagement: int
    average_engagement_rate: float
    top_performing_post: Optional[str]
    reach: int
    impressions: int


class FeedbackSummary(BaseModel):
    """Feedback summary schema."""
    total_feedback: int
    average_rating: float
    feedback_by_type: Dict[str, int]
    open_issues: int
    resolved_issues: int
    response_time_avg: float  # in hours


class UserEngagementMetrics(BaseModel):
    """User engagement metrics schema."""
    daily_active_users: int
    weekly_active_users: int
    monthly_active_users: int
    session_duration_avg: int  # in seconds
    bounce_rate: float
    retention_rate: float
    churn_rate: float 