"""
User Models - Database models for user management and authentication.

This module defines:
- User: Core user account information
- NFTHolder: NFT ownership tracking and verification
- UserSession: Session management and tracking
- Pydantic schemas for API requests/responses
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, validator

from ..database import Base


# SQLAlchemy Models
class User(Base):
    """Core user account model."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    wallet_address = Column(String(44), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # User preferences and settings
    preferences = Column(JSON, default=dict)
    notification_settings = Column(JSON, default=dict)
    
    # Usage tracking
    total_signals_received = Column(Integer, default=0)
    total_blinks_created = Column(Integer, default=0)
    total_trades_executed = Column(Integer, default=0)
    
    # Relationships
    nft_holder = relationship("NFTHolder", back_populates="user", uselist=False)
    sessions = relationship("UserSession", back_populates="user")
    signal_interactions = relationship("SignalInteraction", back_populates="user")
    blinks = relationship("Blink", back_populates="creator")
    trades = relationship("Trade", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, wallet={self.wallet_address[:8]}...)>"


class NFTHolder(Base):
    """NFT ownership tracking and verification model."""
    
    __tablename__ = "nft_holders"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    wallet_address = Column(String(44), nullable=False, index=True)
    
    # NFT ownership details
    nft_count = Column(Integer, default=0, nullable=False)
    verified_nfts = Column(JSON, default=list)  # List of verified NFT mint addresses
    
    # Verification tracking
    verified_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_verified = Column(DateTime, default=datetime.utcnow, nullable=False)
    verification_count = Column(Integer, default=1, nullable=False)
    
    # Access level and features
    access_level = Column(String(20), default="premium", nullable=False)  # premium, vip, whale
    premium_features = Column(JSON, default=dict)
    
    # Premium usage tracking
    signals_accessed_today = Column(Integer, default=0)
    last_signal_access = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="nft_holder")
    
    def __repr__(self):
        return f"<NFTHolder(user_id={self.user_id}, nfts={self.nft_count})>"


class UserSession(Base):
    """User session tracking and management model."""
    
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session details
    session_token = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Session metadata
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    device_info = Column(JSON, default=dict)
    
    # Session status
    is_active = Column(Boolean, default=True, nullable=False)
    logged_out_at = Column(DateTime, nullable=True)
    logout_reason = Column(String(50), nullable=True)  # manual, expired, security
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, active={self.is_active})>"


class SignalInteraction(Base):
    """User interactions with signals (views, clicks, trades)."""
    
    __tablename__ = "signal_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=False)
    
    # Interaction details
    interaction_type = Column(String(20), nullable=False)  # view, click, trade, share
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Interaction metadata
    source = Column(String(20), nullable=True)  # web, mobile, api
    metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="signal_interactions")
    signal = relationship("Signal", back_populates="interactions")
    
    def __repr__(self):
        return f"<SignalInteraction(user_id={self.user_id}, type={self.interaction_type})>"


# Pydantic Schemas for API
class UserBase(BaseModel):
    """Base user schema."""
    wallet_address: str = Field(..., min_length=32, max_length=44)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    notification_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserCreate(UserBase):
    """Schema for creating a new user."""
    pass


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None


class UserResponse(UserBase):
    """Schema for user API responses."""
    id: str
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    total_signals_received: int
    total_blinks_created: int
    total_trades_executed: int
    
    class Config:
        from_attributes = True


class NFTHolderBase(BaseModel):
    """Base NFT holder schema."""
    wallet_address: str = Field(..., min_length=32, max_length=44)
    nft_count: int = Field(ge=0)
    access_level: str = Field(default="premium")


class NFTHolderCreate(NFTHolderBase):
    """Schema for creating NFT holder record."""
    user_id: str
    verified_nfts: List[str] = Field(default_factory=list)


class NFTHolderUpdate(BaseModel):
    """Schema for updating NFT holder information."""
    nft_count: Optional[int] = Field(None, ge=0)
    verified_nfts: Optional[List[str]] = None
    access_level: Optional[str] = None
    premium_features: Optional[Dict[str, Any]] = None


class NFTHolderResponse(NFTHolderBase):
    """Schema for NFT holder API responses."""
    id: str
    user_id: str
    verified_nfts: List[str]
    verified_at: datetime
    last_verified: datetime
    verification_count: int
    premium_features: Dict[str, Any]
    signals_accessed_today: int
    last_signal_access: Optional[datetime]
    
    class Config:
        from_attributes = True


class UserSessionBase(BaseModel):
    """Base user session schema."""
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserSessionCreate(UserSessionBase):
    """Schema for creating user session."""
    user_id: str
    session_token: str
    expires_at: datetime


class UserSessionResponse(UserSessionBase):
    """Schema for user session API responses."""
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    is_active: bool
    logged_out_at: Optional[datetime]
    logout_reason: Optional[str]
    
    class Config:
        from_attributes = True


class SignalInteractionBase(BaseModel):
    """Base signal interaction schema."""
    interaction_type: str = Field(..., regex="^(view|click|trade|share)$")
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SignalInteractionCreate(SignalInteractionBase):
    """Schema for creating signal interaction."""
    user_id: str
    signal_id: str


class SignalInteractionResponse(SignalInteractionBase):
    """Schema for signal interaction API responses."""
    id: str
    user_id: str
    signal_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# Utility schemas
class UserStats(BaseModel):
    """User statistics schema."""
    total_users: int
    active_users_24h: int
    nft_holders: int
    premium_users: int
    new_users_today: int


class NFTVerificationResult(BaseModel):
    """NFT verification result schema."""
    has_access: bool
    nft_count: int
    verified_nfts: List[str]
    access_level: str
    verification_timestamp: datetime


class UserActivity(BaseModel):
    """User activity summary schema."""
    user_id: str
    signals_viewed: int
    signals_traded: int
    blinks_created: int
    last_activity: datetime
    activity_score: float 