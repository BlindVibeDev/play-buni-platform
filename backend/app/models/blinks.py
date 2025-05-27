"""
Blinks and Trading Models - Database models for Solana Actions/Blinks and trade tracking.

This module defines:
- Blink: Solana Actions/Blinks generation and management
- Trade: Trade execution and tracking
- Revenue: Fee collection and revenue tracking
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


# Enums for blinks and trades
class BlinkType(str, Enum):
    SWAP = "swap"
    BUY = "buy"
    SELL = "sell"
    LIMIT_ORDER = "limit_order"
    DCA = "dca"


class BlinkStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    USED = "used"
    CANCELLED = "cancelled"


class TradeStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TradeType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


# SQLAlchemy Models
class Blink(Base):
    """Solana Actions/Blinks model for trading signal integration."""
    
    __tablename__ = "blinks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    
    # Blink identification
    blink_url = Column(String(255), unique=True, nullable=False, index=True)
    short_code = Column(String(20), unique=True, nullable=False, index=True)
    
    # Blink details
    title = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    blink_type = Column(String(20), nullable=False)  # swap, buy, sell, limit_order, dca
    
    # Trading parameters
    input_token = Column(String(44), nullable=False)  # Input token mint address
    output_token = Column(String(44), nullable=False)  # Output token mint address
    input_symbol = Column(String(20), nullable=True)
    output_symbol = Column(String(20), nullable=True)
    
    # Price and amount settings
    suggested_amount = Column(Numeric(20, 8), nullable=True)
    min_amount = Column(Numeric(20, 8), nullable=True)
    max_amount = Column(Numeric(20, 8), nullable=True)
    price_impact_threshold = Column(Numeric(5, 3), default=0.05, nullable=False)  # 5% default
    
    # Fee configuration
    platform_fee_percentage = Column(Numeric(5, 4), default=0.01, nullable=False)  # 1% default
    treasury_wallet = Column(String(44), nullable=False)
    
    # Jupiter integration
    jupiter_quote_params = Column(JSON, default=dict)
    slippage_tolerance = Column(Numeric(5, 3), default=0.005, nullable=False)  # 0.5% default
    
    # Blink metadata
    image_url = Column(String(255), nullable=True)
    icon_url = Column(String(255), nullable=True)
    tags = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    
    # Status and tracking
    status = Column(String(20), default="active", nullable=False, index=True)
    is_public = Column(Boolean, default=True, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    total_volume = Column(Numeric(20, 8), default=0, nullable=False)
    total_fees_collected = Column(Numeric(20, 8), default=0, nullable=False)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    creator = relationship("User", back_populates="blinks")
    signal = relationship("Signal")
    trades = relationship("Trade", back_populates="blink")
    revenue_entries = relationship("Revenue", back_populates="blink")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_blinks_creator_created', 'creator_id', 'created_at'),
        Index('idx_blinks_status_expires', 'status', 'expires_at'),
        Index('idx_blinks_tokens', 'input_token', 'output_token'),
        Index('idx_blinks_public_active', 'is_public', 'status'),
    )
    
    def __repr__(self):
        return f"<Blink(id={self.id}, code={self.short_code}, type={self.blink_type})>"


class Trade(Base):
    """Trade execution and tracking model."""
    
    __tablename__ = "trades"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    blink_id = Column(UUID(as_uuid=True), ForeignKey("blinks.id"), nullable=True)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    
    # Trade identification
    transaction_signature = Column(String(88), unique=True, nullable=True, index=True)
    jupiter_transaction_id = Column(String(100), nullable=True)
    
    # Trade details
    trade_type = Column(String(20), nullable=False)  # market, limit, stop_loss, take_profit
    trade_status = Column(String(20), default="pending", nullable=False, index=True)
    
    # Token information
    input_token = Column(String(44), nullable=False)
    output_token = Column(String(44), nullable=False)
    input_symbol = Column(String(20), nullable=True)
    output_symbol = Column(String(20), nullable=True)
    
    # Trade amounts
    input_amount = Column(Numeric(20, 8), nullable=False)
    output_amount = Column(Numeric(20, 8), nullable=True)
    expected_output = Column(Numeric(20, 8), nullable=True)
    
    # Price information
    input_price_usd = Column(Numeric(20, 8), nullable=True)
    output_price_usd = Column(Numeric(20, 8), nullable=True)
    execution_price = Column(Numeric(20, 8), nullable=True)
    price_impact = Column(Numeric(8, 4), nullable=True)
    
    # Fee breakdown
    platform_fee = Column(Numeric(20, 8), default=0, nullable=False)
    jupiter_fee = Column(Numeric(20, 8), default=0, nullable=False)
    network_fee = Column(Numeric(20, 8), default=0, nullable=False)
    total_fees = Column(Numeric(20, 8), default=0, nullable=False)
    
    # Slippage and execution
    slippage_tolerance = Column(Numeric(5, 3), nullable=False)
    actual_slippage = Column(Numeric(8, 4), nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    submitted_at = Column(DateTime, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Metadata
    user_wallet = Column(String(44), nullable=False)
    route_info = Column(JSON, default=dict)  # Jupiter route information
    metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    blink = relationship("Blink", back_populates="trades")
    signal = relationship("Signal")
    revenue_entry = relationship("Revenue", back_populates="trade", uselist=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_trades_user_created', 'user_id', 'created_at'),
        Index('idx_trades_status_created', 'trade_status', 'created_at'),
        Index('idx_trades_tokens', 'input_token', 'output_token'),
        Index('idx_trades_blink_signal', 'blink_id', 'signal_id'),
    )
    
    def __repr__(self):
        return f"<Trade(id={self.id}, status={self.trade_status}, amount={self.input_amount})>"


class Revenue(Base):
    """Revenue tracking and fee collection model."""
    
    __tablename__ = "revenue"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=False, unique=True)
    blink_id = Column(UUID(as_uuid=True), ForeignKey("blinks.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Revenue details
    fee_amount = Column(Numeric(20, 8), nullable=False)
    fee_token = Column(String(44), nullable=False)
    fee_token_symbol = Column(String(20), nullable=True)
    fee_percentage = Column(Numeric(5, 4), nullable=False)
    
    # USD values
    fee_amount_usd = Column(Numeric(20, 8), nullable=True)
    token_price_usd = Column(Numeric(20, 8), nullable=True)
    
    # Collection details
    treasury_wallet = Column(String(44), nullable=False)
    collection_transaction = Column(String(88), nullable=True)
    collected_at = Column(DateTime, nullable=True)
    
    # Revenue categorization
    revenue_type = Column(String(20), default="trading_fee", nullable=False)  # trading_fee, subscription, premium
    revenue_source = Column(String(20), nullable=False)  # blink, direct, api
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Status
    status = Column(String(20), default="pending", nullable=False)  # pending, collected, failed
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Relationships
    trade = relationship("Trade", back_populates="revenue_entry")
    blink = relationship("Blink", back_populates="revenue_entries")
    user = relationship("User")
    
    # Indexes for revenue analysis
    __table_args__ = (
        Index('idx_revenue_created_type', 'created_at', 'revenue_type'),
        Index('idx_revenue_token_amount', 'fee_token', 'fee_amount'),
        Index('idx_revenue_user_created', 'user_id', 'created_at'),
        Index('idx_revenue_status_processed', 'status', 'processed_at'),
    )
    
    def __repr__(self):
        return f"<Revenue(id={self.id}, amount={self.fee_amount}, token={self.fee_token_symbol})>"


# Pydantic Schemas for API
class BlinkBase(BaseModel):
    """Base blink schema."""
    title: str = Field(..., max_length=100)
    description: Optional[str] = None
    blink_type: BlinkType
    input_token: str = Field(..., min_length=32, max_length=44)
    output_token: str = Field(..., min_length=32, max_length=44)
    input_symbol: Optional[str] = Field(None, max_length=20)
    output_symbol: Optional[str] = Field(None, max_length=20)
    suggested_amount: Optional[Decimal] = Field(None, gt=0)
    min_amount: Optional[Decimal] = Field(None, gt=0)
    max_amount: Optional[Decimal] = Field(None, gt=0)
    platform_fee_percentage: Decimal = Field(default=Decimal("0.01"), ge=0, le=0.1)
    slippage_tolerance: Decimal = Field(default=Decimal("0.005"), ge=0, le=0.1)


class BlinkCreate(BlinkBase):
    """Schema for creating a new blink."""
    signal_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    image_url: Optional[str] = None
    icon_url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    jupiter_quote_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BlinkUpdate(BaseModel):
    """Schema for updating blink information."""
    title: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    status: Optional[BlinkStatus] = None
    expires_at: Optional[datetime] = None
    is_public: Optional[bool] = None


class BlinkResponse(BlinkBase):
    """Schema for blink API responses."""
    id: str
    creator_id: str
    signal_id: Optional[str]
    blink_url: str
    short_code: str
    treasury_wallet: str
    status: BlinkStatus
    is_public: bool
    usage_count: int
    total_volume: Decimal
    total_fees_collected: Decimal
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    image_url: Optional[str]
    icon_url: Optional[str]
    tags: List[str]
    
    class Config:
        from_attributes = True


class TradeBase(BaseModel):
    """Base trade schema."""
    trade_type: TradeType
    input_token: str = Field(..., min_length=32, max_length=44)
    output_token: str = Field(..., min_length=32, max_length=44)
    input_amount: Decimal = Field(..., gt=0)
    slippage_tolerance: Decimal = Field(..., ge=0, le=0.1)
    user_wallet: str = Field(..., min_length=32, max_length=44)


class TradeCreate(TradeBase):
    """Schema for creating a new trade."""
    blink_id: Optional[str] = None
    signal_id: Optional[str] = None
    expected_output: Optional[Decimal] = None
    route_info: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TradeUpdate(BaseModel):
    """Schema for updating trade information."""
    trade_status: Optional[TradeStatus] = None
    transaction_signature: Optional[str] = None
    output_amount: Optional[Decimal] = None
    execution_price: Optional[Decimal] = None
    actual_slippage: Optional[Decimal] = None
    executed_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class TradeResponse(TradeBase):
    """Schema for trade API responses."""
    id: str
    user_id: str
    blink_id: Optional[str]
    signal_id: Optional[str]
    transaction_signature: Optional[str]
    trade_status: TradeStatus
    input_symbol: Optional[str]
    output_symbol: Optional[str]
    output_amount: Optional[Decimal]
    expected_output: Optional[Decimal]
    execution_price: Optional[Decimal]
    price_impact: Optional[Decimal]
    platform_fee: Decimal
    total_fees: Decimal
    actual_slippage: Optional[Decimal]
    created_at: datetime
    executed_at: Optional[datetime]
    confirmed_at: Optional[datetime]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class RevenueBase(BaseModel):
    """Base revenue schema."""
    fee_amount: Decimal = Field(..., gt=0)
    fee_token: str = Field(..., min_length=32, max_length=44)
    fee_percentage: Decimal = Field(..., ge=0, le=0.1)
    revenue_type: str = Field(default="trading_fee")
    revenue_source: str


class RevenueCreate(RevenueBase):
    """Schema for creating revenue record."""
    trade_id: str
    blink_id: Optional[str] = None
    user_id: str
    treasury_wallet: str
    fee_amount_usd: Optional[Decimal] = None
    token_price_usd: Optional[Decimal] = None


class RevenueUpdate(BaseModel):
    """Schema for updating revenue information."""
    collection_transaction: Optional[str] = None
    collected_at: Optional[datetime] = None
    status: Optional[str] = None
    processed_at: Optional[datetime] = None


class RevenueResponse(RevenueBase):
    """Schema for revenue API responses."""
    id: str
    trade_id: str
    blink_id: Optional[str]
    user_id: str
    fee_token_symbol: Optional[str]
    fee_amount_usd: Optional[Decimal]
    treasury_wallet: str
    collection_transaction: Optional[str]
    status: str
    created_at: datetime
    collected_at: Optional[datetime]
    processed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


# Utility schemas
class BlinkStats(BaseModel):
    """Blink statistics schema."""
    total_blinks: int
    active_blinks: int
    total_usage: int
    total_volume: Decimal
    total_fees: Decimal
    top_performing_blink: Optional[str]
    most_used_token_pair: Optional[str]


class TradeStats(BaseModel):
    """Trade statistics schema."""
    total_trades: int
    successful_trades: int
    failed_trades: int
    success_rate: float
    total_volume: Decimal
    total_fees: Decimal
    average_trade_size: Decimal
    most_traded_token: Optional[str]


class RevenueStats(BaseModel):
    """Revenue statistics schema."""
    total_revenue: Decimal
    revenue_today: Decimal
    revenue_this_month: Decimal
    total_trades_with_fees: int
    average_fee_per_trade: Decimal
    top_revenue_token: Optional[str]
    treasury_balance: Optional[Decimal]


class JupiterQuoteRequest(BaseModel):
    """Jupiter quote request schema."""
    input_mint: str
    output_mint: str
    amount: int
    slippage_bps: int = 50  # 0.5% default
    platform_fee_bps: int = 100  # 1% default


class JupiterSwapRequest(BaseModel):
    """Jupiter swap request schema."""
    quote_response: Dict[str, Any]
    user_public_key: str
    wrap_unwrap_sol: bool = True
    compute_unit_price_micro_lamports: Optional[int] = None 