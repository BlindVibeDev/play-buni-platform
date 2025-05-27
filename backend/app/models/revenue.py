"""
Revenue and Transaction Models
Models for tracking platform revenue, fees, and transaction data
"""
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Boolean, 
    Text, DECIMAL, Index, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from app.database import Base

class Revenue(Base):
    """Revenue tracking model"""
    __tablename__ = "revenue"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(50), nullable=False)  # hourly, daily, weekly, monthly
    
    # Revenue breakdown
    blinks_fees = Column(DECIMAL(20, 8), default=0)
    trading_fees = Column(DECIMAL(20, 8), default=0)
    premium_fees = Column(DECIMAL(20, 8), default=0)
    total_revenue = Column(DECIMAL(20, 8), nullable=False)
    
    # Transaction metrics
    transaction_count = Column(Integer, default=0)
    transaction_volume = Column(DECIMAL(20, 8), default=0)
    
    # Treasury data
    treasury_balance = Column(DECIMAL(20, 8), default=0)
    
    # Additional data
    data = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_revenue_timestamp', 'timestamp'),
        Index('idx_revenue_period_type', 'period_type'),
        Index('idx_revenue_created_at', 'created_at'),
    )

class Transaction(Base):
    """Individual transaction tracking"""
    __tablename__ = "transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signature = Column(String(200), nullable=False, unique=True, index=True)
    
    # Transaction details
    transaction_type = Column(String(50), nullable=False)  # blinks, swap, fee_collection
    user_id = Column(String(200), nullable=True, index=True)
    
    # Token and amounts
    token = Column(String(100), nullable=False)
    amount = Column(DECIMAL(20, 8), nullable=False)
    platform_fee = Column(DECIMAL(20, 8), default=0)
    network_fee = Column(DECIMAL(20, 8), default=0)
    
    # Status and timing
    status = Column(String(50), default='pending')  # pending, completed, failed
    timestamp = Column(DateTime, nullable=False, index=True)
    confirmed_at = Column(DateTime, nullable=True)
    
    # Additional metadata
    metadata = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_transaction_signature', 'signature'),
        Index('idx_transaction_user_id', 'user_id'),
        Index('idx_transaction_timestamp', 'timestamp'),
        Index('idx_transaction_type', 'transaction_type'),
        Index('idx_transaction_status', 'status'),
    )

class FeeCollection(Base):
    """Fee collection tracking"""
    __tablename__ = "fee_collections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(UUID(as_uuid=True), ForeignKey('transactions.id'), nullable=False)
    
    # Fee details
    fee_type = Column(String(50), nullable=False)  # platform_fee, referral_fee, etc.
    fee_amount = Column(DECIMAL(20, 8), nullable=False)
    fee_token = Column(String(100), nullable=False)
    
    # Collection status
    collected = Column(Boolean, default=False)
    collected_at = Column(DateTime, nullable=True)
    collection_signature = Column(String(200), nullable=True)
    
    # Treasury tracking
    treasury_wallet = Column(String(200), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    transaction = relationship("Transaction", backref="fee_collections")
    
    # Indexes
    __table_args__ = (
        Index('idx_fee_collection_transaction_id', 'transaction_id'),
        Index('idx_fee_collection_type', 'fee_type'),
        Index('idx_fee_collection_collected', 'collected'),
        Index('idx_fee_collection_created_at', 'created_at'),
    ) 