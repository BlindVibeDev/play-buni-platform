"""
Solana Blinks API Router for Play Buni Platform

This router handles Solana Actions/Blinks generation, execution, and management.
Enables users to execute trades directly from signals with embedded fee collection.

Features:
- Blink generation from trading signals
- Solana Actions execution with Jupiter integration
- Fee collection and treasury management
- Transaction monitoring and tracking
- Blink sharing and social distribution
"""

import json
import base64
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.core.access_control import (
    get_current_user,
    get_premium_user,
    get_admin_user,
    require_permission,
    PermissionType
)
from app.models.signals import Signal
from app.models.blinks import Blink, Trade
from app.models.users import User
from app.services.solana_actions import solana_actions_service, SolanaAction, ActionType, BlinkStatus
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Pydantic models for API requests/responses
class CreateBlinkRequest(BaseModel):
    """Request model for creating a Blink"""
    signal_id: int
    action_type: str = Field(default="swap", description="Type of action (swap, buy, sell)")
    input_token: str = Field(default="USDC", description="Input token symbol")
    suggested_amounts: Optional[List[float]] = Field(default=None, description="Suggested amounts for quick actions")
    max_slippage: float = Field(default=1.0, description="Maximum slippage percentage", ge=0.1, le=10.0)


class ExecuteActionRequest(BaseModel):
    """Request model for executing a Solana Action"""
    account: str = Field(description="User wallet address")
    amount: Optional[float] = Field(default=None, description="Trade amount")
    slippage: Optional[float] = Field(default=0.5, description="Slippage tolerance", ge=0.1, le=10.0)
    priority_fee: Optional[int] = Field(default=0, description="Priority fee in lamports")


class BlinkResponse(BaseModel):
    """Response model for Blinks"""
    id: int
    signal_id: int
    blink_id: str
    title: str
    description: str
    icon_url: str
    action_url: str
    blink_url: str
    status: str
    created_at: datetime
    expires_at: Optional[datetime]
    action_data: Optional[Dict[str, Any]] = None


class TradeResponse(BaseModel):
    """Response model for trades"""
    id: int
    user_wallet: str
    signal_id: Optional[int]
    blink_id: Optional[str]
    transaction_type: str
    input_token: str
    output_token: str
    input_amount: float
    output_amount: Optional[float]
    fee_amount: float
    fee_percentage: float
    status: str
    transaction_signature: Optional[str]
    created_at: datetime


class TreasuryResponse(BaseModel):
    """Response model for treasury information"""
    total_fees_collected: float
    sol_balance: float
    usdc_balance: float
    total_trades: int
    total_volume: float
    last_updated: datetime


@router.post("/create", response_model=BlinkResponse)
async def create_blink(
    request: CreateBlinkRequest,
    current_user: User = Depends(get_premium_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new Solana Blink from a trading signal
    
    Premium users can create Blinks to share trading signals with embedded execution.
    """
    try:
        # Verify signal exists and is accessible
        result = await db.execute(
            select(Signal).where(Signal.id == request.signal_id)
        )
        signal = result.scalar_one_or_none()
        
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        # Check if signal is still valid
        if signal.expires_at and signal.expires_at < datetime.utcnow():
            raise HTTPException(status_code=400, detail="Signal has expired")
        
        # Create Solana Action based on type
        async with solana_actions_service as service:
            if request.action_type.lower() == "buy":
                action = await service.create_buy_action(
                    signal_id=request.signal_id,
                    base_currency=request.input_token,
                    suggested_amounts=request.suggested_amounts
                )
            else:
                action = await service.create_swap_action(
                    signal_id=request.signal_id,
                    input_token=request.input_token,
                    output_token=signal.symbol,
                    max_slippage=request.max_slippage
                )
            
            if not action:
                raise HTTPException(status_code=500, detail="Failed to create Solana Action")
            
            # Generate Blink URL
            action_url = action.links["actions"][0]["href"]
            blink_url = service.generate_blink_url(action_url)
            
            # Save Blink to database
            action_id = action_url.split("/")[-2]  # Extract action ID from URL
            blink_id = await service.save_blink_to_database(
                signal_id=request.signal_id,
                action_id=action_id,
                action_data=action,
                blink_url=blink_url
            )
            
            if not blink_id:
                raise HTTPException(status_code=500, detail="Failed to save Blink")
            
            # Return Blink response
            return BlinkResponse(
                id=blink_id,
                signal_id=request.signal_id,
                blink_id=action_id,
                title=action.title,
                description=action.description,
                icon_url=action.icon,
                action_url=action_url,
                blink_url=blink_url,
                status=BlinkStatus.ACTIVE.value,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),
                action_data=action.__dict__
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Blink: {e}")
        raise HTTPException(status_code=500, detail="Failed to create Blink")


@router.get("/{blink_id}", response_model=Dict[str, Any])
async def get_blink_action(blink_id: str):
    """
    Get Solana Action for a Blink (implements Solana Actions specification)
    
    This endpoint returns the Solana Action metadata that wallets use to display
    and execute the action.
    """
    try:
        async with get_db() as db:
            # Get Blink from database
            result = await db.execute(
                select(Blink).where(Blink.blink_id == blink_id)
            )
            blink = result.scalar_one_or_none()
            
            if not blink:
                return JSONResponse(
                    status_code=404,
                    content={
                        "type": "action",
                        "title": "Blink Not Found",
                        "icon": "https://playbuni.com/static/error.png",
                        "description": "The requested Blink could not be found.",
                        "label": "Error",
                        "disabled": True,
                        "error": "Blink not found"
                    }
                )
            
            # Check if Blink is still active
            if blink.status != BlinkStatus.ACTIVE.value:
                return JSONResponse(
                    status_code=410,
                    content={
                        "type": "action",
                        "title": "Blink Unavailable",
                        "icon": blink.icon_url,
                        "description": f"This Blink is {blink.status}.",
                        "label": "Unavailable",
                        "disabled": True,
                        "error": f"Blink is {blink.status}"
                    }
                )
            
            # Check if Blink has expired
            if blink.expires_at and blink.expires_at < datetime.utcnow():
                return JSONResponse(
                    status_code=410,
                    content={
                        "type": "action",
                        "title": "Blink Expired",
                        "icon": blink.icon_url,
                        "description": "This Blink has expired.",
                        "label": "Expired",
                        "disabled": True,
                        "error": "Blink has expired"
                    }
                )
            
            # Return action data
            return blink.action_data
            
    except Exception as e:
        logger.error(f"Error getting Blink action {blink_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "type": "action",
                "title": "Server Error",
                "icon": "https://playbuni.com/static/error.png",
                "description": "An error occurred while loading this Blink.",
                "label": "Error",
                "disabled": True,
                "error": "Internal server error"
            }
        )


@router.post("/{blink_id}/execute")
async def execute_blink_action(
    blink_id: str,
    request: ExecuteActionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Execute a Solana Action (implements Solana Actions specification)
    
    This endpoint processes the action execution and returns a transaction
    for the user to sign and submit.
    """
    try:
        # Validate request
        if not request.account:
            raise HTTPException(status_code=400, detail="Account address is required")
        
        # Get Blink from database
        result = await db.execute(
            select(Blink).where(Blink.blink_id == blink_id)
        )
        blink = result.scalar_one_or_none()
        
        if not blink:
            raise HTTPException(status_code=404, detail="Blink not found")
        
        # Check Blink status
        if blink.status != BlinkStatus.ACTIVE.value:
            raise HTTPException(status_code=410, detail=f"Blink is {blink.status}")
        
        if blink.expires_at and blink.expires_at < datetime.utcnow():
            raise HTTPException(status_code=410, detail="Blink has expired")
        
        # Execute the action
        async with solana_actions_service as service:
            transaction = await service.execute_swap_action(
                action_id=blink_id,
                user_wallet=request.account,
                amount=request.amount or 10.0,  # Default amount
                slippage=request.slippage or 0.5,
                priority_fee=request.priority_fee or 0
            )
            
            if not transaction:
                raise HTTPException(status_code=500, detail="Failed to create transaction")
            
            # Return transaction for signing
            return {
                "transaction": transaction,
                "message": "Transaction created successfully. Please sign and submit."
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing Blink action {blink_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute action")


@router.get("/", response_model=List[BlinkResponse])
async def get_user_blinks(
    limit: int = Query(20, description="Number of Blinks to return", le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's Blinks"""
    try:
        # Build query
        conditions = []
        
        if status:
            conditions.append(Blink.status == status)
        
        # For now, return all Blinks (in production, filter by user)
        query = select(Blink)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(Blink.created_at)).limit(limit)
        
        result = await db.execute(query)
        blinks = result.scalars().all()
        
        # Convert to response format
        blink_responses = []
        for blink in blinks:
            blink_responses.append(BlinkResponse(
                id=blink.id,
                signal_id=blink.signal_id,
                blink_id=blink.blink_id,
                title=blink.title,
                description=blink.description,
                icon_url=blink.icon_url,
                action_url=blink.action_url,
                blink_url=blink.blink_url,
                status=blink.status,
                created_at=blink.created_at,
                expires_at=blink.expires_at,
                action_data=blink.action_data
            ))
        
        return blink_responses
        
    except Exception as e:
        logger.error(f"Error getting user Blinks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve Blinks")


@router.get("/trades/", response_model=List[TradeResponse])
async def get_trades(
    limit: int = Query(20, description="Number of trades to return", le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get trading history"""
    try:
        # Build query
        conditions = []
        
        if status:
            conditions.append(Trade.status == status)
        
        # Filter by user wallet (if user is authenticated)
        if hasattr(current_user, 'wallet_address'):
            conditions.append(Trade.user_wallet == current_user.wallet_address)
        
        query = select(Trade)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(Trade.created_at)).limit(limit)
        
        result = await db.execute(query)
        trades = result.scalars().all()
        
        # Convert to response format
        trade_responses = []
        for trade in trades:
            trade_responses.append(TradeResponse(
                id=trade.id,
                user_wallet=trade.user_wallet,
                signal_id=trade.signal_id,
                blink_id=trade.blink_id,
                transaction_type=trade.transaction_type,
                input_token=trade.input_token,
                output_token=trade.output_token,
                input_amount=float(trade.input_amount),
                output_amount=float(trade.output_amount) if trade.output_amount else None,
                fee_amount=float(trade.fee_amount),
                fee_percentage=float(trade.fee_percentage),
                status=trade.status,
                transaction_signature=trade.transaction_signature,
                created_at=trade.created_at
            ))
        
        return trade_responses
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trades")


@router.get("/treasury/balance", response_model=TreasuryResponse)
async def get_treasury_balance(
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get treasury balance and statistics (Admin only)
    
    Returns current treasury balance, total fees collected, and trading statistics.
    """
    try:
        # Get treasury balance from Solana Actions service
        async with solana_actions_service as service:
            balance = await service.get_treasury_balance()
        
        # Get trading statistics from database
        trades_result = await db.execute(
            select(
                func.count(Trade.id).label('total_trades'),
                func.sum(Trade.fee_amount).label('total_fees'),
                func.sum(Trade.input_amount).label('total_volume')
            ).where(Trade.status == 'completed')
        )
        
        stats = trades_result.first()
        
        return TreasuryResponse(
            total_fees_collected=float(stats.total_fees) if stats.total_fees else 0.0,
            sol_balance=balance.get("SOL", 0.0),
            usdc_balance=balance.get("USDC", 0.0),
            total_trades=stats.total_trades or 0,
            total_volume=float(stats.total_volume) if stats.total_volume else 0.0,
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting treasury balance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve treasury balance")


@router.post("/{blink_id}/disable")
async def disable_blink(
    blink_id: str,
    current_user: User = Depends(get_premium_user),
    db: AsyncSession = Depends(get_db)
):
    """Disable a Blink"""
    try:
        # Get Blink
        result = await db.execute(
            select(Blink).where(Blink.blink_id == blink_id)
        )
        blink = result.scalar_one_or_none()
        
        if not blink:
            raise HTTPException(status_code=404, detail="Blink not found")
        
        # Update status
        await db.execute(
            "UPDATE blinks SET status = %s, updated_at = %s WHERE blink_id = %s",
            (BlinkStatus.DISABLED.value, datetime.utcnow(), blink_id)
        )
        await db.commit()
        
        return {"message": "Blink disabled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling Blink {blink_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable Blink")


@router.get("/analytics/summary")
async def get_blinks_analytics(
    days: int = Query(7, description="Number of days to analyze", le=30),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get Blinks and trading analytics"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get Blinks statistics
        blinks_result = await db.execute(
            select(
                func.count(Blink.id).label('total_blinks'),
                func.count(Blink.id).filter(Blink.status == BlinkStatus.ACTIVE.value).label('active_blinks'),
                func.count(Blink.id).filter(Blink.status == BlinkStatus.EXECUTED.value).label('executed_blinks')
            ).where(Blink.created_at >= start_date)
        )
        
        blinks_stats = blinks_result.first()
        
        # Get trading statistics
        trades_result = await db.execute(
            select(
                func.count(Trade.id).label('total_trades'),
                func.sum(Trade.fee_amount).label('total_fees'),
                func.sum(Trade.input_amount).label('total_volume'),
                func.avg(Trade.fee_amount).label('avg_fee')
            ).where(Trade.created_at >= start_date)
        )
        
        trades_stats = trades_result.first()
        
        # Calculate execution rate
        execution_rate = 0.0
        if blinks_stats.total_blinks > 0:
            execution_rate = (blinks_stats.executed_blinks or 0) / blinks_stats.total_blinks
        
        return {
            "period_days": days,
            "blinks": {
                "total": blinks_stats.total_blinks or 0,
                "active": blinks_stats.active_blinks or 0,
                "executed": blinks_stats.executed_blinks or 0,
                "execution_rate": round(execution_rate, 3)
            },
            "trading": {
                "total_trades": trades_stats.total_trades or 0,
                "total_volume": float(trades_stats.total_volume) if trades_stats.total_volume else 0.0,
                "total_fees": float(trades_stats.total_fees) if trades_stats.total_fees else 0.0,
                "avg_fee": float(trades_stats.avg_fee) if trades_stats.avg_fee else 0.0
            },
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting Blinks analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@router.post("/webhook/transaction-update")
async def handle_transaction_update(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle transaction status updates from Solana
    
    This webhook endpoint receives updates about transaction confirmations
    and updates trade status accordingly.
    """
    try:
        data = await request.json()
        
        transaction_signature = data.get("signature")
        status = data.get("status")
        
        if not transaction_signature or not status:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Update trade status
        await db.execute(
            "UPDATE trades SET status = %s, transaction_signature = %s, updated_at = %s "
            "WHERE transaction_data LIKE %s",
            (status, transaction_signature, datetime.utcnow(), f"%{transaction_signature}%")
        )
        await db.commit()
        
        logger.info(f"Updated transaction {transaction_signature} status to {status}")
        
        return {"message": "Transaction status updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling transaction update: {e}")
        raise HTTPException(status_code=500, detail="Failed to process transaction update") 