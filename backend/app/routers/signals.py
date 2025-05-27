"""
Trading Signals API Router for Play Buni Platform

This router provides endpoints for accessing trading signals based on user
access levels and NFT holdings. Integrates with the signal generation engine
and distribution system.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from pydantic import BaseModel

from app.core.database import get_db
from app.core.access_control import (
    get_current_user,
    check_signal_access,
    get_premium_user,
    get_admin_user,
    require_permission,
    PermissionType,
    get_user_access_level,
    AccessLevel
)
from app.models.signals import Signal, SignalPerformance, SignalQueue
from app.models.users import User
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Pydantic models for API responses
class SignalResponse(BaseModel):
    """Response model for trading signals"""
    id: int
    symbol: str
    mint_address: str
    signal_type: str
    signal_strength: str
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    confidence_score: float
    risk_score: float
    timeframe: str
    reasoning: List[str]
    generated_at: datetime
    expires_at: Optional[datetime]
    metadata: Optional[Dict[str, Any]] = None


class SignalPerformanceResponse(BaseModel):
    """Response model for signal performance"""
    signal_id: int
    actual_return: Optional[float]
    max_return: Optional[float]
    min_return: Optional[float]
    hit_target: Optional[bool]
    hit_stop_loss: Optional[bool]
    duration_minutes: Optional[int]
    status: str
    created_at: datetime


class SignalAnalyticsResponse(BaseModel):
    """Response model for signal analytics"""
    total_signals: int
    successful_signals: int
    success_rate: float
    avg_confidence: float
    avg_return: float
    signal_types: Dict[str, int]
    top_performers: List[Dict[str, Any]]
    period_days: int


@router.get("/", response_model=List[SignalResponse])
async def get_signals(
    limit: int = Query(10, description="Number of signals to return", le=50),
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence score", ge=0, le=1),
    current_user: User = Depends(check_signal_access),
    db: AsyncSession = Depends(get_db)
):
    """
    Get trading signals based on user access level
    
    Free users get limited access to signals, while premium users get full access.
    """
    try:
        # Get user's access level to determine signal filtering
        access_level = await get_user_access_level(current_user, db)
        
        # Build base query
        query = select(Signal).where(
            and_(
                Signal.expires_at > datetime.utcnow(),
                Signal.generated_at >= datetime.utcnow() - timedelta(hours=24)
            )
        )
        
        # Apply access level filtering
        if access_level == AccessLevel.FREE:
            # Free users only get high-confidence signals
            query = query.where(Signal.confidence_score >= 0.7)
            limit = min(limit, 5)  # Limit free users to 5 signals
        elif access_level == AccessLevel.PREMIUM:
            # Premium users get medium+ confidence signals
            query = query.where(Signal.confidence_score >= 0.5)
        # VIP and ADMIN get all signals
        
        # Apply additional filters
        if signal_type:
            query = query.where(Signal.signal_type == signal_type.lower())
        
        if symbol:
            query = query.where(Signal.symbol.ilike(f"%{symbol}%"))
        
        if min_confidence:
            query = query.where(Signal.confidence_score >= min_confidence)
        
        # Order by confidence and generation time
        query = query.order_by(
            desc(Signal.confidence_score), 
            desc(Signal.generated_at)
        ).limit(limit)
        
        result = await db.execute(query)
        signals = result.scalars().all()
        
        # Convert to response format
        signal_responses = []
        for signal in signals:
            # Include metadata only for premium+ users
            metadata = None
            if access_level in [AccessLevel.PREMIUM, AccessLevel.VIP, AccessLevel.ADMIN]:
                metadata = signal.metadata
            
            signal_responses.append(SignalResponse(
                id=signal.id,
                symbol=signal.symbol,
                mint_address=signal.mint_address,
                signal_type=signal.signal_type,
                signal_strength=signal.signal_strength,
                entry_price=float(signal.entry_price),
                target_price=float(signal.target_price) if signal.target_price else None,
                stop_loss=float(signal.stop_loss) if signal.stop_loss else None,
                confidence_score=float(signal.confidence_score),
                risk_score=float(signal.risk_score),
                timeframe=signal.timeframe,
                reasoning=signal.reasoning,
                generated_at=signal.generated_at,
                expires_at=signal.expires_at,
                metadata=metadata
            ))
        
        logger.info(f"Returned {len(signal_responses)} signals for user {current_user.wallet_address} (access: {access_level.value})")
        return signal_responses
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signals")


@router.get("/premium", response_model=List[SignalResponse])
async def get_premium_signals(
    limit: int = Query(20, description="Number of signals to return", le=100),
    include_expired: bool = Query(False, description="Include expired signals"),
    current_user: User = Depends(get_premium_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get premium trading signals for NFT holders
    
    Premium users get access to higher quality signals with more details.
    """
    try:
        # Build query for premium signals
        conditions = [Signal.confidence_score >= 0.5]
        
        if not include_expired:
            conditions.append(Signal.expires_at > datetime.utcnow())
        
        conditions.append(Signal.generated_at >= datetime.utcnow() - timedelta(hours=48))
        
        query = select(Signal).where(and_(*conditions)).order_by(
            desc(Signal.confidence_score), 
            desc(Signal.generated_at)
        ).limit(limit)
        
        result = await db.execute(query)
        signals = result.scalars().all()
        
        # Convert to response format with full details
        signal_responses = []
        for signal in signals:
            signal_responses.append(SignalResponse(
                id=signal.id,
                symbol=signal.symbol,
                mint_address=signal.mint_address,
                signal_type=signal.signal_type,
                signal_strength=signal.signal_strength,
                entry_price=float(signal.entry_price),
                target_price=float(signal.target_price) if signal.target_price else None,
                stop_loss=float(signal.stop_loss) if signal.stop_loss else None,
                confidence_score=float(signal.confidence_score),
                risk_score=float(signal.risk_score),
                timeframe=signal.timeframe,
                reasoning=signal.reasoning,
                generated_at=signal.generated_at,
                expires_at=signal.expires_at,
                metadata=signal.metadata
            ))
        
        logger.info(f"Returned {len(signal_responses)} premium signals for user {current_user.wallet_address}")
        return signal_responses
        
    except Exception as e:
        logger.error(f"Error getting premium signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve premium signals")


@router.get("/live", response_model=List[SignalResponse])
async def get_live_signals(
    current_user: User = Depends(get_premium_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get live/active trading signals for real-time trading
    
    Returns only currently active signals that haven't expired.
    """
    try:
        # Get only active, non-expired signals
        query = select(Signal).where(
            and_(
                Signal.expires_at > datetime.utcnow(),
                Signal.generated_at >= datetime.utcnow() - timedelta(hours=6),  # Last 6 hours
                Signal.confidence_score >= 0.6  # High confidence only
            )
        ).order_by(desc(Signal.generated_at)).limit(10)
        
        result = await db.execute(query)
        signals = result.scalars().all()
        
        signal_responses = []
        for signal in signals:
            signal_responses.append(SignalResponse(
                id=signal.id,
                symbol=signal.symbol,
                mint_address=signal.mint_address,
                signal_type=signal.signal_type,
                signal_strength=signal.signal_strength,
                entry_price=float(signal.entry_price),
                target_price=float(signal.target_price) if signal.target_price else None,
                stop_loss=float(signal.stop_loss) if signal.stop_loss else None,
                confidence_score=float(signal.confidence_score),
                risk_score=float(signal.risk_score),
                timeframe=signal.timeframe,
                reasoning=signal.reasoning,
                generated_at=signal.generated_at,
                expires_at=signal.expires_at,
                metadata=signal.metadata
            ))
        
        return signal_responses
        
    except Exception as e:
        logger.error(f"Error getting live signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve live signals")


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal_by_id(
    signal_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific signal by ID"""
    try:
        result = await db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = result.scalar_one_or_none()
        
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        # Check access level for metadata
        access_level = await get_user_access_level(current_user, db)
        metadata = None
        if access_level in [AccessLevel.PREMIUM, AccessLevel.VIP, AccessLevel.ADMIN]:
            metadata = signal.metadata
        
        return SignalResponse(
            id=signal.id,
            symbol=signal.symbol,
            mint_address=signal.mint_address,
            signal_type=signal.signal_type,
            signal_strength=signal.signal_strength,
            entry_price=float(signal.entry_price),
            target_price=float(signal.target_price) if signal.target_price else None,
            stop_loss=float(signal.stop_loss) if signal.stop_loss else None,
            confidence_score=float(signal.confidence_score),
            risk_score=float(signal.risk_score),
            timeframe=signal.timeframe,
            reasoning=signal.reasoning,
            generated_at=signal.generated_at,
            expires_at=signal.expires_at,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal {signal_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signal")


@router.get("/{signal_id}/performance", response_model=SignalPerformanceResponse)
async def get_signal_performance(
    signal_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get performance data for a specific signal"""
    try:
        result = await db.execute(
            select(SignalPerformance).where(SignalPerformance.signal_id == signal_id)
        )
        performance = result.scalar_one_or_none()
        
        if not performance:
            raise HTTPException(status_code=404, detail="Signal performance not found")
        
        return SignalPerformanceResponse(
            signal_id=performance.signal_id,
            actual_return=float(performance.actual_return) if performance.actual_return else None,
            max_return=float(performance.max_return) if performance.max_return else None,
            min_return=float(performance.min_return) if performance.min_return else None,
            hit_target=performance.hit_target,
            hit_stop_loss=performance.hit_stop_loss,
            duration_minutes=performance.duration_minutes,
            status=performance.status,
            created_at=performance.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal performance {signal_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signal performance")


@router.get("/analytics/summary", response_model=SignalAnalyticsResponse)
async def get_signals_analytics(
    days: int = Query(7, description="Number of days to analyze", le=30),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive analytics for signals"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get signals from the specified period
        signals_result = await db.execute(
            select(Signal).where(Signal.generated_at >= start_date)
        )
        signals = signals_result.scalars().all()
        
        # Get performance data
        performance_result = await db.execute(
            select(SignalPerformance).join(Signal).where(
                Signal.generated_at >= start_date
            )
        )
        performances = performance_result.scalars().all()
        
        total_signals = len(signals)
        
        if total_signals == 0:
            return SignalAnalyticsResponse(
                total_signals=0,
                successful_signals=0,
                success_rate=0,
                avg_confidence=0,
                avg_return=0,
                signal_types={},
                top_performers=[],
                period_days=days
            )
        
        # Calculate metrics
        successful_signals = sum(1 for p in performances if p.actual_return and p.actual_return > 0)
        success_rate = successful_signals / len(performances) if performances else 0
        avg_confidence = sum(float(s.confidence_score) for s in signals) / total_signals
        avg_return = sum(float(p.actual_return) for p in performances if p.actual_return) / len(performances) if performances else 0
        
        # Count signal types
        signal_types = {}
        for signal in signals:
            signal_type = signal.signal_type
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        
        # Get top performers
        top_performers = []
        for perf in sorted(performances, key=lambda p: p.actual_return or 0, reverse=True)[:5]:
            signal = next((s for s in signals if s.id == perf.signal_id), None)
            if signal:
                top_performers.append({
                    "signal_id": signal.id,
                    "symbol": signal.symbol,
                    "return": float(perf.actual_return) if perf.actual_return else 0,
                    "confidence": float(signal.confidence_score)
                })
        
        return SignalAnalyticsResponse(
            total_signals=total_signals,
            successful_signals=successful_signals,
            success_rate=round(success_rate, 3),
            avg_confidence=round(avg_confidence, 3),
            avg_return=round(avg_return, 3),
            signal_types=signal_types,
            top_performers=top_performers,
            period_days=days
        )
        
    except Exception as e:
        logger.error(f"Error getting signals analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signals analytics")


@router.post("/generate")
async def trigger_signal_generation(
    symbol: Optional[str] = Query(None, description="Generate signal for specific symbol"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_admin_user)
):
    """
    Manually trigger signal generation (Admin only)
    
    This endpoint allows administrators to manually trigger signal generation
    for testing or immediate signal needs.
    """
    try:
        # Import here to avoid circular imports
        from app.services.signal_engine import signal_generation_engine
        
        if symbol:
            # Generate signal for specific symbol
            logger.info(f"Manual signal generation requested for {symbol} by admin {current_user.wallet_address}")
            return {"message": f"Signal generation triggered for {symbol}", "status": "queued"}
        else:
            # Trigger full generation cycle
            background_tasks.add_task(signal_generation_engine.run_signal_generation_cycle)
            logger.info(f"Manual signal generation cycle triggered by admin {current_user.wallet_address}")
            return {"message": "Signal generation cycle triggered", "status": "running"}
        
    except Exception as e:
        logger.error(f"Error triggering signal generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger signal generation")


@router.get("/queue/status")
async def get_queue_status(
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get signal queue status (Admin only)
    
    Returns information about pending signals in the distribution queue.
    """
    try:
        # Get queue statistics
        result = await db.execute(
            select(
                SignalQueue.status,
                SignalQueue.channel,
                func.count(SignalQueue.id).label('count')
            ).group_by(SignalQueue.status, SignalQueue.channel)
        )
        
        queue_stats = {}
        for status, channel, count in result:
            if status not in queue_stats:
                queue_stats[status] = {}
            queue_stats[status][channel] = count
        
        # Get pending signals count
        pending_result = await db.execute(
            select(func.count(SignalQueue.id)).where(
                SignalQueue.status == "pending"
            )
        )
        pending_count = pending_result.scalar()
        
        return {
            "pending_signals": pending_count,
            "queue_statistics": queue_stats,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve queue status") 