"""
Celery Application Configuration for Play Buni Platform
Handles background job processing for market monitoring, signal processing, and analytics
"""
import os
import logging
from celery import Celery
from celery.schedules import crontab
from datetime import timedelta

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery application
celery_app = Celery(
    "playbuni_platform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'app.workers.market_monitor',
        'app.workers.signal_processor', 
        'app.workers.analytics_collector',
        'app.workers.nft_verifier',
        'app.workers.revenue_tracker',
        'app.workers.notification_sender'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'app.workers.market_monitor.*': {'queue': 'market_data'},
        'app.workers.signal_processor.*': {'queue': 'signals'},
        'app.workers.analytics_collector.*': {'queue': 'analytics'},
        'app.workers.nft_verifier.*': {'queue': 'nft_verification'},
        'app.workers.revenue_tracker.*': {'queue': 'revenue'},
        'app.workers.notification_sender.*': {'queue': 'notifications'},
        'app.workers.websocket_streaming.*': {'queue': 'websockets'}
    },
    
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Result settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600,
    },
    
    # Task execution settings
    task_soft_time_limit=600,  # 10 minutes soft limit
    task_time_limit=900,       # 15 minutes hard limit
    task_reject_on_worker_lost=True,
    
    # Error handling
    task_retry_delay=60,       # 1 minute retry delay
    task_max_retries=3,
    
    # Queue settings
    task_default_queue='default',
    task_default_exchange='default',
    task_default_exchange_type='direct',
    task_default_routing_key='default',
    
    # Beat scheduler settings (for periodic tasks)
    beat_schedule={
        # Market data monitoring every 30 seconds
        'monitor-market-data': {
            'task': 'app.workers.market_monitor.fetch_market_data',
            'schedule': timedelta(seconds=30),
            'options': {'queue': 'market_data'}
        },
        
        # Process signals every minute
        'process-signals': {
            'task': 'app.workers.signal_processor.process_pending_signals',
            'schedule': timedelta(minutes=1),
            'options': {'queue': 'signals'}
        },
        
        # Analytics collection every 5 minutes
        'collect-analytics': {
            'task': 'app.workers.analytics_collector.collect_platform_analytics',
            'schedule': timedelta(minutes=5),
            'options': {'queue': 'analytics'}
        },
        
        # NFT verification refresh every 10 minutes
        'refresh-nft-verification': {
            'task': 'app.workers.nft_verifier.refresh_nft_status',
            'schedule': timedelta(minutes=10),
            'options': {'queue': 'nft_verification'}
        },
        
        # Revenue tracking every hour
        'track-revenue': {
            'task': 'app.workers.revenue_tracker.calculate_hourly_revenue',
            'schedule': crontab(minute=0),  # Every hour at minute 0
            'options': {'queue': 'revenue'}
        },
        
        # Daily analytics at midnight UTC
        'daily-analytics': {
            'task': 'app.workers.analytics_collector.generate_daily_report',
            'schedule': crontab(hour=0, minute=0),  # Midnight UTC
            'options': {'queue': 'analytics'}
        },
        
        # Cleanup old data weekly on Sunday at 2 AM UTC
        'weekly-cleanup': {
            'task': 'app.workers.analytics_collector.cleanup_old_data',
            'schedule': crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2 AM
            'options': {'queue': 'analytics'}
        },
        
        # Health check for all services every 5 minutes
        'system-health-check': {
            'task': 'app.workers.analytics_collector.system_health_check',
            'schedule': timedelta(minutes=5),
            'options': {'queue': 'analytics'}
        }
    },
    
    # Monitoring settings
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security settings
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# Task annotations for specific configuration
celery_app.conf.task_annotations = {
    'app.workers.market_monitor.fetch_market_data': {
        'rate_limit': '2/s',  # Max 2 calls per second to avoid API limits
        'priority': 9         # High priority for market data
    },
    'app.workers.signal_processor.process_pending_signals': {
        'rate_limit': '10/s', # Process up to 10 signals per second
        'priority': 8         # High priority for signal processing
    },
    'app.workers.nft_verifier.verify_wallet_nfts': {
        'rate_limit': '5/s',  # Solana RPC rate limiting
        'priority': 6         # Medium priority
    },
    'app.workers.revenue_tracker.process_transaction': {
        'rate_limit': '20/s', # Fast transaction processing
        'priority': 7         # Medium-high priority
    },
    'app.workers.notification_sender.send_notification': {
        'rate_limit': '50/s', # Allow many notifications
        'priority': 5         # Medium priority
    }
}

# Custom task base class with error handling
class CallbackTask(celery_app.Task):
    """Custom task base class with enhanced error handling and logging"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {self.name} [{task_id}] succeeded: {retval}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Task {self.name} [{task_id}] failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {self.name} [{task_id}] retrying: {exc}")

# Set custom task base
celery_app.Task = CallbackTask

# Worker startup/shutdown hooks
@celery_app.signals.worker_ready.connect
def worker_ready(sender=None, **kwargs):
    """Called when worker is ready"""
    logger.info(f"Celery worker {sender.hostname} is ready")

@celery_app.signals.worker_shutdown.connect  
def worker_shutdown(sender=None, **kwargs):
    """Called when worker shuts down"""
    logger.info(f"Celery worker {sender.hostname} is shutting down")

# Task failure tracking
@celery_app.signals.task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, einfo=None, **kwargs):
    """Handle task failures for monitoring"""
    logger.error(f"Task failure: {sender.name} [{task_id}] - {exception}")

# Queue monitoring functions
def get_queue_stats():
    """Get statistics for all queues"""
    inspect = celery_app.control.inspect()
    stats = {
        'active': inspect.active(),
        'scheduled': inspect.scheduled(),
        'reserved': inspect.reserved(),
        'stats': inspect.stats(),
        'registered': inspect.registered()
    }
    return stats

def purge_queue(queue_name: str):
    """Purge all tasks from a specific queue"""
    return celery_app.control.purge_queue(queue_name)

def get_active_tasks():
    """Get all currently active tasks"""
    inspect = celery_app.control.inspect()
    return inspect.active()

# Health check function
def health_check():
    """Check if Celery workers are responding"""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        if stats:
            return {"status": "healthy", "workers": len(stats)}
        else:
            return {"status": "no_workers", "workers": 0}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Task management utilities
def cancel_task(task_id: str):
    """Cancel a specific task by ID"""
    celery_app.control.revoke(task_id, terminate=True)

def list_pending_tasks():
    """List all pending tasks across all queues"""
    inspect = celery_app.control.inspect()
    return {
        'scheduled': inspect.scheduled(),
        'reserved': inspect.reserved()
    }

# Development utilities
if settings.environment == "development":
    # Enable debug logging for development
    celery_app.conf.update(
        worker_log_level='DEBUG',
        task_track_started=True,
        task_send_sent_event=True
    )

# Production optimizations
if settings.environment == "production":
    celery_app.conf.update(
        # Production-specific settings
        worker_disable_rate_limits=False,
        task_compression='gzip',
        result_compression='gzip',
        
        # Monitoring and reliability
        worker_send_task_events=True,
        task_send_sent_event=True,
        worker_enable_remote_control=True,
        
        # Performance tuning
        worker_prefetch_multiplier=4,
        task_acks_late=True,
        worker_max_tasks_per_child=1000
    )

# Export the app
__all__ = ['celery_app'] 