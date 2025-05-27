#!/usr/bin/env python3
"""
Celery Startup Script for Play Buni Platform
Provides convenient commands to start Celery workers and beat scheduler
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def start_worker(queues=None, concurrency=None, loglevel="info"):
    """Start a Celery worker"""
    cmd = [
        "celery", 
        "-A", "app.workers.celery_app", 
        "worker"
    ]
    
    if queues:
        cmd.extend(["--queues", queues])
    else:
        # Default queues
        cmd.extend(["--queues", "default,market_data,signals,analytics,nft_verification,revenue,notifications,websockets"])
    
    if concurrency:
        cmd.extend(["--concurrency", str(concurrency)])
    
    cmd.extend(["--loglevel", loglevel])
    cmd.extend(["--prefetch-multiplier", "1"])
    cmd.extend(["--max-tasks-per-child", "1000"])
    
    print(f"Starting Celery worker with command: {' '.join(cmd)}")
    subprocess.run(cmd)

def start_beat(loglevel="info"):
    """Start Celery beat scheduler"""
    cmd = [
        "celery",
        "-A", "app.workers.celery_app",
        "beat",
        "--loglevel", loglevel,
        "--scheduler", "celery.beat:PersistentScheduler"
    ]
    
    print(f"Starting Celery beat with command: {' '.join(cmd)}")
    subprocess.run(cmd)

def start_flower(port=5555):
    """Start Celery Flower monitoring"""
    cmd = [
        "celery",
        "-A", "app.workers.celery_app",
        "flower",
        "--port", str(port)
    ]
    
    print(f"Starting Celery Flower on port {port}")
    subprocess.run(cmd)

def start_all():
    """Start all Celery services (for development only)"""
    print("Starting all Celery services for development...")
    print("Note: In production, run these as separate processes/containers")
    
    # This would typically be run in separate terminals or containers
    print("\nTo start all services properly, run these commands in separate terminals:")
    print("1. python start_celery.py worker")
    print("2. python start_celery.py beat")
    print("3. python start_celery.py flower")

def purge_queues():
    """Purge all Celery queues"""
    cmd = [
        "celery",
        "-A", "app.workers.celery_app",
        "purge",
        "-f"  # Force without confirmation
    ]
    
    print("Purging all Celery queues...")
    subprocess.run(cmd)

def inspect_tasks():
    """Inspect active tasks"""
    cmd = [
        "celery",
        "-A", "app.workers.celery_app",
        "inspect",
        "active"
    ]
    
    print("Inspecting active tasks...")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Celery management for Play Buni Platform")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Start Celery worker")
    worker_parser.add_argument("--queues", "-q", help="Comma-separated list of queues")
    worker_parser.add_argument("--concurrency", "-c", type=int, help="Number of concurrent worker processes")
    worker_parser.add_argument("--loglevel", "-l", default="info", help="Log level")
    
    # Beat command
    beat_parser = subparsers.add_parser("beat", help="Start Celery beat scheduler")
    beat_parser.add_argument("--loglevel", "-l", default="info", help="Log level")
    
    # Flower command
    flower_parser = subparsers.add_parser("flower", help="Start Celery Flower monitoring")
    flower_parser.add_argument("--port", "-p", type=int, default=5555, help="Port for Flower")
    
    # Other commands
    subparsers.add_parser("all", help="Show commands to start all services")
    subparsers.add_parser("purge", help="Purge all queues")
    subparsers.add_parser("inspect", help="Inspect active tasks")
    
    args = parser.parse_args()
    
    if args.command == "worker":
        start_worker(
            queues=args.queues,
            concurrency=args.concurrency,
            loglevel=args.loglevel
        )
    elif args.command == "beat":
        start_beat(loglevel=args.loglevel)
    elif args.command == "flower":
        start_flower(port=args.port)
    elif args.command == "all":
        start_all()
    elif args.command == "purge":
        purge_queues()
    elif args.command == "inspect":
        inspect_tasks()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 