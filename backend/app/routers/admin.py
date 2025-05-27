"""
Admin Router - Administrative endpoints and management.

This module provides:
- System administration
- User management
- Signal management
- Analytics and reporting
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def admin_dashboard():
    """Admin dashboard overview."""
    return {"message": "Admin dashboard - to be implemented"}


@router.get("/users")
async def manage_users():
    """User management interface."""
    return {"message": "User management - to be implemented"} 