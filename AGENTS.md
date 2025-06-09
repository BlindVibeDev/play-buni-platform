# Project Context for AI Agents

## Project Overview
Play Buni Platform is an NFT-gated AI trading signal platform built with FastAPI and Supabase. It delivers Solana trading signals to free users via social channels while NFT holders receive real-time premium access.

## Code Standards
- **Style**: Follow PEP 8 and format with Black.
- **Naming**: Use descriptive snake_case for variables and functions. Classes use PascalCase.
- **Testing**: Aim for 80%+ coverage using pytest and pytest-asyncio.

## Build and Test Instructions
- Install Python 3.11+ dependencies from `backend/requirements.txt`.
- Run FastAPI locally: `uvicorn app.main:app --reload` from the `backend` folder.
- Run tests with `pytest`.

## Special Considerations
- The platform integrates Solana blockchain features and uses async SQLAlchemy with Supabase.
- Celery workers handle background tasks for signal generation and distribution.
