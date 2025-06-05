#!/usr/bin/env python3
"""Quick sanity checks for Play Buni Platform."""

import sys
from pathlib import Path

print("🚀 Play Buni Platform - Quick Test")
print("=" * 40)

# Confirm core dependencies are importable
try:
    import fastapi
    import uvicorn
    import pydantic

    print("✅ Core dependencies imported")
except Exception as exc:
    print(f"❌ Import failure: {exc}")
    sys.exit(1)

# Confirm application modules load
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from app.core.config import Settings

    Settings(
        app_name="Test App",
        secret_key="test",
        database_url="postgresql://user:pass@localhost/db",
        redis_url="redis://localhost:6379/0",
    )
    print("✅ Application modules import")
except Exception as exc:
    print(f"⚠️  Application import issue: {exc}")

print("\n🎉 Quick checks completed")
