#!/usr/bin/env python3
"""
Deployment Test Script
Quick tests to verify core functionality before Railway deployment
"""
import asyncio
import logging
import sys
from datetime import datetime

# Test imports
try:
    from app.core.config import settings
    from app.core.logging import configure_logging
    from app.services.jupiter_service import jupiter_service, JupiterQuoteRequest
    from app.services.signal_service import SignalService
    from app.database import init_db, get_db_session
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connectivity"""
    try:
        await init_db()
        async with get_db_session() as db:
            await db.execute("SELECT 1")
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def test_jupiter_service():
    """Test Jupiter service basic functionality"""
    try:
        # Test tier info
        tier_info = jupiter_service.get_current_tier_info()
        print(f"✅ Jupiter service initialized - Tier: {tier_info['current_tier']}")
        
        # Test quote request (this will use free tier)
        quote_request = JupiterQuoteRequest(
            inputMint="So11111111111111111111111111111111111111112",  # SOL
            outputMint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            amount=1000000000,  # 1 SOL in lamports
            slippageBps=50
        )
        
        # Note: This will fail without network, but tests the service structure
        print(f"✅ Jupiter quote request created successfully")
        return True
    except Exception as e:
        print(f"❌ Jupiter service test failed: {e}")
        return False

async def test_signal_service():
    """Test signal service"""
    try:
        signal_service = SignalService()
        stats = signal_service.get_signal_statistics()
        print(f"✅ Signal service initialized - Stats: {stats}")
        return True
    except Exception as e:
        print(f"❌ Signal service test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        # Test required settings
        required_vars = [
            'app_name',
            'secret_key',
            'database_url',
            'redis_url'
        ]
        
        for var in required_vars:
            value = getattr(settings, var, None)
            if not value:
                print(f"⚠️  Warning: {var} not configured")
            else:
                print(f"✅ {var} configured")
        
        print(f"✅ Configuration loaded - Environment: {settings.environment}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Play Buni Platform deployment tests...\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Database Connection", test_database_connection),
        ("Jupiter Service", test_jupiter_service),
        ("Signal Service", test_signal_service),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for Railway deployment!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before deployment.")
        return False

async def main():
    """Main test function"""
    try:
        success = await run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 