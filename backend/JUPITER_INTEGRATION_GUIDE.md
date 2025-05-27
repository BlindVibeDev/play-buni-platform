# Jupiter API Integration Guide - Play Buni Platform

## Overview

The Play Buni Platform uses an intelligent Jupiter API integration that automatically manages costs and scales with platform growth. This guide explains how to set up and use the multi-tier Jupiter API system.

## Jupiter API Tiers

### 1. Free Tier (Default - Start Here)
- **Cost**: $0/month + 0.2% platform fees on swaps
- **Rate Limits**: 10 requests/second, ~8,640 daily requests
- **Endpoint**: `https://www.jupiterapi.com`
- **Perfect for**: MVP, testing, low-volume trading

### 2. Trial Tier (QuickNode Metis)
- **Cost**: Free trial period
- **Rate Limits**: 600 requests/minute, 50,000 daily requests
- **Features**: Higher limits, dedicated endpoint
- **Perfect for**: Growing platform, moderate volume

### 3. Paid Tier (Premium)
- **Cost**: Variable based on usage
- **Rate Limits**: 1,000+ requests/minute, unlimited daily
- **Features**: Maximum performance, 24/7 support
- **Perfect for**: High-volume production platform

## Auto-Upgrade Logic

The system automatically upgrades tiers based on:

```python
# Upgrade to Trial Tier when:
daily_requests > 5,000 OR monthly_revenue > $100

# Upgrade to Paid Tier when:
daily_requests > 30,000 OR monthly_revenue > $1,000
```

## Setup Instructions

### 1. Environment Configuration

Add to your `.env` file:

```bash
# Jupiter API Configuration (Multi-tier support)
JUPITER_FREE_ENDPOINT=https://www.jupiterapi.com
QUICKNODE_METIS_ENDPOINT=""  # Optional: Add when you get trial access
JUPITER_PAID_ENDPOINT=""     # Optional: Add when you upgrade to paid
```

### 2. Getting Trial Access (When Ready)

1. **Sign up for QuickNode**: [https://quicknode.com](https://quicknode.com)
2. **Add Metis Add-on**: Go to Add-ons → Search "Metis" → Add to endpoint
3. **Get Endpoint URL**: Copy your Metis endpoint URL
4. **Update Environment**: Add the URL to `QUICKNODE_METIS_ENDPOINT`

### 3. Usage in Code

```python
from app.services.jupiter_service import jupiter_service, JupiterQuoteRequest

# Get a quote for SOL to USDC swap
quote_request = JupiterQuoteRequest(
    inputMint="So11111111111111111111111111111111111111112",  # SOL
    outputMint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    amount=1000000000,  # 1 SOL in lamports
    slippageBps=50      # 0.5% slippage
)

# This automatically uses the best available tier
quote = await jupiter_service.get_quote(quote_request)

# Get swap transaction
swap_tx = await jupiter_service.get_swap_transaction(
    quote, 
    user_public_key="your_wallet_address"
)
```

## Monitoring & Cost Management

### API Endpoints for Monitoring

```bash
# Check current tier and usage
GET /api/jupiter/tier-info

# Get detailed usage statistics
GET /api/jupiter/usage-stats

# Get cost projections
GET /api/jupiter/cost-projection

# Manually upgrade tier (admin only)
POST /api/jupiter/force-tier-upgrade
```

### Example Usage Monitoring

```python
# Check current usage
tier_info = jupiter_service.get_current_tier_info()
print(f"Current tier: {tier_info['current_tier']}")
print(f"Daily requests: {tier_info['daily_requests']}")
print(f"Usage: {tier_info['usage_percentage']:.1f}%")
```

## Cost Optimization Strategies

### 1. Start Small
- Begin with the free tier
- Monitor usage patterns
- Let auto-upgrade handle scaling

### 2. Cache Quotes When Possible
```python
# Cache quotes for identical requests (within reason)
# Don't cache for more than 30 seconds due to price volatility
```

### 3. Batch Operations
```python
# Process multiple signals together when possible
# Reduces total API calls
```

### 4. Smart Rate Limiting
```python
# The system automatically handles rate limits
# Falls back to free tier if paid tier fails
# Prevents API call failures
```

## Revenue Milestones & Tier Progression

### Startup Phase (Free Tier)
- **Revenue Target**: $0 - $100/month
- **Features**: Basic platform functionality
- **Limitations**: Lower rate limits may slow signal processing

### Growth Phase (Trial Tier)
- **Revenue Target**: $100 - $1,000/month
- **Benefits**: Higher rate limits, better performance
- **Cost**: Still free during trial period

### Scale Phase (Paid Tier)
- **Revenue Target**: $1,000+/month
- **Benefits**: Maximum performance, unlimited requests
- **ROI**: Platform fees (1%) easily cover API costs

## Emergency Fallback

The system includes automatic fallback:

1. **Primary**: Current tier endpoint
2. **Fallback**: Free tier endpoint
3. **Logging**: All failures logged for debugging

## Getting Help

### Free Tier Support
- GitHub issues
- Community Discord
- Documentation

### Trial/Paid Tier Support
- Priority support channels
- Direct technical assistance
- Custom integration help

## Migration Path

### Current State → Free Tier (Immediate)
```bash
# No setup required - works out of the box
# Just use the existing Jupiter service
```

### Free Tier → Trial Tier (When Ready)
```bash
# 1. Sign up for QuickNode
# 2. Add Metis add-on
# 3. Update QUICKNODE_METIS_ENDPOINT in .env
# 4. System auto-upgrades when usage/revenue thresholds met
```

### Trial Tier → Paid Tier (When Scaling)
```bash
# 1. Contact QuickNode for paid plan
# 2. Update JUPITER_PAID_ENDPOINT in .env
# 3. System auto-upgrades based on usage
```

## Cost Estimates

### Monthly Cost Breakdown

| Platform Revenue | Tier | API Costs | Platform Fees | Net Cost |
|------------------|------|-----------|---------------|----------|
| $0 - $100 | Free | $0 | 0.2% on swaps | ~$0-5 |
| $100 - $1,000 | Trial | $0 | 0% | $0 |
| $1,000+ | Paid | $50-200 | 0% | $50-200 |

### Break-Even Analysis
- **Free Tier**: Profitable immediately
- **Trial Tier**: Profitable immediately  
- **Paid Tier**: Profitable when monthly revenue > $2,000

The system is designed to scale profitably - API costs only increase when your revenue can support them.

## Support

For technical questions about this integration:
1. Check the monitoring endpoints first
2. Review logs in the analytics dashboard
3. Contact the development team with specific error messages

The intelligent tier management ensures you always have the best cost-performance ratio for your current platform size. 