# Railway Deployment Checklist - Play Buni Platform

## ✅ Pre-Deployment Checklist

### 1. Environment Variables Setup
Before deploying to Railway, ensure these environment variables are configured:

#### Required Environment Variables:
```bash
# Database (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# Security
SECRET_KEY=your-super-secret-key-here
SOLANA_PRIVATE_KEY=your-base58-private-key

# Treasury & NFT
TREASURY_WALLET=your-treasury-wallet-address
TREASURY_WALLET_ADDRESS=your-treasury-wallet-address
NFT_COLLECTION_ADDRESS=your-nft-collection-mint-address
NFT_CREATOR_ADDRESS=your-creator-wallet-address

# Redis (Railway will provide this)
REDIS_URL=redis://localhost:6379/0

# API Keys (Get these when ready to scale)
COINGECKO_API_KEY=your-coingecko-api-key (optional for free tier)
BIRDEYE_API_KEY=your-birdeye-api-key (optional)

# Social Media (Skip for now)
TWITTER_API_KEY=skip-for-now
TWITTER_API_SECRET=skip-for-now
DISCORD_BOT_TOKEN=skip-for-now
```

### 2. Railway Services Setup

#### Main Application Service:
- **Name**: play-buni-api
- **Type**: Web Service
- **Repository**: Your GitHub repository
- **Root Directory**: `/backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python startup.py`

#### Redis Service:
- **Name**: play-buni-redis
- **Type**: Redis
- **Connect to main service via**: `REDIS_URL` environment variable

#### Database:
- **Use Supabase** (external service)
- Add `DATABASE_URL` to environment variables

### 3. Deployment Steps

1. **Push Code to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

2. **Create Railway Project**
   - Go to [Railway](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Services**
   - Add Redis service
   - Configure environment variables
   - Set root directory to `/backend`

4. **Initial Deployment**
   - Railway will automatically deploy
   - Monitor logs for any issues
   - Check health endpoint: `https://your-app.railway.app/health`

### 4. Post-Deployment Testing

#### API Endpoints to Test:
```bash
# Health check
GET https://your-app.railway.app/health

# API info
GET https://your-app.railway.app/api/v1

# Jupiter monitoring (requires auth)
GET https://your-app.railway.app/api/v1/jupiter/tier-info

# WebSocket test page
GET https://your-app.railway.app/api/v1/websocket-test
```

#### Database Migration:
- Migrations run automatically via `startup.py`
- Check Railway logs for migration status
- Tables should be created in Supabase

### 5. Scaling Configuration

#### Horizontal Scaling:
- Start with 1 instance
- Scale up based on traffic
- Monitor memory usage (Railway has limits)

#### Background Workers:
- Celery workers are included in main process
- For high load, consider separate worker services
- Monitor Redis for queue buildup

## 🚨 Common Issues & Solutions

### 1. Database Connection Issues
```bash
# Check DATABASE_URL format
# Ensure Supabase allows external connections
# Verify service role key permissions
```

### 2. Redis Connection Issues
```bash
# Ensure Redis service is running
# Check REDIS_URL environment variable
# Verify network connectivity between services
```

### 3. Memory Issues
```bash
# Railway has memory limits
# Reduce Celery concurrency if needed
# Monitor memory usage in Railway dashboard
```

### 4. Import/Module Issues
```bash
# Ensure all dependencies in requirements.txt
# Check Python path in startup.py
# Verify all imports are relative
```

## 📊 Monitoring & Maintenance

### Health Monitoring:
- Health endpoint: `/health`
- Check database connectivity
- Monitor Celery worker status
- Track API response times

### Logs:
- Railway provides real-time logs
- Check for startup errors
- Monitor background task execution
- Track API request patterns

### Scaling Triggers:
- CPU usage > 80%
- Memory usage > 90%
- Response time > 2 seconds
- Queue backlog > 100 tasks

## 🎯 Success Criteria

### Deployment is successful when:
- [ ] Health endpoint returns 200 OK
- [ ] Database tables are created
- [ ] WebSocket connections work
- [ ] Jupiter API integration responds
- [ ] Background workers are processing
- [ ] No critical errors in logs
- [ ] API documentation is accessible

### Performance targets:
- [ ] API response time < 500ms
- [ ] WebSocket latency < 100ms
- [ ] Signal processing < 30 seconds
- [ ] 99.9% uptime
- [ ] Memory usage < 512MB

## 🔧 Development vs Production

### Development Features (Enabled):
- API documentation at `/docs`
- Debug mode
- Detailed error messages
- Test endpoints

### Production Features (Disabled):
- Debug mode off
- Error tracking via logs
- Rate limiting enabled
- Security headers

## 📞 Support

### Railway Support:
- Railway Discord community
- Railway documentation
- GitHub issues for platform-specific issues

### Platform Support:
- Check GitHub repository issues
- Review deployment logs
- Monitor health endpoints
- Check environment variable configuration

---

## Quick Start Command

```bash
# Test locally first
cd backend
python startup.py

# Then deploy to Railway with confidence! 🚀
``` 