# 🤖 Play Buni Platform - NFT-Gated AI Trading Signals

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Solana-Blockchain-purple?style=for-the-badge&logo=solana" alt="Solana" />
  <img src="https://img.shields.io/badge/Supabase-Database-green?style=for-the-badge&logo=supabase" alt="Supabase" />
</div>

## 🌟 Overview

Play Buni Platform is an advanced NFT-gated trading signal platform that provides AI-powered Solana trading insights. The platform operates on a dual-channel distribution model where free users receive 2 quality signals per hour on social media, while NFT holders get unlimited real-time access to all signals on the premium platform.

### 🎯 Core Value Proposition
**"Public Signals for Growth → NFT Access for Volume → Fee Revenue for Sustainability"**

- **Free Users**: 48 profitable signals per day via social media Blinks
- **NFT Holders**: Unlimited real-time signals + premium platform access  
- **Platform**: 1% fee on all trades funds signal infrastructure

## 🔑 Key Features

### 🤖 AI Signal Agent
- **24/7 Market Intelligence**: Continuous monitoring of 200+ Solana tokens
- **Multi-Layer Analysis**: Technical indicators, sentiment analysis, whale tracking
- **Confidence Scoring**: Advanced algorithms for signal quality assessment
- **Pattern Recognition**: Historical performance-based decision making

### 🔐 NFT-Gated Access Control
- **Solana NFT Verification**: Secure ownership-based access control
- **Dual Distribution**: Social (2/hour) vs Premium (real-time) channels
- **Session Management**: JWT-based authentication for NFT holders
- **Tier Permissions**: Feature unlocking based on NFT ownership

### 📈 Signal Generation System
- **Real-Time Processing**: FastAPI server with async signal generation
- **Quality Validation**: Multi-stage verification before distribution
- **Performance Tracking**: Continuous monitoring and algorithm optimization
- **Market Adaptation**: Self-improving accuracy based on outcomes

### 🔗 Solana Blinks Integration
- **One-Click Trading**: Seamless execution via shareable blockchain links
- **Automatic Fee Collection**: 1% fee on all trades to treasury wallet
- **Custom Interfaces**: Branded Blink UIs for platform identity
- **Error Handling**: Robust transaction failure management

### 📱 Social Distribution
- **Automated Posting**: Twitter/Discord integration with rate limiting
- **Viral Mechanics**: Performance comparisons to drive NFT adoption
- **Engagement Tracking**: Analytics on signal performance and reach
- **FOMO Creation**: Strategic messaging about premium access benefits

## 🏗️ Architecture

### Signal Distribution Hub
```
┌─────────────────────────────────────────────────────────────┐
│                 SIGNAL DISTRIBUTION HUB                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Signal Engine   │    │ Distribution    │                │
│  │                 │    │ Controller      │                │
│  │ • Real-time AI  │────│                 │────┐           │
│  │ • Market Scan   │    │ • Rate Limiting │    │           │
│  │ • Validation    │    │ • NFT Gating    │    │           │
│  │ • Confidence    │    │ • Channel Route │    │           │
│  └─────────────────┘    └─────────────────┘    │           │
│                                   │             │           │
│  ┌─────────────────┐    ┌─────────▼───────┐    ▼           │
│  │ Social Channel  │    │ Premium Platform│ ┌─────────────┐ │
│  │                 │    │                 │ │Fee Treasury │ │
│  │ • 2/hour limit  │    │ • Real-time     │ │             │ │
│  │ • Public Blinks │    │ • NFT Gated     │ │ • 1% of all │ │
│  │ • User Acquire  │    │ • Full Access   │ │   trades    │ │
│  │ • Growth Focus  │    │ • Retention     │ │ • Automatic │ │
│  └─────────────────┘    └─────────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Backend Architecture (FastAPI + Supabase)
```
backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── core/                   # Core configuration and utilities
│   │   ├── config.py          # Pydantic settings management
│   │   ├── logging.py         # Structured logging setup
│   │   └── security.py        # NFT verification & JWT handling
│   ├── models/                 # Pydantic data models
│   │   ├── signals.py         # Signal data structures
│   │   ├── users.py           # User and NFT holder models
│   │   └── blinks.py          # Blink action models
│   ├── routers/                # API endpoint routers
│   │   ├── auth.py            # NFT-based authentication
│   │   ├── signals.py         # Signal generation & distribution
│   │   ├── blinks.py          # Solana Blinks creation
│   │   ├── social.py          # Social media automation
│   │   ├── premium.py         # NFT holder exclusive features
│   │   └── health.py          # Health check endpoints
│   ├── services/               # Business logic services
│   │   ├── signal_agent.py    # AI signal generation engine
│   │   ├── market_analyzer.py # Technical analysis & data processing
│   │   ├── nft_verifier.py    # Solana NFT ownership verification
│   │   ├── blink_factory.py   # Solana Action generation
│   │   ├── social_publisher.py # Automated social media posting
│   │   └── fee_collector.py   # Revenue tracking & treasury management
│   └── workers/                # Background task workers
│       ├── signal_worker.py   # Continuous signal generation
│       ├── social_worker.py   # Scheduled social posting
│       └── performance_tracker.py # Signal outcome monitoring
├── requirements.txt            # Python dependencies
├── Dockerfile                 # Container configuration
└── alembic/                   # Database migrations
```

### Key Technologies
- **FastAPI**: High-performance async web framework with automatic API docs
- **Supabase**: PostgreSQL database with real-time subscriptions
- **Solana Web3.py**: Blockchain interaction and NFT verification
- **Jupiter API**: DEX aggregation for trading with fee collection
- **Redis**: Caching, rate limiting, and background job queue
- **Discord/Twitter APIs**: Social media automation and distribution
- **Celery**: Distributed task queue for background processing

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Supabase account and project
- Solana RPC access (Helius/QuickNode recommended)
- Discord/Twitter developer accounts
- Redis instance (local or cloud)

### 1. Clone Repository
```bash
git clone <repository-url>
cd "Play Buni Platform"
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp ../env.example .env
# Edit .env with your configuration
```

### 3. Environment Configuration

Copy `env.example` to `.env` and configure:

```bash
# Supabase Configuration
SUPABASE_URL="your_supabase_project_url"
SUPABASE_ANON_KEY="your_supabase_anon_key"
SUPABASE_SERVICE_ROLE_KEY="your_supabase_service_role_key"

# Solana Configuration
SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
NFT_COLLECTION_ADDRESS="your_nft_collection_mint_address"
TREASURY_WALLET_ADDRESS="your_treasury_wallet_address"
BOT_WALLET_PRIVATE_KEY="your_bot_wallet_private_key"

# Social Media APIs
DISCORD_WEBHOOK_URL="your_discord_webhook_url"
TWITTER_API_KEY="your_twitter_api_key"
TWITTER_API_SECRET="your_twitter_api_secret"
TWITTER_ACCESS_TOKEN="your_twitter_access_token"
TWITTER_ACCESS_SECRET="your_twitter_access_secret"

# External Market Data
JUPITER_API_KEY="your_jupiter_api_key"
COINGECKO_API_KEY="your_coingecko_api_key"
BIRDEYE_API_KEY="your_birdeye_api_key"

# Redis Configuration
REDIS_URL="redis://localhost:6379"

# Security
JWT_SECRET_KEY="your_super_secret_jwt_key"
API_KEY_SALT="your_api_key_salt"
```

### 4. Database Setup
```bash
# Run database migrations
alembic upgrade head

# Seed initial data (optional)
python scripts/seed_database.py
```

### 5. Run Development Server
```bash
# Start Redis (if running locally)
redis-server

# Start background workers
celery -A app.workers.signal_worker worker --loglevel=info

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access Application
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Signal Feed**: http://localhost:8000/signals/feed
- **Premium Dashboard**: http://localhost:8000/premium/dashboard

### Developer Setup
Use the following steps for a local developer environment:
1. From the `backend` folder create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If some packages fail to build in your environment, install the minimal
   dependencies needed for the test suite:
   ```bash
   pip install fastapi pydantic pydantic-settings uvicorn structlog aiohttp sqlalchemy
   ```
3. Copy `env.example` to `.env` and update the values.
4. Run the test suite to ensure everything works:
   ```bash
   pytest
   ```

## 📊 Platform Features

### 🤖 AI Signal Agent
The core intelligence system that powers all signal generation:

**Market Intelligence**
- Continuous monitoring of 200+ Solana tokens
- Real-time price feeds from Jupiter, Raydium, Orca
- Volume analysis and liquidity depth tracking
- Social sentiment monitoring across Twitter/Discord

**Analysis Engine**
- Multi-layer technical analysis (RSI, MACD, Bollinger Bands)
- Pattern recognition based on historical performance
- Whale activity and smart money flow tracking
- Market context and volatility assessment

**Decision Framework**
- Conservative entry with aggressive profit-taking
- Risk assessment and confidence scoring
- Timing optimization for maximum user success
- Performance feedback loop for continuous improvement

### 🔐 NFT-Gated Access Control
Secure and transparent access management:

**Verification System**
- Real-time Solana NFT ownership verification
- Support for multiple NFT collections
- Automatic access revocation on NFT transfer
- Session management with JWT tokens

**Access Tiers**
- Free Users: 2 signals per hour on social media
- NFT Holders: Unlimited real-time platform access
- Premium Features: Exclusive analytics and community access
- Future Tiers: Additional NFT collections for enhanced features

### 📈 Signal Generation & Distribution
Sophisticated signal processing and delivery:

**Generation Process**
- Real-time market scanning every 30 seconds
- Multi-factor confidence scoring (0-100%)
- Quality validation before distribution
- Performance tracking and outcome analysis

**Distribution Channels**
- Social Media: Automated Twitter/Discord posting (2/hour)
- Premium Platform: Real-time signal feed for NFT holders
- Push Notifications: Instant alerts for high-confidence signals
- API Access: Programmatic signal consumption for developers

### 🔗 Solana Blinks Integration
Seamless trading execution with revenue generation:

**Blink Creation**
- Custom Solana Actions for each signal
- Embedded 1% fee collection to treasury
- Branded UI with platform identity
- Error handling and transaction retry logic

**Revenue Model**
- 1% fee on all trade executions
- Automatic treasury wallet collection
- Transparent fee disclosure to users
- Revenue analytics and reporting

### 📱 Social Media Automation
Viral growth through strategic content distribution:

**Content Strategy**
- Profit-focused messaging with clear CTAs
- Performance comparisons to drive FOMO
- Educational content mixed with signals
- Community engagement and response management

**Platform Integration**
- Twitter: Automated tweet posting with Blinks
- Discord: Webhook integration for community channels
- Telegram: Bot integration for signal distribution
- Analytics: Engagement tracking and optimization

## 🛠️ Development

### Project Structure
```
Play Buni Platform/
├── backend/                    # FastAPI backend application
│   ├── app/                   # Main application code
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Container configuration
│   └── alembic/              # Database migrations
├── frontend/                   # React dashboard (future)
│   ├── src/                  # React components
│   ├── public/               # Static assets
│   └── package.json          # Node.js dependencies
├── infrastructure/             # Deployment configurations
│   ├── docker-compose.yml    # Local development setup
│   ├── nginx/                # Reverse proxy configuration
│   └── monitoring/           # Logging and metrics
├── scripts/                    # Utility scripts
│   ├── seed_database.py      # Database seeding
│   ├── deploy.sh            # Deployment automation
│   └── backup.sh            # Data backup utilities
├── docs/                      # Documentation
│   ├── api.md               # API documentation
│   ├── deployment.md        # Deployment guide
│   └── architecture.md      # System architecture
├── env.example               # Environment template
└── README.md                 # This file
```

### API Endpoints

#### Authentication & Access Control
- `POST /auth/verify-nft` - Verify NFT ownership and create session
- `POST /auth/refresh` - Refresh JWT token
- `GET /auth/profile` - Get user profile and access level
- `DELETE /auth/logout` - Invalidate session

#### Signal Management
- `GET /signals/` - Get public signals (rate limited)
- `GET /signals/premium` - Get real-time signals (NFT holders only)
- `GET /signals/{signal_id}` - Get specific signal details
- `POST /signals/feedback` - Submit signal performance feedback

#### Blinks & Trading
- `POST /blinks/create` - Create trading Blink for signal
- `GET /blinks/{blink_id}` - Get Blink details and status
- `POST /blinks/execute` - Execute trade via Blink
- `GET /blinks/history` - Get user's trading history

#### Social Media
- `POST /social/post-signal` - Manually post signal to social media
- `GET /social/status` - Get posting status and queue
- `GET /social/analytics` - Get engagement metrics

#### Premium Features (NFT Holders Only)
- `GET /premium/dashboard` - Premium dashboard data
- `GET /premium/analytics` - Advanced performance analytics
- `GET /premium/community` - Community features and discussions
- `POST /premium/alerts` - Configure custom signal alerts

#### Health & Monitoring
- `GET /health/` - Comprehensive health check
- `GET /health/ready` - Readiness probe for load balancers
- `GET /health/live` - Liveness probe for monitoring
- `GET /metrics` - Prometheus metrics endpoint

## 🔧 Configuration

### Required Environment Variables

See `env.example` for complete configuration template with descriptions.

### Database Schema

The platform uses Supabase (PostgreSQL) with the following core tables:

**Signals Management**
- `signals`: Core signal data with confidence scores and targets
- `signal_performance`: Tracking outcomes and ROI for each signal
- `signal_queue`: Social media posting queue with rate limiting

**User Management**
- `users`: User profiles and authentication data
- `nft_holders`: NFT ownership verification and access levels
- `user_sessions`: JWT session management and tracking

**Trading & Revenue**
- `blinks`: Generated Solana Actions and execution tracking
- `trades`: User trade executions and performance
- `revenue`: Fee collection and treasury management

**Social & Analytics**
- `social_posts`: Posted signals and engagement metrics
- `platform_analytics`: System performance and user behavior
- `feedback`: User feedback and signal quality ratings

### Solana Integration

**RPC Configuration**
- Mainnet/Devnet support with automatic failover
- Rate limiting and request optimization
- Connection pooling for high throughput

**NFT Verification**
- Collection-based access control
- Real-time ownership verification
- Support for multiple NFT standards

**Blinks & Actions**
- Jupiter integration for optimal swap routing
- Automatic fee embedding in all transactions
- Custom UI generation for branded experience

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=3
```

### Production Deployment
```bash
# Deploy to VPS
./scripts/deploy.sh production

# Monitor deployment
./scripts/monitor.sh

# Backup data
./scripts/backup.sh
```

### Environment-Specific Configurations

**Development**
- Local Redis and PostgreSQL
- Debug logging enabled
- Hot reload for development
- Mock social media APIs

**Staging**
- Solana Devnet for testing
- Limited social media posting
- Performance monitoring
- Automated testing suite

**Production**
- Solana Mainnet with backup RPCs
- Full social media integration
- Comprehensive monitoring
- Automated backups and scaling

## 📈 Revenue Model & Economics

### Fee Structure
- **Universal Fee**: 1% on all Blink trade executions
- **Transparent Collection**: Clearly displayed in all interfaces
- **Automatic Processing**: Seamless integration with Solana Actions
- **Treasury Management**: Automated collection and allocation

### Revenue Streams
1. **Primary**: Trading fees from Blink executions
2. **Secondary**: Premium API access for developers
3. **Future**: Revenue sharing with NFT holders
4. **Potential**: White-label licensing to other platforms

### Growth Projections
- **Month 1**: $1K-5K revenue from early adopters
- **Month 6**: $25K-50K revenue with established user base
- **Month 12**: $100K-250K revenue with viral growth
- **Year 2**: $500K+ revenue with platform maturity

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow coding standards and add tests
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request with detailed description

### Code Standards
- Python: Follow PEP 8 with Black formatting
- FastAPI: Use async/await patterns consistently
- Testing: Minimum 80% code coverage required
- Documentation: Docstrings for all public functions

### Testing
```bash
# Run test suite
pytest

# Run with coverage
pytest --cov=app

# Run specific test category
pytest -m "unit" # or "integration"
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support & Community

- **Documentation**: Comprehensive API docs at `/docs` endpoint
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discord**: Join our community Discord for real-time support
- **Email**: support@playbuni.com for direct assistance

## 🔮 Roadmap

### Phase 1: Core Platform (Q1 2025)
- ✅ NFT-gated signal distribution
- ✅ Solana Blinks integration
- 🔄 Social media automation
- 🔄 Premium dashboard development

### Phase 2: Advanced Features (Q2 2025)
- 📋 Mobile app development
- 📋 Advanced analytics dashboard
- 📋 Community features and governance
- 📋 API marketplace for developers

### Phase 3: Platform Expansion (Q3 2025)
- 📋 Multi-chain support (Ethereum, Base)
- 📋 White-label licensing program
- 📋 Institutional features and APIs
- 📋 Revenue sharing with NFT holders

### Phase 4: Ecosystem Growth (Q4 2025)
- 📋 Partner integrations and collaborations
- 📋 Advanced AI features and personalization
- 📋 Global expansion and localization
- 📋 IPO preparation and scaling

---

<div align="center">
  <p>Built with ❤️ by the Play Buni Team</p>
  <p>Powered by Solana • FastAPI • Supabase • AI</p>
  <p><strong>Making Profitable Trading Accessible to Everyone</strong></p>
</div> 