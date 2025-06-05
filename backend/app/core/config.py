from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Application configuration settings."""

    # Application Configuration
    app_name: str = Field(default="Play Buni Platform", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")

    # Database Configuration (Supabase)
    supabase_url: str = Field(env="SUPABASE_URL")
    supabase_anon_key: str = Field(env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(env="SUPABASE_SERVICE_ROLE_KEY")
    database_url: str = Field(env="DATABASE_URL")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # Solana Configuration
    solana_rpc_url: str = Field(
        default="https://api.mainnet-beta.solana.com", env="SOLANA_RPC_URL"
    )
    solana_ws_url: str = Field(
        default="wss://api.mainnet-beta.solana.com", env="SOLANA_WS_URL"
    )
    solana_private_key: str = Field(env="SOLANA_PRIVATE_KEY")
    treasury_wallet: str = Field(env="TREASURY_WALLET")

    # NFT Collection Configuration
    nft_collection_address: str = Field(env="NFT_COLLECTION_ADDRESS")
    nft_creator_address: str = Field(env="NFT_CREATOR_ADDRESS")
    required_nft_collection: Optional[str] = Field(
        default=None, env="REQUIRED_NFT_COLLECTION"
    )
    vip_nft_threshold: int = Field(default=10, env="VIP_NFT_THRESHOLD")
    admin_wallets: str = Field(default="", env="ADMIN_WALLETS")  # Comma-separated list

    # Jupiter API Configuration (Multi-tier support)
    jupiter_free_endpoint: str = Field(
        default="https://www.jupiterapi.com", env="JUPITER_FREE_ENDPOINT"
    )
    quicknode_metis_endpoint: Optional[str] = Field(
        default=None, env="QUICKNODE_METIS_ENDPOINT"
    )
    jupiter_paid_endpoint: Optional[str] = Field(
        default=None, env="JUPITER_PAID_ENDPOINT"
    )
    jupiter_api_url: str = Field(
        default="https://quote-api.jup.ag/v6", env="JUPITER_API_URL"
    )
    jupiter_swap_api_url: str = Field(
        default="https://quote-api.jup.ag/v6/swap", env="JUPITER_SWAP_API_URL"
    )

    # Authentication & Security
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Social Media APIs
    twitter_api_key: str = Field(env="TWITTER_API_KEY")
    twitter_api_secret: str = Field(env="TWITTER_API_SECRET")
    twitter_access_token: str = Field(env="TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret: str = Field(env="TWITTER_ACCESS_TOKEN_SECRET")
    twitter_bearer_token: str = Field(env="TWITTER_BEARER_TOKEN")

    discord_bot_token: str = Field(env="DISCORD_BOT_TOKEN")
    discord_channel_id: str = Field(env="DISCORD_CHANNEL_ID")

    # External APIs
    coingecko_api_key: Optional[str] = Field(default=None, env="COINGECKO_API_KEY")
    birdeye_api_key: Optional[str] = Field(default=None, env="BIRDEYE_API_KEY")
    dexscreener_api_url: str = Field(
        default="https://api.dexscreener.com/latest", env="DEXSCREENER_API_URL"
    )
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # Monitoring & Logging
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Celery Configuration
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND"
    )

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    premium_rate_limit_per_minute: int = Field(
        default=1000, env="PREMIUM_RATE_LIMIT_PER_MINUTE"
    )
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")

    # Signal Configuration
    signal_processing_interval: int = Field(
        default=30, env="SIGNAL_PROCESSING_INTERVAL"
    )
    public_signals_per_hour: int = Field(default=2, env="PUBLIC_SIGNALS_PER_HOUR")
    max_tokens_to_monitor: int = Field(default=200, env="MAX_TOKENS_TO_MONITOR")

    # WebSocket and Real-time Configuration
    market_data_update_interval: int = Field(
        default=30, env="MARKET_DATA_UPDATE_INTERVAL"
    )  # seconds
    websocket_heartbeat_interval: int = Field(
        default=120, env="WEBSOCKET_HEARTBEAT_INTERVAL"
    )  # seconds
    connection_cleanup_interval: int = Field(
        default=300, env="CONNECTION_CLEANUP_INTERVAL"
    )  # seconds

    # Fee Configuration
    platform_fee_percentage: float = Field(default=1.0, env="PLATFORM_FEE_PERCENTAGE")
    min_trade_amount_sol: float = Field(default=0.01, env="MIN_TRADE_AMOUNT_SOL")

    # Blinks Configuration
    base_url: str = Field(default="https://api.playbuni.com", env="BASE_URL")
    blinks_enabled: bool = Field(default=True, env="BLINKS_ENABLED")
    max_blink_duration_hours: int = Field(default=24, env="MAX_BLINK_DURATION_HOURS")

    # Treasury Configuration
    treasury_wallet_address: str = Field(env="TREASURY_WALLET_ADDRESS")
    platform_fee_bps: int = Field(
        default=100, env="PLATFORM_FEE_BPS"
    )  # 1% fee (100 basis points)
    min_trade_amount: float = Field(
        default=1.0, env="MIN_TRADE_AMOUNT"
    )  # Minimum trade amount in USD
    max_trade_amount: float = Field(
        default=100000.0, env="MAX_TRADE_AMOUNT"
    )  # Maximum trade amount in USD

    @property
    def admin_wallets_list(self) -> List[str]:
        """Parse admin wallets from comma-separated string."""
        if not self.admin_wallets:
            return []
        return [
            wallet.strip() for wallet in self.admin_wallets.split(",") if wallet.strip()
        ]

    @property
    def allowed_origins(self) -> List[str]:
        """Get allowed CORS origins."""
        if self.debug:
            return ["*"]
        return [
            "https://playbuni.com",
            "https://www.playbuni.com",
            "https://app.playbuni.com",
        ]

    @property
    def allowed_hosts(self) -> List[str]:
        """Get allowed hosts for TrustedHostMiddleware."""
        if self.debug:
            return ["*"]
        return [
            "playbuni.com",
            "www.playbuni.com",
            "app.playbuni.com",
            "api.playbuni.com",
        ]

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
