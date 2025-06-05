"""
NFT Verification Service for Play Buni Platform

This service handles Solana NFT ownership verification using the latest
Metaplex Token Metadata program and Solana Python SDK.

Features:
- Real-time NFT ownership verification
- Metaplex metadata parsing
- Collection verification
- Batch verification for multiple NFTs
- Caching for performance optimization
- Rate limiting protection
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TokenAccountOpts
from solders.pubkey import Pubkey
from solders.rpc.responses import GetAccountInfoResp
import base58

from app.core.config import settings
from app.core.cache import cache_manager
from app.core.logging import get_logger

logger = get_logger(__name__)

# Metaplex Token Metadata Program ID
TOKEN_METADATA_PROGRAM_ID = Pubkey.from_string(
    "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
)

# SPL Token Program ID
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

# Associated Token Program ID
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string(
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
)


class NFTStandard(Enum):
    """NFT Token Standards"""

    NON_FUNGIBLE = "NonFungible"
    NON_FUNGIBLE_EDITION = "NonFungibleEdition"
    FUNGIBLE_ASSET = "FungibleAsset"
    FUNGIBLE = "Fungible"
    PROGRAMMABLE_NON_FUNGIBLE = "ProgrammableNonFungible"


class VerificationStatus(Enum):
    """NFT Verification Status"""

    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    INVALID_OWNER = "invalid_owner"
    INVALID_METADATA = "invalid_metadata"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"


@dataclass
class NFTMetadata:
    """NFT Metadata Structure"""

    mint: str
    name: str
    symbol: str
    uri: str
    update_authority: str
    creators: List[Dict[str, Any]]
    seller_fee_basis_points: int
    primary_sale_happened: bool
    is_mutable: bool
    token_standard: Optional[NFTStandard]
    collection: Optional[Dict[str, Any]]
    uses: Optional[Dict[str, Any]]

    # Off-chain metadata
    description: Optional[str] = None
    image: Optional[str] = None
    animation_url: Optional[str] = None
    external_url: Optional[str] = None
    attributes: Optional[List[Dict[str, Any]]] = None
    properties: Optional[Dict[str, Any]] = None


@dataclass
class NFTOwnership:
    """NFT Ownership Information"""

    mint: str
    owner: str
    amount: int
    delegate: Optional[str]
    state: str
    is_native: bool
    rent_exempt_reserve: Optional[int]
    delegated_amount: int
    close_authority: Optional[str]


@dataclass
class VerificationResult:
    """NFT Verification Result"""

    status: VerificationStatus
    is_verified: bool
    mint: str
    owner: Optional[str]
    metadata: Optional[NFTMetadata]
    ownership: Optional[NFTOwnership]
    error_message: Optional[str]
    verified_at: datetime
    collection_verified: bool = False


class NFTVerificationService:
    """
    NFT Verification Service

    Handles real-time verification of Solana NFT ownership using the latest
    Metaplex Token Metadata program and Solana RPC methods.
    """

    def __init__(self):
        self.rpc_client = AsyncClient(settings.SOLANA_RPC_URL)
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = {}  # Simple rate limiting

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        await self.rpc_client.close()

    def _get_metadata_pda(self, mint_pubkey: Pubkey) -> Pubkey:
        """
        Get the Program Derived Address (PDA) for NFT metadata

        Args:
            mint_pubkey: The mint public key

        Returns:
            The metadata PDA
        """
        seeds = [b"metadata", bytes(TOKEN_METADATA_PROGRAM_ID), bytes(mint_pubkey)]

        pda, _ = Pubkey.find_program_address(seeds, TOKEN_METADATA_PROGRAM_ID)
        return pda

    def _get_associated_token_address(self, owner: Pubkey, mint: Pubkey) -> Pubkey:
        """
        Get the Associated Token Account address

        Args:
            owner: Owner's public key
            mint: Mint public key

        Returns:
            Associated token account address
        """
        seeds = [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)]

        pda, _ = Pubkey.find_program_address(seeds, ASSOCIATED_TOKEN_PROGRAM_ID)
        return pda

    async def _fetch_account_info(self, pubkey: Pubkey) -> Optional[GetAccountInfoResp]:
        """
        Fetch account information from Solana RPC

        Args:
            pubkey: Public key to fetch

        Returns:
            Account info response or None if not found
        """
        try:
            response = await self.rpc_client.get_account_info(pubkey)
            if response.value is None:
                return None
            return response
        except Exception as e:
            logger.error(f"Error fetching account info for {pubkey}: {e}")
            return None

    def _parse_metadata_account(self, account_data: bytes) -> Optional[NFTMetadata]:
        """
        Parse Metaplex metadata account data

        Args:
            account_data: Raw account data bytes

        Returns:
            Parsed NFT metadata or None if parsing fails
        """
        try:
            # Skip the first byte (account discriminator)
            data = account_data[1:]

            # Parse metadata structure
            # This is a simplified parser - in production, use the official Metaplex SDK
            offset = 0

            # Update Authority (32 bytes)
            update_authority = base58.b58encode(data[offset : offset + 32]).decode()
            offset += 32

            # Mint (32 bytes)
            mint = base58.b58encode(data[offset : offset + 32]).decode()
            offset += 32

            # Name length (4 bytes) + name
            name_len = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            name = data[offset : offset + name_len].decode("utf-8").rstrip("\x00")
            offset += name_len

            # Symbol length (4 bytes) + symbol
            symbol_len = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            symbol = data[offset : offset + symbol_len].decode("utf-8").rstrip("\x00")
            offset += symbol_len

            # URI length (4 bytes) + URI
            uri_len = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            uri = data[offset : offset + uri_len].decode("utf-8").rstrip("\x00")
            offset += uri_len

            # Seller fee basis points (2 bytes)
            seller_fee_basis_points = int.from_bytes(
                data[offset : offset + 2], "little"
            )
            offset += 2

            # Creators (optional)
            has_creators = data[offset] == 1
            offset += 1
            creators = []

            if has_creators:
                creator_count = int.from_bytes(data[offset : offset + 4], "little")
                offset += 4

                for _ in range(creator_count):
                    creator_address = base58.b58encode(
                        data[offset : offset + 32]
                    ).decode()
                    offset += 32
                    verified = data[offset] == 1
                    offset += 1
                    share = data[offset]
                    offset += 1

                    creators.append(
                        {
                            "address": creator_address,
                            "verified": verified,
                            "share": share,
                        }
                    )

            # Primary sale happened (1 byte)
            primary_sale_happened = data[offset] == 1
            offset += 1

            # Is mutable (1 byte)
            is_mutable = data[offset] == 1
            offset += 1

            # Optional: edition nonce (1 byte) - skip if present
            if len(data) > offset:
                _ = data[offset]
                offset += 1

            token_standard = None
            if len(data) > offset:
                try:
                    token_standard_idx = data[offset]
                    token_standard = list(NFTStandard)[token_standard_idx]
                except Exception:
                    token_standard = None
                offset += 1

            collection = None
            if len(data) > offset:
                has_collection = data[offset] == 1
                offset += 1
                if has_collection and len(data) >= offset + 33:
                    collection_key = base58.b58encode(
                        data[offset : offset + 32]
                    ).decode()
                    offset += 32
                    verified = data[offset] == 1
                    offset += 1
                    collection = {"key": collection_key, "verified": bool(verified)}

            return NFTMetadata(
                mint=mint,
                name=name,
                symbol=symbol,
                uri=uri,
                update_authority=update_authority,
                creators=creators,
                seller_fee_basis_points=seller_fee_basis_points,
                primary_sale_happened=primary_sale_happened,
                is_mutable=is_mutable,
                token_standard=token_standard,
                collection=collection,
                uses=None,  # Not parsed
            )

        except Exception as e:
            logger.error(f"Error parsing metadata account: {e}")
            return None

    async def _fetch_off_chain_metadata(self, uri: str) -> Dict[str, Any]:
        """
        Fetch off-chain metadata from URI

        Args:
            uri: Metadata URI

        Returns:
            Off-chain metadata dictionary
        """
        if not self.session:
            return {}

        try:
            # Check cache first
            cache_key = f"metadata_uri:{uri}"
            cached = await cache_manager.get(cache_key)
            if cached:
                return json.loads(cached)

            async with self.session.get(uri, timeout=10) as response:
                if response.status == 200:
                    metadata = await response.json()

                    # Cache for 1 hour
                    await cache_manager.set(
                        cache_key, json.dumps(metadata), expire=3600
                    )

                    return metadata
                else:
                    logger.warning(
                        f"Failed to fetch metadata from {uri}: {response.status}"
                    )
                    return {}

        except Exception as e:
            logger.error(f"Error fetching off-chain metadata from {uri}: {e}")
            return {}

    async def _get_token_account_by_owner(
        self, owner: Pubkey, mint: Pubkey
    ) -> Optional[NFTOwnership]:
        """
        Get token account information for a specific owner and mint

        Args:
            owner: Owner's public key
            mint: Mint public key

        Returns:
            NFT ownership information or None
        """
        try:
            # Get associated token account
            ata = self._get_associated_token_address(owner, mint)
            account_info = await self._fetch_account_info(ata)

            if not account_info or not account_info.value:
                return None

            # Parse token account data
            data = account_info.value.data
            if len(data) < 165:  # Token account should be 165 bytes
                return None

            # Parse token account structure
            mint_bytes = data[0:32]
            owner_bytes = data[32:64]
            amount = int.from_bytes(data[64:72], "little")

            # Verify mint matches
            if mint_bytes != bytes(mint):
                return None

            # Verify owner matches
            if owner_bytes != bytes(owner):
                return None

            return NFTOwnership(
                mint=str(mint),
                owner=str(owner),
                amount=amount,
                delegate=None,  # Would need additional parsing
                state="initialized",
                is_native=False,
                rent_exempt_reserve=None,
                delegated_amount=0,
                close_authority=None,
            )

        except Exception as e:
            logger.error(f"Error getting token account: {e}")
            return None

    async def verify_nft_ownership(
        self, mint_address: str, owner_address: str
    ) -> VerificationResult:
        """
        Verify NFT ownership for a specific mint and owner

        Args:
            mint_address: NFT mint address
            owner_address: Wallet address to verify

        Returns:
            Verification result with detailed information
        """
        try:
            # Convert addresses to Pubkey objects
            mint_pubkey = Pubkey.from_string(mint_address)
            owner_pubkey = Pubkey.from_string(owner_address)

            # Check rate limiting
            rate_key = f"{owner_address}:{mint_address}"
            if rate_key in self.rate_limiter:
                last_check = self.rate_limiter[rate_key]
                if datetime.now() - last_check < timedelta(seconds=1):
                    return VerificationResult(
                        status=VerificationStatus.RATE_LIMITED,
                        is_verified=False,
                        mint=mint_address,
                        owner=owner_address,
                        metadata=None,
                        ownership=None,
                        error_message="Rate limited",
                        verified_at=datetime.now(),
                    )

            self.rate_limiter[rate_key] = datetime.now()

            # Check cache first
            cache_key = f"nft_verification:{mint_address}:{owner_address}"
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                result_data = json.loads(cached_result)
                return VerificationResult(**result_data)

            # Get metadata PDA
            metadata_pda = self._get_metadata_pda(mint_pubkey)

            # Fetch metadata account
            metadata_account = await self._fetch_account_info(metadata_pda)
            if not metadata_account or not metadata_account.value:
                return VerificationResult(
                    status=VerificationStatus.NOT_FOUND,
                    is_verified=False,
                    mint=mint_address,
                    owner=owner_address,
                    metadata=None,
                    ownership=None,
                    error_message="NFT metadata not found",
                    verified_at=datetime.now(),
                )

            # Parse metadata
            metadata = self._parse_metadata_account(metadata_account.value.data)
            if not metadata:
                return VerificationResult(
                    status=VerificationStatus.INVALID_METADATA,
                    is_verified=False,
                    mint=mint_address,
                    owner=owner_address,
                    metadata=None,
                    ownership=None,
                    error_message="Invalid metadata format",
                    verified_at=datetime.now(),
                )

            # Fetch off-chain metadata
            if metadata.uri:
                off_chain_data = await self._fetch_off_chain_metadata(metadata.uri)
                metadata.description = off_chain_data.get("description")
                metadata.image = off_chain_data.get("image")
                metadata.animation_url = off_chain_data.get("animation_url")
                metadata.external_url = off_chain_data.get("external_url")
                metadata.attributes = off_chain_data.get("attributes", [])
                metadata.properties = off_chain_data.get("properties", {})

            # Verify ownership
            ownership = await self._get_token_account_by_owner(
                owner_pubkey, mint_pubkey
            )
            if not ownership or ownership.amount == 0:
                return VerificationResult(
                    status=VerificationStatus.INVALID_OWNER,
                    is_verified=False,
                    mint=mint_address,
                    owner=owner_address,
                    metadata=metadata,
                    ownership=ownership,
                    error_message="NFT not owned by specified address",
                    verified_at=datetime.now(),
                )

            # Successful verification
            result = VerificationResult(
                status=VerificationStatus.VERIFIED,
                is_verified=True,
                mint=mint_address,
                owner=owner_address,
                metadata=metadata,
                ownership=ownership,
                error_message=None,
                verified_at=datetime.now(),
                collection_verified=bool(metadata.collection),
            )

            # Cache result for 5 minutes
            await cache_manager.set(
                cache_key, json.dumps(result.__dict__, default=str), expire=300
            )

            return result

        except Exception as e:
            logger.error(f"Error verifying NFT ownership: {e}")
            return VerificationResult(
                status=VerificationStatus.NETWORK_ERROR,
                is_verified=False,
                mint=mint_address,
                owner=owner_address,
                metadata=None,
                ownership=None,
                error_message=str(e),
                verified_at=datetime.now(),
            )

    async def verify_multiple_nfts(
        self, mint_addresses: List[str], owner_address: str
    ) -> List[VerificationResult]:
        """
        Verify ownership of multiple NFTs for a single owner

        Args:
            mint_addresses: List of NFT mint addresses
            owner_address: Wallet address to verify

        Returns:
            List of verification results
        """
        tasks = [
            self.verify_nft_ownership(mint_address, owner_address)
            for mint_address in mint_addresses
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        verified_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                verified_results.append(
                    VerificationResult(
                        status=VerificationStatus.NETWORK_ERROR,
                        is_verified=False,
                        mint=mint_addresses[i],
                        owner=owner_address,
                        metadata=None,
                        ownership=None,
                        error_message=str(result),
                        verified_at=datetime.now(),
                    )
                )
            else:
                verified_results.append(result)

        return verified_results

    async def get_nfts_by_owner(
        self, owner_address: str, limit: int = 100
    ) -> List[str]:
        """
        Get all NFT mint addresses owned by a wallet

        Args:
            owner_address: Wallet address
            limit: Maximum number of NFTs to return

        Returns:
            List of NFT mint addresses
        """
        try:
            owner_pubkey = Pubkey.from_string(owner_address)

            # Get all token accounts for the owner
            response = await self.rpc_client.get_token_accounts_by_owner(
                owner_pubkey, TokenAccountOpts(program_id=TOKEN_PROGRAM_ID)
            )

            if not response.value:
                return []

            nft_mints = []
            for account in response.value[:limit]:
                try:
                    # Parse token account data
                    data = account.account.data
                    if len(data) >= 72:
                        # Get mint address (first 32 bytes)
                        mint_bytes = data[0:32]
                        mint_address = base58.b58encode(mint_bytes).decode()

                        # Get amount (bytes 64-72)
                        amount = int.from_bytes(data[64:72], "little")

                        # Only include tokens with amount = 1 (potential NFTs)
                        if amount == 1:
                            nft_mints.append(mint_address)

                except Exception as e:
                    logger.warning(f"Error parsing token account: {e}")
                    continue

            return nft_mints

        except Exception as e:
            logger.error(f"Error getting NFTs by owner: {e}")
            return []

    async def verify_collection_membership(
        self, mint_address: str, collection_address: str
    ) -> bool:
        """
        Verify if an NFT belongs to a specific collection

        Args:
            mint_address: NFT mint address
            collection_address: Collection mint address

        Returns:
            True if NFT belongs to collection
        """
        try:
            mint_pubkey = Pubkey.from_string(mint_address)
            metadata_pda = self._get_metadata_pda(mint_pubkey)

            metadata_account = await self._fetch_account_info(metadata_pda)
            if not metadata_account or not metadata_account.value:
                return False

            metadata = self._parse_metadata_account(metadata_account.value.data)
            if not metadata or not metadata.collection:
                return False

            return metadata.collection.get(
                "key"
            ) == collection_address and metadata.collection.get("verified", False)

        except Exception as e:
            logger.error(f"Error verifying collection membership: {e}")
            return False


# Global service instance
nft_verification_service = NFTVerificationService()


async def verify_wallet_nft_holdings(
    wallet_address: str,
    required_collection: Optional[str] = None,
    min_nft_count: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify if a wallet holds the required NFTs for platform access

    Args:
        wallet_address: Wallet address to verify
        required_collection: Optional collection address requirement
        min_nft_count: Minimum number of NFTs required

    Returns:
        Tuple of (is_verified, verification_details)
    """
    async with nft_verification_service as service:
        try:
            # Get all NFTs owned by the wallet
            nft_mints = await service.get_nfts_by_owner(wallet_address, limit=200)

            if len(nft_mints) < min_nft_count:
                return False, {
                    "verified": False,
                    "nft_count": len(nft_mints),
                    "required_count": min_nft_count,
                    "error": "Insufficient NFT holdings",
                }

            # If collection requirement specified, verify collection membership
            if required_collection:
                collection_nfts = []
                for mint in nft_mints:
                    is_member = await service.verify_collection_membership(
                        mint, required_collection
                    )
                    if is_member:
                        collection_nfts.append(mint)

                if len(collection_nfts) < min_nft_count:
                    return False, {
                        "verified": False,
                        "nft_count": len(nft_mints),
                        "collection_nft_count": len(collection_nfts),
                        "required_count": min_nft_count,
                        "required_collection": required_collection,
                        "error": "Insufficient collection NFT holdings",
                    }

                return True, {
                    "verified": True,
                    "nft_count": len(nft_mints),
                    "collection_nft_count": len(collection_nfts),
                    "verified_nfts": collection_nfts[:10],  # Return first 10
                    "collection": required_collection,
                }

            # No collection requirement, just verify NFT count
            return True, {
                "verified": True,
                "nft_count": len(nft_mints),
                "verified_nfts": nft_mints[:10],  # Return first 10
            }

        except Exception as e:
            logger.error(f"Error verifying wallet NFT holdings: {e}")
            return False, {"verified": False, "error": str(e)}
