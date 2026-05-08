"""
Spotify OAuth 2.0 Authentication Handler

Supports both Client Credentials Flow (app-only) and Authorization Code Flow (user-specific).
Handles token refresh and expiration automatically.
"""

import os
import json
import time
import base64
import httpx
from typing import Dict, Any, Optional
from pathlib import Path


class SpotifyAuthManager:
    """Manages Spotify OAuth 2.0 authentication and token refresh."""
    
    def __init__(self, 
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 refresh_token: Optional[str] = None,
                 cache_path: Optional[str] = None):
        """
        Initialize Spotify authentication manager.
        
        Args:
            client_id: Spotify app client ID (from env if not provided)
            client_secret: Spotify app client secret (from env if not provided)
            refresh_token: Optional refresh token for user authentication
            cache_path: Path to cache tokens (default: ~/.spotify_token_cache.json)
        """
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.refresh_token = refresh_token or os.getenv("SPOTIFY_REFRESH_TOKEN", "")
        
        # Token cache
        self.cache_path = cache_path or os.path.expanduser("~/.spotify_token_cache.json")
        self.access_token = None
        self.token_expires_at = 0
        
        # Try to load cached token
        self._load_cached_token()
    
    def _load_cached_token(self):
        """Load cached access token if still valid."""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    cache = json.load(f)
                    
                if cache.get('expires_at', 0) > time.time() + 60:  # 1 min buffer
                    self.access_token = cache.get('access_token')
                    self.token_expires_at = cache.get('expires_at', 0)
                    print(f"✅ Loaded cached Spotify token (expires in {int(self.token_expires_at - time.time())}s)")
        except Exception as e:
            print(f"⚠️  Failed to load cached token: {e}")
    
    def _save_token_cache(self, access_token: str, expires_in: int):
        """Save access token to cache."""
        try:
            self.access_token = access_token
            self.token_expires_at = time.time() + expires_in
            
            cache = {
                'access_token': access_token,
                'expires_at': self.token_expires_at
            }
            
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
                
            print(f"💾 Cached Spotify token (expires in {expires_in}s)")
        except Exception as e:
            print(f"⚠️  Failed to cache token: {e}")
    
    def _get_client_credentials_token(self) -> Optional[str]:
        """
        Get access token using Client Credentials Flow (app-only access).
        This flow doesn't require user authorization but has limited access.
        """
        if not self.client_id or not self.client_secret:
            print("❌ SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET required for authentication")
            return None
        
        try:
            # Encode credentials
            credentials = f"{self.client_id}:{self.client_secret}"
            credentials_b64 = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {credentials_b64}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "client_credentials"
            }
            
            response = httpx.post(
                "https://accounts.spotify.com/api/token",
                headers=headers,
                data=data,
                timeout=30.0
            )
            
            response.raise_for_status()
            token_info = response.json()
            
            access_token = token_info['access_token']
            expires_in = token_info['expires_in']
            
            self._save_token_cache(access_token, expires_in)
            
            print(f"✅ Obtained Spotify access token (Client Credentials)")
            return access_token
            
        except Exception as e:
            print(f"❌ Failed to get Spotify access token: {e}")
            return None
    
    def _refresh_access_token(self) -> Optional[str]:
        """
        Refresh access token using refresh token (Authorization Code Flow).
        This provides full access to user data.
        """
        if not self.client_id or not self.client_secret or not self.refresh_token:
            print("⚠️  No refresh token available, falling back to client credentials")
            return self._get_client_credentials_token()
        
        try:
            credentials = f"{self.client_id}:{self.client_secret}"
            credentials_b64 = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {credentials_b64}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token
            }
            
            response = httpx.post(
                "https://accounts.spotify.com/api/token",
                headers=headers,
                data=data,
                timeout=30.0
            )
            
            response.raise_for_status()
            token_info = response.json()
            
            access_token = token_info['access_token']
            expires_in = token_info['expires_in']
            
            self._save_token_cache(access_token, expires_in)
            
            print(f"✅ Refreshed Spotify access token")
            return access_token
            
        except Exception as e:
            print(f"❌ Failed to refresh Spotify token: {e}")
            # Fallback to client credentials
            return self._get_client_credentials_token()
    
    def get_access_token(self) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            Valid access token or None if authentication fails
        """
        # Check if we have a valid cached token
        if self.access_token and self.token_expires_at > time.time() + 60:
            return self.access_token
        
        # Check for manually provided token (backwards compatibility)
        manual_token = os.getenv("SPOTIFY_ACCESS_TOKEN", "")
        if manual_token:
            print("ℹ️  Using manually provided SPOTIFY_ACCESS_TOKEN")
            return manual_token
        
        # Try to refresh or get new token
        if self.refresh_token:
            return self._refresh_access_token()
        else:
            return self._get_client_credentials_token()
    
    def is_authenticated(self) -> bool:
        """Check if we can get a valid access token."""
        return self.get_access_token() is not None


# Global auth manager instance
_auth_manager = None


def get_spotify_auth_manager() -> SpotifyAuthManager:
    """Get or create global Spotify auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = SpotifyAuthManager()
    return _auth_manager


def get_spotify_access_token() -> Optional[str]:
    """
    Convenience function to get a valid Spotify access token.
    
    Returns:
        Valid access token or None
    """
    return get_spotify_auth_manager().get_access_token()
