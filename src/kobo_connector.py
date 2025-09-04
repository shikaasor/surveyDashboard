"""
KoboToolbox API connector for fetching form data.

This module handles authentication and data retrieval from KoboToolbox API,
including export settings resolution and XLSX download with proper error handling.
"""

import os
import time
import requests
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils.logging_utils import setup_logger, log_api_request, log_error_with_context

logger = setup_logger('kobo_connector')


class KoboAPIError(Exception):
    """Custom exception for KoboToolbox API errors."""
    pass


class KoboAPI:
    """
    KoboToolbox API client for authentication and data retrieval.
    
    Handles export settings resolution, XLSX download, caching, and retry logic
    as per FR-1, FR-2, FR-3 requirements.
    """
    
    def __init__(self, base_url: str, asset_uid: str, api_token: str):
        """
        Initialize KoboAPI client.
        
        Args:
            base_url: KoboToolbox server base URL
            asset_uid: Asset UID for the form
            api_token: API authentication token
        """
        self.base_url = base_url.rstrip('/')
        self.asset_uid = asset_uid
        self.api_token = api_token
        
        # Setup session with retry strategy
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {api_token}",
            "User-Agent": "KoboAnalyticsDashboard/1.0"
        })
        
        # Configure retry strategy for resilience (NFR-3)
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized KoboAPI client for asset: {asset_uid}")
    
    def _make_request(self, url: str, method: str = "GET", **kwargs) -> requests.Response:
        """
        Make authenticated request with proper error handling.
        
        Args:
            url: Request URL
            method: HTTP method
            **kwargs: Additional request parameters
            
        Returns:
            Response object
            
        Raises:
            KoboAPIError: If request fails
        """
        start_time = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            log_api_request(logger, method, url, response.status_code, response_time)
            
            return response
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            log_api_request(logger, method, url, 
                          getattr(e.response, 'status_code', None), response_time)
            
            context = {
                'url': url,
                'method': method,
                'asset_uid': self.asset_uid
            }
            log_error_with_context(logger, e, context)
            
            raise KoboAPIError(f"API request failed: {str(e)}") from e
    
    def get_export_settings(self, export_settings_uid: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve export settings for the asset.
        
        Args:
            export_settings_uid: Specific export settings UID (optional)
            
        Returns:
            Export settings dictionary
            
        Raises:
            KoboAPIError: If no export settings found
        """
        if export_settings_uid:
            url = f"{self.base_url}/api/v2/assets/{self.asset_uid}/export-settings/{export_settings_uid}/"
        else:
            url = f"{self.base_url}/api/v2/assets/{self.asset_uid}/export-settings/"
        
        response = self._make_request(url)
        data = response.json()
        
        if export_settings_uid:
            return data
        else:
            # Return the first export setting if multiple exist
            if "results" in data and data["results"]:
                logger.info(f"Found {len(data['results'])} export settings, using first one")
                return data["results"][0]
            else:
                raise KoboAPIError("No export settings found for this asset")
    
    def download_data_xlsx(self, export_settings_uid: Optional[str] = None, 
                          cache_dir: str = "data/cache") -> Tuple[str, Dict[str, Any]]:
        """
        Download XLSX data with caching support.
        
        Args:
            export_settings_uid: Specific export settings UID (optional)
            cache_dir: Directory for caching downloaded files
            
        Returns:
            Tuple of (cache_file_path, export_settings)
            
        Raises:
            KoboAPIError: If download fails
        """
        # Get export settings to find XLSX URL
        settings = self.get_export_settings(export_settings_uid)
        
        # Get XLSX URL
        xlsx_url = settings.get("data_url_xlsx")
        if not xlsx_url:
            raise KoboAPIError("No XLSX export URL found in export settings")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = os.path.join(cache_dir, f"{self.asset_uid}_{timestamp}.xlsx")
        
        # Download the file
        logger.info(f"Downloading XLSX data from KoboToolbox...")
        response = self._make_request(xlsx_url)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            f.write(response.content)
        
        file_size = len(response.content)
        logger.info(f"Downloaded {file_size:,} bytes to {cache_file}")
        
        # Also create a "latest" symlink for easy access
        latest_file = os.path.join(cache_dir, f"{self.asset_uid}_latest.xlsx")
        if os.path.exists(latest_file):
            os.remove(latest_file)
        
        # Copy instead of symlink for Windows compatibility
        with open(cache_file, 'rb') as src, open(latest_file, 'wb') as dst:
            dst.write(src.read())
        
        return cache_file, settings
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/v2/assets/{self.asset_uid}/"
            response = self._make_request(url)
            
            logger.info("API connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            return False
    
    def get_asset_info(self) -> Dict[str, Any]:
        """
        Get basic information about the asset.
        
        Returns:
            Asset information dictionary
        """
        url = f"{self.base_url}/api/v2/assets/{self.asset_uid}/"
        response = self._make_request(url)
        return response.json()
    
    def cleanup_old_cache_files(self, cache_dir: str = "data/cache", 
                               keep_count: int = 5) -> None:
        """
        Clean up old cache files, keeping only the most recent ones.
        
        Args:
            cache_dir: Cache directory path
            keep_count: Number of recent files to keep
        """
        if not os.path.exists(cache_dir):
            return
        
        # Find all cache files for this asset
        cache_files = []
        prefix = f"{self.asset_uid}_"
        
        for filename in os.listdir(cache_dir):
            if filename.startswith(prefix) and filename.endswith('.xlsx') and 'latest' not in filename:
                filepath = os.path.join(cache_dir, filename)
                cache_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (newest first)
        cache_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old files
        for filepath, _ in cache_files[keep_count:]:
            try:
                os.remove(filepath)
                logger.info(f"Removed old cache file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {filepath}: {str(e)}")
        
        if len(cache_files) > keep_count:
            logger.info(f"Cleaned up {len(cache_files) - keep_count} old cache files")