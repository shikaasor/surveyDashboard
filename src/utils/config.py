"""
Configuration utility for loading environment variables and settings.

This module handles loading configuration from .env files and environment variables,
with proper defaults and validation.
"""

import os
from typing import Dict, Optional, Any
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file if it exists
load_dotenv()


def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dictionary containing configuration values
        
    Raises:
        ValueError: If required configuration is missing
    """
    config = {
        # KoboToolbox Configuration
        'KOBO_BASE_URL': os.getenv('KOBO_BASE_URL', 'https://kf.kobotoolbox.org'),
        'KOBO_ASSET_UID': os.getenv('KOBO_ASSET_UID'),
        'KOBO_API_TOKEN': os.getenv('KOBO_API_TOKEN'),
        'KOBO_EXPORT_SETTINGS_UID': os.getenv('KOBO_EXPORT_SETTINGS_UID'),
        
        # Application Configuration
        'REFRESH_INTERVAL': int(os.getenv('REFRESH_INTERVAL', '3600')),
        'ADMIN_PASSWORD': os.getenv('ADMIN_PASSWORD', 'admin'),
        
        # Logging Configuration
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'SENTRY_DSN': os.getenv('SENTRY_DSN'),
        
        # Deployment Configuration
        'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development'),
        'DEBUG': os.getenv('DEBUG', 'false').lower() == 'true',
    }
    
    # Validate required configuration
    required_fields = ['KOBO_ASSET_UID', 'KOBO_API_TOKEN']
    missing_fields = [field for field in required_fields if not config[field]]
    
    if missing_fields:
        st.error(f"Missing required configuration: {', '.join(missing_fields)}")
        st.info("Please check your .env file or environment variables.")
        st.stop()
    
    return config


def get_config_value(key: str, default: Optional[Any] = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = load_config()
    return config.get(key, default)


def is_development() -> bool:
    """Check if running in development environment."""
    return get_config_value('ENVIRONMENT') == 'development'


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return get_config_value('DEBUG', False)