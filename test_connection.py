#!/usr/bin/env python3
"""
KoboToolbox Connection Test Utility

This script tests the connection to KoboToolbox API and validates
that we can access export settings and download data.

Run this before starting the main dashboard to ensure everything works.
"""

import os
import sys
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kobo_connector import KoboAPI, KoboAPIError
from utils.config import load_config
from utils.logging_utils import setup_logger

def test_kobo_connection():
    """Test KoboToolbox API connection step by step."""
    
    print("🔍 KoboToolbox Connection Test")
    print("=" * 50)
    
    # Set up logging
    logger = setup_logger('connection_test')
    
    try:
        # Load configuration
        print("\n1. Loading configuration...")
        config = load_config()
        
        required_vars = ['KOBO_BASE_URL', 'KOBO_ASSET_UID', 'KOBO_API_TOKEN','KOBO_EXPORT_SETTINGS_UID']
        missing_vars = [var for var in required_vars if not config.get(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("\n💡 Please check your .env file or environment variables:")
            print("   - KOBO_BASE_URL (e.g., https://kf.kobotoolbox.org)")  
            print("   - KOBO_ASSET_UID (your form's asset UID)")
            print("   - KOBO_API_TOKEN (API token with view_submissions permission)")
            return False
        
        print(f"✅ Configuration loaded:")
        print(f"   - Base URL: {config['KOBO_BASE_URL']}")
        print(f"   - Asset UID: {config['KOBO_ASSET_UID']}")
        print(f"   - API Token: {'*' * 8}...{config['KOBO_API_TOKEN'][-4:]}")
        print(f"   - EXPORT UID: {config['KOBO_EXPORT_SETTINGS_UID']}")
        
        # Initialize API client
        print("\n2. Initializing KoboAPI client...")
        kobo = KoboAPI(
            base_url=config['KOBO_BASE_URL'],
            asset_uid=config['KOBO_ASSET_UID'], 
            api_token=config['KOBO_API_TOKEN'],
            export_settings_uid=config['KOBO_EXPORT_SETTINGS_UID']
        )
        print("✅ KoboAPI client initialized")
        
        # Test basic connection
        print("\n3. Testing API connection...")
        if kobo.test_connection():
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
            return False
        
        # Get asset information
        print("\n4. Retrieving asset information...")
        try:
            asset_info = kobo.get_asset_info()
            print(f"✅ Asset info retrieved:")
            print(f"   - Name: {asset_info.get('name', 'N/A')}")
            print(f"   - Owner: {asset_info.get('owner__username', 'N/A')}")
            print(f"   - Deployment ID: {asset_info.get('deployment__identifier', 'N/A')}")
            print(f"   - Submissions: {asset_info.get('deployment__submission_count', 0)}")
        except KoboAPIError as e:
            print(f"❌ Failed to get asset info: {e}")
            return False
        
        # Test export settings discovery
        print("\n5. Discovering export settings...")
        try:
            export_settings = kobo.get_export_settings()
            print(f"✅ Export settings found:")
            print(f"   - UID: {export_settings.get('uid', 'N/A')}")
            print(f"   - Type: {export_settings.get('export_settings', {}).get('type', 'N/A')}")
            print(f"   - Last modified: {export_settings.get('date_modified', 'N/A')}")
            
            # Check if XLSX URL is available
            xlsx_url = export_settings.get('data_url_xlsx')
            if xlsx_url:
                print(f"✅ XLSX export URL available")
            else:
                print("⚠️  No XLSX export URL found - you may need to create export settings in KoboToolbox")
                
        except KoboAPIError as e:
            print(f"❌ Failed to get export settings: {e}")
            print("\n💡 You may need to:")
            print("   1. Create export settings in KoboToolbox UI")
            print("   2. Or specify KOBO_EXPORT_SETTINGS_UID in your .env")
            return False
        
        # Test XLSX download (small test)
        print("\n6. Testing XLSX download...")
        try:
            xlsx_file, settings = kobo.download_data_xlsx()
            print(f"✅ XLSX download successful:")
            print(f"   - File saved: {xlsx_file}")
            
            # Check file size
            file_size = os.path.getsize(xlsx_file)
            print(f"   - File size: {file_size:,} bytes")
            
            if file_size > 0:
                print("✅ Downloaded file contains data")
            else:
                print("⚠️  Downloaded file is empty - check if there are submissions")
                
        except KoboAPIError as e:
            print(f"❌ XLSX download failed: {e}")
            return False
        
        # Test cache cleanup
        print("\n7. Testing cache management...")
        try:
            kobo.cleanup_old_cache_files(keep_count=2)
            print("✅ Cache cleanup successful")
        except Exception as e:
            print(f"⚠️  Cache cleanup warning: {e}")
        
        print("\n🎉 All tests passed! KoboToolbox integration is ready.")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.error(f"Connection test failed with error: {e}", exc_info=True)
        return False

def main():
    """Main function to run connection tests."""
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  No .env file found!")
        print("\n📝 Please create a .env file with your KoboToolbox credentials:")
        print("   1. Copy .env.example to .env")
        print("   2. Edit .env with your actual values")
        print("   3. Run this test again")
        return
    
    success = test_kobo_connection()
    
    if success:
        print("\n🚀 Ready to start the dashboard!")
        print("   Run: streamlit run app.py")
    else:
        print("\n🔧 Please fix the issues above before proceeding.")
        print("   Check the logs for more details.")

if __name__ == "__main__":
    main()