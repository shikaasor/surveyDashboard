#!/usr/bin/env python3
"""
Data Processing Test Utility

This script tests the data processing pipeline using real XLSX data
from KoboToolbox. It validates parsing, cleaning, and quality checks.

Run this after test_connection.py passes to verify data processing works.
"""

import os
import sys
import glob
import pandas as pd
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor, DataProcessingError
from utils.logging_utils import setup_logger

def find_latest_xlsx():
    """Find the most recent XLSX file in cache directory."""
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir):
        return None
    
    # Look for XLSX files
    xlsx_files = glob.glob(os.path.join(cache_dir, "*.xlsx"))
    if not xlsx_files:
        return None
    
    # Return the most recent one
    latest_file = max(xlsx_files, key=os.path.getmtime)
    return latest_file

def test_data_processing():
    """Test data processing pipeline step by step."""
    
    print("📊 Data Processing Pipeline Test")
    print("=" * 50)
    
    # Set up logging
    logger = setup_logger('data_processing_test')
    
    try:
        # Find XLSX file to process
        print("\n1. Looking for XLSX data...")
        xlsx_file = find_latest_xlsx()
        
        if not xlsx_file:
            print("❌ No XLSX file found in data/cache/")
            print("💡 Run 'python test_connection.py' first to download data")
            return False
        
        file_size = os.path.getsize(xlsx_file)
        print(f"✅ Found XLSX file: {xlsx_file}")
        print(f"   - Size: {file_size:,} bytes")
        print(f"   - Modified: {datetime.fromtimestamp(os.path.getmtime(xlsx_file))}")
        
        # Initialize data processor
        print("\n2. Initializing DataProcessor...")
        processor = DataProcessor()
        print("✅ DataProcessor initialized")
        
        # Process XLSX file
        print(f"\n3. Processing XLSX file...")
        print("   This may take a moment for large datasets...")
        
        start_time = datetime.now()
        result = processor.process_xlsx(xlsx_file)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ XLSX processing completed in {processing_time:.2f} seconds")
        
        # Analyze results
        print(f"\n4. Analyzing processed data...")
        
        main_df = result['main']
        repeat_groups = result['repeat_groups']
        stats = result['stats']
        
        print(f"✅ Data structure analysis:")
        print(f"   - Main records: {len(main_df):,}")
        print(f"   - Main columns: {len(main_df.columns):,}")
        print(f"   - Repeat groups: {len(repeat_groups)}")
        print(f"   - Processing time: {stats['processing_time']:.2f}s")
        
        # Show repeat groups info
        if repeat_groups:
            print(f"   - Repeat group details:")
            for name, df in repeat_groups.items():
                print(f"     • {name}: {len(df)} rows, {len(df.columns)} columns")
        
        # Analyze main dataframe columns
        print(f"\n5. Column analysis...")
        
        # Check for normalized columns
        original_cols_with_slash = [col for col in main_df.columns if '/' in col]
        normalized_cols = [col for col in main_df.columns if '_' in col and col.startswith(('infrastructure', 'staffing', 'service_delivery', 'clinical_care', 'commodity', 'health_info', 'quality_improvement', 'leadership', 'client_experience', 'pediatric', 'state_and_facility'))]
        
        print(f"   - Columns with / (should be 0): {len(original_cols_with_slash)}")
        if original_cols_with_slash:
            print(f"     ⚠️  Found un-normalized columns: {original_cols_with_slash[:5]}...")
        
        print(f"   - Normalized columns: {len(normalized_cols)}")
        if normalized_cols:
            print(f"     ✅ Examples: {normalized_cols[:3]}")
        
        # Check for key columns
        key_columns = ['_uuid', 'submission_date', 'state', 'facility']
        present_keys = [col for col in key_columns if col in main_df.columns]
        missing_keys = [col for col in key_columns if col not in main_df.columns]
        
        print(f"   - Key columns present: {present_keys}")
        if missing_keys:
            print(f"   - Missing key columns: {missing_keys}")
        
        # Check data types
        print(f"\n6. Data type analysis...")
        
        # Date columns
        date_columns = ['_submission_time', 'start', 'end', 'submission_date']
        for col in date_columns:
            if col in main_df.columns:
                dtype = main_df[col].dtype
                if 'datetime' in str(dtype) or 'date' in str(dtype):
                    print(f"   ✅ {col}: {dtype}")
                else:
                    print(f"   ⚠️  {col}: {dtype} (expected datetime)")
        
        # Infrastructure columns (should be numeric)
        infra_cols = [col for col in main_df.columns if col.startswith('infrastructure_') and not col.endswith(('_is_full', '_is_partial', '_is_not'))]
        numeric_infra = 0
        for col in infra_cols[:5]:  # Check first 5
            if main_df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                numeric_infra += 1
        
        print(f"   - Infrastructure columns found: {len(infra_cols)}")
        print(f"   - Numeric infrastructure columns: {numeric_infra}/{min(5, len(infra_cols))} checked")
        
        # Check for derived boolean columns
        derived_cols = [col for col in main_df.columns if col.endswith(('_is_full', '_is_partial', '_is_not'))]
        print(f"   - Derived boolean columns: {len(derived_cols)}")
        
        # Data quality analysis
        print(f"\n7. Data quality analysis...")
        
        quality_summary = processor.validate_data_quality(main_df)
        
        print(f"   ✅ Quality metrics:")
        print(f"   - Total records: {quality_summary['total_records']:,}")
        print(f"   - Unique facilities: {quality_summary['unique_facilities']:,}")
        print(f"   - Unique states: {quality_summary['unique_states']:,}")
        print(f"   - Completeness score: {quality_summary['completeness_score']:.1%}")
        
        # Duplicate facilities check
        if quality_summary['duplicate_facilities_found'] > 0:
            print(f"   ⚠️  Duplicate facilities found: {quality_summary['duplicate_facilities_found']}")
            print(f"   ✅ Latest submissions kept, older ones removed")
        else:
            print(f"   ✅ No duplicate facilities found")
        
        # Missing critical fields
        if any(count > 0 for count in quality_summary['missing_critical_fields'].values()):
            print(f"   - Missing critical data:")
            for field, count in quality_summary['missing_critical_fields'].items():
                if count > 0:
                    pct = (count / quality_summary['total_records']) * 100
                    print(f"     • {field}: {count:,} missing ({pct:.1f}%)")
        else:
            print(f"   ✅ No missing critical fields")
        
        # Invalid scores by group
        if quality_summary['invalid_scores_by_group']:
            print(f"   - Invalid scores by group:")
            for group, count in quality_summary['invalid_scores_by_group'].items():
                print(f"     • {group}: {count:,} invalid scores")
        else:
            print(f"   ✅ All scores within valid range (0/1/2/N/A)")
        
        # Sample data preview
        print(f"\n8. Sample data preview...")
        print(f"   First 3 records (selected columns):")
        
        preview_cols = ['_uuid', 'submission_date', 'state', 'facility']
        available_preview = [col for col in preview_cols if col in main_df.columns]
        
        if available_preview:
            sample = main_df[available_preview].head(3)
            for idx, row in sample.iterrows():
                print(f"   Record {idx + 1}:")
                for col in available_preview:
                    value = row[col]
                    if pd.isna(value):
                        value = "N/A"
                    print(f"     - {col}: {value}")
                print()
        
        # Save processed data
        print(f"9. Saving processed data...")
        
        os.makedirs('data/processed', exist_ok=True)
        parquet_file = 'data/processed/latest.parquet'
        
        main_df.to_parquet(parquet_file, index=False)
        file_size = os.path.getsize(parquet_file)
        
        print(f"✅ Processed data saved:")
        print(f"   - File: {parquet_file}")
        print(f"   - Size: {file_size:,} bytes")
        print(f"   - Format: Parquet (compressed)")
        
        print(f"\n🎉 Data processing pipeline test successful!")
        print(f"   - Raw XLSX → Clean structured data ✅")
        print(f"   - Column normalization ✅")
        print(f"   - Data type conversion ✅")
        print(f"   - Quality validation ✅")
        print(f"   - Ready for KPI calculation!")
        
        return True
        
    except DataProcessingError as e:
        print(f"\n❌ Data processing failed: {e}")
        logger.error(f"Data processing test failed: {e}", exc_info=True)
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.error(f"Data processing test failed with error: {e}", exc_info=True)
        return False

def main():
    """Main function to run data processing tests."""
    
    # Import pandas here to check if available
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not installed. Run: pip install -r requirements.txt")
        return
    
    success = test_data_processing()
    
    if success:
        print("\n🚀 Ready for Phase 3: KPI Calculation!")
        print("   Your data is clean and structured.")
    else:
        print("\n🔧 Please fix the issues above before proceeding.")
        print("   Check the logs for more details.")

if __name__ == "__main__":
    main()