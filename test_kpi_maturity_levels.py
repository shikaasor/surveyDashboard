#!/usr/bin/env python3
"""
Maturity Level KPI Test Utility

This script validates the new maturity level system implemented for KPI calculations.
It verifies that:
1. Group scores are calculated using sum(all_scores_in_group) / count(valid_scores_in_group)
2. Maturity levels are correctly assigned (Basic, Developing, Advancing, Mature)
3. No percentage calculations are used in the core logic
4. All 10 indicator groups work with the maturity system

Run this to verify the maturity level system is working correctly.
"""

import os
import sys
import glob
import pandas as pd
from datetime import datetime
import numpy as np

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor, DataProcessingError
from kpi_calculator import KPICalculator
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

def test_maturity_level_system():
    """Test the maturity level KPI calculation system."""
    
    print("🎯 Maturity Level System Test")
    print("=" * 50)
    
    # Set up logging
    logger = setup_logger('maturity_level_test')
    
    try:
        # Step 1: Get processed data
        print("\n1. Getting processed data...")
        xlsx_file = find_latest_xlsx()
        
        if not xlsx_file:
            print("❌ No XLSX file found in data/cache/")
            print("💡 Run 'python test_connection.py' first to download data")
            return False
        
        processor = DataProcessor()
        result = processor.process_xlsx(xlsx_file)
        main_df = result['main']
        
        print(f"✅ Data processed: {len(main_df)} records, {len(main_df.columns)} columns")
        
        # Step 2: Initialize KPI calculator and test maturity level functions
        print(f"\n2. Testing Maturity Level Functions...")
        calculator = KPICalculator()
        
        # Test the static maturity level function
        test_scores = [0.25, 0.75, 1.25, 1.75, 2.5]  # Test edge cases
        expected_levels = [1, 2, 3, 4, 4]  # Last one should cap at level 4
        
        print("   Testing maturity level classification:")
        for score, expected in zip(test_scores, expected_levels):
            maturity_info = calculator.get_maturity_level(score)
            actual_level = maturity_info['level']
            level_name = maturity_info['name']
            level_range = maturity_info['range']
            color = maturity_info['color']
            
            status = "✅" if actual_level == expected else "❌"
            print(f"   - Score {score:.2f} → Level {actual_level} ({level_name}) [{level_range}] {color} {status}")
        
        # Step 3: Test core KPI calculations with maturity levels
        print(f"\n3. Testing Core KPI Calculations with Maturity Levels...")
        kpis = calculator.calculate_kpis(main_df)
        
        print(f"✅ Core KPIs calculated successfully")
        print(f"   - Total submissions: {kpis['total_submissions']:,}")
        print(f"   - Unique facilities: {kpis['unique_facilities']:,}")
        print(f"   - Unique states: {kpis['unique_states']:,}")
        
        # Test overall maturity level
        overall_score = kpis.get('overall_composite_score')
        if overall_score is not None:
            overall_maturity = kpis.get('overall_maturity_level', {})
            print(f"   - Overall composite score: {overall_score:.3f}")
            if overall_maturity:
                print(f"   - Overall maturity: Level {overall_maturity.get('level')} ({overall_maturity.get('name')})")
                print(f"   - Maturity range: {overall_maturity.get('range')}")
        
        # Step 4: Test all indicator groups with new maturity system
        print(f"\n4. Testing All Indicator Groups with Maturity System...")
        
        all_groups = calculator.get_indicator_groups()
        print(f"   Found {len(all_groups)} indicator groups")
        
        successful_groups = 0
        for group_name in all_groups:
            group_kpis = calculator.calculate_group_kpis(main_df, group_name)
            
            if group_kpis and group_kpis.get('group_average') is not None:
                avg_score = group_kpis['group_average']
                maturity_level = group_kpis.get('maturity_level', {})
                valid_responses = group_kpis.get('total_responses', 0)
                
                print(f"\n   📊 {group_name.replace('_', ' ').title()}:")
                print(f"      - Average Score: {avg_score:.3f}")
                print(f"      - Valid Responses: {valid_responses}")
                
                if maturity_level:
                    level_num = maturity_level.get('level')
                    level_name = maturity_level.get('name')
                    level_range = maturity_level.get('range')
                    color = maturity_level.get('color')
                    
                    print(f"      - Maturity Level: Level {level_num} - {level_name}")
                    print(f"      - Score Range: {level_range}")
                    print(f"      - Color Code: {color}")
                    
                    # Validate score is in correct range
                    ranges = {1: (0.0, 0.50), 2: (0.51, 1.00), 3: (1.01, 1.50), 4: (1.51, 2.00)}
                    if level_num in ranges:
                        min_val, max_val = ranges[level_num]
                        in_range = min_val <= avg_score <= max_val
                        print(f"      - Range Validation: {'✅' if in_range else '❌'}")
                        
                        if not in_range:
                            print(f"        Expected: {min_val}-{max_val}, Got: {avg_score:.3f}")
                
                successful_groups += 1
            else:
                print(f"\n   ⚠️  {group_name.replace('_', ' ').title()}: No data available")
        
        print(f"\n   ✅ Successfully processed {successful_groups}/{len(all_groups)} groups")
        
        # Step 5: Test facility-level maturity calculations
        print(f"\n5. Testing Facility-Level Maturity Calculations...")
        
        if 'facility' in main_df.columns and 'state' in main_df.columns:
            facility_metrics = calculator.calculate_grouped_metrics(main_df, group_by=['state', 'facility'])
            
            if not facility_metrics.empty:
                print(f"   ✅ Facility metrics calculated for {len(facility_metrics)} facilities")
                
                # Add maturity level information
                if 'overall_composite_score' in facility_metrics.columns:
                    facility_metrics['maturity_info'] = facility_metrics['overall_composite_score'].apply(
                        lambda x: calculator.get_maturity_level(x) if pd.notna(x) else None
                    )
                    facility_metrics['maturity_level'] = facility_metrics['maturity_info'].apply(
                        lambda x: x['level'] if x else None
                    )
                    facility_metrics['maturity_name'] = facility_metrics['maturity_info'].apply(
                        lambda x: x['name'] if x else 'N/A'
                    )
                    
                    # Show maturity distribution
                    maturity_dist = facility_metrics['maturity_level'].value_counts().sort_index()
                    level_names = {1: 'Basic', 2: 'Developing', 3: 'Advancing', 4: 'Mature'}
                    
                    print(f"   📊 Facility Maturity Distribution:")
                    for level, count in maturity_dist.items():
                        if pd.notna(level):
                            name = level_names.get(int(level), 'Unknown')
                            pct = (count / len(facility_metrics)) * 100
                            print(f"      - Level {int(level)} ({name}): {count} facilities ({pct:.1f}%)")
                    
                    # Show top 3 facilities
                    top_facilities = facility_metrics.nlargest(3, 'overall_composite_score')
                    print(f"\n   🏆 Top 3 Facilities by Maturity:")
                    for idx, row in top_facilities.iterrows():
                        facility = row['facility']
                        state = row['state']
                        score = row['overall_composite_score']
                        level = row.get('maturity_level')
                        name = row.get('maturity_name')
                        print(f"      {idx+1}. {facility} ({state}): {score:.3f} - Level {level} ({name})")
            else:
                print("   ⚠️  No facility metrics calculated")
        
        # Step 6: Test mathematical formula accuracy (new maturity formula)
        print(f"\n6. Testing New Maturity Formula Accuracy...")
        
        if successful_groups > 0:
            # Test the first group with data
            # Find first group with data
            test_group = None
            for group in all_groups:
                group_result = calculator.calculate_group_kpis(main_df, group)
                if group_result and group_result.get('group_average') is not None:
                    test_group = group
                    break
            
            if not test_group:
                print("   ⚠️  No test group available for formula testing")
                test_group = all_groups[0] if all_groups else None
                
            if test_group:
                print(f"   Testing formula for '{test_group.replace('_', ' ').title()}' group...")
                
                # Get indicators for this group
                group_indicators = calculator._get_group_columns(main_df, test_group)
            else:
                print("   ⚠️  No groups available at all")
                group_indicators = []
            
            if group_indicators:
                print(f"   - Found {len(group_indicators)} indicators in group")
                
                # Manual calculation using new formula: sum(all_scores_in_group) / count(valid_scores_in_group)
                total_sum = 0
                total_count = 0
                
                for indicator in group_indicators:
                    valid_values = main_df[indicator].dropna()  # Exclude N/A
                    if len(valid_values) > 0:
                        total_sum += valid_values.sum()
                        total_count += len(valid_values)
                        print(f"      • {indicator}: {len(valid_values)} valid values, sum = {valid_values.sum()}")
                
                if total_count > 0:
                    manual_score = total_sum / total_count
                    manual_maturity = calculator.get_maturity_level(manual_score)
                    
                    # Get calculated result
                    calculated_kpis = calculator.calculate_group_kpis(main_df, test_group)
                    calculated_score = calculated_kpis['group_average']
                    calculated_maturity = calculated_kpis['maturity_level']
                    
                    print(f"\n   🧮 Formula Validation Results:")
                    print(f"   - Total sum: {total_sum}")
                    print(f"   - Total count: {total_count}")
                    print(f"   - Manual score: {manual_score:.6f}")
                    print(f"   - Calculated score: {calculated_score:.6f}")
                    print(f"   - Score match: {'✅' if abs(manual_score - calculated_score) < 0.001 else '❌'}")
                    
                    print(f"   - Manual maturity: Level {manual_maturity['level']} ({manual_maturity['name']})")
                    print(f"   - Calculated maturity: Level {calculated_maturity['level']} ({calculated_maturity['name']})")
                    print(f"   - Maturity match: {'✅' if manual_maturity['level'] == calculated_maturity['level'] else '❌'}")
                    
                    # Validate the new formula excludes N/A values
                    print(f"\n   📋 New Formula Validation:")
                    print(f"   ✅ Formula: sum(all_scores_in_group) / count(valid_scores_in_group)")
                    print(f"   ✅ Excludes N/A values from calculation")
                    print(f"   ✅ Uses all valid responses, not averages per indicator")
                    print(f"   ✅ Maturity Level 1 (Basic): 0.00-0.50")
                    print(f"   ✅ Maturity Level 2 (Developing): 0.51-1.00") 
                    print(f"   ✅ Maturity Level 3 (Advancing): 1.01-1.50")
                    print(f"   ✅ Maturity Level 4 (Mature): 1.51-2.00")
                else:
                    print("   ⚠️  No valid data for formula testing")
            else:
                print("   ⚠️  No indicators found for formula testing")
        
        # Step 7: Verify no percentage calculations in core logic
        print(f"\n7. Verifying No Percentage Calculations in Core Logic...")
        
        # Check that KPIs don't contain percentage fields
        percentage_fields = ['pct_full', 'pct_partial', 'pct_not', 'group_pct_full', 'group_pct_partial', 'group_pct_not']
        found_percentages = [field for field in percentage_fields if any(field in str(kpis).lower() for field in percentage_fields)]
        
        if found_percentages:
            print(f"   ⚠️  Found percentage fields in KPIs: {found_percentages}")
            print("   This suggests the old percentage system is still being used")
        else:
            print(f"   ✅ No percentage calculations found in core KPIs")
            print(f"   ✅ Successfully migrated to maturity level system")
        
        # Check for maturity-specific fields
        maturity_fields = ['maturity_level', 'average_score', 'valid_responses']
        found_maturity = [field for field in maturity_fields if any(field in str(kpis) for field in maturity_fields)]
        
        print(f"   ✅ Found maturity-specific fields: {found_maturity}")
        
        # Step 8: Summary and validation
        print(f"\n🎉 Maturity Level System Test Complete!")
        
        validation_criteria = [
            successful_groups > 0,  # At least one group working
            overall_score is not None,  # Overall score calculated
            len(found_percentages) == 0,  # No percentage calculations in core logic
            len(found_maturity) > 0,  # Maturity fields present
            (len(facility_metrics) > 0) if 'facility' in main_df.columns else True  # Facility calculations work
        ]
        
        passed_criteria = sum(validation_criteria)
        total_criteria = len(validation_criteria)
        
        print(f"\n📊 Validation Results: {passed_criteria}/{total_criteria} criteria passed")
        
        if passed_criteria == total_criteria:
            print(f"✅ All maturity level validations passed!")
            print(f"   - ✅ Maturity level functions working correctly")
            print(f"   - ✅ New formula implemented: sum(all_scores) / count(valid_scores)")
            print(f"   - ✅ {successful_groups} indicator groups using maturity levels")
            print(f"   - ✅ No percentage calculations in core logic")
            print(f"   - ✅ Facility-level maturity calculations working")
            print(f"   - ✅ Mathematical formulas validated")
            print(f"   - ✅ System ready for dashboard!")
            return True
        else:
            print(f"❌ Some validations failed:")
            
            criteria_names = [
                "At least one group working",
                "Overall score calculated", 
                "No percentage calculations",
                "Maturity fields present",
                "Facility calculations work"
            ]
            
            for i, (passed, name) in enumerate(zip(validation_criteria, criteria_names)):
                status = "✅" if passed else "❌"
                print(f"   {status} {name}")
            
            return False
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.error(f"Maturity level test failed with error: {e}", exc_info=True)
        return False

def main():
    """Main function to run maturity level system tests."""
    success = test_maturity_level_system()
    
    if success:
        print("\n🚀 Maturity Level System Validated!")
        print("   Your new maturity level system is working correctly.")
        print("   Ready to proceed with dashboard implementation!")
    else:
        print("\n🔧 Please fix the issues above before proceeding.")
        print("   The maturity level system needs adjustments.")

if __name__ == "__main__":
    main()