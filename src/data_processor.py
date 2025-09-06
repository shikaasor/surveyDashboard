"""
Data processing pipeline for KoboToolbox XLSX data.

This module handles parsing, cleaning, and transforming XLSX data from KoboToolbox
into structured datasets ready for analysis and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re

from .utils.logging_utils import setup_logger, log_data_processing, log_error_with_context

logger = setup_logger('data_processor')


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


class DataProcessor:
    """
    Data processor for KoboToolbox XLSX files.
    
    Handles multi-sheet parsing, column normalization, data type conversion,
    and quality checks as per FR-6, FR-7, FR-8, FR-9 requirements.
    
    Uses group-based indicator structure with 0/1/2/N/A scoring system:
    - 0 = Not implemented
    - 1 = Partially implemented  
    - 2 = Fully implemented
    - N/A = Not applicable
    """
    
    # Define indicator groups based on KoboToolbox survey structure
    INDICATOR_GROUPS = {
        'infrastructure': 'group_infrastructure',
        'staffing': 'staffing_human_resources', 
        'service_delivery': 'service_delivery_processes',
        'clinical_care': 'clinical_care_quality',
        'commodity': 'commodity_management',
        'health_info': 'health_information_systems',
        'quality_improvement': 'quality_improvement',
        'leadership': 'leadership_governance_sustainability',
        'client_experience': 'client_experience_satisfaction',
        'pediatric': 'pediatric_adolescent_services'
    }
    
    def __init__(self):
        """Initialize data processor."""
        self.processing_stats = {}
        logger.info("Initialized DataProcessor with group-based indicator structure")
    
    def process_xlsx(self, xlsx_file: str) -> Dict[str, pd.DataFrame]:
        """
        Process XLSX file into clean DataFrames.
        
        Args:
            xlsx_file: Path to XLSX file
            
        Returns:
            Dictionary containing processed DataFrames
            
        Raises:
            DataProcessingError: If processing fails
        """
        start_time = datetime.now()
        
        try:
            # Read the Excel file - handle multiple sheets for repeat groups
            logger.info(f"Reading XLSX file: {xlsx_file}")
            xls = pd.ExcelFile(xlsx_file)
            
            logger.info(f"Found {len(xls.sheet_names)} sheets: {xls.sheet_names}")
            
            # Main form data is usually in the first sheet
            main_df = pd.read_excel(xls, sheet_name=0)
            logger.info(f"Main sheet has {len(main_df)} rows and {len(main_df.columns)} columns")
            
            # Process repeat groups from other sheets if they exist
            repeat_dfs = {}
            for sheet_name in xls.sheet_names[1:]:
                try:
                    repeat_df = pd.read_excel(xls, sheet_name=sheet_name)
                    repeat_dfs[sheet_name] = repeat_df
                    logger.info(f"Repeat group '{sheet_name}': {len(repeat_df)} rows, {len(repeat_df.columns)} columns")
                except Exception as e:
                    logger.warning(f"Failed to read sheet '{sheet_name}': {str(e)}")
            
            # Clean and process the main dataframe
            clean_df = self._clean_main_dataframe(main_df)
            
            # Process repeat groups
            clean_repeat_dfs = {}
            for name, df in repeat_dfs.items():
                try:
                    clean_repeat_dfs[name] = self._clean_repeat_dataframe(df, name)
                except Exception as e:
                    logger.warning(f"Failed to clean repeat group '{name}': {str(e)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            log_data_processing(logger, "XLSX processing", len(clean_df), processing_time)
            
            # Store processing stats
            self.processing_stats = {
                'main_records': len(clean_df),
                'main_columns': len(clean_df.columns),
                'repeat_groups': len(clean_repeat_dfs),
                'processing_time': processing_time,
                'timestamp': datetime.now()
            }
            
            return {
                'main': clean_df,
                'repeat_groups': clean_repeat_dfs,
                'stats': self.processing_stats
            }
            
        except Exception as e:
            context = {'xlsx_file': xlsx_file}
            log_error_with_context(logger, e, context)
            raise DataProcessingError(f"Failed to process XLSX file: {str(e)}") from e
    
    def _clean_main_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize the main dataframe.
        
        Args:
            df: Raw dataframe from XLSX
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning main dataframe...")
        
        # Make a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Normalize column names: replace / with _ (FR-6)
        original_columns = clean_df.columns.tolist()
        clean_df.columns = [self._normalize_column_name(col) for col in clean_df.columns]
        
        logger.info(f"Normalized {len(original_columns)} column names")
        
        # Convert date columns to datetime (FR-7)
        self._convert_date_columns(clean_df)
        
        # Add derived submission_date from _submission_time or end (FR-8)
        self._add_derived_date_fields(clean_df)
        
        # Normalize state and facility names (FR-8)
        self._normalize_location_fields(clean_df)
        
        # Convert score columns to numeric, preserving n/a as None (FR-7)
        self._convert_score_columns(clean_df)
        
        # Add source version field (FR-8)
        self._add_source_version_field(clean_df)
        
        # Add derived indicator columns and get the optimized DataFrame
        clean_df = self._add_derived_indicator_columns_optimized(clean_df)
        
        # Perform data quality checks (FR-9)
        self._mark_data_quality_issues(clean_df)
        
        logger.info(f"Cleaned dataframe: {len(clean_df)} rows, {len(clean_df.columns)} columns")
        
        return clean_df
    
    def _clean_repeat_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """
        Clean repeat group dataframe.
        
        Args:
            df: Repeat group dataframe
            sheet_name: Name of the sheet/repeat group
            
        Returns:
            Cleaned repeat group dataframe
        """
        logger.info(f"Cleaning repeat group: {sheet_name}")
        
        clean_df = df.copy()
        
        # Normalize column names
        clean_df.columns = [self._normalize_column_name(col) for col in clean_df.columns]
        
        # Convert date columns if present
        self._convert_date_columns(clean_df)
        
        # Convert numeric columns if present
        numeric_cols = [col for col in clean_df.columns if 
                       clean_df[col].dtype in ['object'] and 
                       self._could_be_numeric(clean_df[col])]
        
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        
        return clean_df
    
    def _normalize_column_name(self, column_name: str) -> str:
        """
        Normalize column name by replacing / with _ and applying group mappings.
        
        Maps KoboToolbox group names to shorter, cleaner names:
        - group_infrastructure/adequate_space → infrastructure_adequate_space
        - staffing_human_resources/staff_adequacy → staffing_staff_adequacy
        
        Args:
            column_name: Original column name
            
        Returns:
            Normalized column name
        """
        # Replace / with _
        normalized = column_name.replace('/', '_')
        
        # Apply group name mappings
        for short_name, long_name in self.INDICATOR_GROUPS.items():
            if normalized.startswith(f'{long_name}_'):
                normalized = normalized.replace(f'{long_name}_', f'{short_name}_', 1)
                break
        
        # Clean up multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def _convert_date_columns(self, df: pd.DataFrame) -> None:
        """Convert date columns to datetime (FR-7)."""
        date_columns = ['_submission_time', 'start', 'end', 'today']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
                    logger.debug(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to datetime: {str(e)}")
    
    def _add_derived_date_fields(self, df: pd.DataFrame) -> None:
        """Add derived date fields (FR-8)."""
        # Add submission_date from _submission_time (fallback to end)
        if '_submission_time' in df.columns:
            df['submission_date'] = df['_submission_time'].dt.date
        elif 'end' in df.columns:
            df['submission_date'] = df['end'].dt.date
        else:
            logger.warning("No submission time column found for deriving submission_date")
    
    def _normalize_location_fields(self, df: pd.DataFrame) -> None:
        """Normalize state and facility names (FR-8)."""
        # Find state and facility columns with better specificity
        # Look for columns that end with 'state' or 'facility' to avoid ambiguity
        state_cols = [col for col in df.columns if col.lower().endswith('_state') or col.lower() == 'state']
        if not state_cols:
            # Fallback to containing 'state' but not 'facility'
            state_cols = [col for col in df.columns if 'state' in col.lower() and 'facility' not in col.lower()]
        
        facility_cols = [col for col in df.columns if col.lower().endswith('_facility') or col.lower() == 'facility']
        if not facility_cols:
            # Fallback to containing 'facility' but not 'state'  
            facility_cols = [col for col in df.columns if 'facility' in col.lower() and 'state' not in col.lower()]
        
        # Normalize state
        if state_cols:
            source_col = state_cols[0]
            df['state'] = df[source_col].astype(str).str.title().str.strip()
            logger.debug(f"Normalized state from column: {source_col}")
        else:
            logger.warning("No state column found")
        
        # Normalize facility
        if facility_cols:
            source_col = facility_cols[0]
            df['facility'] = df[source_col].astype(str).str.title().str.strip()
            logger.debug(f"Normalized facility from column: {source_col}")
        else:
            logger.warning("No facility column found")
    
    def _convert_score_columns(self, df: pd.DataFrame) -> None:
        """Convert score columns to numeric, preserving N/A as None (FR-7).
        
        Scoring system:
        - 0 = Not implemented
        - 1 = Partially implemented  
        - 2 = Fully implemented
        - N/A = Not applicable (converted to NaN)
        """
        # Get score columns by group
        score_columns = self._get_score_columns(df)
        
        logger.info(f"Found {len(score_columns)} score columns across {len(set(col.split('_')[0] for col in score_columns))} indicator groups")
        
        converted_count = 0
        for col in score_columns:
            original_dtype = df[col].dtype
            original_values = df[col].value_counts(dropna=False)
            
            # Convert to numeric, keeping 'N/A', 'n/a', 'na', etc. as NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Validate converted values are in valid range {0, 1, 2}
            valid_values = df[col].dropna()
            invalid_mask = ~valid_values.isin([0, 1, 2])
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                logger.warning(f"Column {col}: {invalid_count} invalid scores (not in 0/1/2 range)")
            
            # Log conversion stats
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.debug(f"Column {col}: {null_count} N/A values (excluded from calculations)")
            
            if original_dtype == 'object':
                converted_count += 1
                
        logger.info(f"Converted {converted_count} columns from text to numeric scores")
    
    def _get_score_columns(self, df: pd.DataFrame) -> List[str]:
        """Get all score columns by group, excluding comment/evidence fields."""
        score_columns = []
        
        for group_name in self.INDICATOR_GROUPS.keys():
            # Find columns that start with this group prefix
            group_cols = [col for col in df.columns if col.startswith(f'{group_name}_')]
            
            # Filter out comment/evidence/additional columns
            filtered_cols = []
            for col in group_cols:
                if any(keyword in col.lower() for keyword in ['comment', 'evidence', 'additional']):
                    continue
                filtered_cols.append(col)
            
            score_columns.extend(filtered_cols)
            logger.debug(f"Group '{group_name}': {len(filtered_cols)} score columns")
        
        return score_columns
    
    def _add_derived_indicator_columns(self, df: pd.DataFrame) -> None:
        """Add derived boolean columns for indicator analysis by group.
        
        Creates boolean flags for each score:
        - {col}_is_full = True when score = 2 (Fully implemented)
        - {col}_is_partial = True when score = 1 (Partially implemented) 
        - {col}_is_not = True when score = 0 (Not implemented)
        - N/A values are excluded (False for all flags)
        """
        # Get score columns for derivation
        score_columns = self._get_score_columns(df)
        numeric_score_cols = [col for col in score_columns if 
                             df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']]
        
        logger.info(f"Creating derived boolean columns for {len(numeric_score_cols)} indicators")
        
        # Create all derived columns at once to avoid DataFrame fragmentation
        derived_data = {}
        derived_cols_by_group = {}
        
        for col in numeric_score_cols:
            # Determine which group this column belongs to
            group_name = col.split('_')[0]
            if group_name not in derived_cols_by_group:
                derived_cols_by_group[group_name] = []
            
            # Create boolean indicators based on 0/1/2 scoring
            derived_data[f'{col}_is_full'] = (df[col] == 2)
            derived_data[f'{col}_is_partial'] = (df[col] == 1) 
            derived_data[f'{col}_is_not'] = (df[col] == 0)
            
            derived_cols_by_group[group_name].extend([
                f'{col}_is_full',
                f'{col}_is_partial', 
                f'{col}_is_not'
            ])
        
        # This is the old method that causes fragmentation - keeping for compatibility
        # The new optimized method is _add_derived_indicator_columns_optimized
        pass
    
    def _add_derived_indicator_columns_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived boolean columns efficiently without DataFrame fragmentation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            New DataFrame with derived columns added
        """
        # Get score columns for derivation
        score_columns = self._get_score_columns(df)
        numeric_score_cols = [col for col in score_columns if 
                             df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']]
        
        logger.info(f"Creating derived boolean columns for {len(numeric_score_cols)} indicators")
        
        if not numeric_score_cols:
            return df
        
        # Create all derived columns at once to avoid DataFrame fragmentation
        derived_dfs = []
        derived_cols_by_group = {}
        
        for col in numeric_score_cols:
            # Determine which group this column belongs to
            group_name = col.split('_')[0]
            if group_name not in derived_cols_by_group:
                derived_cols_by_group[group_name] = []
            
            # Create boolean indicators based on 0/1/2 scoring
            col_derived = pd.DataFrame({
                f'{col}_is_full': (df[col] == 2),
                f'{col}_is_partial': (df[col] == 1),
                f'{col}_is_not': (df[col] == 0)
            }, index=df.index)
            
            derived_dfs.append(col_derived)
            
            derived_cols_by_group[group_name].extend([
                f'{col}_is_full',
                f'{col}_is_partial', 
                f'{col}_is_not'
            ])
        
        # Concatenate all derived columns at once
        if derived_dfs:
            all_derived = pd.concat(derived_dfs, axis=1)
            result_df = pd.concat([df, all_derived], axis=1)
            
            # Log summary by group
            for group_name, derived_cols in derived_cols_by_group.items():
                base_indicators = len(derived_cols) // 3  # 3 derived cols per base indicator
                logger.debug(f"Group '{group_name}': {base_indicators} indicators → {len(derived_cols)} derived columns")
            
            total_derived = len(all_derived.columns)
            logger.info(f"Added {total_derived} derived boolean columns across {len(derived_cols_by_group)} groups")
            
            return result_df
        else:
            logger.info("No numeric score columns found for derived indicators")
            return df
    
    def _add_source_version_field(self, df: pd.DataFrame) -> None:
        """Add source version field from __version__ (FR-8)."""
        if '__version__' in df.columns:
            df['source_version'] = df['__version__']
        else:
            df['source_version'] = None
            logger.warning("No __version__ column found")
    
    def _mark_data_quality_issues(self, df: pd.DataFrame) -> None:
        """Mark and resolve data quality issues (FR-9)."""
        # Check for duplicate facilities and keep the one with the latest end date
        if 'facility' in df.columns and 'end' in df.columns:
            # Find duplicate facilities
            duplicate_facilities = df[df.duplicated(subset=['facility'], keep=False)]['facility'].unique()
            
            if len(duplicate_facilities) > 0:
                logger.warning(f"Found duplicate facilities: {duplicate_facilities.tolist()}")
                
                # For each duplicate facility, keep only the record with the latest end date
                rows_to_drop = []
                for facility in duplicate_facilities:
                    facility_rows = df[df['facility'] == facility].copy()
                    
                    # Sort by end date (latest first) and keep only the first one
                    facility_rows = facility_rows.sort_values('end', ascending=False)
                    latest_row_idx = facility_rows.index[0]
                    
                    # Mark all other rows for removal
                    other_rows = facility_rows.index[1:]
                    rows_to_drop.extend(other_rows)
                    
                    logger.info(f"Facility '{facility}': keeping latest record from {facility_rows.loc[latest_row_idx, 'end']}, removing {len(other_rows)} older records")
                
                # Remove duplicate rows in-place
                df.drop(rows_to_drop, inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                logger.info(f"Removed {len(rows_to_drop)} duplicate facility records, kept latest submissions")
            else:
                logger.info("No duplicate facilities found")
        else:
            missing_cols = []
            if 'facility' not in df.columns:
                missing_cols.append('facility')
            if 'end' not in df.columns:
                missing_cols.append('end')
            logger.warning(f"Cannot check for duplicate facilities: missing columns {missing_cols}")
        
        # Check for missing critical fields
        critical_fields = ['state', 'facility']
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                if missing_count > 0:
                    logger.warning(f"Missing {field}: {missing_count} records ({missing_count/len(df)*100:.1f}%)")
            else:
                logger.warning(f"Critical field '{field}' not found")
        
        # Check for out-of-range scores
        infrastructure_cols = [col for col in df.columns if 
                              col.startswith('infrastructure_') and 
                              not any(col.endswith(suffix) for suffix in ['_is_full', '_is_partial', '_is_not'])]
        
        for col in infrastructure_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                # Check for values not in {0, 1, 2} (excluding NaN which represents n/a)
                valid_mask = df[col].isna() | df[col].isin([0, 1, 2])
                invalid_count = (~valid_mask).sum()
                df[f'{col}_invalid_score'] = ~valid_mask
                if invalid_count > 0:
                    logger.warning(f"Invalid scores in {col}: {invalid_count} records")
    
    def _could_be_numeric(self, series: pd.Series) -> bool:
        """Check if a series could be converted to numeric."""
        try:
            # Try converting a sample
            sample = series.dropna().head(10)
            if sample.empty:
                return False
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of processing statistics.
        
        Returns:
            Processing summary dictionary
        """
        return self.processing_stats.copy() if hasattr(self, 'processing_stats') else {}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return summary.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Data quality summary
        """
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'duplicate_facilities_found': 0,
            'duplicate_facilities_removed': 0,
            'unique_facilities': 0,
            'unique_states': 0,
            'missing_critical_fields': {},
            'invalid_scores_by_group': {},
            'completeness_score': 0.0
        }
        
        # Check for facility uniqueness (main quality check)
        if 'facility' in df.columns:
            facility_counts = df['facility'].value_counts()
            duplicates = facility_counts[facility_counts > 1]
            summary['duplicate_facilities_found'] = len(duplicates)
            summary['unique_facilities'] = df['facility'].nunique()
            
            if len(duplicates) > 0:
                logger.info(f"Quality check: {len(duplicates)} facilities have multiple submissions")
        
        if 'state' in df.columns:
            summary['unique_states'] = df['state'].nunique()
        
        # Count missing critical fields
        critical_fields = ['state', 'facility']
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                summary['missing_critical_fields'][field] = missing_count
            else:
                summary['missing_critical_fields'][field] = len(df)  # All missing if column doesn't exist
        
        # Count invalid scores by group
        invalid_cols = [col for col in df.columns if col.endswith('_invalid_score')]
        invalid_by_group = {}
        
        for col in invalid_cols:
            field_name = col.replace('_invalid_score', '')
            invalid_count = df[col].sum()
            
            if invalid_count > 0:
                group_name = field_name.split('_')[0] if '_' in field_name else 'unknown'
                if group_name not in invalid_by_group:
                    invalid_by_group[group_name] = 0
                invalid_by_group[group_name] += invalid_count
        
        summary['invalid_scores_by_group'] = invalid_by_group
        
        # Calculate overall completeness score based on critical fields only
        total_critical_missing = sum(summary['missing_critical_fields'].values())
        critical_fields_count = len([f for f in critical_fields if f in df.columns])
        
        if critical_fields_count > 0 and len(df) > 0:
            max_possible_missing = len(df) * critical_fields_count
            completeness = 1 - (total_critical_missing / max_possible_missing)
            summary['completeness_score'] = max(0.0, completeness)
        
        return summary