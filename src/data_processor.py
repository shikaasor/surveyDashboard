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
    """
    
    def __init__(self):
        """Initialize data processor."""
        self.processing_stats = {}
        logger.info("Initialized DataProcessor")
    
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
        
        # Add derived indicator columns (is_full, is_partial, is_not)
        self._add_derived_indicator_columns(clean_df)
        
        # Add source version field (FR-8)
        self._add_source_version_field(clean_df)
        
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
        Normalize column name by replacing / with _ and cleaning up.
        
        Args:
            column_name: Original column name
            
        Returns:
            Normalized column name
        """
        # Replace / with _
        normalized = column_name.replace('/', '_')
        
        # Remove leading group names if verbose (e.g., group_infrastructure/... → infrastructure_...)
        if normalized.startswith('group_'):
            parts = normalized.split('_', 1)
            if len(parts) > 1:
                normalized = parts[1]
        
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
        # Find state and facility columns
        state_cols = [col for col in df.columns if 'state' in col.lower()]
        facility_cols = [col for col in df.columns if 'facility' in col.lower()]
        
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
        """Convert score columns to numeric, preserving n/a as None (FR-7)."""
        # Identify score columns (typically infrastructure_* with 0/1/2/n/a values)
        infrastructure_cols = [col for col in df.columns if col.startswith('infrastructure_')]
        
        for col in infrastructure_cols:
            # Convert to numeric, keeping 'n/a' and similar as NaN
            original_values = df[col].value_counts()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Log conversion stats
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.debug(f"Column {col}: {null_count} values converted to NaN (likely n/a)")
    
    def _add_derived_indicator_columns(self, df: pd.DataFrame) -> None:
        """Add derived boolean columns for indicator analysis."""
        infrastructure_cols = [col for col in df.columns if 
                              col.startswith('infrastructure_') and 
                              df[col].dtype in ['int64', 'float64']]
        
        for col in infrastructure_cols:
            # Create boolean indicators
            df[f'{col}_is_full'] = (df[col] == 2)
            df[f'{col}_is_partial'] = (df[col] == 1)
            df[f'{col}_is_not'] = (df[col] == 0)
        
        logger.debug(f"Added derived columns for {len(infrastructure_cols)} indicators")
    
    def _add_source_version_field(self, df: pd.DataFrame) -> None:
        """Add source version field from __version__ (FR-8)."""
        if '__version__' in df.columns:
            df['source_version'] = df['__version__']
        else:
            df['source_version'] = None
            logger.warning("No __version__ column found")
    
    def _mark_data_quality_issues(self, df: pd.DataFrame) -> None:
        """Mark data quality issues (FR-9)."""
        # Check for duplicates based on _uuid
        if '_uuid' in df.columns:
            df['is_duplicate'] = df.duplicated(subset=['_uuid'], keep='first')
            duplicate_count = df['is_duplicate'].sum()
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate records")
        else:
            df['is_duplicate'] = False
            logger.warning("No _uuid column found for duplicate detection")
        
        # Check for missing critical fields
        critical_fields = ['state', 'facility', 'submission_date']
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                df[f'missing_{field}'] = df[field].isna()
                if missing_count > 0:
                    logger.warning(f"Missing {field}: {missing_count} records ({missing_count/len(df)*100:.1f}%)")
            else:
                df[f'missing_{field}'] = True
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
            'duplicates': 0,
            'missing_critical_fields': {},
            'invalid_scores': {},
            'completeness_score': 0.0
        }
        
        # Count duplicates
        if 'is_duplicate' in df.columns:
            summary['duplicates'] = df['is_duplicate'].sum()
        
        # Count missing critical fields
        critical_fields = ['state', 'facility', 'submission_date']
        for field in critical_fields:
            missing_col = f'missing_{field}'
            if missing_col in df.columns:
                summary['missing_critical_fields'][field] = df[missing_col].sum()
        
        # Count invalid scores
        invalid_cols = [col for col in df.columns if col.endswith('_invalid_score')]
        for col in invalid_cols:
            field_name = col.replace('_invalid_score', '')
            summary['invalid_scores'][field_name] = df[col].sum()
        
        # Calculate overall completeness score
        total_critical_missing = sum(summary['missing_critical_fields'].values())
        if len(critical_fields) > 0 and len(df) > 0:
            completeness = 1 - (total_critical_missing / (len(df) * len(critical_fields)))
            summary['completeness_score'] = max(0.0, completeness)
        
        return summary