"""
KPI calculation engine for implementation score analysis.

This module computes KPIs and metrics from cleaned survey data following
the exact mathematical definitions in PRD Section 9.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .utils.logging_utils import setup_logger, log_data_processing

logger = setup_logger('kpi_calculator')


class KPICalculator:
    """
    KPI calculator for implementation score analysis.
    
    Implements exact formulas from PRD Section 9:
    - Average score (excluding n/a): avg_score_x = mean(v for v in S[x] if v in {0,1,2})
    - % fully implemented: pct_full_x = count(v==2)/count(v in {0,1,2})
    - % partially implemented: pct_partial_x = count(v==1)/count(v in {0,1,2})
    - % not implemented: pct_not_x = count(v==0)/count(v in {0,1,2})
    - Composite score: composite_avg = mean(avg_score_x for x in infrastructure_indicators)
    """
    
    def __init__(self):
        """Initialize KPI calculator."""
        logger.info("Initialized KPICalculator")
    
    def calculate_kpis(self, clean_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate core KPIs from clean data (FR-11).
        
        Args:
            clean_df: Cleaned dataframe with processed indicators
            
        Returns:
            Dictionary containing calculated KPIs
        """
        start_time = datetime.now()
        
        if clean_df.empty:
            logger.warning("Empty dataframe provided for KPI calculation")
            return self._empty_kpi_result()
        
        results = {}
        
        # Overall metrics
        results['total_submissions'] = len(clean_df)
        results['unique_facilities'] = clean_df['facility'].nunique() if 'facility' in clean_df.columns else 0
        results['unique_states'] = clean_df['state'].nunique() if 'state' in clean_df.columns else 0
        
        # Identify infrastructure indicator columns
        infrastructure_cols = self._get_infrastructure_columns(clean_df)
        
        if not infrastructure_cols:
            logger.warning("No infrastructure columns found for KPI calculation")
            return self._empty_kpi_result()
        
        # Calculate average scores excluding n/a (PRD Section 9)
        avg_scores = {}
        for col in infrastructure_cols:
            # Filter out NaN values (converted from n/a) - only include {0,1,2}
            valid_values = clean_df[col].dropna()
            if len(valid_values) > 0:
                avg_scores[col] = valid_values.mean()
        
        results['infrastructure_scores'] = avg_scores
        
        # Calculate overall infrastructure score (composite average)
        if avg_scores:
            results['overall_infrastructure_score'] = np.mean(list(avg_scores.values()))
        else:
            results['overall_infrastructure_score'] = None
        
        # Calculate implementation percentages (PRD Section 9)
        implementation_stats = {}
        for col in infrastructure_cols:
            # Count values excluding NaN (only {0,1,2})
            valid_count = clean_df[col].count()  # count() excludes NaN
            
            if valid_count > 0:
                full_count = (clean_df[col] == 2).sum()
                partial_count = (clean_df[col] == 1).sum()
                not_count = (clean_df[col] == 0).sum()
                
                implementation_stats[col] = {
                    'pct_full': (full_count / valid_count) * 100,
                    'pct_partial': (partial_count / valid_count) * 100,
                    'pct_not': (not_count / valid_count) * 100,
                    'count': valid_count,
                    'full_count': full_count,
                    'partial_count': partial_count,
                    'not_count': not_count
                }
        
        results['implementation_stats'] = implementation_stats
        
        # Calculate overall implementation percentages
        if implementation_stats:
            all_pct_full = [stats['pct_full'] for stats in implementation_stats.values()]
            all_pct_partial = [stats['pct_partial'] for stats in implementation_stats.values()]
            all_pct_not = [stats['pct_not'] for stats in implementation_stats.values()]
            
            results['overall_pct_full'] = np.mean(all_pct_full)
            results['overall_pct_partial'] = np.mean(all_pct_partial)
            results['overall_pct_not'] = np.mean(all_pct_not)
        else:
            results['overall_pct_full'] = 0
            results['overall_pct_partial'] = 0
            results['overall_pct_not'] = 0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        log_data_processing(logger, "KPI calculation", len(clean_df), processing_time)
        
        return results
    
    def calculate_grouped_metrics(self, clean_df: pd.DataFrame, 
                                group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate metrics grouped by specified columns (FR-11).
        
        Args:
            clean_df: Cleaned dataframe
            group_by: List of columns to group by
            
        Returns:
            DataFrame with grouped metrics
        """
        if clean_df.empty:
            return pd.DataFrame()
        
        if not group_by:
            group_by = ['state', 'facility']
        
        # Filter group_by to only include existing columns
        existing_group_cols = [col for col in group_by if col in clean_df.columns]
        if not existing_group_cols:
            logger.warning(f"None of the grouping columns {group_by} exist in dataframe")
            return pd.DataFrame()
        
        # Identify indicator columns
        infrastructure_cols = self._get_infrastructure_columns(clean_df)
        
        if not infrastructure_cols:
            logger.warning("No infrastructure columns found for grouped metrics")
            return pd.DataFrame()
        
        grouped_metrics = []
        
        # Create groups
        try:
            grouped = clean_df.groupby(existing_group_cols)
        except Exception as e:
            logger.error(f"Failed to create groups: {str(e)}")
            return pd.DataFrame()
        
        # Calculate metrics for each group
        for group_names, group_df in grouped:
            # Convert to tuple if single groupby value
            if not isinstance(group_names, tuple):
                group_names = (group_names,)
            
            # Create base record with group identifiers
            record = dict(zip(existing_group_cols, group_names))
            
            # Add count
            record['submission_count'] = len(group_df)
            
            # Calculate metrics for each indicator
            indicator_averages = []
            for col in infrastructure_cols:
                valid_values = group_df[col].dropna()
                valid_count = len(valid_values)
                
                if valid_count > 0:
                    # Average score
                    avg_score = valid_values.mean()
                    record[f'{col}_avg'] = avg_score
                    indicator_averages.append(avg_score)
                    
                    # Implementation percentages (exact PRD formulas)
                    full_count = (group_df[col] == 2).sum()
                    partial_count = (group_df[col] == 1).sum()
                    not_count = (group_df[col] == 0).sum()
                    
                    record[f'{col}_pct_full'] = (full_count / valid_count) * 100
                    record[f'{col}_pct_partial'] = (partial_count / valid_count) * 100
                    record[f'{col}_pct_not'] = (not_count / valid_count) * 100
                else:
                    # No valid data for this indicator
                    record[f'{col}_avg'] = None
                    record[f'{col}_pct_full'] = 0
                    record[f'{col}_pct_partial'] = 0
                    record[f'{col}_pct_not'] = 0
            
            # Add overall infrastructure score (composite average)
            if indicator_averages:
                record['overall_infrastructure_score'] = np.mean(indicator_averages)
            else:
                record['overall_infrastructure_score'] = None
            
            grouped_metrics.append(record)
        
        return pd.DataFrame(grouped_metrics)
    
    def calculate_time_series(self, clean_df: pd.DataFrame, 
                            date_column: str = 'submission_date',
                            freq: str = 'W') -> pd.DataFrame:
        """
        Calculate time series metrics (FR-12).
        
        Args:
            clean_df: Cleaned dataframe
            date_column: Column to use for time grouping
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with time series metrics
        """
        if clean_df.empty or date_column not in clean_df.columns:
            logger.warning(f"Cannot calculate time series: missing {date_column}")
            return pd.DataFrame()
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(clean_df[date_column]):
            clean_df = clean_df.copy()
            clean_df[date_column] = pd.to_datetime(clean_df[date_column])
        
        # Group by date with specified frequency
        if freq == 'W':
            # Weekly - use period to get week start dates
            clean_df = clean_df.copy()
            clean_df['date_bucket'] = clean_df[date_column].dt.to_period('W').dt.start_time
        elif freq == 'M':
            # Monthly
            clean_df = clean_df.copy()
            clean_df['date_bucket'] = clean_df[date_column].dt.to_period('M').dt.start_time
        else:
            # Daily (default)
            clean_df = clean_df.copy()
            clean_df['date_bucket'] = clean_df[date_column].dt.date
        
        # Group by date bucket
        grouped = clean_df.groupby('date_bucket')
        
        # Calculate metrics per time bucket
        metrics_over_time = []
        
        for date, group_df in grouped:
            record = {
                'date_bucket': date, 
                'submission_count': len(group_df)
            }
            
            # Calculate KPIs for this time period
            period_kpis = self.calculate_kpis(group_df)
            
            # Add key metrics to record
            record['overall_infrastructure_score'] = period_kpis.get('overall_infrastructure_score')
            record['overall_pct_full'] = period_kpis.get('overall_pct_full')
            record['overall_pct_partial'] = period_kpis.get('overall_pct_partial')
            record['overall_pct_not'] = period_kpis.get('overall_pct_not')
            
            metrics_over_time.append(record)
        
        result_df = pd.DataFrame(metrics_over_time)
        
        # Sort by date
        if not result_df.empty:
            result_df = result_df.sort_values('date_bucket')
        
        return result_df
    
    def calculate_indicator_group_summary(self, clean_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate indicator group summaries (FR-13).
        
        Args:
            clean_df: Cleaned dataframe
            
        Returns:
            Dictionary with indicator group summaries
        """
        infrastructure_cols = self._get_infrastructure_columns(clean_df)
        
        if not infrastructure_cols:
            return {}
        
        # For now, we only have infrastructure group, but this can be extended
        groups = {
            'infrastructure': infrastructure_cols
        }
        
        summaries = {}
        
        for group_name, columns in groups.items():
            # Calculate group-level metrics
            group_data = clean_df[columns]
            
            # Overall group statistics
            total_valid_responses = group_data.count().sum()  # Excludes NaN
            total_possible_responses = len(clean_df) * len(columns)
            
            if total_valid_responses > 0:
                # Count responses by category across all indicators in group
                full_responses = (group_data == 2).sum().sum()
                partial_responses = (group_data == 1).sum().sum()
                not_responses = (group_data == 0).sum().sum()
                
                summaries[group_name] = {
                    'indicator_count': len(columns),
                    'total_responses': total_valid_responses,
                    'response_rate': (total_valid_responses / total_possible_responses) * 100,
                    'avg_score': group_data.mean().mean(),  # Average of all values
                    'pct_full': (full_responses / total_valid_responses) * 100,
                    'pct_partial': (partial_responses / total_valid_responses) * 100,
                    'pct_not': (not_responses / total_valid_responses) * 100,
                    'indicators': columns
                }
            else:
                summaries[group_name] = {
                    'indicator_count': len(columns),
                    'total_responses': 0,
                    'response_rate': 0,
                    'avg_score': None,
                    'pct_full': 0,
                    'pct_partial': 0,
                    'pct_not': 0,
                    'indicators': columns
                }
        
        return summaries
    
    def _get_infrastructure_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of infrastructure indicator columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of infrastructure column names
        """
        # Find columns that start with 'infrastructure_' but exclude derived boolean columns
        infrastructure_cols = [
            col for col in df.columns 
            if col.startswith('infrastructure_') and 
            not any(col.endswith(suffix) for suffix in ['_is_full', '_is_partial', '_is_not', '_invalid_score'])
            and df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']  # Only numeric columns
        ]
        
        return infrastructure_cols
    
    def _empty_kpi_result(self) -> Dict[str, Any]:
        """Return empty KPI result structure."""
        return {
            'total_submissions': 0,
            'unique_facilities': 0,
            'unique_states': 0,
            'infrastructure_scores': {},
            'overall_infrastructure_score': None,
            'implementation_stats': {},
            'overall_pct_full': 0,
            'overall_pct_partial': 0,
            'overall_pct_not': 0
        }
    
    def get_kpi_definitions(self) -> Dict[str, str]:
        """
        Get mathematical definitions of KPIs as per PRD Section 9.
        
        Returns:
            Dictionary with KPI definitions
        """
        return {
            'avg_score_x': 'mean(v for v in S[x] if v in {0,1,2})',
            'pct_full_x': 'count(v==2)/count(v in {0,1,2})',
            'pct_partial_x': 'count(v==1)/count(v in {0,1,2})',
            'pct_not_x': 'count(v==0)/count(v in {0,1,2})',
            'composite_avg': 'mean(avg_score_x for x in infrastructure_indicators)',
            'description': 'S = set of selected submissions after filters; x = indicator'
        }