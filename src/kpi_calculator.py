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
    KPI calculator for implementation score analysis with group-based maturity level assessment.
    
    Implements maturity level assessment across indicator groups:
    - Group score: sum(valid_scores) / count(valid_scores) where valid in {0,1,2}
    - Maturity levels:
      * Level 1 (Basic): 0.00-0.50
      * Level 2 (Developing): 0.51-1.00  
      * Level 3 (Advancing): 1.01-1.50
      * Level 4 (Mature): 1.51-2.00
    - Overall composite: mean(group_scores for all groups)
    
    Supports 10 indicator groups:
    Infrastructure, Staffing, Service Delivery, Clinical Care, Commodity Management,
    Health Information Systems, Quality Improvement, Leadership, Client Experience, Pediatric
    """
    
    # Define indicator groups based on data processor structure
    INDICATOR_GROUPS = [
        'infrastructure',
        'staffing', 
        'service_delivery',
        'clinical_care',
        'commodity',
        'health_info',
        'quality_improvement',
        'leadership',
        'client_experience',
        'pediatric'
    ]
    
    def __init__(self):
        """Initialize KPI calculator."""
        logger.info("Initialized KPICalculator with maturity level assessment system")
    
    def get_indicator_groups(self) -> List[str]:
        """
        Get list of all indicator groups (domains).
        
        Returns:
            List of indicator group names
        """
        return self.INDICATOR_GROUPS.copy()
    
    def calculate_group_kpis(self, clean_df: pd.DataFrame, group_name: str) -> Dict[str, Any]:
        """
        Calculate KPIs for a specific indicator group by name.
        
        Args:
            clean_df: Cleaned dataframe
            group_name: Name of the group
            
        Returns:
            Dictionary with group KPIs or None if no data
        """
        if group_name not in self.INDICATOR_GROUPS:
            logger.warning(f"Unknown group name: {group_name}")
            return None
            
        # Get columns for this group
        group_cols = self._get_group_columns(clean_df, group_name)
        
        if not group_cols:
            logger.warning(f"No columns found for group: {group_name}")
            return None
            
        # Calculate KPIs using the private method
        return self._calculate_group_kpis(clean_df, group_cols, group_name)
    
    @staticmethod
    def get_maturity_level(score: float) -> Dict[str, Any]:
        """
        Determine maturity level based on group score.
        
        Args:
            score: Group average score (0.0-2.0)
            
        Returns:
            Dictionary with level info
        """
        if pd.isna(score) or score < 0:
            return {"level": 0, "name": "No Data", "range": "N/A", "color": "#CCCCCC"}
        elif 0.0 <= score <= 0.50:
            return {"level": 1, "name": "Basic", "range": "0.00-0.50", "color": "#FF4444"}
        elif 0.51 <= score <= 1.00:
            return {"level": 2, "name": "Developing", "range": "0.51-1.00", "color": "#FFAA00"}
        elif 1.01 <= score <= 1.50:
            return {"level": 3, "name": "Advancing", "range": "1.01-1.50", "color": "#4488FF"}
        elif 1.51 <= score <= 2.00:
            return {"level": 4, "name": "Mature", "range": "1.51-2.00", "color": "#00AA44"}
        else:
            return {"level": 0, "name": "Invalid", "range": f"{score:.2f}", "color": "#CCCCCC"}
    
    @staticmethod
    def get_maturity_distribution(scores: List[float]) -> Dict[str, Any]:
        """
        Calculate distribution of facilities across maturity levels.
        
        Args:
            scores: List of group scores
            
        Returns:
            Dictionary with distribution stats
        """
        valid_scores = [s for s in scores if pd.notna(s) and s >= 0]
        
        if not valid_scores:
            return {
                "total_facilities": 0,
                "level_counts": {1: 0, 2: 0, 3: 0, 4: 0},
                "level_percentages": {1: 0, 2: 0, 3: 0, 4: 0},
                "average_score": None
            }
        
        # Count facilities at each level
        level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for score in valid_scores:
            level_info = KPICalculator.get_maturity_level(score)
            level_num = level_info["level"]
            if 1 <= level_num <= 4:
                level_counts[level_num] += 1
        
        # Calculate percentages
        total = len(valid_scores)
        level_percentages = {level: (count/total)*100 for level, count in level_counts.items()}
        
        return {
            "total_facilities": total,
            "level_counts": level_counts,
            "level_percentages": level_percentages,
            "average_score": np.mean(valid_scores)
        }
    
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
        
        # Calculate KPIs by indicator group
        results['group_scores'] = {}
        results['group_implementation'] = {}
        all_group_averages = []
        
        for group_name in self.INDICATOR_GROUPS:
            group_cols = self._get_group_columns(clean_df, group_name)
            
            if not group_cols:
                logger.debug(f"No columns found for group: {group_name}")
                continue
            
            # Calculate group-level KPIs
            group_kpis = self._calculate_group_kpis(clean_df, group_cols, group_name)
            
            if group_kpis['avg_scores']:
                results['group_scores'][group_name] = group_kpis
                all_group_averages.append(group_kpis['group_average'])
        
        # Calculate overall composite score across all groups
        if all_group_averages:
            results['overall_composite_score'] = np.mean(all_group_averages)
        else:
            results['overall_composite_score'] = None
            
        # Calculate overall maturity level and distribution
        if all_group_averages:
            overall_maturity = self.get_maturity_level(results['overall_composite_score'])
            results['overall_maturity_level'] = overall_maturity
            
            # Calculate maturity distribution across all groups
            all_group_scores = [data.get('group_average') for data in results['group_scores'].values() 
                               if data.get('group_average') is not None]
            
            if all_group_scores:
                maturity_dist = self.get_maturity_distribution(all_group_scores)
                results['maturity_distribution'] = maturity_dist
        else:
            results['overall_maturity_level'] = self.get_maturity_level(None)
            results['maturity_distribution'] = self.get_maturity_distribution([])
        
        logger.info(f"Calculated KPIs for {len(results['group_scores'])} indicator groups")
        
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
        
        # Get all indicator columns across all groups
        all_indicator_cols = []
        for group_name in self.INDICATOR_GROUPS:
            group_cols = self._get_group_columns(clean_df, group_name)
            all_indicator_cols.extend(group_cols)
        
        if not all_indicator_cols:
            logger.warning("No indicator columns found for grouped metrics")
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
            
            # Calculate metrics for each indicator and by group
            overall_indicator_averages = []
            
            # Calculate by indicator group
            for group_name in self.INDICATOR_GROUPS:
                group_cols = self._get_group_columns(group_df, group_name)
                if not group_cols:
                    continue
                    
                group_averages = []
                for col in group_cols:
                    valid_values = group_df[col].dropna()
                    valid_count = len(valid_values)
                    
                    if valid_count > 0:
                        # Average score
                        avg_score = valid_values.mean()
                        record[f'{col}_avg'] = avg_score
                        group_averages.append(avg_score)
                        
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
                
                # Group-level score using new maturity formula
                if group_averages:
                    # Calculate group score: sum all scores / count all responses
                    total_sum = 0
                    total_count = 0
                    
                    for col in group_cols:
                        valid_values = group_df[col].dropna()
                        if len(valid_values) > 0:
                            total_sum += valid_values.sum()
                            total_count += len(valid_values)
                    
                    group_score = total_sum / total_count if total_count > 0 else None
                    record[f'{group_name}_group_score'] = group_score
                    record[f'{group_name}_group_avg'] = group_score  # Legacy compatibility
                    
                    # Add maturity level
                    maturity_level = self.get_maturity_level(group_score)
                    record[f'{group_name}_maturity_level'] = maturity_level['name']
                    record[f'{group_name}_maturity_level_num'] = maturity_level['level']
                    
                    if group_score is not None:
                        overall_indicator_averages.append(group_score)
                else:
                    record[f'{group_name}_group_score'] = None
                    record[f'{group_name}_group_avg'] = None
                    record[f'{group_name}_maturity_level'] = 'No Data'
                    record[f'{group_name}_maturity_level_num'] = 0
            
            # Add overall composite score and maturity level
            if overall_indicator_averages:
                overall_score = np.mean(overall_indicator_averages)
                record['overall_composite_score'] = overall_score
                
                # Add overall maturity level
                overall_maturity = self.get_maturity_level(overall_score)
                record['overall_maturity_level'] = overall_maturity['name']
                record['overall_maturity_level_num'] = overall_maturity['level']
            else:
                record['overall_composite_score'] = None
                record['overall_maturity_level'] = 'No Data'
                record['overall_maturity_level_num'] = 0
            
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
            record['overall_composite_score'] = period_kpis.get('overall_composite_score')
            record['overall_pct_full'] = period_kpis.get('overall_pct_full')
            record['overall_pct_partial'] = period_kpis.get('overall_pct_partial')
            record['overall_pct_not'] = period_kpis.get('overall_pct_not')
            
            # Add group-specific scores
            for group_name in self.INDICATOR_GROUPS:
                if group_name in period_kpis.get('group_scores', {}):
                    group_data = period_kpis['group_scores'][group_name]
                    record[f'{group_name}_score'] = group_data.get('group_average')
                    record[f'{group_name}_pct_full'] = group_data.get('group_pct_full')
                else:
                    record[f'{group_name}_score'] = None
                    record[f'{group_name}_pct_full'] = 0
            
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
    
    def _get_group_columns(self, df: pd.DataFrame, group_name: str) -> List[str]:
        """
        Get list of indicator columns for a specific group.
        
        Args:
            df: DataFrame to analyze
            group_name: Name of the indicator group (e.g., 'infrastructure')
            
        Returns:
            List of column names for the group
        """
        # Find columns that start with the group name but exclude derived boolean columns
        group_cols = [
            col for col in df.columns 
            if col.startswith(f'{group_name}_') and 
            not any(col.endswith(suffix) for suffix in ['_is_full', '_is_partial', '_is_not', '_invalid_score']) and
            not any(keyword in col.lower() for keyword in ['comment', 'evidence', 'additional']) and
            df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']  # Only numeric columns
        ]
        
        return group_cols
    
    def _get_group_indicators(self, df: pd.DataFrame, group_name: str) -> List[str]:
        """
        Get list of indicator columns for a specific group (alias for _get_group_columns).
        
        Args:
            df: DataFrame to analyze
            group_name: Name of the indicator group
            
        Returns:
            List of column names for the group
        """
        return self._get_group_columns(df, group_name)
    
    def _calculate_group_kpis(self, clean_df: pd.DataFrame, group_cols: List[str], group_name: str) -> Dict[str, Any]:
        """
        Calculate KPIs for a specific indicator group.
        
        Args:
            clean_df: Cleaned dataframe
            group_cols: List of columns in the group
            group_name: Name of the group
            
        Returns:
            Dictionary with group KPIs
        """
        if not group_cols:
            return {'avg_scores': {}, 'group_average': None, 'group_pct_full': 0, 'group_pct_partial': 0, 'group_pct_not': 0}
        
        # Calculate average scores excluding N/A (PRD Section 9)
        avg_scores = {}
        valid_indicators = []
        
        for col in group_cols:
            # Filter out NaN values (converted from N/A) - only include {0,1,2}
            valid_values = clean_df[col].dropna()
            if len(valid_values) > 0:
                avg_score = valid_values.mean()
                avg_scores[col] = avg_score
                valid_indicators.append(col)
        
        # Calculate group score using new formula: sum(all_valid_scores) / count(valid_scores)
        if valid_indicators:
            # Calculate total sum and count across all indicators in the group
            total_sum = 0
            total_count = 0
            
            for col in valid_indicators:
                valid_values = clean_df[col].dropna()  # Exclude N/A
                if len(valid_values) > 0:
                    total_sum += valid_values.sum()
                    total_count += len(valid_values)
            
            # Group average = sum of all scores / count of all valid responses
            group_average = total_sum / total_count if total_count > 0 else None
        else:
            group_average = None
            total_count = 0
        
        # Get maturity level for this group
        maturity_level = self.get_maturity_level(group_average) if group_average is not None else self.get_maturity_level(None)
        
        logger.debug(f"Group '{group_name}': {len(valid_indicators)} indicators, score={group_average:.3f}, level={maturity_level['name']}" if group_average else f"Group '{group_name}': no valid data")
        
        return {
            'avg_scores': avg_scores,  # Individual indicator averages (for compatibility)
            'group_average': group_average,
            'group_score': group_average,  # Same as group_average, clearer naming
            'maturity_level': maturity_level,
            'indicator_count': len(valid_indicators),
            'total_responses': total_count,
            'total_sum': total_sum if valid_indicators else 0
        }
    
    def _get_infrastructure_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of infrastructure indicator columns (legacy method for backwards compatibility).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of infrastructure column names
        """
        return self._get_group_columns(df, 'infrastructure')
    
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
        Get mathematical definitions of maturity level KPIs.
        
        Returns:
            Dictionary with KPI definitions
        """
        return {
            'group_score': 'sum(all_scores_in_group) / count(valid_responses_in_group)',
            'overall_composite': 'mean(group_score for group in all_groups)',
            'maturity_levels': {
                'Level 1 (Basic)': '0.00-0.50 - Minimal implementation',
                'Level 2 (Developing)': '0.51-1.00 - Partial implementation',
                'Level 3 (Advancing)': '1.01-1.50 - Good implementation', 
                'Level 4 (Mature)': '1.51-2.00 - Excellent implementation'
            },
            'facility_ranking': 'rank facilities by overall_composite score and maturity level',
            'state_aggregation': 'mean(facility_scores) grouped by state with maturity levels',
            'trend_analysis': 'time_series(maturity_levels) by date_bucket',
            'description': 'Maturity assessment excludes N/A responses; group = indicator group'
        }
    
    def get_maturity_level_summary(self, clean_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive maturity level summary across all groups and facilities.
        
        Args:
            clean_df: Cleaned dataframe
            
        Returns:
            Dictionary with maturity level summary
        """
        # Calculate facility-level maturity
        facility_metrics = self.calculate_grouped_metrics(clean_df, group_by=['state', 'facility'])
        
        summary = {
            'total_facilities': 0,
            'overall_distribution': {},
            'group_distributions': {},
            'state_distributions': {},
            'maturity_progression': {},
            'improvement_opportunities': []
        }
        
        if not facility_metrics.empty:
            summary['total_facilities'] = len(facility_metrics)
            
            # Overall maturity distribution
            if 'overall_composite_score' in facility_metrics.columns:
                overall_scores = facility_metrics['overall_composite_score'].dropna().tolist()
                summary['overall_distribution'] = self.get_maturity_distribution(overall_scores)
            
            # Group-specific distributions
            for group_name in self.INDICATOR_GROUPS:
                score_col = f'{group_name}_group_score'
                if score_col in facility_metrics.columns:
                    group_scores = facility_metrics[score_col].dropna().tolist()
                    if group_scores:
                        summary['group_distributions'][group_name] = self.get_maturity_distribution(group_scores)
            
            # State-level distributions
            if 'state' in facility_metrics.columns:
                state_summary = {}
                for state in facility_metrics['state'].unique():
                    state_facilities = facility_metrics[facility_metrics['state'] == state]
                    if 'overall_composite_score' in state_facilities.columns:
                        state_scores = state_facilities['overall_composite_score'].dropna().tolist()
                        if state_scores:
                            state_summary[state] = self.get_maturity_distribution(state_scores)
                
                summary['state_distributions'] = state_summary
            
            # Identify improvement opportunities (Level 1 and 2 facilities)
            if 'overall_maturity_level_num' in facility_metrics.columns:
                low_maturity = facility_metrics[facility_metrics['overall_maturity_level_num'].isin([1, 2])]
                summary['improvement_opportunities'] = low_maturity[
                    ['state', 'facility', 'overall_composite_score', 'overall_maturity_level']
                ].to_dict('records')
        
        return summary
    
    def get_group_summary(self, clean_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary of all indicator groups.
        
        Args:
            clean_df: Cleaned dataframe
            
        Returns:
            Dictionary with group summaries
        """
        summary = {
            'total_groups': len(self.INDICATOR_GROUPS),
            'groups_with_data': 0,
            'total_indicators': 0,
            'groups': {}
        }
        
        for group_name in self.INDICATOR_GROUPS:
            group_cols = self._get_group_columns(clean_df, group_name)
            
            if group_cols:
                group_kpis = self._calculate_group_kpis(clean_df, group_cols, group_name)
                
                if group_kpis['avg_scores']:
                    summary['groups_with_data'] += 1
                    summary['total_indicators'] += len(group_cols)
                    summary['groups'][group_name] = {
                        'indicator_count': len(group_cols),
                        'indicators': group_cols,
                        'group_average': group_kpis['group_average'],
                        'pct_full': group_kpis['group_pct_full'],
                        'pct_partial': group_kpis['group_pct_partial'],
                        'pct_not': group_kpis['group_pct_not'],
                        'total_responses': group_kpis['total_responses']
                    }
        
        return summary
    
    def calculate_facility_rankings(self, clean_df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Calculate facility rankings for overall and each group.
        
        Args:
            clean_df: Cleaned dataframe
            top_n: Number of top facilities to return per ranking
            
        Returns:
            Dictionary with ranking DataFrames
        """
        rankings = {}
        
        # Overall facility ranking
        facility_metrics = self.calculate_grouped_metrics(clean_df, group_by=['state', 'facility'])
        
        if not facility_metrics.empty and 'overall_composite_score' in facility_metrics.columns:
            rankings['overall'] = facility_metrics.nlargest(top_n, 'overall_composite_score')[
                ['state', 'facility', 'overall_composite_score', 'submission_count']
            ]
        
        # Rankings by group
        for group_name in self.INDICATOR_GROUPS:
            group_avg_col = f'{group_name}_group_avg'
            if group_avg_col in facility_metrics.columns:
                group_ranking = facility_metrics[facility_metrics[group_avg_col].notna()].nlargest(
                    top_n, group_avg_col
                )[['state', 'facility', group_avg_col, 'submission_count']]
                
                if not group_ranking.empty:
                    rankings[group_name] = group_ranking
        
        return rankings
    
    def calculate_state_rankings(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate state-level performance rankings.
        
        Args:
            clean_df: Cleaned dataframe
            
        Returns:
            DataFrame with state rankings
        """
        state_metrics = self.calculate_grouped_metrics(clean_df, group_by=['state'])
        
        if not state_metrics.empty and 'overall_composite_score' in state_metrics.columns:
            # Sort by overall composite score
            state_rankings = state_metrics.sort_values('overall_composite_score', ascending=False)
            
            # Add rank column
            state_rankings = state_rankings.copy()
            state_rankings['rank'] = range(1, len(state_rankings) + 1)
            
            return state_rankings
        
        return pd.DataFrame()
    
    def identify_improvement_priorities(self, clean_df: pd.DataFrame, bottom_n: int = 5) -> Dict[str, Any]:
        """
        Identify facilities and groups that need priority attention.
        
        Args:
            clean_df: Cleaned dataframe
            bottom_n: Number of bottom performers to identify
            
        Returns:
            Dictionary with improvement priorities
        """
        priorities = {
            'lowest_scoring_facilities': [],
            'lowest_scoring_groups': [],
            'facilities_needing_support': {},
            'states_needing_support': []
        }
        
        # Get facility metrics
        facility_metrics = self.calculate_grouped_metrics(clean_df, group_by=['state', 'facility'])
        
        if not facility_metrics.empty:
            # Lowest scoring facilities overall
            if 'overall_composite_score' in facility_metrics.columns:
                lowest_facilities = facility_metrics.nsmallest(bottom_n, 'overall_composite_score')
                priorities['lowest_scoring_facilities'] = lowest_facilities[
                    ['state', 'facility', 'overall_composite_score']
                ].to_dict('records')
        
        # Lowest scoring groups across all facilities
        kpis = self.calculate_kpis(clean_df)
        group_scores = kpis.get('group_scores', {})
        
        if group_scores:
            # Sort groups by average score
            group_averages = [(name, data.get('group_average', 0)) 
                            for name, data in group_scores.items() 
                            if data.get('group_average') is not None]
            
            group_averages.sort(key=lambda x: x[1])
            
            priorities['lowest_scoring_groups'] = [
                {'group': name, 'average_score': score} 
                for name, score in group_averages[:bottom_n]
            ]
        
        # Facilities needing support in specific groups
        if not facility_metrics.empty:
            for group_name in self.INDICATOR_GROUPS:
                group_avg_col = f'{group_name}_group_avg'
                if group_avg_col in facility_metrics.columns:
                    # Find facilities with scores below 1.0 in this group
                    low_performers = facility_metrics[
                        (facility_metrics[group_avg_col] < 1.0) & 
                        (facility_metrics[group_avg_col].notna())
                    ]
                    
                    if not low_performers.empty:
                        priorities['facilities_needing_support'][group_name] = low_performers[
                            ['state', 'facility', group_avg_col]
                        ].to_dict('records')
        
        # States needing support (below average performance)
        state_rankings = self.calculate_state_rankings(clean_df)
        if not state_rankings.empty:
            mean_score = state_rankings['overall_composite_score'].mean()
            below_average_states = state_rankings[
                state_rankings['overall_composite_score'] < mean_score
            ]
            
            priorities['states_needing_support'] = below_average_states[
                ['state', 'overall_composite_score', 'submission_count']
            ].to_dict('records')
        
        return priorities