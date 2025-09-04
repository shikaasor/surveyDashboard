"""
Reusable Streamlit dashboard components.

This module provides reusable UI components for the KoboToolbox Analytics Dashboard,
including filters, visualizations, and data quality views.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

from .utils.logging_utils import setup_logger

logger = setup_logger('dashboard_components')


class DashboardComponents:
    """
    Reusable Streamlit components for the dashboard.
    
    Provides filters, visualizations, and data views as per FR-14, FR-15, FR-16.
    """
    
    def __init__(self, st_session_state):
        """
        Initialize dashboard components.
        
        Args:
            st_session_state: Streamlit session state for persistence
        """
        self.st = st
        self.session = st_session_state
        logger.info("Initialized DashboardComponents")
    
    def create_filters(self, df: pd.DataFrame) -> None:
        """
        Create global filter sidebar (FR-14).
        
        Args:
            df: DataFrame to create filters from
        """
        if df.empty:
            self.st.sidebar.warning("No data available for filtering")
            return
        
        self.st.sidebar.title("Filters")
        
        # Date range filter
        if 'submission_date' in df.columns:
            date_col = pd.to_datetime(df['submission_date'])
            min_date = date_col.min().date() if not date_col.empty else None
            max_date = date_col.max().date() if not date_col.empty else None
            
            if min_date and max_date:
                date_range = self.st.sidebar.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Handle single date selection
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                elif len(date_range) == 1:
                    start_date = end_date = date_range[0]
                else:
                    start_date = end_date = min_date
                
                # Store in session state
                self.session.start_date = start_date
                self.session.end_date = end_date
        
        # State filter
        if 'state' in df.columns:
            states = sorted(df['state'].dropna().unique().tolist())
            if states:
                selected_states = self.st.sidebar.multiselect(
                    "States",
                    options=states,
                    default=states
                )
                self.session.selected_states = selected_states
        
        # Facility filter - show only facilities from selected states
        if 'facility' in df.columns and hasattr(self.session, 'selected_states'):
            if self.session.selected_states:
                state_facilities = df[df['state'].isin(self.session.selected_states)]['facility'].dropna().unique()
                selected_facilities = self.st.sidebar.multiselect(
                    "Facilities",
                    options=sorted(state_facilities),
                    default=[]
                )
                self.session.selected_facilities = selected_facilities
            else:
                self.session.selected_facilities = []
        
        # Version filter if available
        if 'source_version' in df.columns:
            versions = sorted(df['source_version'].dropna().unique().tolist())
            if versions:
                selected_versions = self.st.sidebar.multiselect(
                    "Form Versions",
                    options=versions,
                    default=versions
                )
                self.session.selected_versions = selected_versions
    
    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filters to dataframe based on session state.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Apply date filter
        if (hasattr(self.session, 'start_date') and hasattr(self.session, 'end_date') and 
            'submission_date' in filtered_df.columns):
            date_col = pd.to_datetime(filtered_df['submission_date'])
            filtered_df = filtered_df[
                (date_col >= pd.Timestamp(self.session.start_date)) &
                (date_col <= pd.Timestamp(self.session.end_date))
            ]
        
        # Apply state filter
        if (hasattr(self.session, 'selected_states') and self.session.selected_states and 
            'state' in filtered_df.columns):
            filtered_df = filtered_df[filtered_df['state'].isin(self.session.selected_states)]
        
        # Apply facility filter
        if (hasattr(self.session, 'selected_facilities') and self.session.selected_facilities and 
            'facility' in filtered_df.columns):
            filtered_df = filtered_df[filtered_df['facility'].isin(self.session.selected_facilities)]
        
        # Apply version filter
        if (hasattr(self.session, 'selected_versions') and self.session.selected_versions and 
            'source_version' in filtered_df.columns):
            filtered_df = filtered_df[filtered_df['source_version'].isin(self.session.selected_versions)]
        
        return filtered_df
    
    def create_kpi_cards(self, kpis: Dict[str, Any]) -> None:
        """
        Create KPI cards display (Overview tab).
        
        Args:
            kpis: KPI dictionary from calculator
        """
        # Create a row of KPI cards
        col1, col2, col3, col4 = self.st.columns(4)
        
        with col1:
            self.st.metric(
                label="Total Submissions",
                value=f"{kpis.get('total_submissions', 0):,}"
            )
        
        with col2:
            score = kpis.get('overall_infrastructure_score', 0)
            if score is not None:
                self.st.metric(
                    label="Avg Infrastructure Score",
                    value=f"{score:.2f}"
                )
            else:
                self.st.metric(
                    label="Avg Infrastructure Score",
                    value="N/A"
                )
        
        with col3:
            pct_full = kpis.get('overall_pct_full', 0)
            self.st.metric(
                label="% Fully Implemented",
                value=f"{pct_full:.1f}%"
            )
        
        with col4:
            self.st.metric(
                label="Facilities Covered",
                value=f"{kpis.get('unique_facilities', 0):,}"
            )
    
    def create_time_series_chart(self, time_series_df: pd.DataFrame, 
                                metric: str = 'submission_count') -> None:
        """
        Create time series chart (Overview tab).
        
        Args:
            time_series_df: Time series data
            metric: Metric to plot
        """
        if time_series_df.empty:
            self.st.warning("No time series data available.")
            return
        
        self.st.subheader(f"Trend Over Time: {metric.replace('_', ' ').title()}")
        
        # Create line chart
        chart_data = time_series_df[['date_bucket', metric]].copy()
        
        if not chart_data.empty:
            # Use Plotly for better interactivity
            fig = px.line(
                chart_data, 
                x='date_bucket', 
                y=metric,
                title=f"{metric.replace('_', ' ').title()} Over Time"
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=metric.replace('_', ' ').title(),
                height=400
            )
            self.st.plotly_chart(fig, use_container_width=True)
        else:
            self.st.warning("No data to display in time series chart.")
    
    def create_facility_comparison(self, grouped_metrics_df: pd.DataFrame, 
                                 metric: str = 'overall_infrastructure_score', 
                                 top_n: int = 10) -> None:
        """
        Create facility comparison chart (Facility Comparison tab).
        
        Args:
            grouped_metrics_df: Grouped metrics dataframe
            metric: Metric to compare
            top_n: Number of top facilities to show
        """
        if grouped_metrics_df.empty:
            self.st.warning("No facility data available for comparison.")
            return
        
        if metric not in grouped_metrics_df.columns:
            self.st.warning(f"Metric '{metric}' not available in data.")
            return
        
        # Sort by the metric and get top N
        sorted_df = grouped_metrics_df.sort_values(by=metric, ascending=False, na_position='last')
        top_facilities = sorted_df.head(top_n).copy()
        
        if top_facilities.empty:
            self.st.warning("No valid facility data for comparison.")
            return
        
        # Convert the metric name for display
        metric_name = metric.replace('_', ' ').title()
        
        # Create facility label
        if 'state' in top_facilities.columns and 'facility' in top_facilities.columns:
            top_facilities['facility_label'] = (
                top_facilities['facility'].astype(str) + 
                " (" + top_facilities['state'].astype(str) + ")"
            )
        else:
            top_facilities['facility_label'] = top_facilities.get('facility', 'Unknown').astype(str)
        
        self.st.subheader(f"Top {len(top_facilities)} Facilities by {metric_name}")
        
        # Create horizontal bar chart for better readability
        fig = px.bar(
            top_facilities,
            x=metric,
            y='facility_label',
            orientation='h',
            title=f"Top Facilities by {metric_name}",
            color=metric,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, len(top_facilities) * 30),
            xaxis_title=metric_name,
            yaxis_title="Facility"
        )
        self.st.plotly_chart(fig, use_container_width=True)
    
    def create_indicator_deep_dive(self, df: pd.DataFrame, kpi_calculator) -> None:
        """
        Create indicator deep dive analysis (Indicator Deep Dive tab).
        
        Args:
            df: Filtered dataframe
            kpi_calculator: KPI calculator instance
        """
        # List all available indicators
        infrastructure_cols = [
            col for col in df.columns 
            if col.startswith('infrastructure_') and 
            not any(col.endswith(suffix) for suffix in ['_is_full', '_is_partial', '_is_not', '_invalid_score'])
            and df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']
        ]
        
        if not infrastructure_cols:
            self.st.warning("No indicator columns found in the data.")
            return
        
        # Format indicator names for display
        indicator_options = {
            col: col.replace('infrastructure_', '').replace('_', ' ').title() 
            for col in infrastructure_cols
        }
        
        selected_indicator = self.st.selectbox(
            "Select Indicator for Analysis",
            options=list(indicator_options.keys()),
            format_func=lambda x: indicator_options[x]
        )
        
        if selected_indicator and selected_indicator in df.columns:
            indicator_name = indicator_options[selected_indicator]
            
            # Create columns for side-by-side charts
            col1, col2 = self.st.columns(2)
            
            with col1:
                # State distribution
                if 'state' in df.columns:
                    self.st.subheader(f"Average Score by State: {indicator_name}")
                    state_distribution = (
                        df.groupby('state')[selected_indicator]
                        .mean()
                        .reset_index()
                        .sort_values(by=selected_indicator, ascending=False)
                    )
                    
                    if not state_distribution.empty:
                        fig = px.bar(
                            state_distribution,
                            x='state',
                            y=selected_indicator,
                            title=f"Average {indicator_name} by State",
                            color=selected_indicator,
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=400)
                        self.st.plotly_chart(fig, use_container_width=True)
                    else:
                        self.st.warning("No state data available.")
            
            with col2:
                # Score distribution
                self.st.subheader(f"Score Distribution: {indicator_name}")
                value_counts = df[selected_indicator].value_counts().reset_index()
                value_counts.columns = ['Score', 'Count']
                
                # Map scores to labels
                score_map = {
                    0: 'Not Implemented', 
                    1: 'Partially Implemented', 
                    2: 'Fully Implemented'
                }
                
                if not value_counts.empty:
                    value_counts['Score_Label'] = value_counts['Score'].map(
                        lambda x: score_map.get(x, 'Invalid Score')
                    )
                    
                    fig = px.pie(
                        value_counts,
                        values='Count',
                        names='Score_Label',
                        title=f"{indicator_name} Distribution",
                        color_discrete_map={
                            'Not Implemented': '#ff4444',
                            'Partially Implemented': '#ffaa44',
                            'Fully Implemented': '#44ff44'
                        }
                    )
                    fig.update_layout(height=400)
                    self.st.plotly_chart(fig, use_container_width=True)
                else:
                    self.st.warning("No score data available.")
            
            # Time trend analysis
            if 'submission_date' in df.columns:
                self.st.subheader(f"Weekly Trend: {indicator_name}")
                
                # Create weekly aggregation
                df_copy = df.copy()
                df_copy['week'] = pd.to_datetime(df_copy['submission_date']).dt.to_period('W').dt.start_time
                weekly_trend = (
                    df_copy.groupby('week')[selected_indicator]
                    .mean()
                    .reset_index()
                )
                
                if not weekly_trend.empty:
                    fig = px.line(
                        weekly_trend,
                        x='week',
                        y=selected_indicator,
                        title=f"Weekly Average {indicator_name}",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    self.st.plotly_chart(fig, use_container_width=True)
                else:
                    self.st.warning("No time trend data available.")
    
    def create_data_quality_view(self, df: pd.DataFrame) -> None:
        """
        Create data quality dashboard (Data Quality tab).
        
        Args:
            df: DataFrame to analyze
        """
        self.st.header("Data Quality")
        
        if df.empty:
            self.st.warning("No data available for quality analysis.")
            return
        
        # Show metadata
        col1, col2, col3 = self.st.columns(3)
        
        with col1:
            self.st.metric("Total Rows", f"{len(df):,}")
        
        with col2:
            self.st.metric("Total Columns", f"{len(df.columns):,}")
        
        with col3:
            if '_submission_time' in df.columns:
                latest = df['_submission_time'].max()
                if pd.notna(latest):
                    self.st.metric("Latest Submission", latest.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    self.st.metric("Latest Submission", "N/A")
        
        # Check for duplicates
        if 'is_duplicate' in df.columns:
            duplicate_count = df['is_duplicate'].sum()
            if duplicate_count > 0:
                self.st.error(f"⚠️ Found {duplicate_count:,} duplicate submissions")
                
                # Show duplicate records
                duplicates = df[df['is_duplicate'] == True]
                display_cols = ['_uuid', 'submission_date', 'state', 'facility']
                available_cols = [col for col in display_cols if col in duplicates.columns]
                
                if available_cols:
                    self.st.dataframe(duplicates[available_cols])
            else:
                self.st.success("✅ No duplicate submissions found")
        
        # Check for missing critical fields
        critical_fields = ['state', 'facility', 'submission_date']
        missing_issues = []
        
        for field in critical_fields:
            missing_col = f'missing_{field}'
            if missing_col in df.columns:
                missing_count = df[missing_col].sum()
                if missing_count > 0:
                    missing_issues.append((field, missing_count))
        
        if missing_issues:
            self.st.error("⚠️ Missing critical data detected:")
            for field, count in missing_issues:
                percentage = (count / len(df)) * 100
                self.st.write(f"- **{field.title()}**: {count:,} missing values ({percentage:.1f}%)")
        else:
            self.st.success("✅ No missing critical fields detected")
        
        # Show overall data completeness
        self.st.subheader("Data Completeness Overview")
        
        # Calculate missingness percentages
        missingness = df.isna().mean().reset_index()
        missingness.columns = ['Column', 'Missing_Percentage']
        missingness['Missing_Percentage'] = missingness['Missing_Percentage'] * 100
        missingness = missingness.sort_values('Missing_Percentage', ascending=False)
        
        # Show only columns with missing values
        missing_data = missingness[missingness['Missing_Percentage'] > 0]
        
        if not missing_data.empty:
            self.st.write("Columns with missing values:")
            
            # Create a bar chart of missingness
            fig = px.bar(
                missing_data.head(20),  # Show top 20 most missing
                x='Missing_Percentage',
                y='Column',
                orientation='h',
                title="Data Completeness Issues (Top 20)",
                color='Missing_Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(missing_data.head(20)) * 25)
            )
            self.st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed table
            self.st.dataframe(missing_data)
        else:
            self.st.success("✅ No missing values detected in any column")
    
    def create_data_view(self, df: pd.DataFrame) -> None:
        """
        Create data preview and download view (Data tab).
        
        Args:
            df: DataFrame to display
        """
        self.st.header("Data Preview")
        
        if df.empty:
            self.st.warning("No data available to display.")
            return
        
        # Show summary statistics
        col1, col2, col3 = self.st.columns(3)
        
        with col1:
            self.st.metric("Records", f"{len(df):,}")
        
        with col2:
            self.st.metric("Columns", f"{len(df.columns):,}")
        
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            self.st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        
        # Show data sample
        self.st.subheader("Data Sample (First 100 rows)")
        self.st.dataframe(df.head(100))
        
        # Download section
        self.st.subheader("Download Data")
        
        col1, col2 = self.st.columns(2)
        
        with col1:
            # CSV download
            csv_data = df.to_csv(index=False)
            self.st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name=f"kobo_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Survey_Data')
            excel_data = output.getvalue()
            
            self.st.download_button(
                label="📋 Download as Excel",
                data=excel_data,
                file_name=f"kobo_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )