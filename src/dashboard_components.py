"""
Reusable Streamlit dashboard components.

This module provides reusable UI components for the KoboToolbox Analytics Dashboard,
including filters, visualizations, and data quality views.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

from .utils.logging_utils import setup_logger
from .kpi_calculator import KPICalculator

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
        # Add logo at the top of sidebar
        self._add_sidebar_logo()
        
        if df.empty:
            self.st.sidebar.warning("No data available for filtering")
            return
        
        self.st.sidebar.title("Filters")
        
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
        
        # Apply state filter
        if (hasattr(self.session, 'selected_states') and self.session.selected_states and 
            'state' in filtered_df.columns):
            filtered_df = filtered_df[filtered_df['state'].isin(self.session.selected_states)]
        
        # Apply facility filter
        if (hasattr(self.session, 'selected_facilities') and self.session.selected_facilities and 
            'facility' in filtered_df.columns):
            filtered_df = filtered_df[filtered_df['facility'].isin(self.session.selected_facilities)]
        
        
        return filtered_df
    
    def create_kpi_cards(self, kpis: Dict[str, Any]) -> None:
        """
        Create KPI cards display with maturity levels (Overview tab).
        
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
            # Use overall composite score instead of infrastructure score
            score = kpis.get('overall_composite_score')
            if score is not None:
                self.st.metric(
                    label="Overall Composite Score",
                    value=f"{score:.2f}"
                )
            else:
                self.st.metric(
                    label="Overall Composite Score",
                    value="N/A"
                )
        
        with col3:
            # Display overall maturity level instead of percentage
            maturity_info = kpis.get('overall_maturity_level', {})
            if maturity_info:
                level_name = maturity_info.get('name', 'Unknown')
                level_number = maturity_info.get('level', 'N/A')
                self.st.metric(
                    label="Maturity Level",
                    value=f"Level {level_number}: {level_name}"
                )
            else:
                self.st.metric(
                    label="Maturity Level",
                    value="N/A"
                )
        
        with col4:
            self.st.metric(
                label="Facilities Covered",
                value=f"{kpis.get('unique_facilities', 0):,}"
            )
    
    def create_facility_comparison(self, grouped_metrics_df: pd.DataFrame, 
                                 metric: str = 'overall_infrastructure_score', 
                                 top_n: int = 10) -> None:
        """
        Create facility comparison chart with maturity levels (Facility Comparison tab).
        
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
        
        # Use existing maturity level columns if available, otherwise calculate
        if 'overall_maturity_level' in top_facilities.columns:
            top_facilities['maturity_name'] = top_facilities['overall_maturity_level']
            # Map maturity names to colors
            color_map = {
                'Basic': '#FF4444',
                'Developing': '#FFAA00', 
                'Advancing': '#4488FF',
                'Mature': '#00AA44',
                'No Data': '#CCCCCC'
            }
            top_facilities['maturity_color'] = top_facilities['maturity_name'].map(color_map).fillna('#CCCCCC')
        else:
            # Fallback to calculation if columns don't exist
            top_facilities['maturity_level'] = top_facilities[metric].apply(
                lambda x: KPICalculator.get_maturity_level(x) if pd.notna(x) else None
            )
            top_facilities['maturity_name'] = top_facilities['maturity_level'].apply(
                lambda x: x['name'] if x else 'N/A'
            )
            top_facilities['maturity_color'] = top_facilities['maturity_level'].apply(
                lambda x: x['color'] if x else '#CCCCCC'
            )
        
        # Convert the metric name for display
        metric_name = metric.replace('_', ' ').title()
        
        # Create facility label with maturity level
        if 'state' in top_facilities.columns and 'facility' in top_facilities.columns:
            top_facilities['facility_label'] = (
                top_facilities['facility'].astype(str) + 
                " (" + top_facilities['state'].astype(str) + ")"
            )
        else:
            top_facilities['facility_label'] = top_facilities.get('facility', 'Unknown').astype(str)
        
        self.st.subheader(f"Top {len(top_facilities)} Facilities by {metric_name}")
        
        # Create horizontal bar chart with maturity level colors
        fig = px.bar(
            top_facilities,
            x=metric,
            y='facility_label',
            orientation='h',
            title=f"Top Facilities by {metric_name} (Colored by Maturity Level)",
            color='maturity_name',
            color_discrete_map={
                'Basic': '#FF4444',
                'Developing': '#FFAA00', 
                'Advancing': '#4488FF',
                'Mature': '#00AA44',
                'N/A': '#CCCCCC'
            },
            hover_data={'maturity_name': True}
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, len(top_facilities) * 35),
            xaxis_title=metric_name,
            yaxis_title="Facility",
            legend_title="Maturity Level"
        )
        self.st.plotly_chart(fig, use_container_width=True)
        
        # Add maturity level distribution summary and domain breakdown
        self.st.subheader("Maturity Level Distribution")
        maturity_dist = top_facilities['maturity_name'].value_counts()
        
        col1, col2 = self.st.columns(2)
        with col1:
            # Pie chart of maturity levels
            fig_pie = px.pie(
                values=maturity_dist.values,
                names=maturity_dist.index,
                title="Overall Maturity Distribution",
                color=maturity_dist.index,
                color_discrete_map={
                    'Basic': '#FF4444',
                    'Developing': '#FFAA00', 
                    'Advancing': '#4488FF',
                    'Mature': '#00AA44',
                    'No Data': '#CCCCCC',
                    'N/A': '#CCCCCC'
                }
            )
            self.st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Summary table
            for level, count in maturity_dist.items():
                percentage = (count / len(top_facilities)) * 100
                self.st.metric(
                    label=f"Level: {level}",
                    value=f"{count} facilities ({percentage:.1f}%)"
                )
        
        # Domain-level maturity heatmap
        domain_maturity_cols = [col for col in grouped_metrics_df.columns 
                               if col.endswith('_maturity_level') and not col.startswith('overall_')]
        
        if domain_maturity_cols and len(grouped_metrics_df) > 1:
            self.st.subheader("Domain-Level Maturity Heatmap")
            
            # Create a simplified heatmap data
            heatmap_data = []
            for _, row in grouped_metrics_df.head(top_n).iterrows():
                facility_name = f"{row.get('facility', 'Unknown')} ({row.get('state', 'Unknown')})"
                for col in domain_maturity_cols:
                    domain_name = col.replace('_maturity_level', '').replace('_', ' ').title()
                    maturity_level = row.get(col, 'No Data')
                    
                    # Convert maturity level to numeric for heatmap
                    level_map = {'Basic': 1, 'Developing': 2, 'Advancing': 3, 'Mature': 4, 'No Data': 0}
                    level_num = level_map.get(maturity_level, 0)
                    
                    heatmap_data.append({
                        'Facility': facility_name,
                        'Domain': domain_name,
                        'Maturity_Level': maturity_level,
                        'Level_Numeric': level_num
                    })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                
                # Create heatmap
                pivot_data = heatmap_df.pivot(index='Facility', columns='Domain', values='Level_Numeric')
                
                fig = px.imshow(
                    pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale=[
                        [0, '#CCCCCC'],    # No Data
                        [0.25, '#FF4444'], # Basic
                        [0.5, '#FFAA00'],  # Developing
                        [0.75, '#4488FF'], # Advancing
                        [1.0, '#00AA44']   # Mature
                    ],
                    title="Domain Maturity Levels by Facility",
                    labels={'color': 'Maturity Level'}
                )
                
                # Update layout for better readability
                fig.update_layout(
                    height=max(400, len(pivot_data.index) * 30),
                    xaxis_title="Domain",
                    yaxis_title="Facility"
                )
                
                # Add custom colorbar labels
                fig.update_coloraxes(
                    colorbar=dict(
                        tickmode='array',
                        tickvals=[0, 1, 2, 3, 4],
                        ticktext=['No Data', 'Basic', 'Developing', 'Advancing', 'Mature']
                    )
                )
                
                self.st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed domain breakdown for top facility
                if not pivot_data.empty:
                    self.st.subheader("Detailed Domain Breakdown (Top Facility)")
                    top_facility_name = pivot_data.index[0]
                    
                    self.st.write(f"**{top_facility_name}** - Domain Details:")
                    
                    top_facility_data = heatmap_df[heatmap_df['Facility'] == top_facility_name]
                    
                    cols = self.st.columns(min(3, len(top_facility_data)))
                    for i, (_, domain_row) in enumerate(top_facility_data.iterrows()):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            domain = domain_row['Domain']
                            level = domain_row['Maturity_Level']
                            level_num = domain_row['Level_Numeric']
                            
                            # Get the color for this level
                            colors = {'Basic': '🔴', 'Developing': '🟡', 'Advancing': '🔵', 'Mature': '🟢', 'No Data': '⚫'}
                            icon = colors.get(level, '⚫')
                            
                            self.st.metric(
                                label=f"{icon} {domain}",
                                value=f"Level {level_num}" if level_num > 0 else "No Data",
                                delta=level
                            )
    
    def create_indicator_deep_dive(self, df: pd.DataFrame, kpi_calculator) -> None:
        """
        Create indicator deep dive analysis with maturity levels (Indicator Deep Dive tab).
        
        Args:
            df: Filtered dataframe
            kpi_calculator: KPI calculator instance
        """
        # Get all indicator groups (domains)
        all_groups = kpi_calculator.get_indicator_groups()
        
        if not all_groups:
            self.st.warning("No indicator groups found in the data.")
            return
        
        # Format group names for display
        group_options = {
            group: group.replace('_', ' ').title() 
            for group in all_groups
        }
        
        selected_group = self.st.selectbox(
            "Select Domain for Analysis",
            options=list(group_options.keys()),
            format_func=lambda x: group_options[x],
            help="Each domain represents a group of related indicators (e.g., Infrastructure, Staffing)"
        )
        
        if selected_group:
            domain_name = group_options[selected_group]
            
            # Calculate domain KPIs for all facilities  
            domain_kpis = kpi_calculator.calculate_group_kpis(df, selected_group)
            
            if not domain_kpis:
                self.st.warning(f"No data available for {domain_name} domain.")
                return
            
            # Display domain-level metrics
            col1, col2, col3 = self.st.columns(3)
            
            with col1:
                avg_score = domain_kpis.get('group_average')
                if avg_score is not None:
                    self.st.metric(
                        label=f"{domain_name} Average Score",
                        value=f"{avg_score:.2f}"
                    )
            
            with col2:
                maturity_info = domain_kpis.get('maturity_level', {})
                if maturity_info:
                    level_name = maturity_info.get('name', 'Unknown')
                    level_number = maturity_info.get('level', 'N/A')
                    color = maturity_info.get('color', '#CCCCCC')
                    
                    # Display with color coding
                    self.st.metric(
                        label=f"{domain_name} Maturity Level",
                        value=f"Level {level_number}: {level_name}"
                    )
                    
                    # Add visual indicator
                    colors = {'Basic': '🔴', 'Developing': '🟡', 'Advancing': '🔵', 'Mature': '🟢'}
                    icon = colors.get(level_name, '⚫')
                    self.st.write(f"{icon} **{level_name}** maturity level")
            
            with col3:
                valid_responses = domain_kpis.get('total_responses', 0)
                self.st.metric(
                    label="Valid Responses",
                    value=f"{valid_responses:,}"
                )
            
            # Create columns for side-by-side analysis
            col1, col2 = self.st.columns(2)
            
            with col1:
                # State-level analysis for this group
                if 'state' in df.columns:
                    self.st.subheader(f"{domain_name} by State")
                    
                    # Calculate group scores by state
                    state_scores = []
                    for state in df['state'].dropna().unique():
                        state_df = df[df['state'] == state]
                        state_kpis = kpi_calculator.calculate_group_kpis(state_df, selected_group)
                        if state_kpis and state_kpis.get('group_average') is not None:
                            maturity_info = state_kpis.get('maturity_level', {})
                            state_scores.append({
                                'state': state,
                                'score': state_kpis['group_average'],
                                'maturity_level': maturity_info.get('name', 'Unknown')
                            })
                    
                    if state_scores:
                        state_df_chart = pd.DataFrame(state_scores)
                        state_df_chart = state_df_chart.sort_values('score', ascending=False)
                        
                        fig = px.bar(
                            state_df_chart,
                            x='state',
                            y='score',
                            title=f"{domain_name} Average Score by State",
                            color='maturity_level',
                            color_discrete_map={
                                'Basic': '#FF4444',
                                'Developing': '#FFAA00', 
                                'Advancing': '#4488FF',
                                'Mature': '#00AA44',
                                'Unknown': '#CCCCCC'
                            }
                        )
                        fig.update_layout(height=400)
                        self.st.plotly_chart(fig, use_container_width=True)
                    else:
                        self.st.warning("No state data available.")
            
            with col2:
                # Individual indicators within this group
                group_indicators = kpi_calculator._get_group_indicators(df, selected_group)
                
                if group_indicators:
                    self.st.subheader(f"Individual Indicators in {domain_name}")
                    
                    indicator_scores = []
                    for indicator in group_indicators:
                        valid_values = df[indicator].dropna()
                        if len(valid_values) > 0:
                            avg_score = valid_values.mean()
                            indicator_scores.append({
                                'indicator': indicator.replace(f'{selected_group}_', '').replace('_', ' ').title(),
                                'score': avg_score,
                                'count': len(valid_values)
                            })
                    
                    if indicator_scores:
                        indicator_df = pd.DataFrame(indicator_scores)
                        indicator_df = indicator_df.sort_values('score', ascending=True)
                        
                        fig = px.bar(
                            indicator_df,
                            x='score',
                            y='indicator',
                            orientation='h',
                            title=f"Individual Indicators in {domain_name}",
                            color='score',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(
                            height=max(400, len(indicator_df) * 30),
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        self.st.plotly_chart(fig, use_container_width=True)
                    else:
                        self.st.warning("No individual indicator data available.")
            
    
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
            
            # Create a copy of the dataframe and strip timezone info from datetime columns
            # Excel doesn't support timezone-aware datetimes
            df_for_excel = df.copy()
            for col in df_for_excel.columns:
                if pd.api.types.is_datetime64_any_dtype(df_for_excel[col]):
                    # Convert timezone-aware datetimes to timezone-unaware
                    if df_for_excel[col].dt.tz is not None:
                        df_for_excel[col] = df_for_excel[col].dt.tz_localize(None)
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_for_excel.to_excel(writer, index=False, sheet_name='Survey_Data')
            excel_data = output.getvalue()
            
            self.st.download_button(
                label="📋 Download as Excel",
                data=excel_data,
                file_name=f"kobo_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def _add_sidebar_logo(self) -> None:
        """Add logo to the top of the sidebar."""
        try:
            # Try to load logo from assets directory
            logo_path = "assets/logo.png"
            if os.path.exists(logo_path):
                # Embed image in white box
                self.st.sidebar.markdown(
                    f"""
                    <div style="
                        background-color: white;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        text-align: center;
                    ">
                        <img src="data:image/png;base64,{self._get_base64_image(logo_path)}" 
                             style="max-width: 180px; height: auto;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Fallback to text-based logo in white box
                self.st.sidebar.markdown(
                    """
                    <div style="
                        background-color: white;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        text-align: center;
                    ">
                        <h2 style="color: #2E86C1; margin: 0;">🏥</h2>
                        <h3 style="color: #2E86C1; margin: 5px 0;">ACE2</h3>
                        <p style="color: #85929E; margin: 0; font-size: 12px;">Site Assessment</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
        except Exception as e:
            # If there's any error, just skip the logo
            logger.debug(f"Could not add sidebar logo: {str(e)}")
            pass
    
    def _get_base64_image(self, image_path: str) -> str:
        """Convert image to base64 string for embedding."""
        import base64
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()