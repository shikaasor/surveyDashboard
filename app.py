"""
KoboToolbox Analytics Dashboard - Main Entry Point

This is the main Streamlit application that provides an interactive dashboard
for KoboToolbox survey data analysis and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
import time
from datetime import datetime, timedelta
import threading
import schedule

# Import custom modules
from src.kobo_connector import KoboAPI
from src.data_processor import DataProcessor
from src.kpi_calculator import KPICalculator
from src.dashboard_components import DashboardComponents
from src.utils.config import load_config
from src.utils.auth import check_password
from src.utils.logging_utils import setup_logger
import plotly.express as px
import plotly.graph_objects as go

# Set up logger
logger = setup_logger()

# Load configuration
config = load_config()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = int(config.get('REFRESH_INTERVAL', 3600))

if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False


def fetch_and_process_data():
    """Fetch data from KoboToolbox and process it for dashboard use."""
    try:
        # Initialize KoboAPI
        kobo = KoboAPI(
            base_url=config['KOBO_BASE_URL'],
            asset_uid=config['KOBO_ASSET_UID'],
            api_token=config['KOBO_API_TOKEN'],
            export_settings_uid=config.get('KOBO_EXPORT_SETTINGS_UID')
        )
        
        # Download XLSX
        xlsx_file, settings = kobo.download_data_xlsx()
        
        # Process data
        processor = DataProcessor()
        data_dict = processor.process_xlsx(xlsx_file)
        
        # Update session state
        st.session_state.data = data_dict['main']
        st.session_state.last_refresh = datetime.now()
        
        # Save processed data to parquet
        os.makedirs('data/processed', exist_ok=True)
        st.session_state.data.to_parquet('data/processed/latest.parquet', index=False)
        
        logger.info(f"Data refreshed successfully: {len(st.session_state.data)} records")
        
        return True
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
        return False


def setup_scheduler():
    """Set up background scheduler for data refresh."""
    schedule.every(st.session_state.refresh_interval).seconds.do(fetch_and_process_data)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    # Start scheduler in a background thread
    threading.Thread(target=run_scheduler, daemon=True).start()


def load_cached_data():
    """Load data from cache if available."""
    try:
        if os.path.exists('data/processed/latest.parquet'):
            data = pd.read_parquet('data/processed/latest.parquet')
            st.session_state.data = data
            file_mtime = os.path.getmtime('data/processed/latest.parquet')
            st.session_state.last_refresh = datetime.fromtimestamp(file_mtime)
            logger.info(f"Loaded cached data: {len(data)} records")
            return True
        return False
    except Exception as e:
        logger.error(f"Error loading cached data: {str(e)}")
        return False


def main():
    """Main dashboard application."""
    st.title("ACE2 Site Assessment Analytics Dashboard")
    
    # Load data if not in session state
    if st.session_state.data is None:
        if not load_cached_data():
            st.info("No cached data available. Fetching fresh data...")
            with st.spinner("Fetching data from KoboToolbox..."):
                success = fetch_and_process_data()
                if not success:
                    st.error("Failed to fetch data. Please check your configuration.")
                    if st.button("Retry"):
                        st.rerun()
                    return
    
    # # Show last refresh time
    # if st.session_state.last_refresh:
    #     st.sidebar.write(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Manual refresh button (admin only)
    if st.session_state.is_admin:
        if st.sidebar.button("Refresh Now"):
            with st.spinner("Refreshing data..."):
                success = fetch_and_process_data()
                if success:
                    st.sidebar.success("Data refreshed successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to refresh data. See logs for details.")
    
    # Initialize components
    components = DashboardComponents(st.session_state)
    
    # Create filters
    components.create_filters(st.session_state.data)
    
    # Filter data based on selections
    filtered_df = components.filter_dataframe(st.session_state.data)
    
    # Initialize KPI calculator
    kpi_calculator = KPICalculator()
    
    # Calculate KPIs
    kpis = kpi_calculator.calculate_kpis(filtered_df)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Facility Comparison", 
        "Indicator Deep Dive", 
        "Data Quality", 
        "Data"
    ])
    
    with tab1:
        st.header("Overview")
        
        # Display KPI cards with maturity levels
        components.create_kpi_cards(kpis)
        
        # Add facility maturity distribution (corrected)
        st.subheader("Facility Maturity Distribution")
        
        # Calculate facility-level metrics to get correct maturity distribution
        facility_metrics = kpi_calculator.calculate_grouped_metrics(
            filtered_df, 
            group_by=['state', 'facility']
        )
        
        if not facility_metrics.empty and 'overall_composite_score' in facility_metrics.columns:
            # Get facility scores and calculate correct distribution
            facility_scores = facility_metrics['overall_composite_score'].dropna().tolist()
            facility_maturity_dist = kpi_calculator.get_maturity_distribution(facility_scores)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of maturity levels across facilities
                if facility_maturity_dist and facility_maturity_dist['total_facilities'] > 0:
                    import plotly.express as px
                    
                    # Prepare data for pie chart
                    level_counts = facility_maturity_dist.get('level_counts', {})
                    # Map level numbers to representative scores
                    level_to_score = {1: 0.25, 2: 0.75, 3: 1.25, 4: 1.75}
                    dist_data = pd.DataFrame([
                        {'Level': f"Level {level}: {kpi_calculator.get_maturity_level(level_to_score[level])['name']}", 
                         'Count': count, 
                         'Color': kpi_calculator.get_maturity_level(level_to_score[level])['color']}
                        for level, count in level_counts.items() if count > 0
                    ])
                    
                    if not dist_data.empty:
                        fig = px.pie(
                            dist_data,
                            values='Count',
                            names='Level',
                            title="Facility Maturity Level Distribution",
                            color='Level',
                            color_discrete_map={row['Level']: row['Color'] for _, row in dist_data.iterrows()}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary metrics for each maturity level
                st.write("**Facility Maturity Summary:**")
                total_facilities = facility_maturity_dist.get('total_facilities', 0)
                level_counts = facility_maturity_dist.get('level_counts', {})
                level_to_score = {1: 0.25, 2: 0.75, 3: 1.25, 4: 1.75}
                
                for level, count in level_counts.items():
                    if count > 0:
                        percentage = (count / total_facilities) * 100 if total_facilities > 0 else 0
                        level_info = kpi_calculator.get_maturity_level(level_to_score[level])
                        st.metric(
                            label=f"Level {level}: {level_info['name']}",
                            value=f"{count} facilities ({percentage:.1f}%)"
                        )
        else:
            st.warning("No facility data available for maturity distribution.")
        
        
        # Display state comparison with maturity levels
        state_metrics = kpi_calculator.calculate_grouped_metrics(
            filtered_df, 
            group_by=['state']
        )
        
        # State Performance Section
        num_states = len(state_metrics) if not state_metrics.empty else 0
        if num_states == 1:
            st.subheader(f"State Summary")
            st.write("*Note: Data currently contains facilities from one state.*")
        else:
            st.subheader("State Performance Comparison")
        
        if not state_metrics.empty:
            # Display with existing maturity level columns from grouped_metrics
            display_cols = ['state', 'overall_composite_score', 'overall_maturity_level', 'submission_count']
            available_cols = [col for col in display_cols if col in state_metrics.columns]
            
            if 'overall_composite_score' in state_metrics.columns:
                display_df = state_metrics[available_cols].copy()
                display_df = display_df.rename(columns={
                    'overall_composite_score': 'Overall Score',
                    'overall_maturity_level': 'Maturity Level', 
                    'submission_count': 'Facilities'
                })
                
                st.dataframe(
                    display_df.sort_values('Overall Score', ascending=False, na_position='last')
                )
            
            # Show domain-level maturity breakdown
            if num_states == 1:
                st.subheader("Domain-Level Performance Summary")
            else:
                st.subheader("Domain-Level Maturity by State")
            
            domain_cols = [col for col in state_metrics.columns if col.endswith('_maturity_level') and not col.startswith('overall_')]
            
            if domain_cols:
                # Create a summary of domain maturity levels
                if num_states == 1:
                    st.write("**Performance across assessment domains:**")
                else:
                    st.write("**Domain Maturity Levels Across States:**")
                
                # Show all states for single state, top 3 for multiple states
                display_states = state_metrics if num_states == 1 else state_metrics.nlargest(3, 'overall_composite_score')
                
                for _, state_row in display_states.iterrows():
                    state_name = state_row['state']
                    overall_level = state_row.get('overall_maturity_level', 'N/A')
                    
                    st.write(f"**{state_name}** (Overall: {overall_level})")
                    
                    # Show domain breakdown
                    domain_info = []
                    for col in domain_cols:
                        domain_name = col.replace('_maturity_level', '').replace('_', ' ').title()
                        level = state_row.get(col, 'N/A')
                        if level and level != 'No Data':
                            domain_info.append(f"{domain_name}: {level}")
                    
                    if domain_info:
                        st.write(f"   - {' | '.join(domain_info)}")
                    else:
                        st.write("   - No domain data available")
    
    with tab2:
        st.header("Facility Comparison")
        
        # Calculate facility metrics
        facility_metrics = kpi_calculator.calculate_grouped_metrics(
            filtered_df,
            group_by=['state', 'facility']
        )
        
        # Display facility comparison with maturity levels
        if 'overall_composite_score' in facility_metrics.columns:
            components.create_facility_comparison(facility_metrics, metric='overall_composite_score')
        else:
            st.warning("No overall composite score available for facility comparison.")
        
        # Display detailed facility table with maturity information
        st.subheader("Facility Performance Details")
        if not facility_metrics.empty:
            # Display using existing maturity columns from grouped_metrics
            display_cols = ['state', 'facility', 'overall_composite_score', 'overall_maturity_level_num', 'overall_maturity_level', 'submission_count']
            available_cols = [col for col in display_cols if col in facility_metrics.columns]
            
            if available_cols:
                display_df = facility_metrics[available_cols].copy()
                display_df = display_df.rename(columns={
                    'overall_composite_score': 'Overall Score',
                    'overall_maturity_level_num': 'Level',
                    'overall_maturity_level': 'Maturity Stage',
                    'submission_count': 'Submissions'
                })
                
                st.dataframe(
                    display_df.sort_values(['state', 'Overall Score'], ascending=[True, False], na_position='last')
                )
            
            # Add domain-level maturity breakdown for facilities
            st.subheader("Domain-Level Maturity Breakdown")
            
            # Get all domain maturity columns
            domain_maturity_cols = [col for col in facility_metrics.columns if col.endswith('_maturity_level') and not col.startswith('overall_')]
            
            if domain_maturity_cols:
                # Show detailed breakdown for top facilities
                if 'overall_composite_score' in facility_metrics.columns:
                    top_facilities = facility_metrics.nlargest(5, 'overall_composite_score')
                    
                    st.write("**Domain Maturity Levels for Top 5 Facilities:**")
                    
                    for idx, facility_row in top_facilities.iterrows():
                        facility_name = facility_row['facility']
                        state_name = facility_row['state']
                        overall_score = facility_row.get('overall_composite_score', 'N/A')
                        overall_level = facility_row.get('overall_maturity_level', 'N/A')
                        
                        st.write(f"**{facility_name} ({state_name})**")
                        st.write(f"Overall: {overall_score:.2f} - {overall_level}")
                        
                        # Show domain breakdown
                        col1, col2 = st.columns(2)
                        domain_count = len(domain_maturity_cols)
                        mid_point = (domain_count + 1) // 2
                        
                        with col1:
                            for col in domain_maturity_cols[:mid_point]:
                                domain_name = col.replace('_maturity_level', '').replace('_', ' ').title()
                                level = facility_row.get(col, 'N/A')
                                score_col = col.replace('_maturity_level', '_group_score')
                                score = facility_row.get(score_col, 'N/A')
                                score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                st.write(f"   • **{domain_name}**: {level} ({score_text})")
                        
                        with col2:
                            for col in domain_maturity_cols[mid_point:]:
                                domain_name = col.replace('_maturity_level', '').replace('_', ' ').title()
                                level = facility_row.get(col, 'N/A')
                                score_col = col.replace('_maturity_level', '_group_score')
                                score = facility_row.get(score_col, 'N/A')
                                score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                st.write(f"   • **{domain_name}**: {level} ({score_text})")
                        
                        st.write("---")
            else:
                st.info("No domain-level maturity data available. This might indicate an issue with the KPI calculation.")
    
    with tab3:
        st.header("Indicator Deep Dive")
        components.create_indicator_deep_dive(filtered_df, kpi_calculator)
    
    with tab4:
        components.create_data_quality_view(filtered_df)
    
    with tab5:
        components.create_data_view(filtered_df)


def admin_page():
    """Admin settings page."""
    st.title("Admin Settings")
    
    # Set refresh interval
    st.session_state.refresh_interval = st.number_input(
        "Refresh Interval (seconds)",
        min_value=300,  # Minimum 5 minutes
        max_value=86400,  # Maximum 1 day
        value=st.session_state.refresh_interval,
        step=300
    )
    
    if st.button("Save Settings"):
        # Update scheduler
        setup_scheduler()
        st.success("Settings saved successfully!")
    
    # Manual refresh button
    if st.button("Refresh Data Now"):
        with st.spinner("Refreshing data..."):
            success = fetch_and_process_data()
            if success:
                st.success("Data refreshed successfully!")
            else:
                st.error("Failed to refresh data. See logs for details.")
    
    # View logs
    st.subheader("Application Logs")
    try:
        if os.path.exists('logs/app.log'):
            with open('logs/app.log', 'r') as f:
                logs = f.readlines()
            st.code(''.join(logs[-50:]))  # Show last 50 lines
        else:
            st.info("No logs available")
    except Exception:
        st.info("No logs available")


if __name__ == "__main__":
    # Set up page config
    st.set_page_config(
        page_title="KoboToolbox Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Check if admin area requested
    query_params = st.query_params
    if 'page' in query_params and query_params['page'] == 'admin':
        # Authenticate admin
        if not st.session_state.is_admin:
            if check_password(config.get('ADMIN_PASSWORD')):
                st.session_state.is_admin = True
                admin_page()
            else:
                st.error("Invalid password")
        else:
            admin_page()
    else:
        # Set up scheduler if not already running
        if 'scheduler_started' not in st.session_state:
            setup_scheduler()
            st.session_state.scheduler_started = True
        
        # Run the main app
        main()