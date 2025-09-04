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
            api_token=config['KOBO_API_TOKEN']
        )
        
        # Get export settings UID
        export_settings_uid = config.get('KOBO_EXPORT_SETTINGS_UID')
        
        # Download XLSX
        xlsx_file, settings = kobo.download_data_xlsx(export_settings_uid)
        
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
    st.title("KoboToolbox Analytics Dashboard")
    
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
    
    # Show last refresh time
    if st.session_state.last_refresh:
        st.sidebar.write(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
        
        # Display KPI cards
        components.create_kpi_cards(kpis)
        
        # Display time series
        time_series = kpi_calculator.calculate_time_series(filtered_df)
        components.create_time_series_chart(time_series)
        
        # Display state comparison
        state_metrics = kpi_calculator.calculate_grouped_metrics(
            filtered_df, 
            group_by=['state']
        )
        
        st.subheader("State Performance")
        if not state_metrics.empty:
            st.dataframe(
                state_metrics[['state', 'overall_infrastructure_score', 'submission_count']]
                .sort_values('overall_infrastructure_score', ascending=False)
            )
    
    with tab2:
        st.header("Facility Comparison")
        
        # Calculate facility metrics
        facility_metrics = kpi_calculator.calculate_grouped_metrics(
            filtered_df,
            group_by=['state', 'facility']
        )
        
        # Display facility comparison
        components.create_facility_comparison(facility_metrics)
        
        # Display detailed facility table
        st.subheader("Facility Performance Details")
        if not facility_metrics.empty:
            st.dataframe(
                facility_metrics[['state', 'facility', 'overall_infrastructure_score', 'submission_count']]
                .sort_values(['state', 'overall_infrastructure_score'], ascending=[True, False])
            )
    
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