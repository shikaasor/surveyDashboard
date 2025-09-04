# KoboToolbox Analytics Dashboard

A Streamlit-based analytics dashboard that automatically fetches KoboToolbox form data, processes it, and presents interactive visualizations for M&E teams and program managers.

## Features

- **Automatic Data Ingestion**: Connects to KoboToolbox API and fetches XLSX exports
- **Data Processing Pipeline**: Cleans and normalizes survey data with quality checks
- **Interactive Dashboard**: Five specialized tabs for different analysis needs
- **KPI Calculations**: Implementation scores and metrics per PRD specifications
- **Real-time Filtering**: Date range, state, facility, and version filters
- **Data Export**: Download cleaned data in CSV/Excel format
- **Admin Controls**: Manual refresh, scheduling, and system monitoring

## Dashboard Tabs

1. **Overview**: KPI cards, trends, and state comparisons
2. **Facility Comparison**: Rankings and performance analysis
3. **Indicator Deep Dive**: Detailed analysis of specific indicators
4. **Data Quality**: Duplicate detection and completeness monitoring  
5. **Data**: Preview and download functionality

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd surveyDashboard

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your KoboToolbox credentials:

```env
KOBO_BASE_URL=https://kf.kobotoolbox.org
KOBO_ASSET_UID=your_asset_uid_here
KOBO_API_TOKEN=your_api_token_here
ADMIN_PASSWORD=your_secure_password
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### 4. Admin Access

Visit `http://localhost:8501/?page=admin` to access:
- Manual data refresh
- Schedule configuration
- System logs and health monitoring

## Configuration Details

### Required Settings

- `KOBO_BASE_URL`: KoboToolbox server URL (default: https://kf.kobotoolbox.org)
- `KOBO_ASSET_UID`: Your form's asset UID from KoboToolbox
- `KOBO_API_TOKEN`: API token with "view_submissions" permission

### Optional Settings

- `KOBO_EXPORT_SETTINGS_UID`: Specific export configuration (uses default if not set)
- `REFRESH_INTERVAL`: Auto-refresh interval in seconds (default: 3600 = 1 hour)
- `ADMIN_PASSWORD`: Password for admin access (default: "admin")
- `LOG_LEVEL`: Logging level (default: "INFO")

## API Token Setup

1. Log into your KoboToolbox account
2. Go to Account Settings → API Token
3. Generate a new token with "view_submissions" permission
4. Copy the token to your `.env` file

## Data Processing

The system automatically:

1. **Fetches XLSX** from KoboToolbox export-settings endpoint
2. **Normalizes columns** by replacing `/` with `_` 
3. **Converts data types** (dates to UTC, scores to numeric)
4. **Adds derived fields** (submission_date, normalized location names)
5. **Performs quality checks** (duplicates, missing critical fields)
6. **Calculates KPIs** per PRD mathematical definitions

## KPI Definitions

Following PRD Section 9 exact formulas:

- **Average Score**: `mean(v for v in S[x] if v in {0,1,2})` (excludes n/a)
- **% Fully Implemented**: `count(v==2)/count(v in {0,1,2})`
- **% Partially Implemented**: `count(v==1)/count(v in {0,1,2})`
- **% Not Implemented**: `count(v==0)/count(v in {0,1,2})`
- **Composite Score**: `mean(avg_score_x for x in infrastructure_indicators)`

## Architecture

```
├── app.py                      # Main Streamlit application
├── src/
│   ├── kobo_connector.py       # KoboToolbox API client
│   ├── data_processor.py       # Data cleaning and processing
│   ├── kpi_calculator.py       # KPI computation engine
│   ├── dashboard_components.py # Reusable UI components
│   └── utils/
│       ├── config.py          # Configuration management
│       ├── auth.py            # Authentication utilities
│       └── logging_utils.py   # Structured logging
├── data/
│   ├── cache/                 # Downloaded XLSX files
│   └── processed/             # Cleaned parquet files
└── logs/                      # Application logs
```

## Performance

- **Dashboard Load**: ≤3s (P50), ≤6s (P95) with cached data
- **Data Processing**: ≤2 minutes for 100k records
- **Auto-refresh**: Configurable (default 1 hour)
- **Caching**: Local parquet files for fast access

## Security

- API tokens stored in environment variables only
- Sensitive data filtered from logs
- Admin access password protected
- No PII exposure in client-side code

## Troubleshooting

### Common Issues

**"Failed to fetch data"**
- Check KOBO_API_TOKEN has correct permissions
- Verify KOBO_ASSET_UID is correct
- Check network connectivity to KoboToolbox

**"No export settings found"**  
- Create export settings in KoboToolbox UI first
- Or specify KOBO_EXPORT_SETTINGS_UID manually

**Dashboard loads slowly**
- Increase refresh interval to reduce processing frequency
- Check data volume (optimize for <100k records)

### Logs

Application logs are available in:
- Console output (when running locally)
- `logs/app.log` (file with rotation)
- Admin page log viewer

### Health Check

The system provides health monitoring at the admin page:
- Last refresh timestamp
- Processing statistics
- Error logs and status

## Development

### Adding New Indicators

1. Ensure columns follow `infrastructure_*` naming pattern
2. Values should be 0/1/2 for implementation levels, n/a for not applicable
3. KPI calculator will automatically include new indicators

### Customizing Visualizations

Modify `src/dashboard_components.py` to add new charts or change layouts.

### Testing

```bash
# Run tests (when implemented)
pytest src/tests/

# Check code quality
ruff check src/ app.py
mypy src/ app.py
```

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub repository
2. Connect Streamlit Community Cloud to repository
3. Add environment variables in Streamlit Cloud settings
4. Deploy automatically

### Other Platforms

Compatible with:
- Render
- Heroku
- Railway
- DigitalOcean App Platform

## Support

For issues and questions:
1. Check troubleshooting section above
2. Review application logs
3. Verify KoboToolbox API connectivity
4. Create issue in project repository

## License

See LICENSE file for details.