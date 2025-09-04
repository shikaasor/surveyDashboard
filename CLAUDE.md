# claude.md: KoboToolbox Analytics Dashboard (Streamlit)

## Goal
Build a Streamlit dashboard that fetches KoboToolbox form data via synchronous exports API, processes it into structured datasets, computes standardized indicators per PRD specifications, and presents interactive dashboards with filtering capabilities.

## Success Criteria
- Dashboard shows KoboToolbox data refreshed within 10 minutes of submission
- Data processing pipeline handles 100k+ records with ≤200 columns  
- Dashboard loads in ≤3 seconds (P50) and ≤6 seconds (P95)
- Implementation score calculations match PRD definitions exactly
- Time series analysis shows trends over time correctly
- Data quality checks identify duplicates and missing critical fields

## Required Components

### Core Files Structure
- app.py: Main Streamlit application entry point
- src/kobo_connector.py: API connections and data retrieval
- src/data_processor.py: XLSX parsing and cleaning per PRD specs
- src/kpi_calculator.py: KPI computation matching PRD definitions
- src/dashboard_components.py: Streamlit visualization components
- src/utils/config.py: Configuration handling
- requirements.txt: Dependencies (streamlit, pandas, openpyxl, requests, python-dotenv)

### API Integration
- KoboToolbox API: https://support.kobotoolbox.org/api.html
- Export settings endpoint: /api/v2/assets/{asset_uid}/export-settings/
- Authentication via Token header
- XLSX download from data_url_xlsx field

### Critical Implementation Notes
- KoboToolbox API token needs "view_submissions" permission
- XLSX contains repeat groups in separate sheets - handle multi-sheet parsing
- Column names use / separators that must be normalized to _ 
- Missing values may be empty strings, NULL, or 'n/a' - normalize consistently
- Use Streamlit session state for data persistence between re-runs
- Implement retry/backoff for API rate limiting

## Data Processing Requirements

### Environment Variables
```
KOBO_BASE_URL=https://kf.kobotoolbox.org
KOBO_ASSET_UID=your_asset_uid
KOBO_API_TOKEN=your_api_token
KOBO_EXPORT_SETTINGS_UID=optional_specific_export_setting_uid
REFRESH_INTERVAL=3600
ADMIN_PASSWORD=secure_password
```

### Data Processing Steps
1. Fetch XLSX via export-settings endpoint
2. Normalize column names: replace / with _
3. Convert dates to UTC datetime, scores to numeric (0,1,2) or None for n/a
4. Add derived fields: submission_date, state, facility (title-cased)
5. Create derived indicator columns (*_is_full, *_is_partial, *_is_not)
6. Data quality checks: duplicates by _uuid, missing critical fields

## KPI Calculations (per PRD section 9)

For indicator x and selected submissions S:

### Average score (excluding n/a)
```
avg_score_x = mean(v for v in S[x] if v in {0,1,2})
```

### Implementation percentages
```
pct_full_x = count(v==2)/count(v in {0,1,2})
pct_partial_x = count(v==1)/count(v in {0,1,2}) 
pct_not_x = count(v==0)/count(v in {0,1,2})
```

### Composite infrastructure score
```
composite_avg = mean(avg_score_x for x in infrastructure_indicators)
```

## Dashboard Requirements (per PRD section 6.4)

### Global Filters
- Date range, state multi-select, facility multi-select, version filter

### Required Tabs
1. **Overview**: KPI cards, recent trend, state comparison, top/bottom facilities
2. **Facility Comparison**: Sortable table and bar charts by indicator group  
3. **Indicator Deep Dive**: Pick indicator → distribution by state/facility over time
4. **Data Quality**: Duplicates, missingness, last refresh time, pipeline status
5. **Data**: Preview cleaned table, download buttons (CSV/XLSX)

### Admin Features
- Simple password protection
- Manual "Refresh Now" button
- Configure refresh schedule  
- View logs and pipeline health

## Performance Requirements (per PRD section 7)

- Dashboard load time ≤ 3s (P50), ≤ 6s (P95) with cached data
- Full refresh pipeline ≤ 2 minutes for 100k rows  
- Retry with exponential backoff on 429/5xx errors
- Fallback to last successful dataset when refresh fails
- TLS for all deployments, API token not exposed client-side


## Validation & Testing

### Essential Tests
```bash
# Syntax check
ruff check src/ app.py --fix
mypy src/ app.py

# Test KoboToolbox API connection works
# Test KPI calculations match PRD formulas exactly
# Test dashboard loads with filtered data
# Test CSV/XLSX downloads work
```

### Deployment Checklist
- API token configured with "view_submissions" permission
- Dashboard loads in ≤3s (P50), ≤6s (P95)
- KPI calculations match PRD section 9 exactly
- All 5 tabs functional: Overview, Facility Comparison, Indicator Deep Dive, Data Quality, Data
- Admin page accessible with password protection
- Error handling graceful with fallback to cached data