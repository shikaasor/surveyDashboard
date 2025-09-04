PRD: Kobo → Analytics Dashboard (Streamlit)
0. Document Purpose

This PRD defines the product vision, scope, requirements, data model, architecture, UX, and delivery plan for an application that fetches KoboToolbox form data (XLSX via synchronous exports), parses and analyzes it, and presents interactive dashboards to stakeholders. It is written from the perspective of a Product Manager and a Senior Software Engineer and is directly actionable by a development team.

1. Background & Problem

Monitoring & Evaluation (M&E) teams collect data with KoboCollect/Enketo, then spend time exporting, merging, cleaning, and manually building charts before sharing insights. This slows decision-making and introduces errors.

Solution: A web app that automatically ingests Kobo submissions, structures them into clean datasets, computes standardized indicators, and exposes interactive dashboards, exports, and scheduled refreshes.

2. Goals & Success Metrics

Goals

Zero-manual pipeline from Kobo to dashboard.

Consistent, well-structured datasets for analysis and export.

Intuitive dashboard with filters and drill-downs for states/facilities/indicators.

Success Metrics

Time to new data visible on dashboard ≤ 10 minutes (driven by Kobo sync cadence).

≥ 90% reduction in manual Excel wrangling for target workflows.

Dashboard load time ≤ 3 seconds for common filters (P50), ≤ 6 seconds (P95) for 100k-row datasets.

Data completeness checks cover ≥ 95% of critical fields.

Non-goals (Phase 1)

Complex role-based multi-tenant SaaS.

Write-back to Kobo or editing submissions.

Advanced geospatial analytics beyond simple choropleths.

3. Users & Use Cases

Personas

Program Manager: Needs high-level KPIs and quick comparisons, downloadable summaries.

M&E Officer / Analyst: Needs filters, cross-tabs, time trends, data-quality checks.

Country/Regional Lead: Needs state- and facility-level performance snapshots.

Data Engineer/Admin: Configures connections, schedules, and monitors pipeline health.

Primary Use Cases

View current KPIs by state/facility with scores (0/1/2/N/A).

Compare facilities and identify low-performing indicators.

Track trends over time (submissions per week/month; indicator progress).

Export cleaned data to CSV/XLSX for ad hoc analysis.

Monitor ingestion status and data quality (duplicates, missingness).

4. Scope

In Scope (MVP)

Connect to a single Kobo project via API token.

Use synchronous export XLSX (handles repeat groups).

Parse and normalize into a tabular model; flatten group paths.

KPI definitions for “implementation score” questions (0/1/2/N/A).

Interactive Streamlit dashboard with filters (date/state/facility).

Download buttons (clean CSV/XLSX).

Scheduled refresh (hourly/daily).

Out of Scope (MVP)

Multi-project management UI.

User management beyond a single admin password (MVP uses basic app password).

Push notifications and email reports (planned later).

5. Assumptions & Constraints

Kobo API token available; user has “Manage project” permission.

Synchronous export cache: ~5-minute cadence (app should not expect sub-minute freshness).

Some surveys include repeat groups; XLSX endpoint chosen over CSV.

Data volume target (MVP): up to 100k records, ≤ 200 columns.

Network reliability variable; pipeline must retry/backoff.

6. Functional Requirements
6.1 Connections & Ingestion

FR-1: Configure Kobo server base URL, Asset UID, and API token via environment variables or .env.

FR-2: Retrieve synchronous export settings and resolve data_url_xlsx.

FR-3: Fetch XLSX with authentication; respect caching (≥5 min).

FR-4: Scheduled refresh cadence configurable (default hourly).

FR-5: Manual “Refresh Now” button in admin view.

6.2 Parsing & Transformation

FR-6: Load XLSX sheet(s), flatten column names: replace / with _ (e.g., group_infrastructure/waiting_area → infrastructure_waiting_area).

FR-7: Standardize types: dates (start, end, _submission_time) to UTC ISO; numeric scores to Int64 or nullable floats; keep n/a as a distinct category when present.

FR-8: Add derived fields:

submission_date from _submission_time (fallback end).

state, facility normalized (title-cased, human-readable).

source_version from __version__ if available.

FR-9: Data quality checks:

Duplicate _uuid detection.

Missing critical fields (state, facility, submission_date).

Out-of-range scores (values not in {0,1,2,'n/a'}).

FR-10: Persist a cleaned dataset (parquet/CSV) and a small SQLite (MVP); PostgreSQL optional in Phase 2.

6.3 Analysis & KPIs

FR-11: Compute per-facility and per-state:

Average implementation score across selected indicators (exclude n/a from average).

% fully implemented (share of responses == 2).

% partially implemented (== 1).

% not implemented (== 0).

FR-12: Time series: submissions by week/month; optional rolling average of key metrics.

FR-13: Indicator group summaries (e.g., infrastructure_* fields aggregated).

6.4 Dashboard (Streamlit)

FR-14: Global filters: date range, state multi-select, facility multi-select, version filter.

FR-15: Tabs:

Overview: KPI cards, recent trend, state map (optional), top/bottom facilities.

Facility Comparison: sortable table and bar charts by indicator group.

Indicator Deep Dive: pick an indicator → distribution by state/facility over time.

Data Quality: duplicates, missingness, last refresh time, pipeline status.

Data: preview cleaned table; download buttons.

FR-16: Download cleaned data (CSV) and current view (filtered CSV).

FR-17: Display metadata: last refresh timestamp, number of submissions, active export settings UID/name.

6.5 Admin & Security

FR-18: Admin page (simple password) to set schedule (cron string), run now, view logs/health.

FR-19: Secrets (API token) stored in environment variables; never logged.

FR-20: Basic access protection (Streamlit auth plugin or reverse-proxy Basic Auth for MVP).

7. Non-Functional Requirements

Performance

NFR-1: Dashboard initial load ≤ 3s (P50), ≤ 6s (P95) with cached data.

NFR-2: Full refresh pipeline (fetch+parse+persist) ≤ 2 minutes for 100k rows on small instance.

Reliability

NFR-3: Retries with exponential backoff on 429/5xx; circuit breaker to avoid hammering API.

NFR-4: Fallback to last successful dataset when refresh fails; show warning banner.

Security & Privacy

NFR-5: TLS for all deployments; API token not exposed client-side.

NFR-6: PII handling: if PII exists, enable column masking toggles (admin-only) and omit masked columns from downloads by default.

Accessibility & Localization

NFR-7: WCAG AA color contrast, keyboard navigation for filters.

NFR-8: Date/time and number formats configurable (default: ISO dates).

Observability

NFR-9: Structured logs; optional Sentry for errors; health endpoint for uptime pings.

8. Data Model

Core Tables (logical, MVP may use dataframes/SQLite)

submissions

_uuid (PK), _id, submission_date, state, facility, __version__, plus all normalized indicator columns.

metrics_cache

Precomputed aggregates keyed by (date_bucket, state, facility, indicator_group).

Column Normalization Rules

Replace / with _ and strip leading group names if verbose (e.g., group_infrastructure/... → infrastructure_...).

Score mapping:

Store raw values as-is.

Derived columns: *_is_full (=1 if value==2), *_is_partial, *_is_not, to simplify % calculations.

Sample Data Dictionary (from provided sample)

state_and_facility_selection_state (string): State.

state_and_facility_selection_facility (string): Facility code/name.

infrastructure_adequate_space (int or 'n/a'): 0/1/2/n/a.

infrastructure_patient_flow (int): 0/1/2.

infrastructure_waiting_area (int): 0/1/2.

infrastructure_physical_infrastructure (int): 0/1/2.

infrastructure_handwashing_stations (int): 0/1/2.

infrastructure_clean_water (int): 0/1/2.

start/end/_submission_time (datetime).

_uuid (string), _status (string), __version__ (string).

9. KPI Definitions (clear math)

Let S be the set of selected submissions after filters; for indicator x:

Average score (excl. n/a)
avg_score_x = mean(v for v in S[x] if v in {0,1,2})

% fully implemented
pct_full_x = count(v==2)/count(v in {0,1,2})

% partially implemented
pct_partial_x = count(v==1)/count(v in {0,1,2})

% not implemented
pct_not_x = count(v==0)/count(v in {0,1,2})

Composite infrastructure score (for a set X of infra indicators)
composite_avg = mean( mean(v for v in col if v in {0,1,2}) for col in X )
(Or weighted average—weights configurable later.)

10. UX / UI Specification

Global

Header: App title, last refresh time, “Refresh” (if admin).

Left sidebar: Filters (date range, state multi-select, facility multi-select, version).

Overview Tab

KPI cards: Total submissions, Avg infrastructure score, % fully implemented, Facilities covered.

Line chart: submissions over time.

Bar chart: average infrastructure score by state.

Optional map: state-level choropleth (later).

Facility Comparison Tab

Table: facilities × selected indicators (sortable, color-scaled).

Bar chart: top/bottom 10 facilities by composite score.

Indicator Deep Dive Tab

Dropdown: choose indicator.

Distribution chart: count/percentage by state/facility.

Trend chart: indicator average over time.

Data Quality Tab

Cards: duplicates, missing critical fields.

Table: duplicate _uuid rows; missingness heatmap (simple % by column).

Data Tab

Dataframe preview; download buttons (all data, filtered).

Empty/Failure States

If no data available: show instructions and “Check connection”.

If refresh fails: banner with last-good timestamp and error summary.

11. Integration Details (Kobo)

Config

KOBO_BASE=https://kf.kobotoolbox.org (or EU server)

ASSET_UID=<uid>

API_TOKEN=<token>

EXPORT_SETTINGS_NAME=<optional> (if multiple named exports)

Flow

Resolve export-settings for asset; select XLSX link (data_url_xlsx).

Fetch XLSX with header Authorization: Token <token>.

Respect 5-minute cache; implement backoff on 429.

Resilience

If export-settings empty, fall back to UI prompt or documented manual export link.

Log request IDs; redact tokens.

12. System Architecture

Components

Streamlit App (frontend + light backend).

Ingestion Module (Python function callable on schedule or button).

Storage: data/clean.parquet + SQLite data/app.db (MVP). Phase 2: PostgreSQL.

Secrets Store: environment variables.

Optional: Background scheduler (APScheduler) inside the app, or platform cron.

Sequence (Refresh)

Scheduler triggers refresh.

GET XLSX → parse → clean → compute metrics → persist parquet/SQLite.

Update last_refresh_at.

Dashboard reads from persisted artifacts (fast).

Deployment

MVP: Streamlit Community Cloud or Render.

Env vars set in platform secrets.

HTTPS via platform.

13. Monitoring & Alerting

Health endpoint (/healthz via Streamlit route) returns JSON: last refresh time, row count.

Error tracking: Sentry (optional).

Uptime monitor: external ping every 5 minutes (align with Kobo cache).