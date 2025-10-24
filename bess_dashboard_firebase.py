"""
BESS Daily Reports Dashboard - Firebase Edition
Interactive dashboard reading from Firestore
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Base paths configuration
def get_base_paths():
    """Configure base paths for data input and report output"""
    home_dir = os.path.expanduser("~")

    # Try Documents first, fall back to Hebrew
    documents = "Documents"
    test_path = os.path.join(home_dir, r"Blenergy\Operations-BL - Documents\O&M\Monitoring\Python Reports")
    if not os.path.exists(test_path):
        documents = "מסמכים"

    paths = {
        'data_base': os.path.join(home_dir, rf"Blenergy\Operations-BL - {documents}\O&M\Monitoring\Python Reports\Monthly Data"),
        'output': os.path.join(home_dir, rf"Blenergy\Operations-BL - {documents}\O&M\Monitoring\Python Reports\Daily Reports"),
        'pythonProject': os.path.join(home_dir, rf"Blenergy\Operations-BL - {documents}\O&M\Monitoring\Python Reports\pythonProject1"),
    }

    # Create output directory if it doesn't exist
    os.makedirs(paths['output'], exist_ok=True)

    return paths

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="BESS Reports Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

project_folder = get_base_paths()['pythonProject']
# Firebase credentials path - UPDATE THIS
FIREBASE_CREDENTIALS_PATH = os.path.join(project_folder, "studio-1515705170-8ba09-firebase-adminsdk-fbsvc-df9fe77732.json")

# Color schemes
COLORS = {
    'good': '#27ae60',
    'warning': '#f39c12',
    'poor': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#95a5a6'
}

# Performance thresholds
THRESHOLDS = {
    'rte_good': 85,
    'rte_warning': 70,
    'availability_good': 95,
    'availability_warning': 90
}

# ============================================================================
# OPTIMIZED FIRESTORE CONFIGURATION
# ============================================================================

# cache until next morning (more sophisticated)
def seconds_until_next_morning(target_hour=7):
    """Calculate seconds until next 7 AM"""
    from datetime import datetime, timedelta
    now = datetime.now()
    next_morning = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if next_morning <= now:
        next_morning += timedelta(days=1)
    return int((next_morning - now).total_seconds())

# Use this for cache TTL
CACHE_TTL = seconds_until_next_morning(7)  # Cache until 7 AM next day

# ============================================================================
# FIREBASE INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_firebase():
    """Initialize Firebase connection (cached to run only once)"""
    try:
        # Check if already initialized
        if firebase_admin._apps:
            return firestore.client()

        # Initialize Firebase
        cred_path = FIREBASE_CREDENTIALS_PATH
        if not cred_path and 'FIREBASE_CREDENTIALS' in os.environ:
            cred_path = os.environ['FIREBASE_CREDENTIALS']

        if not cred_path or not os.path.exists(cred_path):
            st.error("Firebase credentials not found. Please update FIREBASE_CREDENTIALS_PATH in the script.")
            return None

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

        return firestore.client()

    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")
        return None


# ============================================================================
# OPTIMIZED FIRESTORE DATA LOADING
# ============================================================================

@st.cache_data(ttl=CACHE_TTL)
def load_available_dates_optimized(_db):
    """
    Load available dates efficiently
    Instead of querying all documents, we can:
    1. Query only recent dates (last 90 days)
    2. Or keep a master list in a single document

    For now, we'll query but cache for 24 hours
    """
    try:
        reports_ref = _db.collection('daily_reports')

        # Option 1: Query only recent dates (more efficient)
        from datetime import date, timedelta
        cutoff_date = date.today() - timedelta(days=90)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        # Query documents >= cutoff_date
        docs = reports_ref.where('date_str', '>=', cutoff_str).stream()

        dates = []
        for doc in docs:
            try:
                date_obj = datetime.strptime(doc.id, '%Y-%m-%d').date()
                dates.append(date_obj)
            except ValueError:
                continue

        # If no results with query, fall back to full scan (shouldn't happen if data structure is correct)
        if not dates:
            docs = reports_ref.limit(90).stream()
            for doc in docs:
                try:
                    date_obj = datetime.strptime(doc.id, '%Y-%m-%d').date()
                    dates.append(date_obj)
                except ValueError:
                    continue

        return sorted(dates, reverse=True)

    except Exception as e:
        st.error(f"Error loading available dates: {e}")
        return []


@st.cache_data(ttl=CACHE_TTL)
def load_latest_report_date(_db):
    """
    Get only the most recent report date without loading all dates
    More efficient for default view
    """
    try:
        # Get yesterday's date (since reports run at 6 AM for previous day)
        from datetime import date, timedelta
        yesterday = date.today() - timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')

        # Check if yesterday's report exists
        doc_ref = _db.collection('daily_reports').document(yesterday_str)
        doc = doc_ref.get()

        if doc.exists:
            return yesterday

        # If not, find the most recent
        # Query in descending order and get first result
        reports_ref = _db.collection('daily_reports')
        docs = reports_ref.order_by('date_str', direction=firestore.Query.DESCENDING).limit(1).stream()

        for doc in docs:
            try:
                return datetime.strptime(doc.id, '%Y-%m-%d').date()
            except ValueError:
                continue

        return date.today() - timedelta(days=1)

    except Exception as e:
        st.error(f"Error loading latest date: {e}")
        return date.today() - timedelta(days=1)


def format_availability(row):
    """
    Format availability display based on what's available
    Combines energy and stacks availability into readable format
    """
    av_energy = row.get('Availability Energy (%)', 'N/A')
    av_stacks = row.get('Availability Stacks (%)', 'N/A')

    if av_energy != 'N/A' and av_stacks != 'N/A':
        return f"Energy: {av_energy}%, Stacks: {av_stacks}%"
    elif av_stacks != 'N/A':
        return f"{av_stacks}%"
    elif av_energy != 'N/A':
        return f"{av_energy}%"
    else:
        return 'N/A'

@st.cache_data(ttl=CACHE_TTL)
def load_report_summary(_db, report_date):
    """Load summary data for a specific date - UNCHANGED but with longer cache"""
    try:
        date_str = report_date.strftime('%Y-%m-%d')
        doc_ref = _db.collection('daily_reports').document(date_str)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        return None

    except Exception as e:
        st.error(f"Error loading summary: {e}")
        return None


@st.cache_data(ttl=CACHE_TTL)
def load_all_sites_for_date(_db, report_date):
    """Load all site data for a specific date - UNCHANGED but with longer cache"""
    try:
        date_str = report_date.strftime('%Y-%m-%d')
        sites_ref = _db.collection('daily_reports').document(date_str).collection('sites')
        docs = sites_ref.stream()

        sites_data = []
        for doc in docs:
            site_data = doc.to_dict()

            # Convert None back to 'N/A' for display
            for key, value in site_data.items():
                if value is None:
                    site_data[key] = 'N/A'

            sites_data.append(site_data)

        if not sites_data:
            return pd.DataFrame()

        df = pd.DataFrame(sites_data)

        # Ensure standard columns exist
        if 'status' not in df.columns:
            df['status'] = 'OK'

        # Rename columns to match Excel format
        column_mapping = {
            'site': 'Site',
            'group': 'Group',
            'status': 'Status',
            'export_kwh': 'Export (kWh)',
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        return df

    except Exception as e:
        st.error(f"Error loading sites data: {e}")
        return pd.DataFrame()


# LAZY LOADING - Only load when user navigates to historical views
def load_multiple_dates_lazy(_db, date_list):
    """
    Load multiple dates ONLY when requested (no caching on this one initially)
    User explicitly requested historical data
    """
    try:
        # Check session state first
        cache_key = f"historical_data_{date_list[0]}_{date_list[-1]}"

        if cache_key in st.session_state:
            return st.session_state[cache_key]

        all_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, report_date in enumerate(date_list):
            status_text.text(f"Loading {report_date.strftime('%Y-%m-%d')}...")
            progress_bar.progress((idx + 1) / len(date_list))

            df_date = load_all_sites_for_date(_db, report_date)
            if not df_date.empty:
                df_date['Date'] = report_date
                all_data.append(df_date)

        progress_bar.empty()
        status_text.empty()

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)

        # Cache in session state
        st.session_state[cache_key] = result

        return result

    except Exception as e:
        st.error(f"Error loading multiple dates: {e}")
        return pd.DataFrame()


def load_site_history_lazy(_db, site_name, days_back=30):
    """
    Load site history efficiently using batch approach
    """
    try:
        cache_key = f"site_history_{site_name}_{days_back}"

        if cache_key in st.session_state:
            return st.session_state[cache_key]

        # Get available dates (this is cached for 24 hours)
        available_dates = load_available_dates_optimized(_db)

        # Limit to requested days
        end_date = available_dates[0] if available_dates else date.today()
        start_date = end_date - timedelta(days=days_back)
        dates_in_range = [d for d in available_dates if start_date <= d <= end_date]

        # Use the batch loader for just this one site
        with st.spinner(f"Loading {len(dates_in_range)} days of history for {site_name}..."):
            df = load_selected_sites_history(_db, [site_name], dates_in_range)

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values('Date', ascending=False)

        # Cache in session state
        st.session_state[cache_key] = df

        return df

    except Exception as e:
        st.error(f"Error loading site history: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_multiple_sites_batch(_db, site_names, report_date):
    """
    Load multiple sites in fewer queries using batch operations
    """
    date_str = report_date.strftime('%Y-%m-%d')
    sites_ref = _db.collection('daily_reports').document(date_str).collection('sites')

    # Get all sites in one query instead of individual queries
    docs = sites_ref.stream()

    sites_data = {}
    for doc in docs:
        if doc.id in site_names:
            sites_data[doc.id] = doc.to_dict()

    return sites_data


@st.cache_data(ttl=300)  # Shorter cache since it's for specific queries
def load_selected_sites_history(_db, site_names, date_list):
    """
    Load historical data for specific sites only (optimized with batch queries)
    More efficient than loading all sites when you only need a few

    Args:
        _db: Firestore client
        site_names: List of site names to load
        date_list: List of dates to load

    Returns:
        DataFrame with only selected sites' data
    """
    try:
        all_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, report_date in enumerate(date_list):
            status_text.text(f"Loading {report_date.strftime('%Y-%m-%d')}...")
            progress_bar.progress((idx + 1) / len(date_list))

            # Use batch query to get only the sites we need
            date_str = report_date.strftime('%Y-%m-%d')
            sites_ref = _db.collection('daily_reports').document(date_str).collection('sites')

            # Get all sites in ONE query for this date
            docs = sites_ref.stream()

            for doc in docs:
                # Filter to only the sites we want
                if doc.id in site_names:
                    site_data = doc.to_dict()

                    # Convert None back to 'N/A' for display
                    for key, value in site_data.items():
                        if value is None:
                            site_data[key] = 'N/A'

                    site_data['Date'] = report_date
                    site_data['Site'] = doc.id

                    # Rename columns to match Excel format
                    if 'site' in site_data:
                        site_data['Site'] = site_data['site']
                    if 'export_kwh' in site_data:
                        site_data['Export (kWh)'] = site_data['export_kwh']
                    if 'rte' in site_data:
                        site_data['RTE (%)'] = site_data['rte']
                    if 'group' in site_data:
                        site_data['Group'] = site_data['group']
                    if 'status' in site_data:
                        site_data['Status'] = site_data['status']

                    all_data.append(site_data)

        progress_bar.empty()
        status_text.empty()

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)

    except Exception as e:
        st.error(f"Error loading selected sites history: {e}")
        return pd.DataFrame()
# ============================================================================
# UTILITY FUNCTIONS (same as before)
# ============================================================================

def get_performance_color(value, metric_type='rte'):
    """Return color based on performance thresholds"""
    if pd.isna(value) or value == 'N/A':
        return COLORS['secondary']

    try:
        value = float(value)
    except:
        return COLORS['secondary']

    if metric_type == 'rte':
        if value >= THRESHOLDS['rte_good']:
            return COLORS['good']
        elif value >= THRESHOLDS['rte_warning']:
            return COLORS['warning']
        else:
            return COLORS['poor']
    elif metric_type == 'availability':
        if value >= THRESHOLDS['availability_good']:
            return COLORS['good']
        elif value >= THRESHOLDS['availability_warning']:
            return COLORS['warning']
        else:
            return COLORS['poor']

    return COLORS['primary']


def display_metric_card(label, value, delta=None, help_text=None):
    """Display a metric card with optional delta"""
    if delta is not None:
        st.metric(label, value, delta=delta, help=help_text)
    else:
        st.metric(label, value, help=help_text)


# ============================================================================
# VISUALIZATION FUNCTIONS (same as before)
# ============================================================================

def plot_site_trends(df_historical, site_name, metrics=['RTE (%)', 'Availability']):
    """Create line chart showing trends for a specific site"""
    # Filter for the site
    df_site = df_historical[df_historical.get('Site', df_historical.get('site')) == site_name].copy()

    if df_site.empty:
        st.warning(f"No data available for {site_name}")
        return

    # Sort by date
    df_site = df_site.sort_values('Date')

    # Map Firestore field names to display names
    field_mapping = {
        'RTE (%)': 'rte',
        'Availability': 'availability_stacks',
        'Export (kWh)': 'export_kwh'
    }

    # Create subplot for each metric
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=metrics,
        vertical_spacing=0.1
    )

    for idx, metric in enumerate(metrics, start=1):
        # Check both display name and field name
        field_name = field_mapping.get(metric, metric.lower().replace(' ', '_').replace('(%)', '').strip())

        if metric in df_site.columns:
            col_to_use = metric
        elif field_name in df_site.columns:
            col_to_use = field_name
        else:
            continue

        # Convert to numeric
        df_site[f'{metric}_numeric'] = pd.to_numeric(df_site[col_to_use], errors='coerce')

        fig.add_trace(
            go.Scatter(
                x=df_site['Date'],
                y=df_site[f'{metric}_numeric'],
                mode='lines+markers',
                name=metric,
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=idx, col=1
        )

        # Add threshold lines for RTE
        if 'RTE' in metric:
            fig.add_hline(
                y=THRESHOLDS['rte_good'],
                line_dash="dash",
                line_color="green",
                annotation_text="Good (85%)",
                row=idx, col=1
            )

    fig.update_layout(
        height=300 * len(metrics),
        showlegend=False,
        title_text=f"{site_name} - Performance Trends"
    )

    fig.update_xaxes(title_text="Date")

    st.plotly_chart(fig, use_container_width=True)


def plot_group_comparison(df, group_num, metric='RTE (%)'):
    """Create bar chart comparing all sites in a group"""
    df_group = df[df['Group'] == group_num].copy()

    if df_group.empty:
        return

    # Convert metric to numeric
    df_group[f'{metric}_numeric'] = pd.to_numeric(df_group[metric], errors='coerce')
    df_group = df_group.dropna(subset=[f'{metric}_numeric'])

    if df_group.empty:
        st.warning(f"No valid {metric} data for Group {group_num}")
        return

    # Sort by metric value
    df_group = df_group.sort_values(f'{metric}_numeric', ascending=True)

    # Color bars based on performance
    colors = [get_performance_color(val, 'rte' if 'RTE' in metric else 'availability')
              for val in df_group[f'{metric}_numeric']]

    fig = go.Figure(data=[
        go.Bar(
            x=df_group[f'{metric}_numeric'],
            y=df_group['Site'],
            orientation='h',
            marker_color=colors,
            text=df_group[f'{metric}_numeric'].round(1),
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=f"Group {group_num} - {metric} Comparison",
        xaxis_title=metric,
        yaxis_title="Site",
        height=max(300, len(df_group) * 25),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_historical_comparison(df_historical, selected_sites, metric='RTE (%)'):
    """Create line chart comparing multiple sites over time"""
    df_filtered = df_historical[df_historical['Site'].isin(selected_sites)].copy()

    if df_filtered.empty:
        st.warning("No data available for selected sites")
        return

    # Convert metric to numeric
    df_filtered[f'{metric}_numeric'] = pd.to_numeric(df_filtered[metric], errors='coerce')
    df_filtered = df_filtered.sort_values('Date')

    fig = px.line(
        df_filtered,
        x='Date',
        y=f'{metric}_numeric',
        color='Site',
        title=f'{metric} Comparison - Multiple Sites',
        markers=True
    )

    # Add threshold line for RTE
    if 'RTE' in metric:
        fig.add_hline(
            y=THRESHOLDS['rte_good'],
            line_dash="dash",
            line_color="green",
            annotation_text="Target (85%)"
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric,
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_heatmap(df_historical, metric='RTE (%)'):
    """Create heatmap showing metric across sites and dates"""
    # Pivot data
    df_pivot = df_historical.pivot_table(
        index='Site',
        columns='Date',
        values=metric,
        aggfunc='first'
    )

    # Convert to numeric
    df_pivot = df_pivot.apply(pd.to_numeric, errors='coerce')

    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=[d.strftime('%Y-%m-%d') for d in df_pivot.columns],
        y=df_pivot.index,
        colorscale='RdYlGn',
        zmid=85 if 'RTE' in metric else 95,
        text=df_pivot.values.round(1),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=metric)
    ))

    fig.update_layout(
        title=f'{metric} Heatmap - All Sites',
        xaxis_title='Date',
        yaxis_title='Site',
        height=max(400, len(df_pivot) * 20)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Initialize Firebase
    db = initialize_firebase()

    if db is None:
        st.error("Cannot connect to Firebase. Please check your credentials.")
        st.stop()

    # Header
    st.title("⚡ BESS Daily Reports Dashboard")
    st.caption("🔥 Connected to Firebase")

    # Show cache info and refresh button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("---")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

    # Sidebar
    st.sidebar.header("📅 Navigation")

    # View mode selector
    view_mode = st.sidebar.radio(
        "View Mode",
        ["📊 Today's Report", "📈 Historical Trends", "🔍 Site Details", "📉 Group Analysis", "🗓️ Multi-Site Comparison"]
    )

    # ========================================================================
    # VIEW 1: TODAY'S REPORT (ORGANIZED COLUMNS)
    # ========================================================================

    if view_mode == "📊 Today's Report":
        st.header("Today's Report")

        # Load ONLY the latest date first (efficient)
        with st.spinner("Loading latest report..."):
            latest_date = load_latest_report_date(db)

        # Date selector - LAZY LOAD full date list only if user wants
        selected_date = latest_date

        with st.sidebar.expander("📅 View Different Date"):
            if st.checkbox("Show all available dates"):
                available_dates = load_available_dates_optimized(db)
                selected_date = st.selectbox(
                    "Select Date",
                    options=available_dates,
                    index=0,
                    format_func=lambda x: x.strftime("%Y-%m-%d (%A)")
                )
            else:
                st.info(f"Showing: {latest_date.strftime('%Y-%m-%d')}")

        # Load only the selected date's data
        with st.spinner("Loading report..."):
            df_summary = load_all_sites_for_date(db, selected_date)
            summary_doc = load_report_summary(db, selected_date)

        if df_summary.empty:
            st.warning("No data available for this date")
            return

        # Display summary metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        total_sites = len(df_summary)
        missing_sites = len(df_summary[df_summary.get('Status', df_summary.get('status', 'OK')) == 'MISSING'])
        operational_sites = total_sites - missing_sites

        with col1:
            display_metric_card("Total Sites", total_sites)

        with col2:
            display_metric_card("Operational", operational_sites, help_text="Sites with data")

        with col3:
            display_metric_card(
                "Missing Data",
                missing_sites,
                delta=None if missing_sites == 0 else "⚠️",
                help_text="Sites without data"
            )

        # Calculate average RTE
        rte_col = 'RTE (%)' if 'RTE (%)' in df_summary.columns else 'rte'
        rte_numeric = pd.to_numeric(df_summary[rte_col], errors='coerce')
        avg_rte = rte_numeric.mean()

        with col4:
            if not pd.isna(avg_rte):
                display_metric_card("Avg RTE", f"{avg_rte:.1f}%", help_text="Average Round-Trip Efficiency")
            else:
                display_metric_card("Avg RTE", "N/A")

        # Calculate total export
        export_col = 'Export (kWh)' if 'Export (kWh)' in df_summary.columns else 'export_kwh'
        export_numeric = pd.to_numeric(df_summary[export_col], errors='coerce')
        total_export = export_numeric.sum()

        with col5:
            if not pd.isna(total_export):
                display_metric_card("Total Export", f"{total_export:,.0f} kWh", help_text="Total energy discharged")
            else:
                display_metric_card("Total Export", "N/A")

        st.markdown("---")

        # =====================================================================
        # ALL SITES TABLE - ORGANIZED AND FORMATTED
        # =====================================================================

        st.subheader("🗂️ All Sites")

        # Show date above the table
        st.info(f"📅 Report Date: {selected_date.strftime('%A, %B %d, %Y')}")

        # Create organized dataframe with specific column order
        df_display = df_summary.copy()

        # Map Firestore field names to display names
        column_mapping = {
            'site': 'Site',
            'group': 'Group',
            'availability_stacks': 'Availability (Stacks) (%)',
            'availability_energy': 'Availability (Energy) (%)',
            'rte': 'RTE (%)',
            'soc_start_charge': 'SOC Start',
            'soc_end_charge': 'SOC End Charge',
            'soc_end_discharge': 'SOC End Discharge',
            'import_kwh': 'Import (kWh)',
            'export_kwh': 'Export (kWh)',
            'guaranteed_energy': 'Guaranteed (kWh)',
            'status': 'Status',
            'total_stacks': 'Total Stacks',
            'stacks_connected_avg': 'Avg Stacks Connected'
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df_display.columns:
                df_display.rename(columns={old_name: new_name}, inplace=True)

        # Define desired column order (with both availability types)
        desired_columns = [
            'Site',
            'Group',
            'Availability (Stacks) (%)',
            'Availability (Energy) (%)',
            'RTE (%)',
            'SOC Start',
            'SOC End Charge',
            'SOC End Discharge',
            'Import (kWh)',
            'Export (kWh)',
            'Guaranteed (kWh)',
            'Status',
            'Total Stacks',
            'Avg Stacks Connected'
        ]

        # Keep only columns that exist
        display_columns = [col for col in desired_columns if col in df_display.columns]
        df_display = df_display[display_columns]

        # Format numeric columns to 2 decimal places
        numeric_columns = [
            'Availability (Stacks) (%)',
            'Availability (Energy) (%)',
            'RTE (%)',
            'SOC Start',
            'SOC End Charge',
            'SOC End Discharge',
            'Import (kWh)',
            'Export (kWh)',
            'Guaranteed (kWh)',
            'Avg Stacks Connected'
        ]

        for col in numeric_columns:
            if col in df_display.columns:
                # Convert to numeric, preserving None/NaN
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                # Format: round to 2 decimals, show 'N/A' for missing
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
                )

        # Format integer columns (no decimals)
        integer_columns = ['Group', 'Total Stacks']
        for col in integer_columns:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                df_display[col] = df_display[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else 'N/A'
                )

        # Color coding for status
        def highlight_status(row):
            status = row.get('Status', 'OK')
            if status == 'MISSING':
                return ['background-color: #ffe6e6'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_display.style.apply(highlight_status, axis=1),
            use_container_width=True,
            height=400
        )

        # Download button
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"BESS_Report_{selected_date}.csv",
            mime="text/csv"
        )

        # =====================================================================
        # GROUP DETAILS TABS - ORGANIZED
        # =====================================================================

        st.markdown("---")
        st.subheader("📊 Group Details")

        tab1, tab2, tab3 = st.tabs(["Group 1", "Group 2", "Group 3"])

        # Helper function to format group dataframe
        def format_group_dataframe(df_group, group_num):
            """Format and organize group-specific dataframe"""
            if df_group.empty:
                return df_group

            df_formatted = df_group.copy()

            # Map column names
            for old_name, new_name in column_mapping.items():
                if old_name in df_formatted.columns:
                    df_formatted.rename(columns={old_name: new_name}, inplace=True)

            # Group-specific column orders
            if group_num == 1:
                # Group 1: Only availability by stacks (no energy availability, no RTE)
                group_columns = [
                    'Site',
                    'Availability (Stacks) (%)',
                    'Avg Stacks Connected',
                    'Total Stacks',
                    'Status'
                ]
            elif group_num == 2 or group_num == 3:
                # Groups 2 & 3: Full RTE data with both availability types
                group_columns = [
                    'Site',
                    'Availability (Stacks) (%)',
                    'Availability (Energy) (%)',
                    'RTE (%)',
                    'SOC Start',
                    'SOC End Charge',
                    'SOC End Discharge',
                    'Import (kWh)',
                    'Export (kWh)',
                    'Guaranteed (kWh)',
                    'Avg Stacks Connected',
                    'Total Stacks',
                    'Status'
                ]

            # Keep only existing columns
            existing_columns = [col for col in group_columns if col in df_formatted.columns]
            df_formatted = df_formatted[existing_columns]

            # Format numeric columns
            for col in numeric_columns:
                if col in df_formatted.columns:
                    df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce')
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
                    )

            # Format integer columns
            for col in integer_columns:
                if col in df_formatted.columns:
                    df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce')
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: f"{int(x)}" if pd.notna(x) else 'N/A'
                    )

            return df_formatted

        with tab1:
            st.caption(f"Group 1 - Availability by Stacks | Date: {selected_date.strftime('%Y-%m-%d')}")
            df_g1 = df_summary[df_summary.get('Group', df_summary.get('group', 0)) == 1].copy()

            if not df_g1.empty:
                df_g1_display = format_group_dataframe(df_g1, 1)

                st.dataframe(
                    df_g1_display.style.apply(highlight_status, axis=1),
                    use_container_width=True
                )

                # Chart - use availability_stacks for Group 1
                if 'availability_stacks' in df_g1.columns or 'Availability (Stacks)' in df_g1.columns:
                    df_g1_plot = df_g1.copy()

                    # Get the availability column (check both possible names)
                    avail_col = 'availability_stacks' if 'availability_stacks' in df_g1_plot.columns else 'Availability (Stacks)'
                    df_g1_plot['Availability_num'] = pd.to_numeric(df_g1_plot[avail_col], errors='coerce')
                    df_g1_plot = df_g1_plot.dropna(subset=['Availability_num'])

                    if not df_g1_plot.empty:
                        # Prepare for plotting
                        df_g1_plot['Site'] = df_g1_plot.get('Site', df_g1_plot.get('site', ''))
                        df_g1_plot['Group'] = 1
                        df_g1_plot['RTE (%)'] = df_g1_plot['Availability_num']  # Reuse RTE column for plotting

                        plot_group_comparison(df_g1_plot, 1, 'RTE (%)')
            else:
                st.info("No Group 1 sites in this report")

        with tab2:
            st.caption(f"Group 2 - CATL Sites (Full RTE) | Date: {selected_date.strftime('%Y-%m-%d')}")
            df_g2 = df_summary[df_summary.get('Group', df_summary.get('group', 0)) == 2].copy()

            if not df_g2.empty:
                df_g2_display = format_group_dataframe(df_g2, 2)

                st.dataframe(
                    df_g2_display.style.apply(highlight_status, axis=1),
                    use_container_width=True
                )

                # Chart - use RTE for Group 2
                rte_col = 'rte' if 'rte' in df_g2.columns else 'RTE'
                if rte_col in df_g2.columns:
                    df_g2_plot = df_g2.copy()
                    df_g2_plot['RTE_num'] = pd.to_numeric(df_g2_plot[rte_col], errors='coerce')
                    df_g2_plot = df_g2_plot.dropna(subset=['RTE_num'])

                    if not df_g2_plot.empty:
                        df_g2_plot['Site'] = df_g2_plot.get('Site', df_g2_plot.get('site', ''))
                        df_g2_plot['Group'] = 2
                        df_g2_plot['RTE (%)'] = df_g2_plot['RTE_num']

                        plot_group_comparison(df_g2_plot, 2, 'RTE (%)')
            else:
                st.info("No Group 2 sites in this report")

        with tab3:
            st.caption(f"Group 3 - Powin Sites (Full RTE) | Date: {selected_date.strftime('%Y-%m-%d')}")
            df_g3 = df_summary[df_summary.get('Group', df_summary.get('group', 0)) == 3].copy()

            if not df_g3.empty:
                df_g3_display = format_group_dataframe(df_g3, 3)

                st.dataframe(
                    df_g3_display.style.apply(highlight_status, axis=1),
                    use_container_width=True
                )

                # Chart - use RTE for Group 3
                rte_col = 'rte' if 'rte' in df_g3.columns else 'RTE'
                if rte_col in df_g3.columns:
                    df_g3_plot = df_g3.copy()
                    df_g3_plot['RTE_num'] = pd.to_numeric(df_g3_plot[rte_col], errors='coerce')
                    df_g3_plot = df_g3_plot.dropna(subset=['RTE_num'])

                    if not df_g3_plot.empty:
                        df_g3_plot['Site'] = df_g3_plot.get('Site', df_g3_plot.get('site', ''))
                        df_g3_plot['Group'] = 3
                        df_g3_plot['RTE (%)'] = df_g3_plot['RTE_num']

                        plot_group_comparison(df_g3_plot, 3, 'RTE (%)')
            else:
                st.info("No Group 3 sites in this report")

    # ========================================================================
    # VIEW 2: HISTORICAL TRENDS (LAZY LOAD)
    # ========================================================================

    elif view_mode == "📈 Historical Trends":
        st.header("Historical Trends")

        # Load available dates (cached for 24 hours)
        with st.spinner("Loading available dates..."):
            available_dates = load_available_dates_optimized(db)

        if not available_dates:
            st.error("No reports found in Firestore")
            return

        # Date range selector
        days_back = st.sidebar.slider("Days to analyze", 7, 90, 30)

        # Calculate date range
        end_date = available_dates[0]
        start_date = end_date - timedelta(days=days_back)
        dates_in_range = [d for d in available_dates if start_date <= d <= end_date]

        st.info(f"📊 Date range: {start_date} to {end_date} ({len(dates_in_range)} days)")

        # Check if data already loaded in session
        cache_key = f"historical_{start_date}_{end_date}"
        data_loaded = cache_key in st.session_state

        # Load button
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            if st.button("📥 Load Historical Data", type="primary", disabled=data_loaded):
                with st.spinner(f"Loading {len(dates_in_range)} days of data..."):
                    df_historical = load_multiple_dates_lazy(db, dates_in_range)

                    if df_historical.empty:
                        st.warning("No historical data available")
                        return

                    st.session_state[cache_key] = df_historical
                    st.rerun()

        with col2:
            if data_loaded:
                st.success("✅ Data loaded")

        with col3:
            if data_loaded and st.button("🗑️ Clear Data"):
                del st.session_state[cache_key]
                st.rerun()

        # Check if data is loaded
        if cache_key not in st.session_state:
            st.warning("👆 Click 'Load Historical Data' to view trends")
            st.info("💡 This will load data for all sites across the selected date range")
            return

        df_historical = st.session_state[cache_key]

        # Metric selector
        metric = st.sidebar.selectbox(
            "Select Metric",
            ["RTE (%)", "Export (kWh)", "Availability"]
        )

        # Display heatmap
        st.subheader(f"📊 {metric} Heatmap")
        st.info("Darker green = better performance, Red = poor performance")
        plot_heatmap(df_historical, metric)

        # Statistics
        st.markdown("---")
        st.subheader("📈 Trend Statistics")

        stats_data = []
        for site in df_historical['Site'].unique():
            df_site = df_historical[df_historical['Site'] == site].copy()
            metric_values = pd.to_numeric(df_site[metric], errors='coerce').dropna()

            if len(metric_values) > 0:
                stats_data.append({
                    'Site': site,
                    'Average': metric_values.mean(),
                    'Min': metric_values.min(),
                    'Max': metric_values.max(),
                    'Std Dev': metric_values.std(),
                    'Data Points': len(metric_values)
                })

        df_stats = pd.DataFrame(stats_data).round(2)
        st.dataframe(df_stats, use_container_width=True)

        # Download statistics
        csv = df_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Statistics",
            data=csv,
            file_name=f"BESS_Statistics_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )

    # ========================================================================
    # VIEW 3: SITE DETAILS (LAZY LOAD HISTORY)
    # ========================================================================

    elif view_mode == "🔍 Site Details":
        st.header("Site Details")

        # Load latest report to get site list (single date = minimal reads)
        with st.spinner("Loading site list..."):
            latest_date = load_latest_report_date(db)
            df_latest = load_all_sites_for_date(db, latest_date)

        if df_latest.empty:
            st.error("No data available")
            return

        site_list = sorted(df_latest['Site'].unique())

        # Site selector
        selected_site = st.sidebar.selectbox("Select Site", site_list)

        # Date range
        days_back = st.sidebar.slider("Days of history", 7, 60, 30)

        # Check if history loaded
        cache_key = f"site_{selected_site}_{days_back}"
        history_loaded = cache_key in st.session_state

        # Load button
        col1, col2 = st.columns([3, 3])
        with col1:
            if st.button(f"📥 Load {days_back}-Day History", type="primary", disabled=history_loaded):
                df_site = load_site_history_lazy(db, selected_site, days_back)
                st.session_state[cache_key] = df_site
                st.rerun()

        with col2:
            if history_loaded:
                st.success("✅ History loaded")
                if st.button("🗑️ Clear History"):
                    del st.session_state[cache_key]
                    st.rerun()

        # Always show current status (from latest report)
        latest_data = df_latest[df_latest['Site'] == selected_site].iloc[0]

        st.subheader(f"📍 {selected_site}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Group", f"Group {latest_data.get('Group', 'N/A')}")

        with col2:
            rte_val = latest_data.get('RTE (%)', 'N/A')
            if rte_val != 'N/A':
                st.metric("Latest RTE", f"{float(rte_val):.1f}%")
            else:
                st.metric("Latest RTE", "N/A")

        with col3:
            export_val = latest_data.get('Export (kWh)', 'N/A')
            if export_val != 'N/A':
                st.metric("Latest Export", f"{float(export_val):,.0f} kWh")
            else:
                st.metric("Latest Export", "N/A")

        with col4:
            status = latest_data.get('Status', 'OK')
            st.metric("Status", status)

        # Show history if loaded
        if cache_key not in st.session_state:
            st.markdown("---")
            st.info("👆 Click 'Load History' to view performance trends")
            return

        df_site = st.session_state[cache_key]

        if df_site.empty:
            st.warning(f"No historical data available for {selected_site}")
            return

        # Plot trends
        st.markdown("---")
        st.subheader("📈 Performance Trends")

        # Convert column names for plotting
        df_site_plot = df_site.copy()
        df_site_plot['Site'] = selected_site
        df_site_plot['RTE (%)'] = df_site_plot.get('rte', 'N/A')
        df_site_plot['Export (kWh)'] = df_site_plot.get('export_kwh', 'N/A')
        df_site_plot['Availability'] = df_site_plot.get('availability_stacks',
                                                        df_site_plot.get('availability_energy', 'N/A'))

        plot_site_trends(df_site_plot, selected_site, ['RTE (%)', 'Export (kWh)', 'Availability'])

        # Show data table
        st.markdown("---")
        st.subheader("📋 Historical Data")

        st.dataframe(df_site, use_container_width=True, height=400)

        # Download button
        csv = df_site.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Site History",
            data=csv,
            file_name=f"BESS_{selected_site}_History.csv",
            mime="text/csv"
        )

    # ========================================================================
    # VIEW 4: GROUP ANALYSIS (LAZY LOAD)
    # ========================================================================

    elif view_mode == "📉 Group Analysis":
        st.header("Group Analysis")

        # Group selector
        selected_group = st.sidebar.selectbox("Select Group", [1, 2, 3])

        # Date range
        days_back = st.sidebar.slider("Days of history", 7, 60, 30)

        # Load available dates
        with st.spinner("Loading available dates..."):
            available_dates = load_available_dates_optimized(db)

        if not available_dates:
            st.error("No reports found")
            return

        # Calculate date range
        end_date = available_dates[0]
        start_date = end_date - timedelta(days=days_back)
        dates_in_range = [d for d in available_dates if start_date <= d <= end_date]

        st.info(f"📊 Analysis period: {len(dates_in_range)} days")

        # Check if data loaded
        cache_key = f"group_{selected_group}_{start_date}_{end_date}"
        data_loaded = cache_key in st.session_state

        # Load button
        col1, col2 = st.columns([3, 3])
        with col1:
            if st.button("📥 Load Group Data", type="primary", disabled=data_loaded):
                with st.spinner("Loading group data..."):
                    df_historical = load_multiple_dates_lazy(db, dates_in_range)

                    if df_historical.empty:
                        st.warning("No data available")
                        return

                    # Filter for selected group
                    df_group = df_historical[df_historical['Group'] == selected_group].copy()

                    if df_group.empty:
                        st.warning(f"No data available for Group {selected_group}")
                        return

                    st.session_state[cache_key] = df_group
                    st.rerun()

        with col2:
            if data_loaded:
                st.success("✅ Data loaded")
                if st.button("🗑️ Clear Data"):
                    del st.session_state[cache_key]
                    st.rerun()

        if cache_key not in st.session_state:
            st.warning("👆 Click 'Load Group Data' to view analysis")
            return

        df_group = st.session_state[cache_key]

        st.subheader(f"Group {selected_group} Overview")

        # Group statistics
        col1, col2, col3, col4 = st.columns(4)

        sites_in_group = df_group['Site'].nunique()

        with col1:
            st.metric("Sites in Group", sites_in_group)

        # Calculate group averages
        rte_numeric = pd.to_numeric(df_group['RTE (%)'], errors='coerce')
        avg_rte = rte_numeric.mean()

        with col2:
            if not pd.isna(avg_rte):
                st.metric("Avg RTE", f"{avg_rte:.1f}%")
            else:
                st.metric("Avg RTE", "N/A")

        export_numeric = pd.to_numeric(df_group['Export (kWh)'], errors='coerce')
        total_export = export_numeric.sum()

        with col3:
            if not pd.isna(total_export):
                st.metric("Total Export", f"{total_export:,.0f} kWh")
            else:
                st.metric("Total Export", "N/A")

        # Best performing site
        if not rte_numeric.empty:
            df_avg_by_site = df_group.groupby('Site').agg({
                'RTE (%)': lambda x: pd.to_numeric(x, errors='coerce').mean()
            }).reset_index()
            best_site = df_avg_by_site.loc[df_avg_by_site['RTE (%)'].idxmax(), 'Site']

            with col4:
                st.metric("Best Performer", best_site)

        # Plot group comparison for latest date
        st.markdown("---")
        st.subheader("📊 Current Performance Comparison")

        latest_date = df_group['Date'].max()
        df_latest = df_group[df_group['Date'] == latest_date]
        plot_group_comparison(df_latest, selected_group, 'RTE (%)')

        # Group trend over time
        st.markdown("---")
        st.subheader("📈 Group Average Trend")

        # Calculate daily group average
        df_group_avg = df_group.groupby('Date').agg({
            'RTE (%)': lambda x: pd.to_numeric(x, errors='coerce').mean()
        }).reset_index()

        df_group_avg = df_group_avg.sort_values('Date')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_group_avg['Date'],
            y=df_group_avg['RTE (%)'],
            mode='lines+markers',
            name=f'Group {selected_group} Avg',
            line=dict(width=3, color=COLORS['primary']),
            marker=dict(size=8)
        ))

        fig.add_hline(
            y=THRESHOLDS['rte_good'],
            line_dash="dash",
            line_color="green",
            annotation_text="Target"
        )

        fig.update_layout(
            title=f"Group {selected_group} - Average RTE Trend",
            xaxis_title="Date",
            yaxis_title="Average RTE (%)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Individual site trends
        st.markdown("---")
        st.subheader("🔍 Individual Site Trends")

        sites_in_group_list = sorted(df_group['Site'].unique())

        for site in sites_in_group_list:
            with st.expander(f"📍 {site}"):
                # Prepare data for plotting
                df_site_plot = df_group[df_group['Site'] == site].copy()
                plot_site_trends(df_site_plot, site, ['RTE (%)'])

    # ========================================================================
    # VIEW 5: MULTI-SITE COMPARISON (OPTIMIZED WITH BATCH QUERIES)
    # ========================================================================

    elif view_mode == "🗓️ Multi-Site Comparison":
        st.header("Multi-Site Comparison")

        # Load latest report to get site list
        with st.spinner("Loading site list..."):
            latest_date = load_latest_report_date(db)
            df_latest = load_all_sites_for_date(db, latest_date)

        if df_latest.empty:
            st.error("No data available")
            return

        site_list = sorted(df_latest['Site'].unique())

        # Site selector (multi-select)
        selected_sites = st.sidebar.multiselect(
            "Select Sites (up to 10)",
            site_list,
            default=site_list[:5] if len(site_list) >= 5 else site_list,
            max_selections=10
        )

        if not selected_sites:
            st.info("Please select at least one site from the sidebar")
            return

        # Date range
        days_back = st.sidebar.slider("Days of history", 7, 60, 30)

        # Metric selector
        metric = st.sidebar.selectbox(
            "Select Metric",
            ["RTE (%)", "Export (kWh)", "Availability"]
        )

        # Load available dates
        with st.spinner("Loading available dates..."):
            available_dates = load_available_dates_optimized(db)

        if not available_dates:
            st.error("No reports found")
            return

        # Calculate date range
        end_date = available_dates[0]
        start_date = end_date - timedelta(days=days_back)
        dates_in_range = [d for d in available_dates if start_date <= d <= end_date]

        st.info(f"📊 Will compare {len(selected_sites)} sites over {len(dates_in_range)} days")

        # Check if data loaded
        cache_key = f"comparison_{'-'.join(sorted(selected_sites))}_{start_date}_{end_date}"
        data_loaded = cache_key in st.session_state

        # Load button
        col1, col2 = st.columns([3, 3])
        with col1:
            if st.button("📥 Load Comparison Data", type="primary", disabled=data_loaded):
                # *** OPTIMIZED: Load only selected sites instead of all sites ***
                with st.spinner(f"Loading data for {len(selected_sites)} sites..."):
                    df_historical = load_selected_sites_history(db, selected_sites, dates_in_range)

                    if df_historical.empty:
                        st.warning("No data available")
                        return

                    st.session_state[cache_key] = df_historical
                    st.rerun()

        with col2:
            if data_loaded:
                st.success("✅ Data loaded")
                if st.button("🗑️ Clear Data"):
                    del st.session_state[cache_key]
                    st.rerun()

        if cache_key not in st.session_state:
            st.warning("👆 Click 'Load Comparison Data' to view analysis")
            return

        df_historical = st.session_state[cache_key]

        # Plot comparison
        st.subheader(f"📈 {metric} Comparison")
        plot_historical_comparison(df_historical, selected_sites, metric)

        # Statistics table
        st.markdown("---")
        st.subheader("📊 Comparison Statistics")

        stats_data = []
        for site in selected_sites:
            df_site = df_historical[df_historical['Site'] == site].copy()
            metric_values = pd.to_numeric(df_site[metric], errors='coerce').dropna()

            if len(metric_values) > 0:
                stats_data.append({
                    'Site': site,
                    'Average': metric_values.mean(),
                    'Min': metric_values.min(),
                    'Max': metric_values.max(),
                    'Latest': metric_values.iloc[0] if len(metric_values) > 0 else None,
                    'Trend': '📈' if len(metric_values) > 1 and metric_values.iloc[0] > metric_values.iloc[-1] else '📉'
                })

        df_stats = pd.DataFrame(stats_data).round(2)
        st.dataframe(df_stats, use_container_width=True)

        # Download statistics
        csv = df_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Comparison Statistics",
            data=csv,
            file_name=f"BESS_Comparison_{'-'.join(selected_sites[:3])}.csv",
            mime="text/csv"
        )

        # Side-by-side data
        st.markdown("---")
        st.subheader("📋 Detailed Data")

        # Create columns for each site (max 3 columns)
        num_cols = min(len(selected_sites), 3)
        cols = st.columns(num_cols)

        for idx, site in enumerate(selected_sites):
            with cols[idx % num_cols]:
                st.markdown(f"**{site}**")
                df_site = df_historical[df_historical['Site'] == site].copy()
                df_site = df_site.sort_values('Date', ascending=False)
                st.dataframe(
                    df_site[['Date', metric]].head(10),
                    use_container_width=True,
                    height=300
                )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.success("🔥 Connected to Firestore")

    # Show cache info
    cache_expiry = datetime.now() + timedelta(seconds=CACHE_TTL)
    st.sidebar.info(f"💾 Cache expires: {cache_expiry.strftime('%H:%M')}")

    # Show available dates count
    try:
        dates_count = len(load_available_dates_optimized(db))
        st.sidebar.info(f"📊 Available reports: {dates_count}")

        latest = load_latest_report_date(db)
        st.sidebar.info(f"📅 Latest report: {latest.strftime('%Y-%m-%d')}")
    except:
        pass


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()