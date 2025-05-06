import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Email Metrics Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a6fff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #343a40;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a6fff;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .stDownloadButton button {
        background-color: #4a6fff;
        color: white;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def convert_binary_data(df):
    """Convert yes/no values to 1/0 for metrics columns"""
    binary_columns = ['EmailsSent', 'EmailsOpened', 'EmailsClicked']
    
    for col in binary_columns:
        if col in df.columns:
            # Convert to lowercase strings first
            df[col] = df[col].astype(str).str.lower()
            
            # Handle different variations of yes/no, true/false
            yes_values = ['yes', 'true', '1', 'y']
            no_values = ['no', 'false', '0', 'n']
            
            # Create a mask for each type of value
            yes_mask = df[col].isin(yes_values)
            no_mask = df[col].isin(no_values)
            
            # Apply the conversion
            df.loc[yes_mask, col] = 1
            df.loc[no_mask, col] = 0
            
            # Convert remaining values if possible
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            except:
                st.warning(f"Some values in {col} could not be converted to binary values.")
    
    return df

def calculate_metrics(df):
    """Calculate metrics based on binary email data"""
    # Ensure data is in correct format
    df = convert_binary_data(df)
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Calculate individual metrics for each row
    result_df['OpenRate'] = np.where(result_df['EmailsSent'] > 0, 
                                    result_df['EmailsOpened'] / result_df['EmailsSent'] * 100, 
                                    0)
    result_df['ClickRate'] = np.where(result_df['EmailsSent'] > 0, 
                                     result_df['EmailsClicked'] / result_df['EmailsSent'] * 100, 
                                     0)
    result_df['CTOR'] = np.where(result_df['EmailsOpened'] > 0, 
                               result_df['EmailsClicked'] / result_df['EmailsOpened'] * 100, 
                               0)
    
    # Calculate engagement score (weighted combination of opens and clicks)
    # 40% weight to opens, 60% weight to clicks
    result_df['EngagementScore'] = (0.4 * result_df['OpenRate'] + 0.6 * result_df['ClickRate'])
    
    return result_df

def generate_fake_dates(df, start_date=None, end_date=None):
    """Add synthetic dates to the dataframe for time series visualization"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    if end_date is None:
        end_date = datetime.now()
    
    # Create a copy of the dataframe
    time_df = df.copy()
    
    # Generate random dates within the range
    date_range = (end_date - start_date).days
    if date_range <= 0:
        date_range = 90  # Default to 90 days if dates are invalid
    
    # Fix: Convert numpy.int32 to Python int
    random_days = np.random.randint(0, date_range, size=len(df))
    time_df['Date'] = [start_date + timedelta(days=int(days)) for days in random_days]
    
    # Sort by date
    time_df = time_df.sort_values('Date')
    
    return time_df

def generate_industry_benchmarks():
    """Generate fake industry benchmarks for comparison"""
    return {
        'OpenRate': {
            'Marketing': 21.5,
            'Technology': 19.8,
            'Finance': 17.2,
            'Healthcare': 18.9,
            'Education': 23.4,
            'Overall': 20.2
        },
        'ClickRate': {
            'Marketing': 9.2,
            'Technology': 8.7,
            'Finance': 7.1,
            'Healthcare': 7.8,
            'Education': 10.3,
            'Overall': 8.6
        },
        'CTOR': {
            'Marketing': 42.8,
            'Technology': 43.9,
            'Finance': 41.3,
            'Healthcare': 41.3,
            'Education': 44.0,
            'Overall': 42.6
        }
    }

# App Header
st.markdown('<div class="main-header">📧 Email Metrics Dashboard</div>', unsafe_allow_html=True)
st.markdown("Upload your email metrics data to generate insights and visualizations.")

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    # Load Sample Data option
    load_sample = st.button("Load Sample Data")
    
    # Add separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Dark Mode Toggle
    theme_mode = st.toggle("Dark Mode", False)
    
    # Date Range for Time Series
    st.subheader("Time Series Settings")
    
    # Set default dates
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=90)
    
    start_date = st.date_input("Start Date", value=default_start_date)
    end_date = st.date_input("End Date", value=default_end_date)
    
    # Add separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display industry benchmarks in sidebar
    st.subheader("Industry Benchmarks")
    benchmarks = generate_industry_benchmarks()
    
    st.markdown(f"""
    | Metric | Average |
    | ------ | ------- |
    | Open Rate | {benchmarks['OpenRate']['Overall']}% |
    | Click Rate | {benchmarks['ClickRate']['Overall']}% |
    | CTOR | {benchmarks['CTOR']['Overall']}% |
    """)

# Data loading
@st.cache_data
def create_sample_data():
    data = {
        'Company': ['Datadog', 'Datadog', 'Software Mansion', 'MESCIUS inc'] + ['Acme Corp', 'TechGiant', 'GlobalFirm'] * 5,
        'JobRole': ['Senior Product Marketing Manager', 'Director, Partner Marketing', 
                   'Head of Digital Marketing', 'Marketing Communications Manager'] + 
                   ['CEO', 'Manager', 'Employee'] * 5,
        'PersonName': [
            'Bridgitte Kwong', 'Manuela Rojas', 'Patryk Mamczur', 'Caitlyn Depp',
            'John Smith', 'Jessica Brown', 'David Wilson', 'Emily Davis',
            'Michael Lee', 'Sarah Johnson', 'Robert Chen', 'Amanda White',
            'Thomas Moore', 'Lisa Garcia', 'James Taylor', 'Jennifer Martin',
            'William Adams', 'Olivia King'
        ],
        'EmailsSent': ['yes'] * 18,
        'EmailsOpened': ['yes', 'no', 'yes', 'yes'] + ['yes', 'no'] * 7,
        'EmailsClicked': ['no', 'no', 'no', 'yes'] + ['yes', 'no'] * 7
    }
    return pd.DataFrame(data)

df = None

# Load data based on user choice
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Company', 'JobRole', 'PersonName', 'EmailsSent', 'EmailsOpened', 'EmailsClicked']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your CSV has the following columns: Company, JobRole, PersonName, EmailsSent, EmailsOpened, EmailsClicked")
            df = None
        else:
            st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = None
elif load_sample:
    df = create_sample_data()
    st.success("Sample data loaded successfully!")

# Process data if available
if df is not None:
    try:
        # Convert binary data and calculate metrics
        metrics_df = calculate_metrics(df)
        
        # Generate synthetic dates for time series
        time_series_df = generate_fake_dates(metrics_df, start_date, end_date)
        
        # Create filters
        st.markdown('<div class="sub-header">Filters</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            companies = ['All Companies'] + sorted(df['Company'].unique().tolist())
            selected_company = st.selectbox("Company", companies)
        
        with col2:
            roles = ['All Roles'] + sorted(df['JobRole'].unique().tolist())
            selected_role = st.selectbox("Job Role", roles)
        
        with col3:
            metric_type = st.selectbox("Metric Type", ["Open Rate", "Click Rate", "Click-to-Open Rate", "All Metrics"])
        
        # Apply filters
        filtered_df = metrics_df.copy()
        if selected_company != 'All Companies':
            filtered_df = filtered_df[filtered_df['Company'] == selected_company]
        if selected_role != 'All Roles':
            filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
        
        # Also apply filters to time series data
        filtered_time_df = time_series_df.copy()
        if selected_company != 'All Companies':
            filtered_time_df = filtered_time_df[filtered_time_df['Company'] == selected_company]
        if selected_role != 'All Roles':
            filtered_time_df = filtered_time_df[filtered_time_df['JobRole'] == selected_role]
        
        # Calculate summary metrics for filtered data
        total_sent = filtered_df['EmailsSent'].sum()
        total_opened = filtered_df['EmailsOpened'].sum()
        total_clicked = filtered_df['EmailsClicked'].sum()
        
        open_rate = 0 if total_sent == 0 else (total_opened / total_sent) * 100
        click_rate = 0 if total_sent == 0 else (total_clicked / total_sent) * 100
        ctor_rate = 0 if total_opened == 0 else (total_clicked / total_opened) * 100
        avg_engagement = filtered_df['EngagementScore'].mean()
        
        # Display summary metrics
        st.markdown('<div class="sub-header">Summary Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_sent}</div>
                <div class="metric-label">Total Emails Sent</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{open_rate:.1f}%</div>
                <div class="metric-label">Open Rate</div>
                <div class="tooltip">ℹ️
                    <span class="tooltiptext">Percentage of emails that were opened</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{click_rate:.1f}%</div>
                <div class="metric-label">Click Rate</div>
                <div class="tooltip">ℹ️
                    <span class="tooltiptext">Percentage of emails that were clicked</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_engagement:.1f}</div>
                <div class="metric-label">Engagement Score</div>
                <div class="tooltip">ℹ️
                    <span class="tooltiptext">Weighted score combining opens (40%) and clicks (60%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main charts section
        st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Bar Charts", "Line Charts", "Heat Maps", "Data Table"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Company Performance Bar Chart
                if not filtered_df.empty:
                    company_metrics = filtered_df.groupby('Company').agg({
                        'EmailsSent': 'sum',
                        'EmailsOpened': 'sum',
                        'EmailsClicked': 'sum'
                    }).reset_index()
                    
                    company_metrics['OpenRate'] = np.where(company_metrics['EmailsSent'] > 0, 
                                                        company_metrics['EmailsOpened'] / company_metrics['EmailsSent'] * 100, 
                                                        0)
                    company_metrics['ClickRate'] = np.where(company_metrics['EmailsSent'] > 0, 
                                                        company_metrics['EmailsClicked'] / company_metrics['EmailsSent'] * 100, 
                                                        0)
                    
                    fig_company = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    if metric_type in ["Open Rate", "All Metrics"]:
                        fig_company.add_trace(
                            go.Bar(
                                x=company_metrics['Company'],
                                y=company_metrics['OpenRate'],
                                name="Open Rate (%)",
                                marker_color='rgba(74, 111, 255, 0.7)',
                                text=company_metrics['OpenRate'].round(1).astype(str) + '%',
                                textposition='outside'
                            ),
                            secondary_y=False
                        )
                    
                    if metric_type in ["Click Rate", "All Metrics"]:
                        fig_company.add_trace(
                            go.Bar(
                                x=company_metrics['Company'],
                                y=company_metrics['ClickRate'],
                                name="Click Rate (%)",
                                marker_color='rgba(255, 99, 132, 0.7)',
                                text=company_metrics['ClickRate'].round(1).astype(str) + '%',
                                textposition='outside'
                            ),
                            secondary_y=False
                        )
                    
                    fig_company.update_layout(
                        title_text="Performance by Company",
                        barmode='group',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    if not company_metrics.empty:
                        max_value = max(company_metrics['OpenRate'].max() if 'OpenRate' in company_metrics else 0, 
                                      company_metrics['ClickRate'].max() if 'ClickRate' in company_metrics else 0)
                        fig_company.update_yaxes(title_text="Rate (%)", range=[0, max_value * 1.2])
                    
                    st.plotly_chart(fig_company, use_container_width=True)
                else:
                    st.info("No data available for company performance chart with current filters.")
            
            with col2:
                # Role Performance Bar Chart
                if not filtered_df.empty:
                    role_metrics = filtered_df.groupby('JobRole').agg({
                        'EmailsSent': 'sum',
                        'EmailsOpened': 'sum',
                        'EmailsClicked': 'sum'
                    }).reset_index()
                    
                    role_metrics['OpenRate'] = np.where(role_metrics['EmailsSent'] > 0, 
                                                    role_metrics['EmailsOpened'] / role_metrics['EmailsSent'] * 100, 
                                                    0)
                    role_metrics['ClickRate'] = np.where(role_metrics['EmailsSent'] > 0, 
                                                    role_metrics['EmailsClicked'] / role_metrics['EmailsSent'] * 100, 
                                                    0)
                    
                    fig_role = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    if metric_type in ["Open Rate", "All Metrics"]:
                        fig_role.add_trace(
                            go.Bar(
                                x=role_metrics['JobRole'],
                                y=role_metrics['OpenRate'],
                                name="Open Rate (%)",
                                marker_color='rgba(74, 111, 255, 0.7)',
                                text=role_metrics['OpenRate'].round(1).astype(str) + '%',
                                textposition='outside'
                            ),
                            secondary_y=False
                        )
                    
                    if metric_type in ["Click Rate", "All Metrics"]:
                        fig_role.add_trace(
                            go.Bar(
                                x=role_metrics['JobRole'],
                                y=role_metrics['ClickRate'],
                                name="Click Rate (%)",
                                marker_color='rgba(255, 99, 132, 0.7)',
                                text=role_metrics['ClickRate'].round(1).astype(str) + '%',
                                textposition='outside'
                            ),
                            secondary_y=False
                        )
                    
                    fig_role.update_layout(
                        title_text="Performance by Job Role",
                        barmode='group',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    if not role_metrics.empty:
                        max_value = max(role_metrics['OpenRate'].max() if 'OpenRate' in role_metrics else 0, 
                                      role_metrics['ClickRate'].max() if 'ClickRate' in role_metrics else 0)
                        fig_role.update_yaxes(title_text="Rate (%)", range=[0, max_value * 1.2])
                    
                    st.plotly_chart(fig_role, use_container_width=True)
                else:
                    st.info("No data available for role performance chart with current filters.")
        
        with tab2:
            st.subheader("Trend Analysis")
            
            # Time series charts using the synthetic dates
            if not filtered_time_df.empty:
                # Fix: Make sure Date is datetime type
                filtered_time_df['Date'] = pd.to_datetime(filtered_time_df['Date'])
                # Group by date and calculate aggregated metrics
                # Fix: Use resample instead of Grouper for more reliability
                time_metrics = filtered_time_df.set_index('Date').resample('W').agg({
                    'EmailsSent': 'sum',
                    'EmailsOpened': 'sum',
                    'EmailsClicked': 'sum'
                }).reset_index()
                
                # Only proceed if we have data
                if not time_metrics.empty:
                    time_metrics['OpenRate'] = np.where(time_metrics['EmailsSent'] > 0, 
                                                    time_metrics['EmailsOpened'] / time_metrics['EmailsSent'] * 100, 
                                                    0)
                    time_metrics['ClickRate'] = np.where(time_metrics['EmailsSent'] > 0, 
                                                    time_metrics['EmailsClicked'] / time_metrics['EmailsSent'] * 100, 
                                                    0)
                    time_metrics['CTOR'] = np.where(time_metrics['EmailsOpened'] > 0, 
                                                time_metrics['EmailsClicked'] / time_metrics['EmailsOpened'] * 100, 
                                                0)
                    
                    # Line chart for metrics over time
                    fig_time = go.Figure()
                    
                    if metric_type in ["Open Rate", "All Metrics"]:
                        fig_time.add_trace(go.Scatter(
                            x=time_metrics['Date'],
                            y=time_metrics['OpenRate'],
                            mode='lines+markers',
                            name='Open Rate (%)',
                            line=dict(color='rgba(74, 111, 255, 0.9)', width=3),
                            marker=dict(size=8)
                        ))
                    
                    if metric_type in ["Click Rate", "All Metrics"]:
                        fig_time.add_trace(go.Scatter(
                            x=time_metrics['Date'],
                            y=time_metrics['ClickRate'],
                            mode='lines+markers',
                            name='Click Rate (%)',
                            line=dict(color='rgba(255, 99, 132, 0.9)', width=3),
                            marker=dict(size=8)
                        ))
                    
                    if metric_type in ["Click-to-Open Rate", "All Metrics"]:
                        fig_time.add_trace(go.Scatter(
                            x=time_metrics['Date'],
                            y=time_metrics['CTOR'],
                            mode='lines+markers',
                            name='CTOR (%)',
                            line=dict(color='rgba(50, 168, 82, 0.9)', width=3),
                            marker=dict(size=8)
                        ))
                    
                    fig_time.update_layout(
                        title='Email Metrics Trends Over Time (Weekly)',
                        xaxis_title='Date',
                        yaxis_title='Rate (%)',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    # Comparison line chart if a specific company is selected
                    if selected_company != 'All Companies' and 'Company' in time_series_df.columns:
                        st.subheader(f"Performance Comparison: {selected_company} vs Others")

                        # Fix: Ensure consistent datetime format
                        time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
                        
                        # Calculate metrics for the selected company
                        selected_company_df = time_series_df[time_series_df['Company'] == selected_company].copy()
                        
                        if not selected_company_df.empty:
                            # Fix: Use resample instead of Grouper
                            company_time = selected_company_df.set_index('Date').resample('W').agg({
                                'EmailsSent': 'sum',
                                'EmailsOpened': 'sum',
                                'EmailsClicked': 'sum'
                            }).reset_index()
                        
                            # Only proceed if we have data
                            if not company_time.empty:
                                company_time['OpenRate'] = np.where(company_time['EmailsSent'] > 0, 
                                                            company_time['EmailsOpened'] / company_time['EmailsSent'] * 100, 
                                                            0)
                                company_time['ClickRate'] = np.where(company_time['EmailsSent'] > 0, 
                                                            company_time['EmailsClicked'] / company_time['EmailsSent'] * 100, 
                                                            0)
                                
                                # Calculate metrics for all other companies
                                other_companies_df = time_series_df[time_series_df['Company'] != selected_company].copy()
                                if not other_companies_df.empty:
                                    # Fix: Use resample instead of Grouper
                                    other_time = other_companies_df.set_index('Date').resample('W').agg({
                                        'EmailsSent': 'sum',
                                        'EmailsOpened': 'sum',
                                        'EmailsClicked': 'sum'
                                    }).reset_index()
                                
                                    if not other_time.empty:
                                        other_time['OpenRate'] = np.where(other_time['EmailsSent'] > 0, 
                                                                other_time['EmailsOpened'] / other_time['EmailsSent'] * 100, 
                                                                0)
                                        other_time['ClickRate'] = np.where(other_time['EmailsSent'] > 0, 
                                                                other_time['EmailsClicked'] / other_time['EmailsSent'] * 100, 
                                                                0)
                                        
                                        # Create comparison chart
                                        fig_comp = go.Figure()
                                        
                                        # Add traces based on metric type
                                        if metric_type in ["Open Rate", "All Metrics"]:
                                            fig_comp.add_trace(go.Scatter(
                                                x=company_time['Date'],
                                                y=company_time['OpenRate'],
                                                mode='lines+markers',
                                                name=f'{selected_company} Open Rate',
                                                line=dict(color='rgba(74, 111, 255, 0.9)', width=3),
                                                marker=dict(size=8)
                                            ))
                                            
                                            fig_comp.add_trace(go.Scatter(
                                                x=other_time['Date'],
                                                y=other_time['OpenRate'],
                                                mode='lines+markers',
                                                name='Others Open Rate',
                                                line=dict(color='rgba(74, 111, 255, 0.4)', width=2, dash='dash'),
                                                marker=dict(size=6)
                                            ))
                                        
                                        if metric_type in ["Click Rate", "All Metrics"]:
                                            fig_comp.add_trace(go.Scatter(
                                                x=company_time['Date'],
                                                y=company_time['ClickRate'],
                                                mode='lines+markers',
                                                name=f'{selected_company} Click Rate',
                                                line=dict(color='rgba(255, 99, 132, 0.9)', width=3),
                                                marker=dict(size=8)
                                            ))
                                            
                                            fig_comp.add_trace(go.Scatter(
                                                x=other_time['Date'],
                                                y=other_time['ClickRate'],
                                                mode='lines+markers',
                                                name='Others Click Rate',
                                                line=dict(color='rgba(255, 99, 132, 0.4)', width=2, dash='dash'),
                                                marker=dict(size=6)
                                            ))
                                        
                                        fig_comp.update_layout(
                                            title=f'Performance Comparison: {selected_company} vs Others',
                                            xaxis_title='Date',
                                            yaxis_title='Rate (%)',
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            ),
                                            hovermode='x unified'
                                        )
                                        
                                        st.plotly_chart(fig_comp, use_container_width=True)
                                    else:
                                        st.info("No comparison data available for other companies.")
                            else:
                                st.info(f"No time series data available for {selected_company}.")
                else:
                    st.info("No time series data available with current filters.")
            else:
                st.info("No time series data available with current filters.")
        
        with tab3:
            st.subheader("Heat Map Analysis")
            
            # Create a pivot table for the heat map (Company vs JobRole)
            if not filtered_df.empty:
                if metric_type == "Open Rate":
                    metric_col = 'OpenRate'
                    title = 'Open Rate (%) by Company and Job Role'
                elif metric_type == "Click Rate":
                    metric_col = 'ClickRate'
                    title = 'Click Rate (%) by Company and Job Role'
                elif metric_type == "Click-to-Open Rate":
                    metric_col = 'CTOR'
                    title = 'Click-to-Open Rate (%) by Company and Job Role'
                else:
                    # Default to engagement score for "All Metrics"
                    metric_col = 'EngagementScore'
                    title = 'Engagement Score by Company and Job Role'
                
                # Generate pivot table
                try:
                    pivot_df = pd.pivot_table(
                        filtered_df, 
                        values=metric_col, 
                        index='Company', 
                        columns='JobRole', 
                        aggfunc='mean',
                        fill_value=0
                    )
                    # Create heatmap
                    fig_heatmap = px.imshow(
                        pivot_df,
                        text_auto='.1f',
                        aspect="auto",
                        color_continuous_scale='Blues',
                        title=title
                    )
                    
                    fig_heatmap.update_layout(
                        height=500,
                        xaxis_title='Job Role',
                        yaxis_title='Company'
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not generate heat map: {e}")
                    st.info("Heat map requires multiple companies and job roles in the filtered data.")
            
            else:
                st.info("No data available for heat map with current filters.")
            
        with tab4:
            st.subheader("Detailed Data")

            # Display the filtered data with metrics
            if not filtered_df.empty:
                # Select columns to display
                display_cols = ['Company', 'JobRole', 'PersonName', 'EmailsSent', 'EmailsOpened', 'EmailsClicked', 'OpenRate', 'ClickRate', 'EngagementScore']

                # Format numeric columns
                display_df = filtered_df[display_cols].copy()
                display_df['OpenRate'] = display_df['OpenRate'].round(1).astype(str) + '%'
                display_df['ClickRate'] = display_df['ClickRate'].round(1).astype(str) + '%'
                display_df['EngagementScore'] = display_df['EngagementScore'].round(2)

                # Rename columns for better readability
                display_df.columns = ['Company', 'Job Role', 'Name', 'Emails Sent', 'Emails Opened', 'Emails Clicked', 'Open Rate', 'Click Rate', 'Engagement Score']

                # Display the table
                st.dataframe(display_df, use_container_width=True)
                
                # Add CSV download button
                csv = display_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="email_metrics.csv" class="stDownloadButton">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No data available with current filters.")

            # Performance insights
            st.subheader("Performance Insights")

            if not filtered_df.empty:
                # Calculate top performers
                top_companies = filtered_df.groupby('Company')['EngagementScore'].mean().sort_values(ascending=False)
                top_roles = filtered_df.groupby('JobRole')['EngagementScore'].mean().sort_values(ascending=False)
                
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top Engaging Companies:**")
                    if not top_companies.empty:
                        top_company_df = top_companies.head(5).reset_index()
                        top_company_df.columns = ['Company', 'Avg. Engagement Score']
                        top_company_df['Avg. Engagement Score'] = top_company_df['Avg. Engagement Score'].round(2)
                        st.dataframe(top_company_df, use_container_width=True)
                    else:
                        st.info("No company engagement data available.")

                with col2:
                    st.markdown("**Top Engaging Job Roles:**")
                    if not top_roles.empty:
                        top_role_df = top_roles.head(5).reset_index()
                        top_role_df.columns = ['Job Role', 'Avg. Engagement Score']
                        top_role_df['Avg. Engagement Score'] = top_role_df['Avg. Engagement Score'].round(2)
                        st.dataframe(top_role_df, use_container_width=True)
                    else:
                        st.info("No job role engagement data available.")
                
                # Key insights
                st.markdown("### Key Insights")
                
                # Generate basic insights based on the data
                insights = []
                
                # Overall performance insight
                benchmarks = generate_industry_benchmarks()
                industry_open_rate = benchmarks['OpenRate']['Overall']
                industry_click_rate = benchmarks['ClickRate']['Overall']

                if open_rate > industry_open_rate:
                    insights.append(f"✅ Your open rate ({open_rate:.1f}%) is above the industry average ({industry_open_rate}%).")
                else:
                    insights.append(f"⚠️ Your open rate ({open_rate:.1f}%) is below the industry average ({industry_open_rate}%).")
                    
                if click_rate > industry_click_rate:
                    insights.append(f"✅ Your click rate ({click_rate:.1f}%) is above the industry average ({industry_click_rate}%).")
                else:
                    insights.append(f"⚠️ Your click rate ({click_rate:.1f}%) is below the industry average ({industry_click_rate}%).")
                
                # Best and worst performing segments
                if not top_companies.empty and not top_companies.head(1).empty:
                    best_company = top_companies.head(1).index[0]
                    best_company_score = top_companies.head(1).values[0]
                    insights.append(f"🏆 {best_company} has the highest engagement score ({best_company_score:.2f}).")
                
                if not top_roles.empty and not top_roles.head(1).empty:
                    best_role = top_roles.head(1).index[0]
                    best_role_score = top_roles.head(1).values[0]
                    insights.append(f"👑 {best_role} is the most engaged job role ({best_role_score:.2f}).")
                
                # Display insights
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Recommendations based on insights
                st.markdown("### Recommendations")

                recommendations = [
                    "Segment your audience more granularly based on engagement patterns.",
                    "A/B test subject lines to improve open rates.",
                    f"Focus on personalized content for {best_role if not top_roles.empty and len(top_roles) > 0 else 'top-performing roles'}."
                ]
                
                if open_rate < industry_open_rate:
                    recommendations.append("Improve subject lines and send times to boost open rates.")
                
                if click_rate < industry_click_rate:
                    recommendations.append("Enhance email content and CTAs to increase click rates.")
                
                for recommendation in recommendations:
                    st.markdown(f"- {recommendation}")
            
            else:
                st.info("No data available for insights with current filters.")
    
    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
        st.info("Please check your data format and try again.")

else:
    # Display instructions when no data is loaded
    st.markdown("""
    ## Welcome to the Email Metrics Dashboard 📧
    
    This dashboard helps you visualize and analyze your email campaign performance.
    
    ### Getting Started:
    1. Use the sidebar to upload your CSV data or load sample data
    2. Your CSV should include these columns:
       - Company
       - JobRole
       - PersonName
       - EmailsSent (yes/no)
       - EmailsOpened (yes/no)
       - EmailsClicked (yes/no)
    3. Once data is loaded, you can filter by company, job role, and metric type
    
    ### Key Features:
    - Performance analysis with interactive charts
    - Time series trends
    - Heat maps showing engagement patterns
    - Detailed data tables with export functionality
    - Actionable insights and recommendations
    
    Need sample data? Click the "Load Sample Data" button in the sidebar to get started.
    """)
    
    # Sample data format display
    st.markdown("### Sample Data Format")
    sample_format = pd.DataFrame({
        'Company': ['Acme Corp', 'TechGiant'],
        'JobRole': ['Manager', 'Director'],
        'PersonName': ['John Smith', 'Jane Doe'],
        'EmailsSent': ['yes', 'yes'],
        'EmailsOpened': ['yes', 'no'],
        'EmailsClicked': ['no', 'no']
    })
    st.dataframe(sample_format, use_container_width=True)
