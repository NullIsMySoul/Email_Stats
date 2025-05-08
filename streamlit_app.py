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
    .engagement-stats {
        font-size: 1.1rem;
        color: #343a40;
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .stats-highlight {
        font-weight: bold;
        color: #4a6fff;
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
                <div class="metric-label">Open Rate ({total_opened}/{total_sent})</div>
                <div class="tooltip">ℹ️
                    <span class="tooltiptext">Percentage of emails that were opened</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{click_rate:.1f}%</div>
                <div class="metric-label">Click Rate ({total_clicked}/{total_sent})</div>
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
        
        tab1, tab2 = st.tabs(["Bar Charts", "Data Table"])
        
        with tab1:
            # Company-level detailed engagement metrics
            if selected_company == 'All Companies':
                # Calculate company-level metrics
                company_engagement = filtered_df.groupby('Company').agg({
                    'EmailsSent': 'sum',
                    'EmailsOpened': 'sum',
                    'EmailsClicked': 'sum',
                    'PersonName': 'count'
                }).reset_index()
                
                # Rename PersonName column to TotalPeople
                company_engagement.rename(columns={'PersonName': 'TotalPeople'}, inplace=True)
                
                # Calculate rates
                company_engagement['OpenRate'] = np.where(company_engagement['EmailsSent'] > 0, 
                                                      company_engagement['EmailsOpened'] / company_engagement['EmailsSent'] * 100, 
                                                      0)
                company_engagement['ClickRate'] = np.where(company_engagement['EmailsSent'] > 0, 
                                                       company_engagement['EmailsClicked'] / company_engagement['EmailsSent'] * 100, 
                                                       0)
                
                # Company metrics visualization
                st.subheader("Company Engagement Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart for open rates
                    if metric_type in ["Open Rate", "All Metrics"]:
                        fig_open = go.Figure(go.Bar(
                            x=company_engagement['Company'],
                            y=company_engagement['OpenRate'],
                            text=[f"{rate:.1f}% ({opened}/{total} people)" for rate, opened, total in 
                                  zip(company_engagement['OpenRate'], company_engagement['EmailsOpened'], company_engagement['TotalPeople'])],
                            textposition='outside',
                            marker_color='rgba(74, 111, 255, 0.7)'
                        ))
                        
                        fig_open.update_layout(
                            title='Open Rate by Company',
                            xaxis_title='Company',
                            yaxis_title='Open Rate (%)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_open, use_container_width=True)
                
                with col2:
                    # Bar chart for click rates
                    if metric_type in ["Click Rate", "All Metrics"]:
                        fig_click = go.Figure(go.Bar(
                            x=company_engagement['Company'],
                            y=company_engagement['ClickRate'],
                            text=[f"{rate:.1f}% ({clicked}/{total} people)" for rate, clicked, total in 
                                  zip(company_engagement['ClickRate'], company_engagement['EmailsClicked'], company_engagement['TotalPeople'])],
                            textposition='outside',
                            marker_color='rgba(255, 99, 132, 0.7)'
                        ))
                        
                        fig_click.update_layout(
                            title='Click Rate by Company',
                            xaxis_title='Company',
                            yaxis_title='Click Rate (%)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_click, use_container_width=True)
                
                # Display company engagement details table
                with st.expander("View Company Engagement Details", expanded=True):
                    # Format the table for display
                    display_company_engagement = company_engagement.copy()
                    display_company_engagement['Open Rate'] = display_company_engagement['OpenRate'].round(1).astype(str) + '%'
                    display_company_engagement['Click Rate'] = display_company_engagement['ClickRate'].round(1).astype(str) + '%'
                    display_company_engagement['Open Details'] = [f"{opened}/{total} people" for opened, total in 
                                                                zip(display_company_engagement['EmailsOpened'], display_company_engagement['TotalPeople'])]
                    display_company_engagement['Click Details'] = [f"{clicked}/{total} people" for clicked, total in 
                                                                 zip(display_company_engagement['EmailsClicked'], display_company_engagement['TotalPeople'])]
                    
                    # Select and rename columns for display
                    display_cols = ['Company', 'TotalPeople', 'Open Rate', 'Open Details', 'Click Rate', 'Click Details']
                    display_names = ['Company', 'Total People', 'Open Rate', 'Opens', 'Click Rate', 'Clicks']
                    display_company_table = display_company_engagement[display_cols].copy()
                    display_company_table.columns = display_names
                    
                    st.dataframe(display_company_table, use_container_width=True)
            
            # Job Role-level detailed engagement metrics
            if selected_role == 'All Roles':
                # Calculate role-level metrics
                role_engagement = filtered_df.groupby('JobRole').agg({
                    'EmailsSent': 'sum',
                    'EmailsOpened': 'sum',
                    'EmailsClicked': 'sum',
                    'PersonName': 'count'
                }).reset_index()
                
                # Rename PersonName column to TotalPeople
                role_engagement.rename(columns={'PersonName': 'TotalPeople'}, inplace=True)
                
                # Calculate rates
                role_engagement['OpenRate'] = np.where(role_engagement['EmailsSent'] > 0, 
                                                   role_engagement['EmailsOpened'] / role_engagement['EmailsSent'] * 100, 
                                                   0)
                role_engagement['ClickRate'] = np.where(role_engagement['EmailsSent'] > 0, 
                                                    role_engagement['EmailsClicked'] / role_engagement['EmailsSent'] * 100, 
                                                    0)
                
                # Job Role metrics visualization
                st.subheader("Job Role Engagement Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart for open rates by role
                    if metric_type in ["Open Rate", "All Metrics"]:
                        fig_role_open = go.Figure(go.Bar(
                            x=role_engagement['JobRole'],
                            y=role_engagement['OpenRate'],
                            text=[f"{rate:.1f}% ({opened}/{total} people)" for rate, opened, total in 
                                  zip(role_engagement['OpenRate'], role_engagement['EmailsOpened'], role_engagement['TotalPeople'])],
                            textposition='outside',
                            marker_color='rgba(74, 111, 255, 0.7)'
                        ))
                        
                        fig_role_open.update_layout(
                            title='Open Rate by Job Role',
                            xaxis_title='Job Role',
                            yaxis_title='Open Rate (%)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_role_open, use_container_width=True)
                
                with col2:
                    # Bar chart for click rates by role
                    if metric_type in ["Click Rate", "All Metrics"]:
                        fig_role_click = go.Figure(go.Bar(
                            x=role_engagement['JobRole'],
                            y=role_engagement['ClickRate'],
                            text=[f"{rate:.1f}% ({clicked}/{total} people)" for rate, clicked, total in 
                                  zip(role_engagement['ClickRate'], role_engagement['EmailsClicked'], role_engagement['TotalPeople'])],
                            textposition='outside',
                            marker_color='rgba(255, 99, 132, 0.7)'
                        ))
                        
                        fig_role_click.update_layout(
                            title='Click Rate by Job Role',
                            xaxis_title='Job Role',
                            yaxis_title='Click Rate (%)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_role_click, use_container_width=True)
                
                # Display role engagement details table
                with st.expander("View Job Role Engagement Details", expanded=True):
                    # Format the table for display
                    display_role_engagement = role_engagement.copy()
                    display_role_engagement['Open Rate'] = display_role_engagement['OpenRate'].round(1).astype(str) + '%'
                    display_role_engagement['Click Rate'] = display_role_engagement['ClickRate'].round(1).astype(str) + '%'
                    display_role_engagement['Open Details'] = [f"{opened}/{total} people" for opened, total in 
                                                            zip(display_role_engagement['EmailsOpened'], display_role_engagement['TotalPeople'])]
                    display_role_engagement['Click Details'] = [f"{clicked}/{total} people" for clicked, total in 
                                                             zip(display_role_engagement['EmailsClicked'], display_role_engagement['TotalPeople'])]
                    
                    # Select and rename columns for display
                    display_cols = ['JobRole', 'TotalPeople', 'Open Rate', 'Open Details', 'Click Rate', 'Click Details']
                    display_names = ['Job Role', 'Total People', 'Open Rate', 'Opens', 'Click Rate', 'Clicks']
                    display_role_table = display_role_engagement[display_cols].copy()
                    display_role_table.columns = display_names
                    
                    st.dataframe(display_role_table, use_container_width=True)
            
            # If specific company is selected, show job roles within that company
            if selected_company != 'All Companies':
                st.subheader(f"Job Role Breakdown for {selected_company}")
                
                # Filter data for selected company
                company_data = filtered_df[filtered_df['Company'] == selected_company]
                
                # Group by job role
                company_roles = company_data.groupby('JobRole').agg({
                    'EmailsSent': 'sum',
                    'EmailsOpened': 'sum',
                    'EmailsClicked': 'sum',
                    'PersonName': 'count'
                }).reset_index()
                
                # Rename PersonName column to TotalPeople
                company_roles.rename(columns={'PersonName': 'TotalPeople'}, inplace=True)
                
                # Calculate rates
                company_roles['OpenRate'] = np.where(company_roles['EmailsSent'] > 0, 
                                                  company_roles['EmailsOpened'] / company_roles['EmailsSent'] * 100, 
                                                  0)
                company_roles['ClickRate'] = np.where(company_roles['EmailsSent'] > 0, 
                                                   company_roles['EmailsClicked'] / company_roles['EmailsSent'] * 100, 
                                                   0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart for open rates by role within company
                    if metric_type in ["Open Rate", "All Metrics"]:
                        fig_comp_role_open = go.Figure(go.Bar(
                            x=company_roles['JobRole'],
                            y=company_roles['OpenRate'],
                            text=[f"{rate:.1f}% ({opened}/{total} people)" for rate, opened, total in 
                                  zip(company_roles['OpenRate'], company_roles['EmailsOpened'], company_roles['TotalPeople'])],
                            textposition='outside',
                            marker_color='rgba(74, 111, 255, 0.7)'
                        ))
                        
                        fig_comp_role_open.update_layout(
                            title=f'Open Rate by Job Role at {selected_company}',
                            xaxis_title='Job Role',
                            yaxis_title='Open Rate (%)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_comp_role_open, use_container_width=True)
                
                with col2:
                    # Bar chart for click rates by role within company
                    if metric_type in ["Click Rate", "All Metrics"]:
                        fig_comp_role_click = go.Figure(go.Bar(
                            x=company_roles['JobRole'],
                            y=company_roles['ClickRate'],
                            text=[f"{rate:.1f}% ({clicked}/{total} people)" for rate, clicked, total in 
                                  zip(company_roles['ClickRate'], company_roles['EmailsClicked'], company_roles['TotalPeople'])],
                            textposition='outside',
                            marker_color='rgba(255, 99, 132, 0.7)'
                        ))
                        
                        fig_comp_role_click.update_layout(
                            title=f'Click Rate by Job Role at {selected_company}',
                            xaxis_title='Job Role',
                            yaxis_title='Click Rate (%)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_comp_role_click, use_container_width=True)
                
                # Display who engaged from this company
                st.subheader(f"Individuals who engaged at {selected_company}")
                
                # Get people who opened emails
                openers = company_data[company_data['EmailsOpened'] == 1][['PersonName', 'JobRole']]
                openers['Action'] = 'Opened'
                
                # Get people who clicked emails
                clickers = company_data[company_data['EmailsClicked'] == 1][['PersonName', 'JobRole']]
                clickers['Action'] = 'Clicked'
                
                # Combine and display in a table
                if not openers.empty or not clickers.empty:
                    engaged_people = pd.concat([openers, clickers]).drop_duplicates()
                    engaged_people.columns = ['Name', 'Job Role', 'Action']
                    
                    st.dataframe(engaged_people, use_container_width=True)
                else:
                    st.info(f"No one from {selected_company} has engaged with your emails yet.")
        
        with tab2:
            st.subheader("Email Engagement Details")
            
            # Display company and job role details about who opened/clicked
            company_job_engagement = filtered_df.groupby(['Company', 'JobRole']).agg({
                'EmailsSent': 'sum',
                'EmailsOpened': 'sum',
                'EmailsClicked': 'sum',
                'PersonName': lambda x: ', '.join(filtered_df.loc[filtered_df.loc[x.index]['EmailsOpened'] == 1, 'PersonName'])
            }).reset_index()
            
            # Rename columns
            company_job_engagement.rename(columns={
                'PersonName': 'People Who Opened'
            }, inplace=True)
            
            # Add people who clicked
            company_job_engagement['People Who Clicked'] = filtered_df.groupby(['Company', 'JobRole']).apply(
                lambda x: ', '.join(x.loc[x['EmailsClicked'] == 1, 'PersonName'])
            ).reset_index(drop=True)
            
            # Count total people per company/role combination
            people_count = filtered_df.groupby(['Company', 'JobRole']).size().reset_index(name='Total People')
            company_job_engagement = company_job_engagement.merge(people_count, on=['Company', 'JobRole'])
            
            # Calculate rates
            company_job_engagement['Open Rate'] = np.where(company_job_engagement['EmailsSent'] > 0,
                                                        (company_job_engagement['EmailsOpened'] / company_job_engagement['EmailsSent'] * 100).round(1).astype(str) + '%',
                                                        '0.0%')
            
            company_job_engagement['Click Rate'] = np.where(company_job_engagement['EmailsSent'] > 0,
                                                         (company_job_engagement['EmailsClicked'] / company_job_engagement['EmailsSent'] * 100).round(1).astype(str) + '%',
                                                         '0.0%')
            
            # Add engagement details
            company_job_engagement['Open Details'] = company_job_engagement['EmailsOpened'].astype(str) + '/' + company_job_engagement['Total People'].astype(str)
            company_job_engagement['Click Details'] = company_job_engagement['EmailsClicked'].astype(str) + '/' + company_job_engagement['Total People'].astype(str)
            
            # Select and rename columns for display
            display_cols = ['Company', 'JobRole', 'Total People', 'Open Rate', 'Open Details', 'People Who Opened', 
                           'Click Rate', 'Click Details', 'People Who Clicked']
            display_names = ['Company', 'Job Role', 'Total People', 'Open Rate', 'Opens', 'People Who Opened', 
                            'Click Rate', 'Clicks', 'People Who Clicked']
            
            detailed_table = company_job_engagement[display_cols].copy()
            detailed_table.columns = display_names
            
            # Display the table
            st.dataframe(detailed_table, use_container_width=True)

            # Add export functionality
            csv = detailed_table.to_csv(index=False)

            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="email_engagement_report.csv">Download CSV Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Insights and recommendations section
        st.markdown('<div class="sub-header">Insights & Recommendations</div>', unsafe_allow_html=True)
        
        with st.expander("View Insights", expanded=True):
            if not filtered_df.empty:
                # Calculate engagement metrics for insights
                total_recipients = len(filtered_df)
                engaged_recipients = len(filtered_df[filtered_df['EmailsOpened'] == 1])
                click_recipients = len(filtered_df[filtered_df['EmailsClicked'] == 1])
                
                # Industry benchmarks for comparison
                industry_benchmarks = generate_industry_benchmarks()
                
                # Generate insights based on the data
                st.markdown(f"""
                <div class="engagement-stats">
                    <h3>Overall Engagement</h3>
                    <ul>
                        <li>Your email campaign reached <span class="stats-highlight">{total_recipients}</span> recipients.</li>
                        <li>Your open rate is <span class="stats-highlight">{open_rate:.1f}%</span>, which is 
                            {'<span class="stats-highlight">above</span>' if open_rate > industry_benchmarks['OpenRate']['Overall'] else '<span class="stats-highlight">below</span>'} 
                            the industry average of {industry_benchmarks['OpenRate']['Overall']}%.</li>
                        <li>Your click rate is <span class="stats-highlight">{click_rate:.1f}%</span>, which is 
                            {'<span class="stats-highlight">above</span>' if click_rate > industry_benchmarks['ClickRate']['Overall'] else '<span class="stats-highlight">below</span>'} 
                            the industry average of {industry_benchmarks['ClickRate']['Overall']}%.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate recommendations based on metrics
                st.markdown("""
                <div class="engagement-stats">
                    <h3>Recommendations</h3>
                    <ul>
                """, unsafe_allow_html=True)
                
                # Open rate recommendations
                if open_rate < industry_benchmarks['OpenRate']['Overall']:
                    st.markdown("""
                    <li>To improve your <b>open rate</b>:
                        <ul>
                            <li>Experiment with more compelling subject lines</li>
                            <li>Test different sending times</li>
                            <li>Segment your audience for more targeted emails</li>
                            <li>Clean your email list to remove inactive subscribers</li>
                        </ul>
                    </li>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <li>Your <b>open rate</b> is performing well. Keep:
                        <ul>
                            <li>Using effective subject lines</li>
                            <li>Maintaining your sender reputation</li>
                            <li>Sending at optimal times</li>
                        </ul>
                    </li>
                    """, unsafe_allow_html=True)
                
                # Click rate recommendations
                if click_rate < industry_benchmarks['ClickRate']['Overall']:
                    st.markdown("""
                    <li>To improve your <b>click rate</b>:
                        <ul>
                            <li>Make your call-to-action buttons more prominent</li>
                            <li>Ensure your content is relevant to the audience</li>
                            <li>Test different content formats (images, videos, etc.)</li>
                            <li>Use personalization to increase relevance</li>
                        </ul>
                    </li>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <li>Your <b>click rate</b> is performing well. Continue:
                        <ul>
                            <li>Using clear and compelling calls-to-action</li>
                            <li>Creating valuable and relevant content</li>
                            <li>Keeping your emails concise and focused</li>
                        </ul>
                    </li>
                    """, unsafe_allow_html=True)
                
                # Targeting recommendations
                if selected_company == 'All Companies' and selected_role == 'All Roles':
                    # Find top performing segments
                    if 'company_engagement' in locals():
                        top_company = company_engagement.sort_values('ClickRate', ascending=False).iloc[0]
                        top_company_name = top_company['Company']
                        top_company_click_rate = top_company['ClickRate']
                        
                        st.markdown(f"""
                        <li>Consider focusing more on <b>{top_company_name}</b>:
                            <ul>
                                <li>This company shows the highest engagement with a click rate of {top_company_click_rate:.1f}%</li>
                                <li>Consider creating more targeted content for this audience</li>
                            </ul>
                        </li>
                        """, unsafe_allow_html=True)
                    
                    if 'role_engagement' in locals():
                        top_role = role_engagement.sort_values('ClickRate', ascending=False).iloc[0]
                        top_role_name = top_role['JobRole']
                        top_role_click_rate = top_role['ClickRate']
                        
                        st.markdown(f"""
                        <li>The <b>{top_role_name}</b> role responds best to your emails:
                            <ul>
                                <li>This role shows a click rate of {top_role_click_rate:.1f}%</li>
                                <li>Consider tailoring future content to address their specific needs</li>
                            </ul>
                        </li>
                        """, unsafe_allow_html=True)
                
                st.markdown("""
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No data available to generate insights. Please upload data or use the sample data.")
        
        # Advanced analytics section
        st.markdown('<div class="sub-header">Advanced Analytics</div>', unsafe_allow_html=True)
        
        # Engagement correlation analysis
        advanced_tab1, advanced_tab2 = st.tabs(["Engagement Correlation", "Time Series Simulation"])
        
        with advanced_tab1:
            st.subheader("Correlation between Job Roles and Engagement")
            
            if not filtered_df.empty:
                # Create a pivot table for job roles vs engagement
                role_pivot = pd.pivot_table(
                    filtered_df,
                    values=['EmailsOpened', 'EmailsClicked'],
                    index='JobRole',
                    aggfunc=np.sum
                ).reset_index()
                
                # Add total emails for each role
                role_counts = filtered_df.groupby('JobRole').size().reset_index(name='TotalEmails')
                role_pivot = role_pivot.merge(role_counts, on='JobRole')
                
                # Calculate rates
                role_pivot['OpenRate'] = (role_pivot['EmailsOpened'] / role_pivot['TotalEmails'] * 100).round(1)
                role_pivot['ClickRate'] = (role_pivot['EmailsClicked'] / role_pivot['TotalEmails'] * 100).round(1)
                
                # Create correlation heatmap
                fig = px.scatter(
                    role_pivot,
                    x='OpenRate',
                    y='ClickRate',
                    size='TotalEmails',
                    color='JobRole',
                    hover_name='JobRole',
                    text='JobRole',
                    title='Job Role Engagement Analysis'
                )
                
                fig.update_layout(
                    xaxis_title='Open Rate (%)',
                    yaxis_title='Click Rate (%)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="engagement-stats">
                    <h3>Interpretation</h3>
                    <p>This bubble chart shows the relationship between open rates and click rates across different job roles:</p>
                    <ul>
                        <li>Bubble size represents the number of emails sent to that role</li>
                        <li>Roles in the upper right corner have high open and click rates (most engaged)</li>
                        <li>Roles in the lower left corner have low open and click rates (least engaged)</li>
                        <li>Roles with high open but low click rates may need more compelling content</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No data available for correlation analysis.")
        
        with advanced_tab2:
            st.subheader("Email Campaign Simulation")
            
            # Simulation parameters
            st.markdown("Simulate future email campaign performance based on current metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sim_emails = st.number_input("Number of emails to send", min_value=100, max_value=10000, value=1000, step=100)
                sim_improvement = st.slider("Estimated improvement (%)", min_value=-20, max_value=50, value=5, step=5)
            
            with col2:
                sim_period = st.selectbox("Time period", ["Next week", "Next month", "Next quarter"])
                sim_segment = st.selectbox("Target segment", ["All recipients", "High engagement segment", "Low engagement segment"])
            
            # Run simulation button
            if st.button("Run Simulation"):
                # Get current metrics
                current_open_rate = open_rate / 100  # Convert to decimal
                current_click_rate = click_rate / 100  # Convert to decimal
                
                # Apply improvement factor
                improved_open_rate = current_open_rate * (1 + sim_improvement/100)
                improved_click_rate = current_click_rate * (1 + sim_improvement/100)
                
                # Cap rates at realistic values
                improved_open_rate = min(improved_open_rate, 0.95)
                improved_click_rate = min(improved_click_rate, 0.8)
                
                # Calculate expected outcomes
                expected_opens = int(sim_emails * improved_open_rate)
                expected_clicks = int(sim_emails * improved_click_rate)
                
                # Create time series data for visualization
                if sim_period == "Next week":
                    dates = [datetime.now() + timedelta(days=i) for i in range(7)]
                    daily_emails = [int(sim_emails / 7) for _ in range(7)]
                elif sim_period == "Next month":
                    dates = [datetime.now() + timedelta(days=i) for i in range(0, 30, 3)]
                    daily_emails = [int(sim_emails / 10) for _ in range(10)]
                else:  # Next quarter
                    dates = [datetime.now() + timedelta(days=i) for i in range(0, 90, 9)]
                    daily_emails = [int(sim_emails / 10) for _ in range(10)]
                
                # Add some randomness to make it realistic
                daily_opens = [int(emails * improved_open_rate * np.random.uniform(0.9, 1.1)) for emails in daily_emails]
                daily_clicks = [int(emails * improved_click_rate * np.random.uniform(0.9, 1.1)) for emails in daily_emails]
                
                # Create dataframe for plotting
                sim_df = pd.DataFrame({
                    'Date': dates,
                    'Emails': daily_emails,
                    'Opens': daily_opens,
                    'Clicks': daily_clicks
                })
                
                # Cumulative data
                sim_df['Cum_Emails'] = sim_df['Emails'].cumsum()
                sim_df['Cum_Opens'] = sim_df['Opens'].cumsum()
                sim_df['Cum_Clicks'] = sim_df['Clicks'].cumsum()
                
                # Create time series plot
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=('Daily Metrics', 'Cumulative Performance'),
                                   vertical_spacing=0.15,
                                   shared_xaxes=True)
                
                # Daily metrics
                fig.add_trace(
                    go.Bar(x=sim_df['Date'], y=sim_df['Emails'], name='Emails Sent', marker_color='rgba(156, 165, 196, 0.7)'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=sim_df['Date'], y=sim_df['Opens'], name='Opens', marker_color='rgba(74, 111, 255, 0.7)'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=sim_df['Date'], y=sim_df['Clicks'], name='Clicks', marker_color='rgba(255, 99, 132, 0.7)'),
                    row=1, col=1
                )
                
                # Cumulative metrics
                fig.add_trace(
                    go.Scatter(x=sim_df['Date'], y=sim_df['Cum_Emails'], name='Cum. Emails', line=dict(width=3, color='rgb(156, 165, 196)')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=sim_df['Date'], y=sim_df['Cum_Opens'], name='Cum. Opens', line=dict(width=3, color='rgb(74, 111, 255)')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=sim_df['Date'], y=sim_df['Cum_Clicks'], name='Cum. Clicks', line=dict(width=3, color='rgb(255, 99, 132)')),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    height=600,
                    title_text=f"Email Campaign Simulation ({sim_period})",
                    legend=dict(orientation="h", y=1.1),
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Projected Opens",
                        value=f"{expected_opens:,}",
                        delta=f"{sim_improvement}% from current"
                    )
                
                with col2:
                    st.metric(
                        label="Projected Clicks",
                        value=f"{expected_clicks:,}",
                        delta=f"{sim_improvement}% from current"
                    )
                
                with col3:
                    roi_estimate = expected_clicks * 50  # Assuming $50 value per click
                    st.metric(
                        label="Estimated ROI",
                        value=f"${roi_estimate:,}",
                        delta=f"Based on {expected_clicks} clicks"
                    )
                
                # Add simulation insights
                st.markdown(f"""
                <div class="engagement-stats">
                    <h3>Simulation Insights</h3>
                    <p>Based on your current metrics and an estimated improvement of {sim_improvement}%, here's what you can expect:</p>
                    <ul>
                        <li>From {sim_emails:,} emails, you'll likely get {expected_opens:,} opens and {expected_clicks:,} clicks</li>
                        <li>Your projected open rate will be {improved_open_rate*100:.1f}% (vs current {open_rate:.1f}%)</li>
                        <li>Your projected click rate will be {improved_click_rate*100:.1f}% (vs current {click_rate:.1f}%)</li>
                        <li>If each click is worth about $50 in potential revenue, this campaign could generate approximately ${roi_estimate:,}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Adjust the parameters above and click 'Run Simulation' to see projected results.")
    
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.info("Please check your data format and try again.")
else:
    # Display welcome message when no data is loaded
    st.info("Welcome to the Email Metrics Dashboard! Please upload your email engagement data or load the sample data to get started.")
    
    st.markdown("""
    ### How to use this dashboard:
    
    1. **Upload your data**: Use the uploader in the sidebar to upload a CSV file with your email metrics.
    2. **Explore metrics**: View summary metrics and detailed analytics about your email campaign performance.
    3. **Filter data**: Use the filters to narrow down your analysis by company, job role, or metric type.
    4. **Get insights**: Check the Insights & Recommendations section for actionable tips.
    5. **Export data**: Download your results as a CSV for further analysis or reporting.
    
    ### Required CSV Format:
    
    Your CSV should include the following columns:
    - `Company`: The company name of each recipient
    - `JobRole`: The job role or title of each recipient
    - `PersonName`: The name of each recipient
    - `EmailsSent`: Whether an email was sent (yes/no or 1/0)
    - `EmailsOpened`: Whether the email was opened (yes/no or 1/0)
    - `EmailsClicked`: Whether the email was clicked (yes/no or 1/0)
    """)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("© 2025 Email Metrics Dashboard | Built with Streamlit", unsafe_allow_html=True)
