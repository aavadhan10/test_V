import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Set page configuration
st.set_page_config(
    page_title="Dynamic Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve aesthetics
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 16px;
    }
    /* Make charts stand out in expanders */
    .st-emotion-cache-1r4qj8v {
        border: 1px solid #ddd;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    /* Make expanders a bit more prominent */
    .st-emotion-cache-eqpbcq {
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define standardized color scheme to use across all visualizations
def get_color_scheme():
    """Return standardized color schemes for consistency"""
    return {
        'primary': '#4e8df5',       # Main color for primary metrics (blue)
        'secondary': '#4CAF50',     # Secondary color (green)
        'accent': '#FF9800',        # Accent color (orange)
        'neutral': '#607D8B',       # Neutral color (blue-grey)
        'sequence': px.colors.sequential.Blues,  # Sequential color scale
        'categorical': px.colors.qualitative.Safe,  # Categorical colors
        'diverging': px.colors.diverging.RdBu,  # Diverging color scale
    }

def infer_data_types(df):
    """
    Infer the data types of each column in a DataFrame and categorize them.
    
    Returns:
    - dict: Dictionary with keys for different data types and values as lists of column names
    """
    data_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'boolean': [],
        'id': [],
        'potential_date_strings': []
    }
    
    # First pass to identify potential ID columns and date strings
    for col in df.columns:
        # Check for ID-like column names
        if col.lower().endswith('id') or col.lower() == 'id' or 'identifier' in col.lower():
            data_types['id'].append(col)
            continue
        
        # Check if string column might contain dates
        if df[col].dtype == 'object':
            # Sample some values to check for date patterns
            sample = df[col].dropna().head(10).astype(str)
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # yyyy-mm-dd
                r'\d{2}/\d{2}/\d{4}',   # mm/dd/yyyy or dd/mm/yyyy
                r'\d{2}-\d{2}-\d{4}',   # mm-dd-yyyy or dd-mm-yyyy
                r'\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',  # dd Mon yyyy
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2},?\s+\d{4}'  # Mon dd, yyyy
            ]
            
            if any(sample.str.contains(pattern, regex=True).any() for pattern in date_patterns):
                data_types['potential_date_strings'].append(col)
                continue
    
    # Second pass for other types
    for col in df.columns:
        # Skip already categorized columns
        if any(col in data_types[key] for key in ['id', 'potential_date_strings']):
            continue
            
        # Check data type
        if pd.api.types.is_numeric_dtype(df[col]):
            data_types['numeric'].append(col)
        elif pd.api.types.is_datetime64_dtype(df[col]):
            data_types['datetime'].append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            data_types['boolean'].append(col)
        elif df[col].dtype == 'object':
            # Check if categorical (limited unique values) or text
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.2 and df[col].nunique() < 50:  # Heuristic for categorical data
                data_types['categorical'].append(col)
            else:
                data_types['text'].append(col)
    
    return data_types

def convert_date_columns(df, date_columns):
    """Convert string columns that likely contain dates to datetime"""
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # If successful, extract date components
            if not df[col].isna().all():
                df[f'{col}_Year'] = df[col].dt.year
                df[f'{col}_Month'] = df[col].dt.month
                df[f'{col}_MonthName'] = df[col].dt.strftime('%b')
                df[f'{col}_Day'] = df[col].dt.day
                df[f'{col}_MonthYear'] = df[col].dt.strftime('%b %Y')
        except:
            pass  # If conversion fails, leave as is
    return df

def create_scrollable_bar_chart(df, x, y, title, x_label=None, y_label=None, color=None, orientation='v', height=500, show_top_n=10):
    """
    Create a bar chart that can be scrolled to show all data while defaulting to show top N items
    """
    colors = get_color_scheme()
    
    # If color is specified as a key in our scheme, use it
    if color and color in colors:
        chart_color = colors[color]
    else:
        # Default to primary color
        chart_color = colors['primary']
    
    # Sort data appropriately based on orientation
    if orientation == 'v':
        # For vertical bars, sort by y value
        df_sorted = df.sort_values(y, ascending=False)
        x_title = x_label if x_label else x
        y_title = y_label if y_label else y
    else:
        # For horizontal bars, sort by x value
        df_sorted = df.sort_values(x, ascending=False)
        # Swap x and y labels for horizontal orientation
        x_title = y_label if y_label else y
        y_title = x_label if x_label else x
    
    # Create a copy for display to avoid modifying original
    display_df = df_sorted.copy()
    
    # Limit to top N for initial view if specified
    if show_top_n and len(display_df) > show_top_n:
        display_df = display_df.head(show_top_n)
    
    # Create figure
    if orientation == 'v':
        fig = px.bar(
            display_df, 
            x=x, 
            y=y,
            title=title,
            labels={x: x_title, y: y_title},
            color_discrete_sequence=[chart_color] if isinstance(chart_color, str) else chart_color
        )
    else:
        fig = px.bar(
            display_df, 
            y=x, 
            x=y,
            title=title,
            labels={y: x_title, x: y_title},
            color_discrete_sequence=[chart_color] if isinstance(chart_color, str) else chart_color,
            orientation='h'
        )
    
    # Add layout settings for scrolling
    fig.update_layout(
        height=height,
        xaxis_tickangle=-45 if orientation == 'v' else 0,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        autosize=True
    )
    
    # Create expander with information about scrolling
    with st.expander(f"{title} - Showing top {show_top_n if show_top_n else 'all'} of {len(df)} items", expanded=True):
        # Add note about scrolling if there's more data
        if show_top_n and len(df) > show_top_n:
            st.info(f"Showing top {show_top_n} items. Use the interactive chart controls to explore all {len(df)} items.")
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Option to show full data
        if st.checkbox(f"Show all {len(df)} items in table format"):
            st.dataframe(df_sorted, use_container_width=True, height=min(400, len(df) * 35))

def format_number(num, prefix=""):
    """Format numbers with comma separators and optional prefix"""
    if isinstance(num, (int, float)):
        return f"{prefix}{num:,.0f}"
    return "N/A"

def format_currency(num):
    """Format numbers as currency"""
    if isinstance(num, (int, float)):
        return f"${num:,.2f}"
    return "N/A"

def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file"""
    try:
        # Try to read with pandas
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        return None

def create_overview_section(df, data_types):
    """Create an overview section with key metrics and basic visualizations"""
    colors = get_color_scheme()
    
    st.header("Data Overview")
    
    # Basic data information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", format_number(len(df)))
    with col2:
        st.metric("Total Columns", format_number(len(df.columns)))
    with col3:
        st.metric("Numeric Columns", format_number(len(data_types['numeric'])))
    with col4:
        st.metric("Categorical Columns", format_number(len(data_types['categorical'])))
    
    # Sample data
    with st.expander("Sample Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Data summary
    with st.expander("Data Summary", expanded=True):
        # Create a summary of numeric columns
        if data_types['numeric']:
            st.subheader("Numeric Data Summary")
            numeric_summary = df[data_types['numeric']].describe().T
            # Add additional metrics
            if not numeric_summary.empty:
                numeric_summary['missing'] = df[data_types['numeric']].isna().sum()
                numeric_summary['missing_pct'] = (df[data_types['numeric']].isna().sum() / len(df)) * 100
                st.dataframe(numeric_summary, use_container_width=True)
        
        # Create a summary of categorical columns
        if data_types['categorical']:
            st.subheader("Categorical Data Summary")
            cat_summary = pd.DataFrame({
                'unique_values': df[data_types['categorical']].nunique(),
                'missing': df[data_types['categorical']].isna().sum(),
                'missing_pct': (df[data_types['categorical']].isna().sum() / len(df)) * 100
            })
            st.dataframe(cat_summary, use_container_width=True)
    
    # Visualizations for overview section
    st.subheader("Quick Visualizations")
    
    # If we have numeric columns, select top 2 for visualizations
    if len(data_types['numeric']) >= 2:
        col1, col2 = st.columns(2)
        
        # Pick the first numeric column for histogram
        with col1:
            with st.expander(f"Distribution of {data_types['numeric'][0]}", expanded=True):
                fig = px.histogram(df, x=data_types['numeric'][0], 
                                  title=f"Distribution of {data_types['numeric'][0]}",
                                  color_discrete_sequence=[colors['primary']])
                st.plotly_chart(fig, use_container_width=True)
        
        # Pick the second numeric column for box plot
        with col2:
            with st.expander(f"Box Plot of {data_types['numeric'][1]}", expanded=True):
                fig = px.box(df, y=data_types['numeric'][1], 
                            title=f"Box Plot of {data_types['numeric'][1]}",
                            color_discrete_sequence=[colors['secondary']])
                st.plotly_chart(fig, use_container_width=True)
    
    # If we have one numeric and one categorical column, create a bar chart
    if data_types['numeric'] and data_types['categorical']:
        # Create a bar chart of the first categorical vs the first numeric
        cat_col = data_types['categorical'][0]
        num_col = data_types['numeric'][0]
        
        # Count values in the categorical column
        value_counts = df[cat_col].value_counts().reset_index()
        value_counts.columns = [cat_col, 'Count']
        
        # Sort and take top 15 for better visualization
        value_counts = value_counts.sort_values('Count', ascending=False).head(15)
        
        with st.expander(f"Top Categories by Count - {cat_col}", expanded=True):
            fig = px.bar(value_counts, x=cat_col, y='Count',
                        title=f"Top Categories by Count - {cat_col}",
                        color_discrete_sequence=[colors['accent']])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Create a summary by category
        if len(df) > 1:  # Ensure enough data
            cat_summary = df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count']).reset_index()
            cat_summary = cat_summary.sort_values('sum', ascending=False).head(15)
            
            with st.expander(f"{num_col} by {cat_col}", expanded=True):
                fig = px.bar(cat_summary, x=cat_col, y='sum',
                            title=f"Sum of {num_col} by {cat_col}",
                            color_discrete_sequence=[colors['primary']])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

def create_numeric_analysis(df, data_types):
    """Create analysis section for numeric data"""
    colors = get_color_scheme()
    
    st.header("Numeric Data Analysis")
    
    if not data_types['numeric']:
        st.info("No numeric columns detected in the dataset.")
        return
    
    # Select columns for analysis
    num_cols = st.multiselect("Select numeric columns for analysis", 
                             data_types['numeric'],
                             default=data_types['numeric'][:min(2, len(data_types['numeric']))])
    
    if not num_cols:
        st.warning("Please select at least one numeric column for analysis.")
        return
    
    # Create visualizations for each selected column
    for i, col in enumerate(num_cols):
        st.subheader(f"Analysis of {col}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander(f"Distribution of {col}", expanded=True):
                # Histogram
                fig = px.histogram(df, x=col, 
                                  title=f"Distribution of {col}",
                                  color_discrete_sequence=[colors['primary']])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.expander(f"Box Plot of {col}", expanded=True):
                # Box Plot
                fig = px.box(df, y=col, 
                            title=f"Box Plot of {col}",
                            color_discrete_sequence=[colors['secondary']])
                st.plotly_chart(fig, use_container_width=True)
        
        # If we have categorical columns, create breakdown
        if data_types['categorical']:
            # Select a categorical column for breakdown
            cat_col = st.selectbox(f"Select a category to break down {col} by:", 
                                 data_types['categorical'],
                                 key=f"cat_select_{i}")
            
            if cat_col:
                # Compute breakdown statistics
                breakdown = df.groupby(cat_col)[col].agg(['mean', 'median', 'sum', 'count']).reset_index()
                breakdown = breakdown.sort_values('sum', ascending=False).head(15)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander(f"Sum of {col} by {cat_col}", expanded=True):
                        fig = px.bar(breakdown, x=cat_col, y='sum',
                                    title=f"Sum of {col} by {cat_col}",
                                    color_discrete_sequence=[colors['primary']])
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    with st.expander(f"Average of {col} by {cat_col}", expanded=True):
                        fig = px.bar(breakdown, x=cat_col, y='mean',
                                    title=f"Average of {col} by {cat_col}",
                                    color_discrete_sequence=[colors['accent']])
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
        
        # If we have multiple numeric columns, show correlation
        if len(num_cols) > 1 and len(num_cols) < 20:  # Limit to avoid overloading
            with st.expander("Correlation Matrix", expanded=True):
                # Calculate correlation
                corr = df[num_cols].corr()
                
                # Create heatmap
                fig = px.imshow(corr,
                               title="Correlation Matrix",
                               color_continuous_scale=colors['diverging'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Correlation values:")
                st.dataframe(corr, use_container_width=True)

def create_categorical_analysis(df, data_types):
    """Create analysis section for categorical data"""
    colors = get_color_scheme()
    
    st.header("Categorical Data Analysis")
    
    if not data_types['categorical']:
        st.info("No categorical columns detected in the dataset.")
        return
    
    # Select columns for analysis
    cat_cols = st.multiselect("Select categorical columns for analysis", 
                             data_types['categorical'],
                             default=data_types['categorical'][:min(2, len(data_types['categorical']))])
    
    if not cat_cols:
        st.warning("Please select at least one categorical column for analysis.")
        return
    
    # Create visualizations for each selected column
    for i, col in enumerate(cat_cols):
        st.subheader(f"Analysis of {col}")
        
        # Get value counts
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        
        # Sort and limit for better visualization
        value_counts = value_counts.sort_values('Count', ascending=False)
        
        # Check if too many categories for a regular bar chart
        if len(value_counts) > 20:
            # Use scrollable chart for many categories
            create_scrollable_bar_chart(
                value_counts,
                col,
                'Count',
                f"Distribution of {col}",
                col,
                'Count',
                color='primary',
                height=500,
                show_top_n=15
            )
        else:
            # Regular bar chart for fewer categories
            with st.expander(f"Distribution of {col}", expanded=True):
                fig = px.bar(value_counts, x=col, y='Count',
                            title=f"Distribution of {col}",
                            color_discrete_sequence=[colors['primary']])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Show pie chart for categorical distribution
        with st.expander(f"Percentage Distribution of {col}", expanded=True):
            # Limit categories for pie chart to top 10
            pie_data = value_counts.head(10)
            
            if len(value_counts) > 10:
                # Add an "Other" category if there are more than 10
                other_sum = value_counts.iloc[10:]['Count'].sum()
                other_row = pd.DataFrame({col: ['Other'], 'Count': [other_sum]})
                pie_data = pd.concat([pie_data, other_row])
            
            fig = px.pie(pie_data, names=col, values='Count',
                        title=f"Percentage Distribution of {col}",
                        color_discrete_sequence=colors['categorical'])
            st.plotly_chart(fig, use_container_width=True)
        
        # If we have multiple categorical columns, show cross-tabulation
        if len(cat_cols) > 1:
            cross_col = st.selectbox(f"Select another category to compare with {col}:", 
                                   [c for c in cat_cols if c != col],
                                   key=f"cross_{i}")
            
            if cross_col:
                # Create cross-tabulation
                cross_tab = pd.crosstab(df[col], df[cross_col])
                
                with st.expander(f"Relationship between {col} and {cross_col}", expanded=True):
                    fig = px.imshow(cross_tab,
                                   title=f"Relationship between {col} and {cross_col}",
                                   color_continuous_scale=colors['sequence'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("Cross-tabulation values:")
                    st.dataframe(cross_tab, use_container_width=True)

def create_time_series_analysis(df, data_types):
    """Create analysis section for time series data"""
    colors = get_color_scheme()
    
    st.header("Time Series Analysis")
    
    # Check if we have datetime columns or date-like string columns
    datetime_cols = data_types['datetime'] + [col for col in data_types['potential_date_strings'] 
                                             if f"{col}_Year" in df.columns]
    
    if not datetime_cols:
        st.info("No datetime columns detected in the dataset. Please upload data with date/time information for time series analysis.")
        return
    
    # Select date column
    date_col = st.selectbox("Select date/time column for analysis", datetime_cols)
    
    if not date_col:
        st.warning("Please select a date/time column for analysis.")
        return
    
    # Get the actual datetime column (could be original or transformed)
    if date_col in data_types['potential_date_strings']:
        # Use the column itself if it was successfully converted to datetime
        if pd.api.types.is_datetime64_dtype(df[date_col]):
            datetime_column = date_col
        else:
            st.warning(f"Column {date_col} contains date-like strings but couldn't be converted to datetime format.")
            return
    else:
        datetime_column = date_col
    
    # Select a numeric column to analyze over time
    if not data_types['numeric']:
        st.info("No numeric columns available for time series analysis.")
        return
    
    value_col = st.selectbox("Select numeric column to analyze over time", data_types['numeric'])
    
    if not value_col:
        st.warning("Please select a numeric column to analyze over time.")
        return
    
    # Create time grouping options
    groupby_options = ["Day", "Week", "Month", "Quarter", "Year"]
    time_groupby = st.selectbox("Group by time period", groupby_options, index=2)  # Default to Month
    
    # Set up time grouping
    if time_groupby == "Day":
        df['time_group'] = df[datetime_column].dt.date
    elif time_groupby == "Week":
        df['time_group'] = df[datetime_column].dt.to_period('W').dt.start_time
    elif time_groupby == "Month":
        df['time_group'] = df[datetime_column].dt.to_period('M').dt.start_time
    elif time_groupby == "Quarter":
        df['time_group'] = df[datetime_column].dt.to_period('Q').dt.start_time
    else:  # Year
        df['time_group'] = df[datetime_column].dt.year
    
    # Group data by time period
    time_series_data = df.groupby('time_group')[value_col].agg(['sum', 'mean', 'count']).reset_index()
    time_series_data.columns = ['Period', 'Sum', 'Average', 'Count']
    
    # Ensure the Period column is sorted chronologically
    time_series_data = time_series_data.sort_values('Period')
    
    # Create time series visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander(f"Sum of {value_col} Over Time", expanded=True):
            fig = px.line(time_series_data, x='Period', y='Sum',
                         title=f"Sum of {value_col} by {time_groupby}",
                         markers=True,
                         color_discrete_sequence=[colors['primary']])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.expander(f"Average {value_col} Over Time", expanded=True):
            fig = px.line(time_series_data, x='Period', y='Average',
                         title=f"Average {value_col} by {time_groupby}",
                         markers=True,
                         color_discrete_sequence=[colors['secondary']])
            st.plotly_chart(fig, use_container_width=True)
    
    # Optional: Add categorical breakdown over time
    if data_types['categorical']:
        st.subheader("Time Series Breakdown by Category")
        
        cat_col = st.selectbox("Select a category for time series breakdown", data_types['categorical'])
        
        if cat_col:
            # Group by time period and category
            cat_time_data = df.groupby(['time_group', cat_col])[value_col].sum().reset_index()
            cat_time_data.columns = ['Period', 'Category', 'Value']
            
            # Sort by time
            cat_time_data = cat_time_data.sort_values('Period')
            
            # For more than 10 categories, limit to top ones by total value
            if df[cat_col].nunique() > 10:
                top_categories = df.groupby(cat_col)[value_col].sum().nlargest(8).index.tolist()
                cat_time_data = cat_time_data[cat_time_data['Category'].isin(top_categories)]
            
            with st.expander(f"{value_col} Over Time by {cat_col}", expanded=True):
                fig = px.line(cat_time_data, x='Period', y='Value', color='Category',
                             title=f"{value_col} Over Time by {cat_col}",
                             labels={'Value': value_col, 'Period': time_groupby},
                             color_discrete_sequence=colors['categorical'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Stacked area chart
            with st.expander(f"Stacked {value_col} Over Time by {cat_col}", expanded=True):
                fig = px.area(cat_time_data, x='Period', y='Value', color='Category',
                             title=f"Stacked {value_col} Over Time by {cat_col}",
                             labels={'Value': value_col, 'Period': time_groupby},
                             color_discrete_sequence=colors['categorical'])
                st.plotly_chart(fig, use_container_width=True)

def create_custom_visualizations(df, data_types):
    """Create custom visualization section where users can select columns and chart types"""
    colors = get_color_scheme()
    
    st.header("Custom Visualizations")
    
    # Chart type selection
    chart_types = [
        "Bar Chart", 
        "Line Chart", 
        "Scatter Plot", 
        "Pie Chart", 
        "Histogram", 
        "Box Plot",
        "Heat Map"
    ]
    
    chart_type = st.selectbox("Select chart type", chart_types)
    
    # Select columns based on chart type
    if chart_type == "Bar Chart":
        x_col = st.selectbox("Select category (X-axis)", data_types['categorical'])
        y_col = st.selectbox("Select value (Y-axis)", data_types['numeric'])
        color_col = st.selectbox("Select color category (optional)", ["None"] + data_types['categorical'])
        
        if not x_col or not y_col:
            st.warning("Please select both X and Y columns for the bar chart.")
            return
        
        # Create aggregation options
        agg_options = ["Sum", "Average", "Count", "Median", "Min", "Max"]
        agg_func = st.selectbox("Select aggregation function", agg_options)
        
        # Map to pandas aggregation function
        agg_map = {
            "Sum": "sum",
            "Average": "mean",
            "Count": "count",
            "Median": "median",
            "Min": "min",
            "Max": "max"
        }
        
        # Apply aggregation
        if color_col != "None":
            # Group by both x and color column
            result = df.groupby([x_col, color_col])[y_col].agg(agg_map[agg_func]).reset_index()
            
            # Create the bar chart
            fig = px.bar(result, x=x_col, y=y_col, color=color_col,
                         title=f"{agg_func} of {y_col} by {x_col}, colored by {color_col}",
                         labels={x_col: x_col, y_col: f"{agg_func} of {y_col}"},
                         color_discrete_sequence=colors['categorical'])
        else:
            # Group by just x column
            result = df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            # Create the bar chart
            fig = px.bar(result, x=x_col, y=y_col,
                         title=f"{agg_func} of {y_col} by {x_col}",
                         labels={x_col: x_col, y_col: f"{agg_func} of {y_col}"},
                         color_discrete_sequence=[colors['primary']])
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Line Chart":
        # Determine if we have datetime columns
        has_datetime = bool(data_types['datetime']) or any(f"{col}_Year" in df.columns for col in data_types['potential_date_strings'])
        
        if has_datetime:
            # Get available date columns (original or derived)
            date_options = data_types['datetime'] + [col for col in data_types['potential_date_strings'] 
                                                 if f"{col}_Year" in df.columns]
            
            x_col = st.selectbox("Select date/time column (X-axis)", date_options)
            y_col = st.selectbox("Select value (Y-axis)", data_types['numeric'])
            color_col = st.selectbox("Select line category (optional)", ["None"] + data_types['categorical'])
            
            if not x_col or not y_col:
                st.warning("Please select both X and Y columns for the line chart.")
                return
            
            # Choose time grouping
            groupby_options = ["Day", "Week", "Month", "Quarter", "Year"]
            time_groupby = st.selectbox("Group by time period", groupby_options, index=2)  # Default to Month
            
            # Set up time grouping
            if time_groupby == "Day":
                df['time_group'] = df[x_col].dt.date
            elif time_groupby == "Week":
                df['time_group'] = df[x_col].dt.to_period('W').dt.start_time
            elif time_groupby == "Month":
                df['time_group'] = df[x_col].dt.to_period('M').dt.start_time
            elif time_groupby == "Quarter":
                df['time_group'] = df[x_col].dt.to_period('Q').dt.start_time
            else:  # Year
                df['time_group'] = df[x_col].dt.year
            
            # Create aggregation options
            agg_options = ["Sum", "Average", "Count", "Median", "Min", "Max"]
            agg_func = st.selectbox("Select aggregation function", agg_options)
            
            # Map to pandas aggregation function
            agg_map = {
                "Sum": "sum",
                "Average": "mean",
                "Count": "count",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Apply aggregation
            if color_col != "None":
                # Group by time period and color column
                result = df.groupby(['time_group', color_col])[y_col].agg(agg_map[agg_func]).reset_index()
                
                # Create the line chart with color
                fig = px.line(result, x='time_group', y=y_col, color=color_col,
                             title=f"{agg_func} of {y_col} Over Time by {color_col}",
                             labels={'time_group': time_groupby, y_col: f"{agg_func} of {y_col}"},
                             markers=True,
                             color_discrete_sequence=colors['categorical'])
            else:
                # Group by just time period
                result = df.groupby('time_group')[y_col].agg(agg_map[agg_func]).reset_index()
                
                # Create the line chart
                fig = px.line(result, x='time_group', y=y_col,
                             title=f"{agg_func} of {y_col} Over Time",
                             labels={'time_group': time_groupby, y_col: f"{agg_func} of {y_col}"},
                             markers=True,
                             color_discrete_sequence=[colors['primary']])
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # No datetime columns - create line chart with numeric x-axis
            x_col = st.selectbox("Select numeric column (X-axis)", data_types['numeric'])
            y_col = st.selectbox("Select value (Y-axis)", [col for col in data_types['numeric'] if col != x_col])
            color_col = st.selectbox("Select line category (optional)", ["None"] + data_types['categorical'])
            
            if not x_col or not y_col:
                st.warning("Please select both X and Y columns for the line chart.")
                return
            
            # Create the chart
            if color_col != "None":
                fig = px.line(df, x=x_col, y=y_col, color=color_col,
                             title=f"{y_col} vs {x_col} by {color_col}",
                             labels={x_col: x_col, y_col: y_col},
                             markers=True,
                             color_discrete_sequence=colors['categorical'])
            else:
                fig = px.line(df, x=x_col, y=y_col,
                             title=f"{y_col} vs {x_col}",
                             labels={x_col: x_col, y_col: y_col},
                             markers=True,
                             color_discrete_sequence=[colors['primary']])
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis", data_types['numeric'])
        y_col = st.selectbox("Select Y-axis", [col for col in data_types['numeric'] if col != x_col])
        color_col = st.selectbox("Select color category (optional)", ["None"] + data_types['categorical'])
        size_col = st.selectbox("Select size column (optional)", ["None"] + data_types['numeric'])
        
        if not x_col or not y_col:
            st.warning("Please select both X and Y columns for the scatter plot.")
            return
        
        # Create the scatter plot
        if color_col != "None" and size_col != "None":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                            title=f"{y_col} vs {x_col}, colored by {color_col}, sized by {size_col}",
                            labels={x_col: x_col, y_col: y_col, size_col: size_col},
                            color_discrete_sequence=colors['categorical'])
        elif color_col != "None":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                            title=f"{y_col} vs {x_col}, colored by {color_col}",
                            labels={x_col: x_col, y_col: y_col},
                            color_discrete_sequence=colors['categorical'])
        elif size_col != "None":
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col,
                            title=f"{y_col} vs {x_col}, sized by {size_col}",
                            labels={x_col: x_col, y_col: y_col, size_col: size_col},
                            color_discrete_sequence=[colors['primary']])
        else:
            fig = px.scatter(df, x=x_col, y=y_col,
                            title=f"{y_col} vs {x_col}",
                            labels={x_col: x_col, y_col: y_col},
                            color_discrete_sequence=[colors['primary']])
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Pie Chart":
        cat_col = st.selectbox("Select category column", data_types['categorical'])
        value_col = st.selectbox("Select value column", data_types['numeric'])
        
        if not cat_col or not value_col:
            st.warning("Please select both category and value columns for the pie chart.")
            return
        
        # Aggregate data
        pie_data = df.groupby(cat_col)[value_col].sum().reset_index()
        
        # Limit to top categories if there are many
        if df[cat_col].nunique() > 10:
            pie_data = pie_data.nlargest(10, value_col)
            st.info("Showing only the top 10 categories due to the large number of unique values.")
        
        # Create the pie chart
        fig = px.pie(pie_data, names=cat_col, values=value_col,
                    title=f"Distribution of {value_col} by {cat_col}",
                    color_discrete_sequence=colors['categorical'])
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        num_col = st.selectbox("Select numeric column", data_types['numeric'])
        
        if not num_col:
            st.warning("Please select a numeric column for the histogram.")
            return
        
        # Optional color by category
        color_col = st.selectbox("Color by category (optional)", ["None"] + data_types['categorical'])
        
        # Histogram parameters
        n_bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
        
        # Create the histogram
        if color_col != "None":
            fig = px.histogram(df, x=num_col, color=color_col, nbins=n_bins,
                              title=f"Distribution of {num_col} by {color_col}",
                              color_discrete_sequence=colors['categorical'])
        else:
            fig = px.histogram(df, x=num_col, nbins=n_bins,
                              title=f"Distribution of {num_col}",
                              color_discrete_sequence=[colors['primary']])
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Box Plot":
        num_col = st.selectbox("Select numeric column for values", data_types['numeric'])
        
        if not num_col:
            st.warning("Please select a numeric column for the box plot.")
            return
        
        # Optional grouping by category
        group_col = st.selectbox("Group by category (optional)", ["None"] + data_types['categorical'])
        
        # Create the box plot
        if group_col != "None":
            fig = px.box(df, x=group_col, y=num_col,
                        title=f"Box Plot of {num_col} by {group_col}",
                        color=group_col,
                        color_discrete_sequence=colors['categorical'])
        else:
            fig = px.box(df, y=num_col,
                        title=f"Box Plot of {num_col}",
                        color_discrete_sequence=[colors['primary']])
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Heat Map":
        # For heatmap, we need two categorical columns and one numeric
        row_col = st.selectbox("Select row category", data_types['categorical'])
        col_col = st.selectbox("Select column category", 
                               [col for col in data_types['categorical'] if col != row_col])
        value_col = st.selectbox("Select value", data_types['numeric'])
        
        if not row_col or not col_col or not value_col:
            st.warning("Please select row, column, and value columns for the heat map.")
            return
        
        # Create aggregation options
        agg_options = ["Sum", "Average", "Count", "Median", "Min", "Max"]
        agg_func = st.selectbox("Select aggregation function", agg_options)
        
        # Map to pandas aggregation function
        agg_map = {
            "Sum": "sum",
            "Average": "mean",
            "Count": "count",
            "Median": "median",
            "Min": "min",
            "Max": "max"
        }
        
        # Create pivot table
        pivot_data = df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc=agg_map[agg_func],
            fill_value=0
        )
        
        # Limit categories if there are too many
        if pivot_data.shape[0] > 20 or pivot_data.shape[1] > 20:
            st.warning("Too many categories for a readable heatmap. Showing top categories by total value.")
            # Get top categories for rows and columns
            row_totals = pivot_data.sum(axis=1).nlargest(15)
            col_totals = pivot_data.sum(axis=0).nlargest(15)
            
            # Filter pivot table
            pivot_data = pivot_data.loc[row_totals.index, col_totals.index]
        
        # Create the heatmap
        fig = px.imshow(pivot_data,
                        title=f"Heatmap of {agg_func} of {value_col} by {row_col} and {col_col}",
                        labels=dict(x=col_col, y=row_col, color=f"{agg_func} of {value_col}"),
                        color_continuous_scale=colors['sequence'])
        
        st.plotly_chart(fig, use_container_width=True)

def create_prompt_based_visualizations(df, data_types):
    """Create visualizations based on natural language prompts"""
    st.header("Prompt-Based Visualizations")
    
    st.write("""
    Describe the visualization you'd like to create in plain language. For example:
    - "Show me a bar chart of total sales by product category"
    - "Create a line chart of revenue over time"
    - "I want to see a pie chart of customers by region"
    """)
    
    # Get user prompt
    prompt = st.text_area("Enter your visualization request:", 
                         height=100,
                         placeholder="E.g., Show me the top 10 categories by total value...")
    
    if not prompt:
        st.info("Enter a prompt to generate a visualization.")
        return
    
    # Process the prompt when user clicks the button
    if st.button("Generate Visualization"):
        # Simple keyword-based visualization generator
        prompt_lower = prompt.lower()
        
        # Find column references
        potential_columns = []
        for col in df.columns:
            if col.lower() in prompt_lower:
                potential_columns.append(col)
        
        # Detect visualization type
        viz_type = None
        if any(term in prompt_lower for term in ["bar", "column"]):
            viz_type = "bar"
        elif any(term in prompt_lower for term in ["line", "trend", "over time", "time series"]):
            viz_type = "line"
        elif any(term in prompt_lower for term in ["scatter", "correlation", "relationship"]):
            viz_type = "scatter"
        elif any(term in prompt_lower for term in ["pie", "distribution", "proportion", "percentage"]):
            viz_type = "pie"
        elif any(term in prompt_lower for term in ["histogram", "distribution"]):
            viz_type = "histogram"
        elif any(term in prompt_lower for term in ["box", "boxplot", "quartile"]):
            viz_type = "box"
        elif any(term in prompt_lower for term in ["heat", "heatmap", "matrix", "table"]):
            viz_type = "heatmap"
        else:
            # Default to bar if we can't determine
            viz_type = "bar"
        
        # Try to identify column types from prompt and data types
        categorical_cols = []
        numeric_cols = []
        date_cols = []
        
        # Extract potential categorical columns
        for term in ["by", "group", "category", "categories", "segment"]:
            if term in prompt_lower:
                # Find closest column name after this term
                term_pos = prompt_lower.find(term)
                closest_col = None
                min_distance = float('inf')
                
                for col in data_types['categorical']:
                    col_pos = prompt_lower.find(col.lower())
                    if col_pos > term_pos and col_pos - term_pos < min_distance:
                        closest_col = col
                        min_distance = col_pos - term_pos
                
                if closest_col and closest_col not in categorical_cols and min_distance < 50:
                    categorical_cols.append(closest_col)
        
        # Extract potential numeric columns
        for term in ["sum", "total", "average", "count", "value", "amount", "quantity", "measure", "metric"]:
            if term in prompt_lower:
                # Find closest column name after this term
                term_pos = prompt_lower.find(term)
                closest_col = None
                min_distance = float('inf')
                
                for col in data_types['numeric']:
                    col_pos = prompt_lower.find(col.lower())
                    if col_pos > term_pos and col_pos - term_pos < min_distance:
                        closest_col = col
                        min_distance = col_pos - term_pos
                
                if closest_col and closest_col not in numeric_cols and min_distance < 50:
                    numeric_cols.append(closest_col)
        
        # Extract potential date columns for time series
        for term in ["time", "date", "year", "month", "day", "period", "trend"]:
            if term in prompt_lower:
                # Find closest date column
                date_options = data_types['datetime'] + [col for col in data_types['potential_date_strings'] 
                                                     if f"{col}_Year" in df.columns]
                if date_options:
                    date_cols.append(date_options[0])  # Use first date column
        
        # Fallback to get some columns if we couldn't identify them from the prompt
        if not categorical_cols and data_types['categorical']:
            categorical_cols.append(data_types['categorical'][0])
        
        if not numeric_cols and data_types['numeric']:
            numeric_cols.append(data_types['numeric'][0])
        
        # Create appropriate visualization based on prompt analysis
        st.subheader("Generated Visualization")
        
        try:
            colors = get_color_scheme()
            
            if viz_type == "bar":
                if categorical_cols and numeric_cols:
                    # Create bar chart
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    
                    # Check for aggregation in prompt
                    agg_func = "sum"
                    if "average" in prompt_lower or "mean" in prompt_lower:
                        agg_func = "mean"
                    elif "count" in prompt_lower:
                        agg_func = "count"
                    
                    # Check for top N mention
                    top_n = None
                    for i in range(1, 101):
                        if f"top {i}" in prompt_lower or f"{i} top" in prompt_lower:
                            top_n = i
                            break
                    
                    # Aggregate data
                    chart_data = df.groupby(cat_col)[num_col].agg(agg_func).reset_index()
                    chart_data = chart_data.sort_values(num_col, ascending=False)
                    
                    # Limit to top N if specified
                    if top_n and len(chart_data) > top_n:
                        chart_data = chart_data.head(top_n)
                        title = f"Top {top_n} {cat_col} by {agg_func.capitalize()} of {num_col}"
                    else:
                        title = f"{agg_func.capitalize()} of {num_col} by {cat_col}"
                    
                    fig = px.bar(
                        chart_data, 
                        x=cat_col, 
                        y=num_col,
                        title=title,
                        labels={cat_col: cat_col, num_col: f"{agg_func.capitalize()} of {num_col}"},
                        color_discrete_sequence=[colors['primary']]
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not identify appropriate categorical and numeric columns for a bar chart.")
            
            elif viz_type == "line":
                # Check for time series first
                if date_cols and numeric_cols:
                    # Create line chart with date
                    date_col = date_cols[0]
                    num_col = numeric_cols[0]
                    
                    # Choose time grouping based on prompt
                    time_groupby = "Month"  # Default
                    if "day" in prompt_lower:
                        time_groupby = "Day"
                    elif "week" in prompt_lower:
                        time_groupby = "Week"
                    elif "quarter" in prompt_lower:
                        time_groupby = "Quarter"
                    elif "year" in prompt_lower:
                        time_groupby = "Year"
                    
                    # Set up time grouping
                    if time_groupby == "Day":
                        df['time_group'] = df[date_col].dt.date
                    elif time_groupby == "Week":
                        df['time_group'] = df[date_col].dt.to_period('W').dt.start_time
                    elif time_groupby == "Month":
                        df['time_group'] = df[date_col].dt.to_period('M').dt.start_time
                    elif time_groupby == "Quarter":
                        df['time_group'] = df[date_col].dt.to_period('Q').dt.start_time
                    else:  # Year
                        df['time_group'] = df[date_col].dt.year
                    
                    # Check for aggregation in prompt
                    agg_func = "sum"
                    if "average" in prompt_lower or "mean" in prompt_lower:
                        agg_func = "mean"
                    elif "count" in prompt_lower:
                        agg_func = "count"
                    
                    # Check if categorical breakdown is requested
                    if categorical_cols and any(term in prompt_lower for term in ["by", "group", "segment", "category", "breakdown"]):
                        cat_col = categorical_cols[0]
                        
                        # Group by time and category
                        result = df.groupby(['time_group', cat_col])[num_col].agg(agg_func).reset_index()
                        
                        # For many categories, limit to top ones
                        if df[cat_col].nunique() > 10:
                            top_categories = df.groupby(cat_col)[num_col].sum().nlargest(8).index.tolist()
                            result = result[result[cat_col].isin(top_categories)]
                        
                        title = f"{agg_func.capitalize()} of {num_col} Over Time by {cat_col}"
                        fig = px.line(
                            result, 
                            x='time_group', 
                            y=num_col, 
                            color=cat_col,
                            title=title,
                            labels={'time_group': time_groupby, num_col: f"{agg_func.capitalize()} of {num_col}"},
                            markers=True,
                            color_discrete_sequence=colors['categorical']
                        )
                    else:
                        # Just time series without categories
                        result = df.groupby('time_group')[num_col].agg(agg_func).reset_index()
                        
                        title = f"{agg_func.capitalize()} of {num_col} Over Time"
                        fig = px.line(
                            result, 
                            x='time_group', 
                            y=num_col,
                            title=title,
                            labels={'time_group': time_groupby, num_col: f"{agg_func.capitalize()} of {num_col}"},
                            markers=True,
                            color_discrete_sequence=[colors['primary']]
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                elif len(numeric_cols) >= 2:
                    # Create line chart with numeric x-axis
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                    
                    title = f"{y_col} vs {x_col}"
                    fig = px.line(
                        df, 
                        x=x_col, 
                        y=y_col,
                        title=title,
                        labels={x_col: x_col, y_col: y_col},
                        markers=True,
                        color_discrete_sequence=[colors['primary']]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not identify appropriate columns for a line chart.")
            
            elif viz_type == "scatter":
                if len(numeric_cols) >= 2:
                    # Create scatter plot
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
                    
                    # Check if color or size is mentioned
                    use_color = any(term in prompt_lower for term in ["color", "coloured", "colored by"])
                    use_size = any(term in prompt_lower for term in ["size", "sized by"])
                    
                    if use_color and categorical_cols:
                        color_col = categorical_cols[0]
                        
                        if use_size and len(numeric_cols) > 2:
                            size_col = numeric_cols[2]
                            title = f"{y_col} vs {x_col} colored by {color_col} and sized by {size_col}"
                            fig = px.scatter(
                                df, 
                                x=x_col, 
                                y=y_col, 
                                color=color_col, 
                                size=size_col,
                                title=title,
                                labels={x_col: x_col, y_col: y_col},
                                color_discrete_sequence=colors['categorical']
                            )
                        else:
                            title = f"{y_col} vs {x_col} colored by {color_col}"
                            fig = px.scatter(
                                df, 
                                x=x_col, 
                                y=y_col, 
                                color=color_col,
                                title=title,
                                labels={x_col: x_col, y_col: y_col},
                                color_discrete_sequence=colors['categorical']
                            )
                    else:
                        title = f"{y_col} vs {x_col}"
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            title=title,
                            labels={x_col: x_col, y_col: y_col},
                            color_discrete_sequence=[colors['primary']]
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least two numeric columns for a scatter plot.")
            
            elif viz_type == "pie":
                if categorical_cols and numeric_cols:
                    # Create pie chart
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    
                    # Aggregate data
                    pie_data = df.groupby(cat_col)[num_col].sum().reset_index()
                    
                    # Limit to top categories if there are many
                    if df[cat_col].nunique() > 10:
                        pie_data = pie_data.nlargest(10, num_col)
                        st.info("Showing only the top 10 categories due to the large number of unique values.")
                    
                    title = f"Distribution of {num_col} by {cat_col}"
                    fig = px.pie(
                        pie_data, 
                        names=cat_col, 
                        values=num_col,
                        title=title,
                        color_discrete_sequence=colors['categorical']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not identify appropriate categorical and numeric columns for a pie chart.")
            
            elif viz_type == "histogram":
                if numeric_cols:
                    # Create histogram
                    num_col = numeric_cols[0]
                    
                    # Check if we should color by category
                    if categorical_cols and any(term in prompt_lower for term in ["by", "group", "colou", "color"]):
                        cat_col = categorical_cols[0]
                        title = f"Distribution of {num_col} by {cat_col}"
                        fig = px.histogram(
                            df, 
                            x=num_col, 
                            color=cat_col,
                            title=title,
                            nbins=20,
                            color_discrete_sequence=colors['categorical']
                        )
                    else:
                        title = f"Distribution of {num_col}"
                        fig = px.histogram(
                            df, 
                            x=num_col,
                            title=title,
                            nbins=20,
                            color_discrete_sequence=[colors['primary']]
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not identify appropriate numeric column for a histogram.")
            
            elif viz_type == "box":
                if numeric_cols:
                    # Create box plot
                    num_col = numeric_cols[0]
                    
                    # Check if we should group by category
                    if categorical_cols and any(term in prompt_lower for term in ["by", "group", "compare"]):
                        cat_col = categorical_cols[0]
                        title = f"Box Plot of {num_col} by {cat_col}"
                        fig = px.box(
                            df, 
                            x=cat_col, 
                            y=num_col,
                            title=title,
                            color=cat_col,
                            color_discrete_sequence=colors['categorical']
                        )
                    else:
                        title = f"Box Plot of {num_col}"
                        fig = px.box(
                            df, 
                            y=num_col,
                            title=title,
                            color_discrete_sequence=[colors['primary']]
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not identify appropriate numeric column for a box plot.")
            
            elif viz_type == "heatmap":
                if len(categorical_cols) >= 2 and numeric_cols:
                    # Create heatmap
                    row_col = categorical_cols[0]
                    col_col = categorical_cols[1]
                    val_col = numeric_cols[0]
                    
                    # Check for aggregation in prompt
                    agg_func = "sum"
                    if "average" in prompt_lower or "mean" in prompt_lower:
                        agg_func = "mean"
                    elif "count" in prompt_lower:
                        agg_func = "count"
                    
                    # Create pivot table
                    pivot_data = df.pivot_table(
                        index=row_col,
                        columns=col_col,
                        values=val_col,
                        aggfunc=agg_func,
                        fill_value=0
                    )
                    
                    # Limit categories if there are too many
                    if pivot_data.shape[0] > 20 or pivot_data.shape[1] > 20:
                        # Get top categories for rows and columns
                        row_totals = pivot_data.sum(axis=1).nlargest(15)
                        col_totals = pivot_data.sum(axis=0).nlargest(15)
                        
                        # Filter pivot table
                        pivot_data = pivot_data.loc[row_totals.index, col_totals.index]
                    
                    title = f"Heatmap of {agg_func.capitalize()} of {val_col} by {row_col} and {col_col}"
                    fig = px.imshow(
                        pivot_data,
                        title=title,
                        labels=dict(x=col_col, y=row_col, color=f"{agg_func.capitalize()} of {val_col}"),
                        color_continuous_scale=colors['sequence']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least two categorical columns and one numeric column for a heatmap.")
            
            # Add the visualization to the dashboard if user wants
            st.markdown("---")
            
            # Save visualization to session state for later use in dashboard
            if 'saved_visualizations' not in st.session_state:
                st.session_state.saved_visualizations = []
            
            viz_title = st.text_input("Enter a title for this visualization (to save it to your dashboard):", 
                                    value=title if 'title' in locals() else "My Visualization")
            
            if st.button("Add to My Dashboard"):
                # Store the visualization info
                viz_info = {
                    'title': viz_title,
                    'prompt': prompt,
                    'type': viz_type,
                    'fig': fig if 'fig' in locals() else None
                }
                st.session_state.saved_visualizations.append(viz_info)
                st.success(f"Added '{viz_title}' to your dashboard!")
        
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("Try being more specific about the columns and type of visualization you want.")

def create_dashboard(saved_visualizations):
    """Display the user's personalized dashboard with saved visualizations"""
    st.header("My Custom Dashboard")
    
    if not saved_visualizations:
        st.info("Your dashboard is empty. Create and save visualizations to see them here.")
        return
    
    # Display all saved visualizations in a grid
    cols = st.columns(2)  # 2 columns for the dashboard
    
    for i, viz_info in enumerate(saved_visualizations):
        with cols[i % 2]:
            with st.expander(viz_info['title'], expanded=True):
                # Display visualization
                if viz_info['fig'] is not None:
                    st.plotly_chart(viz_info['fig'], use_container_width=True)
                else:
                    st.warning("Visualization data is not available.")
                
                # Show the prompt that created this visualization
                st.caption(f"Prompt: {viz_info['prompt']}")
                
                # Option to remove from dashboard
                if st.button(f"Remove from Dashboard", key=f"remove_{i}"):
                    saved_visualizations.pop(i)
                    st.rerun()

def main():
    # App title and description
    st.title("Dynamic CSV Dashboard Generator")
    st.markdown("""
    Upload any CSV file to automatically generate visualizations and create your own custom dashboard.
    You can use natural language prompts to describe the visualizations you want to see!
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        
        # Show sample data option
        if st.button("Use Sample Data"):
            # Create sample data
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Create sample dates
            start_date = datetime(2022, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(365)]
            
            # Create sample categories
            categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books', 'Food', 'Beauty']
            regions = ['North', 'South', 'East', 'West', 'Central']
            customer_types = ['Retail', 'Wholesale', 'Online']
            
            # Create sample data
            np.random.seed(42)  # For reproducibility
            sample_data = {
                'Date': np.random.choice(dates, 1000),
                'Category': np.random.choice(categories, 1000),
                'Region': np.random.choice(regions, 1000),
                'CustomerType': np.random.choice(customer_types, 1000),
                'Sales': np.random.randint(100, 10000, 1000),
                'Quantity': np.random.randint(1, 100, 1000),
                'Discount': np.random.choice([0, 5, 10, 15, 20], 1000),
                'Profit': np.random.randint(-1000, 5000, 1000)
            }
            
            # Calculate unit price
            sample_data['UnitPrice'] = [s/q for s, q in zip(sample_data['Sales'], sample_data['Quantity'])]
            
            # Create DataFrame
            sample_df = pd.DataFrame(sample_data)
            
            # Set to session state
            st.session_state.df = sample_df
            st.session_state.data_loaded = True
            
            # Infer data types
            st.session_state.data_types = infer_data_types(sample_df)
            
            # Convert date columns
            potential_date_cols = [col for col in st.session_state.data_types['potential_date_strings']]
            st.session_state.df = convert_date_columns(st.session_state.df, potential_date_cols)
            
            st.success("Sample data loaded successfully!")
            st.rerun()
        
        return
    
    # Process the uploaded file
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            # Set to session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # Infer data types
            st.session_state.data_types = infer_data_types(df)
            
            # Convert date columns
            potential_date_cols = [col for col in st.session_state.data_types['potential_date_strings']]
            st.session_state.df = convert_date_columns(st.session_state.df, potential_date_cols)
            
            st.success("File processed successfully!")
    
    # Get data from session state
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        df = st.session_state.df
        data_types = st.session_state.data_types
        
        # Initialize session state for saved visualizations if not already done
        if 'saved_visualizations' not in st.session_state:
            st.session_state.saved_visualizations = []
        
        # Create tabs for different sections
        tabs = st.tabs([
            "Overview", 
            "Numeric Analysis", 
            "Categorical Analysis", 
            "Time Series", 
            "Custom Visualization",
            "Prompt-Based Viz",
            "My Dashboard"
        ])
        
        with tabs[0]:
            create_overview_section(df, data_types)
        
        with tabs[1]:
            create_numeric_analysis(df, data_types)
        
        with tabs[2]:
            create_categorical_analysis(df, data_types)
        
        with tabs[3]:
            create_time_series_analysis(df, data_types)
        
        with tabs[4]:
            create_custom_visualizations(df, data_types)
        
        with tabs[5]:
            create_prompt_based_visualizations(df, data_types)
        
        with tabs[6]:
            create_dashboard(st.session_state.saved_visualizations)
        
        # Option to reset/upload a new file
        if st.sidebar.button("Reset / Upload New File"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def add_drag_drop_layout():
    """Add custom CSS and JavaScript for drag-and-drop dashboard layout"""
    # This is a placeholder for drag-and-drop functionality
    # In a real implementation, we would use a JavaScript library like GridStack or React DnD
    # Since Streamlit doesn't natively support drag-and-drop, this would require custom components
    
    st.markdown("""
    <style>
    /* Styles for draggable elements */
    .draggable {
        cursor: move;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Style for drag-over effect */
    .drag-over {
        background-color: #f0f8ff;
    }
    
    /* Dashboard grid */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        grid-gap: 15px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Note: In a real application, we would include JavaScript for drag and drop
    # functionality, but Streamlit limits what custom JS can be run
    st.markdown("""
    <div class="stMarkdown">
      <p style="color: #666; font-style: italic; font-size: 0.9em; margin-top: 20px;">
        Note: For a full drag-and-drop dashboard experience, consider using a Streamlit 
        Component specifically designed for this purpose, or using a framework like
        Dash or Panel that has built-in drag-and-drop capabilities.
      </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Add drag-and-drop functionality (note: limited in Streamlit)
    add_drag_drop_layout()
    main()
