import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import json
import anthropic
import os
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Claude-Powered Dashboard",
    page_icon="🤖",
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

# *** HARDCODED API KEY - REPLACE WITH YOUR ACTUAL KEY ***
CLAUDE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual Claude API key

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

class ClaudeAnalyzer:
    def __init__(self, api_key=None):
    if api_key is None:
        # Try to get from environment variables
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        # Or from streamlit secrets
        if hasattr(st, 'secrets') and 'CLAUDE_API_KEY' in st.secrets:
            api_key = st.secrets["CLAUDE_API_KEY"]
    
    if not api_key:
        st.warning("Claude API key not found.")
        self.client = None
        self.is_available = False
    else:
        try:
            # Initialize without proxies parameter
            self.client = anthropic.Anthropic(api_key=api_key)
            self.is_available = True
        except Exception as e:
            st.error(f"Error initializing Claude client: {str(e)}")
            self.client = None
            self.is_available = False
    
    # Set default model
    self.model = "claude-3-sonnet-20240229"
    
    def analyze_data(self, df, prompt, max_rows=100):
        """Analyze data using Claude API"""
        if not self.is_available:
            return {"error": "Claude API not available."}
        
        try:
            # Sample the data for large datasets
            sample_df = df.head(max_rows) if len(df) > max_rows else df
            csv_string = sample_df.to_csv(index=False)
            
            # Add statistical summary for larger datasets
            if len(df) > max_rows:
                stats_summary = f"""
                Additional dataset statistics:
                - Total rows: {len(df)} (showing first {max_rows} rows in sample)
                - Columns: {', '.join(df.columns)}
                - Numeric columns stats:
                {df.describe().to_string()}
                """
            else:
                stats_summary = ""
            
            # Create the prompt for Claude
            system_message = """
            You are an expert data analyst. You will be given CSV data and a request for analysis.
            Analyze the data and provide your findings in JSON format with these keys:
            - "summary": A brief summary of the data (1-2 sentences)
            - "key_insights": Array of 3-5 key insights or patterns you observe
            - "recommended_visualizations": Array of objects, each with:
                - "title": Descriptive title for the visualization
                - "description": What this visualization shows
                - "type": One of "bar", "line", "scatter", "pie", "histogram", "box", "heatmap"
                - "x_column": Column name for x-axis (or null for some chart types)
                - "y_column": Column name for y-axis (or values for pie charts)
                - "color_column": Optional column name for color dimension (or null)
                - "agg_function": Aggregation function to use ("sum", "mean", "count", etc.)
                - "filters": Optional array of filter operations to apply
            - "anomalies": Array of potential anomalies or data issues
            - "data_quality": Object with data quality metrics
            
            Respond ONLY with valid JSON. Do not include any explanations or text outside the JSON structure.
            """
            
            full_prompt = f"""
            Here is a CSV dataset:
            
            ```
            {csv_string}
            ```
            
            {stats_summary}
            
            User request: {prompt}
            
            Provide your analysis and recommendations in JSON format as specified.
            """
            
            # Call to Claude API
            response = self.client.messages.create(
                model=self.model,  # Use the specified Claude model
                system=system_message,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Extract JSON from response
            result_text = response.content[0].text
            
            # Clean up the response to extract just valid JSON
            # Sometimes Claude might wrap the JSON in code blocks or add explanations
            json_pattern = r'```(?:json)?(.*?)```|(\{.*\})'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                json_str = match.group(1) or match.group(2)
                json_str = json_str.strip()
            else:
                json_str = result_text
            
            # Parse the JSON
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                st.error(f"Error parsing Claude's response as JSON: {str(e)}")
                st.text("Raw response from Claude:")
                st.text(result_text)
                return {"error": "Failed to parse Claude's response as JSON"}
            
        except Exception as e:
            st.error(f"Error calling Claude API: {str(e)}")
            return {"error": f"Error calling Claude API: {str(e)}"}

    def generate_natural_language_report(self, df, data_types):
        """Generate a natural language report about the data"""
        if not self.is_available:
            return "Claude API not available."
        
        try:
            # Prepare sample of data to send to Claude
            max_rows = 100  # Limit rows to avoid token limits
            sample_df = df.head(max_rows) if len(df) > max_rows else df
            csv_string = sample_df.to_csv(index=False)
            
            # Create summary of data types
            data_types_summary = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_types": {
                    "numeric": data_types['numeric'],
                    "categorical": data_types['categorical'],
                    "datetime": data_types['datetime'] + data_types['potential_date_strings'],
                    "text": data_types['text'],
                    "boolean": data_types['boolean'],
                    "id": data_types['id']
                }
            }
            
            # Generate numeric summary if available
            numeric_summary = None
            if data_types['numeric']:
                numeric_summary = df[data_types['numeric']].describe().to_dict()
            
            # Create the prompt for Claude
            system_message = """
            You are an expert data analyst creating a natural language report about a dataset.
            Write a concise, informative report for a business audience that:
            
            1. Summarizes the key characteristics of the data
            2. Highlights important patterns, trends, or relationships
            3. Points out potential issues or anomalies
            4. Provides 2-3 actionable recommendations based on the data
            
            Use a professional but accessible tone. Limit your report to 500-800 words.
            """
            
            full_prompt = f"""
            Here is information about a dataset:
            
            Sample data (first {min(max_rows, len(df))} rows):
            ```
            {csv_string}
            ```
            
            Data types summary:
            ```
            {json.dumps(data_types_summary, indent=2)}
            ```
            
            {f"Numeric summary statistics: {json.dumps(numeric_summary, indent=2)}" if numeric_summary else ""}
            
            Based on this information, generate a comprehensive data report. 
            Include insights about the data's structure, key metrics, patterns, and potential areas for further investigation.
            """
            
            # Call to Claude API
            response = self.client.messages.create(
                model=self.model,  # Use the specified Claude model
                system=system_message,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Return the report
            return response.content[0].text
            
        except Exception as e:
            st.error(f"Error generating report with Claude API: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def interpret_prompt(self, df, data_types, prompt):
        """Interpret a natural language prompt and convert it to visualization specifications"""
        if not self.is_available:
            return {"error": "Claude API not available."}
        
        try:
            # Prepare summary of data structure
            columns_info = {}
            for col in df.columns:
                col_type = "unknown"
                for dtype, cols in data_types.items():
                    if col in cols:
                        col_type = dtype
                        break
                
                # Add sample values
                if col in df.columns:
                    sample_values = df[col].dropna().head(5).tolist()
                    try:
                        sample_values = [str(val) for val in sample_values]
                    except:
                        sample_values = ["[complex value]"]
                else:
                    sample_values = []
                
                columns_info[col] = {
                    "type": col_type,
                    "sample_values": sample_values
                }
            
            # Create the prompt for Claude
            system_message = """
            You are an expert data visualization assistant. You will be given information about a dataset 
            and a natural language request for a visualization. Your job is to interpret the request and 
            determine the appropriate visualization specifications.
            
            Return ONLY a JSON object with these fields:
            - "visualization_type": One of "bar", "line", "scatter", "pie", "histogram", "box", "heatmap"
            - "x_column": Column name for x-axis (or null for some chart types)
            - "y_column": Column name for y-axis (or values for pie charts)
            - "color_column": Optional column name for color dimension (or null)
            - "agg_function": Aggregation function to use ("sum", "mean", "count", etc.)
            - "title": Suggested title for the visualization
            - "filters": Optional array of filter operations to apply
            - "interpretation": Brief explanation of what you understood from the request
            
            Your response should be valid JSON only. Do not include any text outside the JSON structure.
            """
            
            full_prompt = f"""
            Dataset information:
            - Number of rows: {len(df)}
            - Columns: {json.dumps(columns_info, indent=2)}
            
            User's visualization request: "{prompt}"
            
            Determine the appropriate visualization specifications based on this request.
            """
            
            # Call to Claude API
            response = self.client.messages.create(
                model=self.model,  # Use the specified Claude model
                system=system_message,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Extract JSON from response
            result_text = response.content[0].text
            
            # Clean up the response to extract just valid JSON
            json_pattern = r'```(?:json)?(.*?)```|(\{.*\})'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                json_str = match.group(1) or match.group(2)
                json_str = json_str.strip()
            else:
                json_str = result_text
            
            # Parse the JSON
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                st.error(f"Error parsing Claude's response as JSON: {str(e)}")
                st.text("Raw response from Claude:")
                st.text(result_text)
                return {"error": "Failed to parse Claude's response as JSON"}
            
        except Exception as e:
            st.error(f"Error interpreting prompt with Claude API: {str(e)}")
            return {"error": f"Error interpreting prompt: {str(e)}"}

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

def create_overview_section(df, data_types, claude_analyzer=None):
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
    
    # Claude-powered data report
    if claude_analyzer and claude_analyzer.is_available:
        with st.expander("Claude's Data Analysis Report", expanded=True):
            if st.button("Generate Report with Claude"):
                with st.spinner("Claude is analyzing your data..."):
                    report = claude_analyzer.generate_natural_language_report(df, data_types)
                    st.markdown(report)
    
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

def create_claude_analysis_section(df, data_types, claude_analyzer):
    """Create a section for Claude-powered data analysis"""
    st.header("Claude-Powered Data Analysis")
    
    if not claude_analyzer or not claude_analyzer.is_available:
        st.warning("Claude API is not available.")
        return
    
    # Let user ask Claude to analyze data
    st.subheader("Ask Claude about your data")
    
    analysis_prompt = st.text_area(
        "What would you like Claude to analyze in your data?",
        height=100,
        placeholder="Example: 'Analyze the relationship between categories and sales' or 'Find interesting patterns in this dataset'"
    )
    
    if st.button("Analyze with Claude"):
        if not analysis_prompt:
            st.warning("Please enter a prompt for analysis.")
        else:
            with st.spinner("Claude is analyzing your data..."):
                analysis_results = claude_analyzer.analyze_data(df, analysis_prompt)
                
                if "error" in analysis_results:
                    st.error(analysis_results["error"])
                else:
                    # Display summary
                    st.subheader("Summary")
                    st.write(analysis_results.get("summary", "No summary provided."))
                    
                    # Display key insights
                    st.subheader("Key Insights")
                    insights = analysis_results.get("key_insights", [])
                    if insights:
                        for i, insight in enumerate(insights):
                            st.markdown(f"**{i+1}.** {insight}")
                    else:
                        st.write("No insights provided.")
                    
                    # Display recommended visualizations
                    st.subheader("Recommended Visualizations")
                    viz_recs = analysis_results.get("recommended_visualizations", [])
                    
                    if viz_recs:
                        for i, viz in enumerate(viz_recs):
                            st.markdown(f"### {viz.get('title', f'Visualization {i+1}')}")
                            st.markdown(viz.get('description', 'No description provided.'))
                            
                            # Extract visualization parameters
                            viz_type = viz.get('type', 'bar')
                            x_col = viz.get('x_column')
                            y_col = viz.get('y_column')
                            color_col = viz.get('color_column')
                            agg_func = viz.get('agg_function', 'sum')
                            
                            # Create visualization based on type
                            try:
                                if viz_type == 'bar':
                                    if x_col and y_col:
                                        # For bar charts, we typically group by x and aggregate y
                                        agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                                        agg_data = agg_data.sort_values(y_col, ascending=False)
                                        
                                        # Limit to top 15 for clarity
                                        if len(agg_data) > 15:
                                            agg_data = agg_data.head(15)
                                            
                                        fig = px.bar(
                                            agg_data, 
                                            x=x_col, 
                                            y=y_col,
                                            title=viz.get('title', f"{agg_func.capitalize()} of {y_col} by {x_col}"),
                                            color=color_col if color_col else None
                                        )
                                        fig.update_layout(xaxis_tickangle=-45)
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Missing columns for bar chart: x_col={x_col}, y_col={y_col}")
                                
                                elif viz_type == 'line':
                                    if x_col and y_col:
                                        # Handle datetime x
                                        if x_col in data_types['datetime'] or x_col in [col for col in data_types['potential_date_strings'] if f"{col}_Year" in df.columns]:
                                            # For time series, we need to ensure the x-axis is properly sorted
                                            df_sorted = df.sort_values(x_col)
                                            
                                            # If color column is provided, group by that as well
                                            if color_col:
                                                grouped = df_sorted.groupby([pd.Grouper(key=x_col, freq='M'), color_col])[y_col].agg(agg_func).reset_index()
                                                fig = px.line(
                                                    grouped,
                                                    x=x_col,
                                                    y=y_col,
                                                    color=color_col,
                                                    title=viz.get('title', f"{agg_func.capitalize()} of {y_col} Over Time by {color_col}"),
                                                    markers=True
                                                )
                                            else:
                                                grouped = df_sorted.groupby(pd.Grouper(key=x_col, freq='M'))[y_col].agg(agg_func).reset_index()
                                                fig = px.line(
                                                    grouped,
                                                    x=x_col,
                                                    y=y_col,
                                                    title=viz.get('title', f"{agg_func.capitalize()} of {y_col} Over Time"),
                                                    markers=True
                                                )
                                        else:
                                            # For non-datetime x, just create a regular line chart
                                            fig = px.line(
                                                df,
                                                x=x_col,
                                                y=y_col,
                                                color=color_col if color_col else None,
                                                title=viz.get('title', f"{y_col} vs {x_col}"),
                                                markers=True
                                            )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Missing columns for line chart: x_col={x_col}, y_col={y_col}")
                                
                                elif viz_type == 'scatter':
                                    if x_col and y_col:
                                        fig = px.scatter(
                                            df,
                                            x=x_col,
                                            y=y_col,
                                            color=color_col if color_col else None,
                                            title=viz.get('title', f"{y_col} vs {x_col}")
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Missing columns for scatter plot: x_col={x_col}, y_col={y_col}")
                                
                                elif viz_type == 'pie':
                                    if x_col and y_col:
                                        # For pie charts, we need to aggregate the data
                                        agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                                        
                                        # Limit to top categories for clarity
                                        if len(agg_data) > 10:
                                            agg_data = agg_data.sort_values(y_col, ascending=False).head(10)
                                            st.info("Showing only top 10 categories for clarity")
                                        
                                        fig = px.pie(
                                            agg_data,
                                            names=x_col,
                                            values=y_col,
                                            title=viz.get('title', f"Distribution of {y_col} by {x_col}")
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Missing columns for pie chart: x_col={x_col}, y_col={y_col}")
                                
                                elif viz_type == 'histogram':
                                    if x_col:
                                        fig = px.histogram(
                                            df,
                                            x=x_col,
                                            color=color_col if color_col else None,
                                            title=viz.get('title', f"Distribution of {x_col}")
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Missing column for histogram: x_col={x_col}")
                                
                                elif viz_type == 'box':
                                    if y_col:
                                        fig = px.box(
                                            df,
                                            x=x_col if x_col else None,
                                            y=y_col,
                                            color=color_col if color_col else None,
                                            title=viz.get('title', f"Box Plot of {y_col}" + (f" by {x_col}" if x_col else ""))
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Missing column for box plot: y_col={y_col}")
                                
                                elif viz_type == 'heatmap':
                                    if x_col and y_col:
                                        # Create pivot table for heatmap
                                        pivot_data = df.pivot_table(
                                            index=y_col,
                                            columns=x_col,
                                            values=color_col if color_col else y_col,  # If color_col is provided, use it for values
                                            aggfunc=agg_func,
                                            fill_value=0
                                        )
                                        
                                        # Limit size for readability
                                        if pivot_data.shape[0] > 15 or pivot_data.shape[1] > 15:
                                            # Get top rows and columns by sum
                                            row_sums = pivot_data.sum(axis=1).sort_values(ascending=False).head(15).index
                                            col_sums = pivot_data.sum(axis=0).sort_values(ascending=False).head(15).index
                                            pivot_data = pivot_data.loc[row_sums, col_sums]
                                            st.info("Heatmap limited to top 15 rows and columns for readability")
                                        
                                        fig = px.imshow(
                                            pivot_data,
                                            title=viz.get('title', f"Heatmap of {agg_func.capitalize()} of {color_col if color_col else y_col} by {y_col} and {x_col}"),
                                            color_continuous_scale='Blues'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Missing columns for heatmap: x_col={x_col}, y_col={y_col}")
                                
                                else:
                                    st.warning(f"Unsupported visualization type: {viz_type}")
                                
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
                            
                            # Add option to save this visualization to the dashboard
                            if 'saved_visualizations' not in st.session_state:
                                st.session_state.saved_visualizations = []
                            
                            if st.button(f"Add to Dashboard", key=f"add_claude_viz_{i}"):
                                try:
                                    # Re-create the visualization to save
                                    st.session_state.saved_visualizations.append({
                                        'title': viz.get('title', f"Visualization {i+1}"),
                                        'type': viz_type,
                                        'x_col': x_col,
                                        'y_col': y_col,
                                        'color_col': color_col,
                                        'agg_func': agg_func,
                                        'description': viz.get('description', '')
                                    })
                                    st.success(f"Added '{viz.get('title', f'Visualization {i+1}')}' to your dashboard!")
                                except Exception as e:
                                    st.error(f"Error saving visualization: {str(e)}")
                    
                    else:
                        st.write("No visualizations recommended.")
                    
                    # Display anomalies if any
                    if "anomalies" in analysis_results and analysis_results["anomalies"]:
                        st.subheader("Potential Anomalies")
                        anomalies = analysis_results["anomalies"]
                        for i, anomaly in enumerate(anomalies):
                            st.markdown(f"**{i+1}.** {anomaly}")
                    
                    # Display data quality information if available
                    if "data_quality" in analysis_results and analysis_results["data_quality"]:
                        st.subheader("Data Quality Assessment")
                        data_quality = analysis_results["data_quality"]
                        for key, value in data_quality.items():
                            st.markdown(f"**{key}:** {value}")

def create_prompt_based_visualizations(df, data_types, claude_analyzer=None):
    """Create visualizations based on natural language prompts using Claude"""
    colors = get_color_scheme()
    
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
        if claude_analyzer and claude_analyzer.is_available:
            # Use Claude to interpret the prompt
            with st.spinner("Claude is interpreting your request..."):
                viz_spec = claude_analyzer.interpret_prompt(df, data_types, prompt)
                
                if "error" in viz_spec:
                    st.error(viz_spec["error"])
                else:
                    st.success("Visualization generated based on your request!")
                    
                    # Extract visualization parameters
                    viz_type = viz_spec.get("visualization_type", "bar")
                    x_col = viz_spec.get("x_column")
                    y_col = viz_spec.get("y_column")
                    color_col = viz_spec.get("color_column")
                    agg_func = viz_spec.get("agg_function", "sum")
                    title = viz_spec.get("title", "Visualization")
                    
                    # Display interpretation
                    if "interpretation" in viz_spec:
                        st.info(f"I understood your request as: {viz_spec['interpretation']}")
                    
                    # Create visualization based on type
                    try:
                        if viz_type == 'bar':
                            if x_col and y_col:
                                # For bar charts, we typically group by x and aggregate y
                                agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                                agg_data = agg_data.sort_values(y_col, ascending=False)
                                
                                # Limit to top 15 for clarity
                                if len(agg_data) > 15:
                                    agg_data = agg_data.head(15)
                                    
                                fig = px.bar(
                                    agg_data, 
                                    x=x_col, 
                                    y=y_col,
                                    title=title,
                                    color=color_col if color_col else None,
                                    color_discrete_sequence=[colors['primary']]
                                )
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Missing columns for bar chart: x_col={x_col}, y_col={y_col}")
                        
                        elif viz_type == 'line':
                            if x_col and y_col:
                                # Handle datetime x
                                if x_col in data_types['datetime'] or x_col in [col for col in data_types['potential_date_strings'] if f"{col}_Year" in df.columns]:
                                    # For time series, we need to ensure the x-axis is properly sorted
                                    df_sorted = df.sort_values(x_col)
                                    
                                    # If color column is provided, group by that as well
                                    if color_col:
                                        grouped = df_sorted.groupby([pd.Grouper(key=x_col, freq='M'), color_col])[y_col].agg(agg_func).reset_index()
                                        fig = px.line(
                                            grouped,
                                            x=x_col,
                                            y=y_col,
                                            color=color_col,
                                            title=title,
                                            markers=True,
                                            color_discrete_sequence=colors['categorical']
                                        )
                                    else:
                                        grouped = df_sorted.groupby(pd.Grouper(key=x_col, freq='M'))[y_col].agg(agg_func).reset_index()
                                        fig = px.line(
                                            grouped,
                                            x=x_col,
                                            y=y_col,
                                            title=title,
                                            markers=True,
                                            color_discrete_sequence=[colors['primary']]
                                        )
                                else:
                                    # For non-datetime x, just create a regular line chart
                                    fig = px.line(
                                        df,
                                        x=x_col,
                                        y=y_col,
                                        color=color_col if color_col else None,
                                        title=title,
                                        markers=True,
                                        color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Missing columns for line chart: x_col={x_col}, y_col={y_col}")
                        
                        elif viz_type == 'scatter':
                            if x_col and y_col:
                                fig = px.scatter(
                                    df,
                                    x=x_col,
                                    y=y_col,
                                    color=color_col if color_col else None,
                                    title=title,
                                    color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Missing columns for scatter plot: x_col={x_col}, y_col={y_col}")
                        
                        elif viz_type == 'pie':
                            if x_col and y_col:
                                # For pie charts, we need to aggregate the data
                                agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                                
                                # Limit to top categories for clarity
                                if len(agg_data) > 10:
                                    agg_data = agg_data.sort_values(y_col, ascending=False).head(10)
                                    st.info("Showing only top 10 categories for clarity")
                                
                                fig = px.pie(
                                    agg_data,
                                    names=x_col,
                                    values=y_col,
                                    title=title,
                                    color_discrete_sequence=colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Missing columns for pie chart: x_col={x_col}, y_col={y_col}")
                        
                        elif viz_type == 'histogram':
                            if x_col:
                                fig = px.histogram(
                                    df,
                                    x=x_col,
                                    color=color_col if color_col else None,
                                    title=title,
                                    color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Missing column for histogram: x_col={x_col}")
                        
                        elif viz_type == 'box':
                            if y_col:
                                fig = px.box(
                                    df,
                                    x=x_col if x_col else None,
                                    y=y_col,
                                    color=color_col if color_col else None,
                                    title=title,
                                    color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Missing column for box plot: y_col={y_col}")
                        
                        elif viz_type == 'heatmap':
                            if x_col and y_col:
                                # Create pivot table for heatmap
                                pivot_data = df.pivot_table(
                                    index=y_col,
                                    columns=x_col,
                                    values=color_col if color_col else data_types['numeric'][0],  # Use first numeric column if no color_col
                                    aggfunc=agg_func,
                                    fill_value=0
                                )
                                
                                # Limit size for readability
                                if pivot_data.shape[0] > 15 or pivot_data.shape[1] > 15:
                                    # Get top rows and columns by sum
                                    row_sums = pivot_data.sum(axis=1).sort_values(ascending=False).head(15).index
                                    col_sums = pivot_data.sum(axis=0).sort_values(ascending=False).head(15).index
                                    pivot_data = pivot_data.loc[row_sums, col_sums]
                                    st.info("Heatmap limited to top 15 rows and columns for readability")
                                
                                fig = px.imshow(
                                    pivot_data,
                                    title=title,
                                    color_continuous_scale=colors['sequence']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Missing columns for heatmap: x_col={x_col}, y_col={y_col}")
                        
                        else:
                            st.warning(f"Unsupported visualization type: {viz_type}")
                            
                        # Add the visualization to the dashboard if user wants
                        st.markdown("---")
                        
                        # Save visualization to session state for later use in dashboard
                        if 'saved_visualizations' not in st.session_state:
                            st.session_state.saved_visualizations = []
                        
                        viz_title = st.text_input("Enter a title for this visualization (to save it to your dashboard):", 
                                                value=title)
                        
                        if st.button("Add to My Dashboard"):
                            # Store the visualization info
                            viz_info = {
                                'title': viz_title,
                                'prompt': prompt,
                                'type': viz_type,
                                'x_col': x_col,
                                'y_col': y_col,
                                'color_col': color_col,
                                'agg_func': agg_func
                            }
                            st.session_state.saved_visualizations.append(viz_info)
                            st.success(f"Added '{viz_title}' to your dashboard!")
                            
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                        st.info("Try being more specific in your request.")
        else:
            # Fallback to simple keyword-based visualization (summarized for brevity)
            st.warning("Claude API is not available.")
            # Basic fallback visualization logic

def create_dashboard(df, saved_visualizations):
    """Display the user's personalized dashboard with saved visualizations"""
    colors = get_color_scheme()
    
    st.header("My Custom Dashboard")
    
    if not saved_visualizations:
        st.info("Your dashboard is empty. Create and save visualizations to see them here.")
        return
    
    # Display all saved visualizations in a grid
    cols = st.columns(2)  # 2 columns for the dashboard
    
    for i, viz_info in enumerate(saved_visualizations):
        with cols[i % 2]:
            with st.expander(viz_info['title'], expanded=True):
                # Display visualization based on stored information
                if 'type' in viz_info:
                    # Re-create the visualization from saved parameters
                    viz_type = viz_info.get('type')
                    x_col = viz_info.get('x_col')
                    y_col = viz_info.get('y_col')
                    color_col = viz_info.get('color_col')
                    agg_func = viz_info.get('agg_func', 'sum')
                    
                    try:
                        if viz_type == 'bar':
                            if x_col and y_col:
                                # Re-aggregate the data
                                agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                                agg_data = agg_data.sort_values(y_col, ascending=False).head(15)
                                
                                fig = px.bar(
                                    agg_data, 
                                    x=x_col, 
                                    y=y_col,
                                    title=viz_info['title'],
                                    color=color_col if color_col else None,
                                    color_discrete_sequence=[colors['primary']]
                                )
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Missing parameters for bar chart")
                        
                        elif viz_type == 'line':
                            if x_col and y_col:
                                fig = px.line(
                                    df,
                                    x=x_col,
                                    y=y_col,
                                    color=color_col if color_col else None,
                                    title=viz_info['title'],
                                    markers=True,
                                    color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Missing parameters for line chart")
                        
                        elif viz_type == 'pie':
                            if x_col and y_col:
                                agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                                agg_data = agg_data.sort_values(y_col, ascending=False).head(10)
                                
                                fig = px.pie(
                                    agg_data,
                                    names=x_col,
                                    values=y_col,
                                    title=viz_info['title'],
                                    color_discrete_sequence=colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Missing parameters for pie chart")
                        
                        elif viz_type == 'scatter':
                            if x_col and y_col:
                                fig = px.scatter(
                                    df,
                                    x=x_col,
                                    y=y_col,
                                    color=color_col if color_col else None,
                                    title=viz_info['title'],
                                    color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Missing parameters for scatter plot")
                        
                        elif viz_type == 'histogram':
                            if x_col:
                                fig = px.histogram(
                                    df,
                                    x=x_col,
                                    color=color_col if color_col else None,
                                    title=viz_info['title'],
                                    color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Missing parameters for histogram")
                        
                        elif viz_type == 'box':
                            if y_col:
                                fig = px.box(
                                    df,
                                    x=x_col if x_col else None,
                                    y=y_col,
                                    color=color_col if color_col else None,
                                    title=viz_info['title'],
                                    color_discrete_sequence=[colors['primary']] if not color_col else colors['categorical']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Missing parameters for box plot")
                        
                        elif viz_type == 'heatmap':
                            if x_col and y_col:
                                # Create pivot table for heatmap
                                try:
                                    values_col = color_col if color_col else data_types['numeric'][0]
                                    pivot_data = df.pivot_table(
                                        index=y_col,
                                        columns=x_col,
                                        values=values_col,
                                        aggfunc=agg_func,
                                        fill_value=0
                                    )
                                    
                                    # Limit size for readability
                                    if pivot_data.shape[0] > 15 or pivot_data.shape[1] > 15:
                                        # Get top rows and columns by sum
                                        row_sums = pivot_data.sum(axis=1).sort_values(ascending=False).head(15).index
                                        col_sums = pivot_data.sum(axis=0).sort_values(ascending=False).head(15).index
                                        pivot_data = pivot_data.loc[row_sums, col_sums]
                                        st.info("Heatmap limited to top 15 rows and columns for readability")
                                    
                                    fig = px.imshow(
                                        pivot_data,
                                        title=viz_info['title'],
                                        color_continuous_scale=colors['sequence']
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Error creating heatmap: {str(e)}")
                            else:
                                st.warning("Missing parameters for heatmap")
                                
                        else:
                            st.warning(f"Unsupported visualization type: {viz_type}")
                        
                    except Exception as e:
                        st.error(f"Error recreating visualization: {str(e)}")
                else:
                    st.warning("Visualization data is not available")
                
                # Show the prompt that created this visualization if available
                if 'prompt' in viz_info:
                    st.caption(f"Prompt: {viz_info['prompt']}")
                
                # Show description if available
                if 'description' in viz_info and viz_info['description']:
                    st.markdown(viz_info['description'])
                
                # Option to remove from dashboard
                if st.button(f"Remove from Dashboard", key=f"remove_{i}"):
                    saved_visualizations.pop(i)
                    st.rerun()

def main():
    # App title and description
    st.title("Claude-Powered Dashboard Generator")
    st.markdown("""
    Upload any CSV file to automatically generate visualizations and create your own custom dashboard.
    Leverage Claude's AI to analyze your data and create intelligent visualizations from natural language prompts!
    """)
    
    # Sidebar for model selection
    st.sidebar.header("Claude Configuration")
    claude_model = st.sidebar.selectbox(
        "Select Claude Model",
        options=[
            "claude-3-opus-20240229", 
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620"
        ],
        index=1  # Default to Sonnet
    )
    
    # Initialize Claude analyzer
    claude_analyzer = ClaudeAnalyzer()
    
    # Set the model for all Claude API calls
    if claude_analyzer.is_available:
        claude_analyzer.model = claude_model
        st.sidebar.success(f"✅ Using {claude_model}")
    
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
            
            # Store the selected Claude model
            st.session_state.selected_model = claude_model
            
            st.success("Sample data loaded successfully!")
            st.rerun()
        
        return
    
    # Process the uploaded file
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        with st.status("Processing data...") as status:
            df = process_uploaded_file(uploaded_file)
            
            if df is not None:
                # Show row count
                st.write(f"Found {len(df)} rows and {len(df.columns)} columns in the dataset.")
                
                status.update(label="Analyzing data types...", state="running")
                # Set to session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Infer data types
                st.session_state.data_types = infer_data_types(df)
                
                # Convert date columns
                potential_date_cols = [col for col in st.session_state.data_types['potential_date_strings']]
                st.session_state.df = convert_date_columns(st.session_state.df, potential_date_cols)
                
                # Store the selected Claude model
                st.session_state.selected_model = claude_model
                    
                status.update(label="Data processed successfully!", state="complete")
    
    # Get data from session state
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        df = st.session_state.df
        data_types = st.session_state.data_types
        
        # Apply model setting from session state if available
        if 'selected_model' in st.session_state:
            claude_model = st.session_state.selected_model
            if claude_analyzer.is_available:
                claude_analyzer.model = claude_model
        
        # Display data overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", format_number(len(df)))
        with col2:
            st.metric("Columns", format_number(len(df.columns)))
        with col3:
            st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Initialize session state for saved visualizations if not already done
        if 'saved_visualizations' not in st.session_state:
            st.session_state.saved_visualizations = []
        
        # Create tabs for different sections
        tabs = st.tabs([
            "Overview", 
            "Claude Analysis",
            "Prompt-Based Viz",
            "My Dashboard"
        ])
        
        with tabs[0]:
            create_overview_section(df, data_types, claude_analyzer)
        
        with tabs[1]:
            create_claude_analysis_section(df, data_types, claude_analyzer)
        
        with tabs[2]:
            create_prompt_based_visualizations(df, data_types, claude_analyzer)
        
        with tabs[3]:
            create_dashboard(df, st.session_state.saved_visualizations)
        
        # Option to reset/upload a new file
        if st.sidebar.button("Reset / Upload New File"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    # Set up session state variables if they don't exist
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'data_types' not in st.session_state:
        st.session_state.data_types = None
    
    if 'saved_visualizations' not in st.session_state:
        st.session_state.saved_visualizations = []
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "claude-3-sonnet-20240229"
    
    # Display version info in footer
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #f0f2f6; font-size: 12px;">
        Claude-Powered Dashboard Generator v1.0 | Using Anthropic API | Created 2025
    </div>
    """, unsafe_allow_html=True)
    
    # Run the main app
    main()
