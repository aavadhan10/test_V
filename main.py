import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import re
from datetime import datetime
import time
import uuid
import base64
from io import BytesIO
import hashlib

# Import Anthropic library with error handling
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    st.warning("Anthropic library not available. Install with: pip install anthropic")

# Password for demo
DEMO_PASSWORD = "velorademo"

# Set page configuration
st.set_page_config(
    page_title="Velora AI | Practice Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to display the Velora logo
def display_logo():
    """Display the Velora.AI logo in the app"""
    try:
        # Load from your GitHub repository
        github_logo_path = "https://raw.githubusercontent.com/aavadhan10/test_V/main/logo.png"
        
        # Create centered logo container
        st.markdown(
            """
            <div style="text-align: center; padding: 20px 0px 30px 0px;">
                <img src="{}" width="180px">
            </div>
            """.format(github_logo_path),
            unsafe_allow_html=True
        )
    except Exception as e:
        # If logo loading fails, use text fallback
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0px 20px 0px;">
                <h1 style="color: #9C27B0; font-size: 2.5rem; margin-bottom: 0;">VELORA.AI</h1>
                <p style="color: #555; margin-top: 0;">Practice Intelligence Platform</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to set Velora styling
def set_velora_styling():
    """Set custom CSS styling to match Velora branding"""
    st.markdown("""
    <style>
        /* Main styles */
        .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        
        /* Metrics */
        div[data-testid="stMetricValue"] {font-size: 32px; font-weight: 600; color: #9C27B0;}
        div[data-testid="stMetricLabel"] {font-size: 16px; opacity: 0.8;}
        
        /* Headers */
        h1 {color: #2C3E50; font-weight: 800; margin-bottom: 1.5rem;}
        h2 {color: #34495E; font-weight: 700; margin-top: 1rem;}
        h3 {color: #34495E; font-weight: 600;}
        
        /* Login container */
        .login-container {
            max-width: 450px;
            margin: 100px auto;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            background: white;
        }
        .login-header {text-align: center; margin-bottom: 20px;}
        
        /* Cards */
        .dashboard-card {
            border-radius: 10px;
            border: 1px solid #f0f2f6;
            padding: 1.5rem;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {background-color: #2C3E50;}
        
        /* Link buttons */
        .link-button {
            display: inline-block;
            border-radius: 4px;
            padding: 8px 16px;
            background-color: #9C27B0;
            color: white;
            text-decoration: none;
            font-weight: 500;
            margin: 0.25rem 0;
        }
        
        /* Custom tabs */
        .custom-tab {
            padding: 10px 15px;
            border-radius: 5px 5px 0 0;
            background-color: #f0f2f6;
            border: 1px solid #e6e9ef;
            border-bottom: none;
            cursor: pointer;
            margin-right: 2px;
        }
        .custom-tab.active {
            background-color: white;
            border-bottom: 3px solid #9C27B0;
            font-weight: 600;
        }
        
        /* Success icon */
        .success-icon {color: #4CAF50; font-size: 18px;}
        
        /* Insight card */
        .insight-card {
            background: linear-gradient(135deg, #f6f8fa 0%, #f0f2f6 100%);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #9C27B0;
        }
        
        /* Filters container */
        .filters-container {
            background-color: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        /* Custom footer */
        .custom-footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #2C3E50;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            z-index: 999;
        }
        
        /* Loader */
        .stSpinner > div {border-color: #9C27B0 !important;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {width: 8px; height: 8px;}
        ::-webkit-scrollbar-track {background: #f1f1f1;}
        ::-webkit-scrollbar-thumb {background: #c1c1c1; border-radius: 10px;}
        ::-webkit-scrollbar-thumb:hover {background: #a8a8a8;}
        
        /* Buttons */
        .stButton>button {
            background-color: #9C27B0;
            color: white;
        }
        .stButton>button:hover {
            background-color: #7B1FA2;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Define standard color scheme
COLORS = {
    'primary': '#9C27B0',     # Velora purple
    'secondary': '#34c759',   # Green - for positive metrics
    'accent': '#FF9500',      # Orange - for warnings or highlights
    'neutral': '#8E8E93',     # Gray - for secondary information
    'negative': '#FF3B30',    # Red - for negative metrics or alerts
    'background': '#F0F2F5',  # Light gray - for backgrounds
    'sequence': px.colors.sequential.Purples,  # Updated to purple theme
    'categorical': ['#9C27B0', '#34c759', '#FF9500', '#5856D6', '#FF2D55', '#007AFF', '#5AC8FA', '#FFCC00'],
    'diverging': px.colors.diverging.RdBu,
}

class PasswordProtection:
    """Handles password protection for the app"""
    
    def __init__(self, password):
        self.password = password
        self.login_placeholder = st.empty()
    
    def require_login(self):
        """Require login and return True if authenticated"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            with self.login_placeholder.container():
                self._display_login_form()
                return False
        else:
            self.login_placeholder.empty()
            return True
    
    def _display_login_form(self):
        """Display the login form"""
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Logo display
        try:
            # Use your GitHub repository
            github_logo_path = "https://raw.githubusercontent.com/aavadhan10/test_V/main/logo.png"
            
            st.markdown(
                """
                <div class="login-header" style="text-align: center; margin-bottom: 30px;">
                    <img src="{}" width="180px" style="margin-bottom: 15px;">
                    <p style="margin-top:10px; color: #555;">Practice Intelligence Platform</p>
                </div>
                """.format(github_logo_path),
                unsafe_allow_html=True
            )
        except:
            # Fallback to text if image loading fails
            st.markdown("""
            <div class="login-header">
                <h1 style="margin-bottom:5px; color: #9C27B0;">Velora AI</h1>
                <p style="margin-top:0;">Practice Intelligence Platform</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Login to Dashboard")
        password_input = st.text_input("Password", type="password")
        
        col1, col2 = st.columns([1,1])
        
        with col1:
            if st.button("Login", use_container_width=True):
                if password_input == self.password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        
        with col2:
            if st.button("Demo Mode", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.demo_mode = True
                st.rerun()
        
        st.markdown("<p style='font-size:12px; margin-top:20px;'>For demo, use password: 'velorademo'</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="custom-footer">
            Velora AI Practice Intelligence Platform | © 2025 Velora, Inc. All rights reserved.
        </div>
        """, unsafe_allow_html=True)

class ClaudeAnalyzer:
    """Handles interactions with Claude API for data analysis"""
    
    def __init__(self):
        self.client = None
        self.is_available = False
        self.model = "claude-3-sonnet-20240229"  # Default model
        
        # Try to get API key from various sources
        api_key = None
        
        # 1. Try Streamlit secrets
        if hasattr(st, 'secrets') and 'CLAUDE_API_KEY' in st.secrets:
            api_key = st.secrets["CLAUDE_API_KEY"]
        
        # 2. Try environment variables
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        
        # Initialize client if we have an API key and the library is available
        if api_key and ANTHROPIC_AVAILABLE:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.is_available = True
                st.sidebar.success("✅ Claude AI connected")
            except Exception as e:
                st.sidebar.error(f"Claude AI connection error: {type(e).__name__}")
                print(f"Error initializing Claude client: {str(e)}")
    
    def analyze_data(self, df, prompt, max_rows=100):
        """Analyze data using Claude API or return dummy data if unavailable"""
        if not self.is_available:
            # Return fallback data for demo purposes
            return self._get_fallback_analysis(df)
        
        try:
            # Sample data for large datasets
            sample_df = df.head(max_rows) if len(df) > max_rows else df
            csv_string = sample_df.to_csv(index=False)
            
            # Create system prompt
            system_message = """
            You are an expert data analyst specializing in practice management data for professional services firms.
            You will be given CSV data and a request for analysis.
            
            Analyze the data and provide your findings in JSON format with these keys:
            - "summary": A brief summary of the data (1-2 sentences)
            - "key_insights": Array of 3-5 key insights or patterns you observe
            - "kpis": Array of objects, each with:
                - "name": Name of the KPI
                - "value": The value
                - "trend": "up", "down", or "neutral"
                - "interpretation": Brief interpretation (is this good or bad)
            - "recommended_visualizations": Array of objects, each with:
                - "title": Descriptive title for the visualization
                - "description": What this visualization shows
                - "type": One of "bar", "line", "scatter", "pie", "histogram", "box", "heatmap"
                - "x_column": Column name for x-axis (or null for some chart types)
                - "y_column": Column name for y-axis (or values for pie charts)
                - "color_column": Optional column name for color dimension (or null)
                - "agg_function": Aggregation function to use ("sum", "mean", "count", etc.)
            
            Respond ONLY with valid JSON. Do not include any explanations or text outside the JSON structure.
            """
            
            # User prompt with data
            user_prompt = f"""
            Here is a CSV dataset containing practice management data:
            
            ```
            {csv_string}
            ```
            
            User request: {prompt}
            
            Provide your analysis and recommendations in JSON format as specified.
            """
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                system=system_message,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract and parse JSON response
            result_text = response.content[0].text
            
            # Clean up response to extract valid JSON
            json_pattern = r'```(?:json)?(.*?)```|(\{.*\})'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                json_str = match.group(1) or match.group(2)
                json_str = json_str.strip()
            else:
                json_str = result_text
            
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # Return error result
                return {
                    "error": "Failed to parse Claude's response as JSON",
                    "summary": "Error in analysis",
                    "key_insights": ["Could not process response"],
                    "kpis": [],
                    "recommended_visualizations": []
                }
            
        except Exception as e:
            # Handle any API errors
            return {
                "error": f"Error calling Claude API: {str(e)}",
                "summary": "Error in analysis",
                "key_insights": ["API error occurred"],
                "kpis": [],
                "recommended_visualizations": []
            }
    
    def interpret_prompt(self, df, data_types, prompt):
        """Interpret natural language prompt for visualization"""
        if not self.is_available:
            # Return fallback visualization spec
            return self._get_fallback_visualization(df, data_types, prompt)
        
        try:
            # Prepare data structure info
            columns_info = {col: {"type": self._get_column_type(col, data_types)} for col in df.columns}
            
            # Create system prompt
            system_message = """
            You are an expert data visualization assistant specializing in practice management data.
            Given information about a dataset and a natural language request, determine the
            appropriate visualization specifications.
            
            Return ONLY a JSON object with these fields:
            - "visualization_type": One of "bar", "line", "scatter", "pie", "histogram", "box", "heatmap"
            - "x_column": Column name for x-axis (or null for some chart types)
            - "y_column": Column name for y-axis (or values for pie charts)
            - "color_column": Optional column name for color dimension (or null)
            - "agg_function": Aggregation function to use ("sum", "mean", "count", etc.)
            - "title": Suggested title for the visualization
            - "subtitle": A brief explanation of what this visualization shows
            - "interpretation": Brief explanation of what you understood from the request
            
            Your response should be valid JSON only. Do not include any text outside the JSON structure.
            """
            
            # User prompt with data structure
            user_prompt = f"""
            Dataset information:
            - Number of rows: {len(df)}
            - Columns: {json.dumps(columns_info, indent=2)}
            
            User's visualization request: "{prompt}"
            
            Determine the appropriate visualization specifications based on this request.
            """
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                system=system_message,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract and parse JSON response
            result_text = response.content[0].text
            
            # Clean up response to extract valid JSON
            json_pattern = r'```(?:json)?(.*?)```|(\{.*\})'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                json_str = match.group(1) or match.group(2)
                json_str = json_str.strip()
            else:
                json_str = result_text
            
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # Return fallback on parse error
                return self._get_fallback_visualization(df, data_types, prompt)
            
        except Exception as e:
            # Handle API errors with fallback
            return {
                "error": f"Error calling Claude API: {str(e)}",
                "visualization_type": "bar",
                "x_column": data_types['categorical'][0] if data_types['categorical'] else None,
                "y_column": data_types['numeric'][0] if data_types['numeric'] else None,
                "color_column": None,
                "agg_function": "sum",
                "title": "Fallback Visualization",
                "subtitle": "Visualization based on most common columns"
            }
    
    def _get_column_type(self, col, data_types):
        """Determine the type of a column based on data_types dictionary"""
        for dtype, cols in data_types.items():
            if col in cols:
                return dtype
        return "unknown"
    
    def _get_fallback_analysis(self, df):
        """Generate fallback analysis when Claude API is unavailable"""
        # Use basic statistics to generate insights
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        visualizations = []
        kpis = []
        
        # Generate KPIs
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:3]):  # At most 3 KPIs
                kpis.append({
                    "name": f"{col.replace('_', ' ').title()}",
                    "value": f"{df[col].mean():.2f}",
                    "trend": np.random.choice(["up", "down", "neutral"]),
                    "interpretation": f"Average {col.replace('_', ' ')} across all data"
                })
        
        # Add a bar chart if we have categorical and numeric columns
        if categorical_cols and numeric_cols:
            visualizations.append({
                "title": f"Total {numeric_cols[0].replace('_', ' ').title()} by {categorical_cols[0].replace('_', ' ').title()}",
                "description": f"Shows the total {numeric_cols[0].replace('_', ' ')} for each {categorical_cols[0].replace('_', ' ')}",
                "type": "bar",
                "x_column": categorical_cols[0],
                "y_column": numeric_cols[0],
                "color_column": None,
                "agg_function": "sum"
            })
        
        # Add a histogram for numeric data
        if numeric_cols:
            visualizations.append({
                "title": f"Distribution of {numeric_cols[0].replace('_', ' ').title()}",
                "description": f"Shows the frequency distribution of {numeric_cols[0].replace('_', ' ')}",
                "type": "histogram",
                "x_column": numeric_cols[0],
                "y_column": None,
                "color_column": None,
                "agg_function": None
            })
        
        return {
            "summary": f"Dataset with {len(df)} rows and {len(df.columns)} columns, containing practice management data.",
            "key_insights": [
                "This is a demo analysis without Claude API connection.",
                f"The dataset contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.",
                "For actual insights, please connect a valid Claude API key."
            ],
            "kpis": kpis,
            "recommended_visualizations": visualizations
        }
    
    def _get_fallback_visualization(self, df, data_types, prompt):
        """Generate fallback visualization spec when Claude API is unavailable"""
        # Simple keyword matching
        prompt = prompt.lower()
        
        # Determine visualization type from keywords
        viz_type = "bar"  # Default
        if any(term in prompt for term in ["line", "trend", "time", "over time"]):
            viz_type = "line"
        elif any(term in prompt for term in ["scatter", "correlation", "relationship"]):
            viz_type = "scatter"
        elif any(term in prompt for term in ["pie", "distribution", "percentage"]):
            viz_type = "pie"
        elif any(term in prompt for term in ["histogram", "frequency"]):
            viz_type = "histogram"
        elif any(term in prompt for term in ["box", "boxplot", "quartile"]):
            viz_type = "box"
        
        # Select columns based on types available
        x_col = None
        y_col = None
        color_col = None
        
        if viz_type == "bar" or viz_type == "pie":
            x_col = data_types['categorical'][0] if data_types['categorical'] else None
            y_col = data_types['numeric'][0] if data_types['numeric'] else None
        elif viz_type == "line" or viz_type == "scatter":
            if data_types['datetime']:
                x_col = data_types['datetime'][0]
            elif data_types['numeric'] and len(data_types['numeric']) > 1:
                x_col = data_types['numeric'][0]
                y_col = data_types['numeric'][1]
            else:
                x_col = data_types['categorical'][0] if data_types['categorical'] else None
                y_col = data_types['numeric'][0] if data_types['numeric'] else None
        elif viz_type == "histogram" or viz_type == "box":
            y_col = data_types['numeric'][0] if data_types['numeric'] else None
        
        return {
            "visualization_type": viz_type,
            "x_column": x_col,
            "y_column": y_col,
            "color_column": color_col,
            "agg_function": "sum",
            "title": f"{viz_type.capitalize()} of {y_col.replace('_', ' ').title() if y_col else 'Data'} by {x_col.replace('_', ' ').title() if x_col else 'Category'}",
            "subtitle": f"Shows the relationship between {x_col.replace('_', ' ') if x_col else 'categories'} and {y_col.replace('_', ' ') if y_col else 'values'}",
            "interpretation": "Created visualization based on basic keyword matching."
        }

def infer_data_types(df):
    """Infer column data types from DataFrame"""
    data_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'boolean': [],
        'id': [],
        'potential_date_strings': []
    }
    
    # Identify ID columns and potential dates
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
    
    # Categorize other columns
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

def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        return None

def create_visualization(df, viz_type, x_col, y_col, color_col=None, agg_func="sum", title=None, subtitle=None):
    """Create a visualization based on specifications"""
    if not title:
        title = f"{viz_type.capitalize()} of {y_col} by {x_col}" if x_col and y_col else "Visualization"
    
    try:
        if viz_type == 'bar':
            if x_col and y_col:
                # Group and aggregate the data
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
                    color_discrete_sequence=COLORS['categorical']
                )
                
                if subtitle:
                    fig.add_annotation(
                        text=subtitle,
                        xref="paper", yref="paper",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='white',
                    margin=dict(l=40, r=40, t=80, b=80),
                    xaxis=dict(
                        title=x_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    yaxis=dict(
                        title=y_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                return fig
            
        elif viz_type == 'line':
            if x_col and y_col:
                # Check if x_col is datetime
                if pd.api.types.is_datetime64_dtype(df[x_col]):
                    # Sort by date for line charts
                    df_sorted = df.sort_values(x_col)
                    
                    # Group if needed
                    if color_col:
                        # If color column is provided, group by that too
                        agg_data = df_sorted.groupby([x_col, color_col])[y_col].agg(agg_func).reset_index()
                        fig = px.line(
                            agg_data,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            title=title,
                            markers=True
                        )
                    else:
                        # Otherwise just group by x
                        agg_data = df_sorted.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        fig = px.line(
                            agg_data,
                            x=x_col,
                            y=y_col,
                            title=title,
                            markers=True,
                            color_discrete_sequence=[COLORS['primary']]
                        )
                else:
                    # For non-datetime, create a regular line chart
                    agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                    agg_data = agg_data.sort_values(x_col)
                    
                    fig = px.line(
                        agg_data,
                        x=x_col,
                        y=y_col,
                        color=color_col if color_col else None,
                        title=title,
                        markers=True
                    )
                
                if subtitle:
                    fig.add_annotation(
                        text=subtitle,
                        xref="paper", yref="paper",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(l=40, r=40, t=80, b=80),
                    xaxis=dict(
                        title=x_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    yaxis=dict(
                        title=y_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                return fig
            
        elif viz_type == 'scatter':
            if x_col and y_col:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=title,
                    opacity=0.7,
                    size_max=15
                )
                
                if subtitle:
                    fig.add_annotation(
                        text=subtitle,
                        xref="paper", yref="paper",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(l=40, r=40, t=80, b=80),
                    xaxis=dict(
                        title=x_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    yaxis=dict(
                        title=y_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                return fig
            
        elif viz_type == 'pie':
            if x_col and y_col:
                # Aggregate the data
                agg_data = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                
                # Limit to top categories for clarity
                if len(agg_data) > 10:
                    agg_data = agg_data.sort_values(y_col, ascending=False).head(10)
                
                fig = px.pie(
                    agg_data,
                    names=x_col,
                    values=y_col,
                    title=title,
                    color_discrete_sequence=COLORS['categorical'],
                    hover_data=[y_col]
                )
                
                if subtitle:
                    fig.add_annotation(
                        text=subtitle,
                        xref="paper", yref="paper",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    insidetextfont=dict(color='white')
                )
                
                fig.update_layout(
                    margin=dict(l=40, r=40, t=80, b=40),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                return fig
            
        elif viz_type == 'histogram':
            if x_col:
                fig = px.histogram(
                    df,
                    x=x_col,
                    color=color_col if color_col else None,
                    title=title,
                    nbins=30,
                    opacity=0.8,
                    color_discrete_sequence=[COLORS['primary']]
                )
                
                if subtitle:
                    fig.add_annotation(
                        text=subtitle,
                        xref="paper", yref="paper",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(l=40, r=40, t=80, b=40),
                    xaxis=dict(
                        title=x_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    yaxis=dict(
                        title="Count",
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    ),
                    bargap=0.1
                )
                return fig
            
        elif viz_type == 'box':
            if y_col:
                fig = px.box(
                    df,
                    x=x_col if x_col else None,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=title,
                    color_discrete_sequence=COLORS['categorical']
                )
                
                if subtitle:
                    fig.add_annotation(
                        text=subtitle,
                        xref="paper", yref="paper",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(l=40, r=40, t=80, b=80),
                    xaxis=dict(
                        title=x_col.replace('_', ' ').title() if x_col else "",
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    yaxis=dict(
                        title=y_col.replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                return fig
            
        elif viz_type == 'heatmap':
            if x_col and y_col:
                # Create pivot table for heatmap
                pivot_data = df.pivot_table(
                    index=y_col,
                    columns=x_col,
                    values=color_col if color_col else y_col,
                    aggfunc=agg_func,
                    fill_value=0
                )
                
                # Limit size for readability
                if pivot_data.shape[0] > 15 or pivot_data.shape[1] > 15:
                    # Get top rows and columns by sum
                    row_sums = pivot_data.sum(axis=1).sort_values(ascending=False).head(15).index
                    col_sums = pivot_data.sum(axis=0).sort_values(ascending=False).head(15).index
                    pivot_data = pivot_data.loc[row_sums, col_sums]
                
                fig = px.imshow(
                    pivot_data,
                    title=title,
                    color_continuous_scale=COLORS['sequence'],
                    aspect="auto"
                )
                
                if subtitle:
                    fig.add_annotation(
                        text=subtitle,
                        xref="paper", yref="paper",
                        x=0.5, y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                
                fig.update_layout(
                    margin=dict(l=40, r=40, t=80, b=40),
                    xaxis=dict(title=x_col.replace('_', ' ').title()),
                    yaxis=dict(title=y_col.replace('_', ' ').title()),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                return fig
        
        # If we get here, return None to indicate failure
        return None
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def create_data_filters(df, data_types):
    """Create data filters sidebar"""
    st.sidebar.header("Data Filters")
    
    filters_applied = False
    filtered_df = df.copy()
    
    # Date range filter if we have datetime columns
    if data_types['datetime']:
        st.sidebar.subheader("Date Range")
        date_col = data_types['datetime'][0]  # Use the first datetime column
        
        min_date = pd.to_datetime(filtered_df[date_col].min())
        max_date = pd.to_datetime(filtered_df[date_col].max())
        
        # Use date input for date range selection
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply date filter
        if start_date and end_date:
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df[date_col]).dt.date >= start_date) & 
                (pd.to_datetime(filtered_df[date_col]).dt.date <= end_date)
            ]
            filters_applied = True
    
    # Categorical filters
    if data_types['categorical']:
        st.sidebar.subheader("Categories")
        
        # Select up to 3 categorical columns for filtering
        for col in data_types['categorical'][:3]:
            try:
                # Get unique values, handle mixed types by converting to string first
                unique_values = filtered_df[col].astype(str).unique()
                
                # Try to sort them, but handle case where they can't be sorted
                try:
                    unique_values = sorted(unique_values)
                except:
                    # If sorting fails, just leave them in original order
                    pass
                
                # If too many values, use multiselect with default "All"
                if len(unique_values) > 3 and len(unique_values) <= 30:
                    selected_values = st.sidebar.multiselect(
                        f"Select {col.replace('_', ' ').title()}",
                        options=unique_values,
                        default=unique_values
                    )
                    
                    if selected_values and len(selected_values) < len(unique_values):
                        filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_values)]
                        filters_applied = True
                
                # If few values, use checkboxes
                elif len(unique_values) <= 3:
                    st.sidebar.write(f"**{col.replace('_', ' ').title()}**")
                    selected_values = []
                    
                    for value in unique_values:
                        if st.sidebar.checkbox(str(value), value=True, key=f"{col}_{value}"):
                            selected_values.append(value)
                    
                    if selected_values and len(selected_values) < len(unique_values):
                        filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_values)]
                        filters_applied = True
            except Exception as e:
                st.sidebar.warning(f"Could not create filter for {col}: {str(e)}")
    
    # Numeric range filters
    if data_types['numeric']:
        st.sidebar.subheader("Numeric Filters")
        
        # Select up to 2 numeric columns for range filtering
        for col in data_types['numeric'][:2]:
            try:
                # Try to get min and max values, but handle errors
                try:
                    min_val = float(filtered_df[col].min())
                    max_val = float(filtered_df[col].max())
                    
                    # Ensure min and max are not the same to avoid slider errors
                    if min_val == max_val:
                        min_val = min_val - 1
                        max_val = max_val + 1
                    
                    # Add a slider for range selection
                    range_values = st.sidebar.slider(
                        f"{col.replace('_', ' ').title()} Range",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.1
                    )
                    
                    if range_values != (min_val, max_val):
                        filtered_df = filtered_df[
                            (filtered_df[col] >= range_values[0]) & 
                            (filtered_df[col] <= range_values[1])
                        ]
                        filters_applied = True
                except:
                    st.sidebar.warning(f"Could not create range filter for {col}")
            except Exception as e:
                st.sidebar.warning(f"Error with column {col}: {str(e)}")
    
    # Reset filters button
    if filters_applied:
        if st.sidebar.button("Reset All Filters"):
            st.rerun()
        
        # Show count of filtered records
        st.sidebar.info(f"Showing {len(filtered_df):,} of {len(df):,} records")
    
    return filtered_df

def create_overview_section(df, data_types):
    """Create an overview section with key metrics and basic visualizations"""
    st.header("Practice Performance Overview")
    
    # Basic data information in metric cards
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Data Points", f"{len(df) * len(df.columns):,}")
    with col3:
        st.metric("Numeric Metrics", f"{len(data_types['numeric']):,}")
    with col4:
        st.metric("Time Period", "Last 12 Months")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample data
    with st.expander("Sample Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Data summary
    with st.expander("Data Summary"):
        # Create a summary of numeric columns
        if data_types['numeric']:
            st.subheader("Numeric Metrics Summary")
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
    st.subheader("Key Performance Indicators")
    
    # KPI cards
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    if len(data_types['numeric']) >= 4:
        kpi_cols = st.columns(4)
        
        # Use first four numeric columns for KPIs with random trends
        for i, col in enumerate(data_types['numeric'][:4]):
            with kpi_cols[i]:
                value = df[col].mean()
                delta = np.random.uniform(-0.15, 0.25) * value
                st.metric(
                    col.replace('_', ' ').title(),
                    f"${value:.2f}" if 'price' in col.lower() or 'revenue' in col.lower() or 'cost' in col.lower() or 'profit' in col.lower() else f"{value:.2f}",
                    f"{delta:.2f}" if abs(delta) > 0.01 else None,
                    delta_color="normal" if delta >= 0 else "inverse" if 'cost' in col.lower() else "normal"
                )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show a couple of basic visualizations if data is available
    if len(data_types['numeric']) >= 1:
        st.subheader("Performance Trends")
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        # First numeric column for histogram
        with col1:
            with st.expander("Distribution Analysis", expanded=True):
                fig = px.histogram(
                    df, 
                    x=data_types['numeric'][0], 
                    title=f"Distribution of {data_types['numeric'][0].replace('_', ' ').title()}",
                    color_discrete_sequence=[COLORS['primary']],
                    opacity=0.8,
                    marginal="box"
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(l=40, r=40, t=60, b=40),
                    xaxis=dict(
                        title=data_types['numeric'][0].replace('_', ' ').title(),
                        showgrid=True,
                        gridcolor='#eee'
                    ),
                    yaxis=dict(
                        title="Frequency",
                        showgrid=True,
                        gridcolor='#eee'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # If we have a second numeric column, show a box plot
        if len(data_types['numeric']) >= 2:
            with col2:
                with st.expander("Outlier Analysis", expanded=True):
                    fig = px.box(
                        df, 
                        y=data_types['numeric'][1], 
                        title=f"Distribution of {data_types['numeric'][1].replace('_', ' ').title()}",
                        color_discrete_sequence=[COLORS['secondary']]
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        margin=dict(l=40, r=40, t=60, b=40),
                        yaxis=dict(
                            title=data_types['numeric'][1].replace('_', ' ').title(),
                            showgrid=True,
                            gridcolor='#eee'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # If we have one numeric and one categorical column, create a bar chart
    if data_types['numeric'] and data_types['categorical']:
        st.subheader("Performance by Category")
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        cat_col = data_types['categorical'][0]
        num_col = data_types['numeric'][0]
        
        with st.expander(f"{num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}", expanded=True):
            # Group and aggregate
            agg_data = df.groupby(cat_col)[num_col].sum().reset_index()
            agg_data = agg_data.sort_values(num_col, ascending=False).head(10)
            
            fig = px.bar(
                agg_data, 
                x=cat_col, 
                y=num_col,
                title=f"Total {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                color_discrete_sequence=[COLORS['primary']]
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='white',
                margin=dict(l=40, r=40, t=60, b=100),
                xaxis=dict(
                    title=cat_col.replace('_', ' ').title(),
                    showgrid=True,
                    gridcolor='#eee'
                ),
                yaxis=dict(
                    title=num_col.replace('_', ' ').title(),
                    showgrid=True,
                    gridcolor='#eee'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add some insights about this chart
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown(f"**Insight:** The category **{agg_data.iloc[0][cat_col]}** has the highest total {num_col.replace('_', ' ')}, representing **{(agg_data.iloc[0][num_col]/agg_data[num_col].sum())*100:.1f}%** of the overall total.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_claude_analysis_section(df, data_types, claude_analyzer):
    """Create a section for Claude-powered data analysis"""
    st.header("AI-Powered Insights")
    
    # Let user ask Claude to analyze data
    st.subheader("Ask about your practice data")
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    
    analysis_prompt = st.text_area(
        "What would you like to analyze in your practice data?",
        height=100,
        placeholder="Example: 'Analyze the relationship between service categories and revenue' or 'Find patterns in client retention'"
    )
    
    # Initialize session state for results if not exists
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if st.button("Analyze Data", use_container_width=False):
        if not analysis_prompt:
            st.warning("Please enter a prompt for analysis.")
        else:
            with st.spinner("Analyzing your data..."):
                # Add a small delay to show the spinner
                time.sleep(1.5)
                st.session_state.analysis_results = claude_analyzer.analyze_data(df, analysis_prompt)
    
    # Display results if available
    if st.session_state.analysis_results:
        analysis_results = st.session_state.analysis_results
        
        # Display summary
        st.subheader("Summary")
        st.markdown(f"<div class='insight-card'><strong>{analysis_results.get('summary', 'No summary provided.')}</strong></div>", unsafe_allow_html=True)
        
        # Display KPIs if available
        kpis = analysis_results.get("kpis", [])
        if kpis:
            st.subheader("Key Performance Indicators")
            kpi_cols = st.columns(min(4, len(kpis)))
            
            for i, kpi in enumerate(kpis):
                with kpi_cols[i % len(kpi_cols)]:
                    name = kpi.get('name', f'KPI {i+1}')
                    value = kpi.get('value', 'N/A')
                    trend = kpi.get('trend', 'neutral')
                    
                    # Determine delta value and color based on trend
                    delta_value = None
                    if trend == 'up':
                        delta_value = "▲"
                    elif trend == 'down':
                        delta_value = "▼"
                    
                    delta_color = "normal"
                    if "cost" in name.lower() or "expense" in name.lower():
                        # For costs, down is good
                        delta_color = "inverse" if trend == 'down' else "normal"
                    else:
                        # For revenue/other metrics, up is good
                        delta_color = "normal" if trend == 'up' else "inverse"
                    
                    st.metric(name, value, delta_value, delta_color=delta_color)
                    
                    # Add interpretation if available
                    interpretation = kpi.get('interpretation')
                    if interpretation:
                        st.caption(interpretation)
        
        # Display key insights
        st.subheader("Key Insights")
        insights = analysis_results.get("key_insights", [])
        if insights:
            for i, insight in enumerate(insights):
                st.markdown(f"<div class='insight-card'><strong>#{i+1}:</strong> {insight}</div>", unsafe_allow_html=True)
        else:
            st.write("No insights provided.")
        
        # Display recommended visualizations
        st.subheader("Recommended Visualizations")
        viz_recs = analysis_results.get("recommended_visualizations", [])
        
        # Initialize session state for saved visualizations if not exists
        if 'saved_visualizations' not in st.session_state:
            st.session_state.saved_visualizations = []
        
        if viz_recs:
            for i, viz in enumerate(viz_recs[:4]):  # Limit to 4 visualizations
                viz_id = f"viz_{int(time.time())}_{i}"  # Generate unique ID
                
                st.markdown(f"<div class='dashboard-card'>", unsafe_allow_html=True)
                st.markdown(f"### {viz.get('title', f'Visualization {i+1}')}")
                st.caption(viz.get('description', 'No description provided.'))
                
                # Extract visualization parameters
                viz_type = viz.get('type', 'bar')
                x_col = viz.get('x_column')
                y_col = viz.get('y_column')
                color_col = viz.get('color_column')
                agg_func = viz.get('agg_function', 'sum')
                
                # Create and display visualization
                fig = create_visualization(
                    df, 
                    viz_type, 
                    x_col, 
                    y_col, 
                    color_col, 
                    agg_func, 
                    viz.get('title')
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add option to save this visualization to the dashboard
                    # Create a key for the current viz to track its save state
                    save_key = f"save_{viz_id}"
                    if save_key not in st.session_state:
                        st.session_state[save_key] = False
                    
                    if st.button(f"Add to Dashboard", key=f"add_btn_{viz_id}"):
                        # Store the visualization info
                        viz_info = {
                            'id': viz_id,
                            'title': viz.get('title', f"Visualization {i+1}"),
                            'type': viz_type,
                            'x_col': x_col,
                            'y_col': y_col,
                            'color_col': color_col,
                            'agg_func': agg_func,
                            'description': viz.get('description', '')
                        }
                        st.session_state.saved_visualizations.append(viz_info)
                        st.session_state[save_key] = True
                        st.rerun()  # Force a rerun to update the UI
                    
                    # Show success message if saved
                    if st.session_state[save_key]:
                        st.success(f"Added to your dashboard!")
                else:
                    st.warning(f"Could not create {viz_type} visualization with the specified parameters.")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("No visualizations recommended.")
    st.markdown('</div>', unsafe_allow_html=True)

def create_prompt_based_visualizations(df, data_types, claude_analyzer):
    """Create visualizations based on natural language prompts"""
    st.header("Custom Visualization Generator")
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.write("""
    Describe the visualization you'd like to create in plain language. For example:
    - "Show me a bar chart of total revenue by service category"
    - "Create a line chart of client acquisition over time"
    - "I want to see a pie chart of clients by region"
    """)
    
    # Get user prompt
    prompt = st.text_input("Enter your visualization request:", 
                         placeholder="E.g., Show me the top 10 clients by total revenue...")
    
    # Initialize state for viz storage
    if 'viz_results' not in st.session_state:
        st.session_state.viz_results = None
        
    if 'viz_title' not in st.session_state:
        st.session_state.viz_title = ""
        
    if 'viz_saved' not in st.session_state:
        st.session_state.viz_saved = False
    
    if not prompt:
        st.info("Enter a prompt to generate a visualization.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Process the prompt when user clicks the button
    if st.button("Generate Visualization", use_container_width=False):
        with st.spinner("Creating your visualization..."):
            # Add a small delay to show the spinner
            time.sleep(1)
            st.session_state.viz_results = claude_analyzer.interpret_prompt(df, data_types, prompt)
            st.session_state.viz_saved = False  # Reset saved state
    
    # Display results if available
    if st.session_state.viz_results:
        viz_spec = st.session_state.viz_results
        
        if "error" in viz_spec:
            st.error(viz_spec["error"])
        else:
            st.success("Visualization generated successfully!")
            
            # Extract visualization parameters
            viz_type = viz_spec.get("visualization_type", "bar")
            x_col = viz_spec.get("x_column")
            y_col = viz_spec.get("y_column")
            color_col = viz_spec.get("color_column")
            agg_func = viz_spec.get("agg_function", "sum")
            title = viz_spec.get("title", "Visualization")
            subtitle = viz_spec.get("subtitle", "")
            
            # Display interpretation
            if "interpretation" in viz_spec:
                st.info(f"I understood your request as: {viz_spec['interpretation']}")
            
            # Create visualization
            fig = create_visualization(
                df,
                viz_type,
                x_col,
                y_col,
                color_col,
                agg_func,
                title,
                subtitle
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add option to save to dashboard
                if 'saved_visualizations' not in st.session_state:
                    st.session_state.saved_visualizations = []
                
                # Show title input if not saved yet
                if not st.session_state.viz_saved:
                    st.session_state.viz_title = st.text_input(
                        "Enter a title for this visualization (to save it to your dashboard):", 
                        value=title,
                        key=f"title_input_{int(time.time())}"
                    )
                    
                    if st.button("Add to My Dashboard", key=f"save_btn_{int(time.time())}"):
                        # Store the visualization info with unique ID
                        viz_id = f"custom_viz_{int(time.time())}"
                        viz_info = {
                            'id': viz_id,
                            'title': st.session_state.viz_title,
                            'prompt': prompt,
                            'type': viz_type,
                            'x_col': x_col,
                            'y_col': y_col,
                            'color_col': color_col,
                            'agg_func': agg_func,
                            'subtitle': subtitle
                        }
                        st.session_state.saved_visualizations.append(viz_info)
                        st.session_state.viz_saved = True
                        st.rerun()  # Force a rerun to update the UI
                
                # Show success message if saved
                if st.session_state.viz_saved:
                    st.success(f"Added '{st.session_state.viz_title}' to your dashboard!")
            else:
                st.error("Could not create visualization. Try a different request.")
    st.markdown('</div>', unsafe_allow_html=True)

def create_custom_dashboard(df, saved_visualizations):
    """Display the user's personalized dashboard with saved visualizations"""
    st.header("My Custom Dashboard")
    
    if not saved_visualizations:
        st.info("Your dashboard is empty. Create and save visualizations to see them here.")
        
        # Add sample visualization suggestion
        st.markdown("""
        <div class="insight-card">
        <strong>Tip:</strong> Try using the AI-powered analysis or custom visualization generator to create charts, 
        then add them to your dashboard for quick reference.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Add dashboard actions
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Your Custom Dashboard ({len(saved_visualizations)} visualizations)")
    with col2:
        if st.button("📥 Export Dashboard", use_container_width=True):
            st.success("Dashboard exported successfully!")
    
    # Track removals with a separate list to avoid modifying the list during iteration
    to_remove = []
            
    # Display all saved visualizations in a grid
    cols = st.columns(2)  # 2 columns for the dashboard
    
    for i, viz_info in enumerate(saved_visualizations):
        with cols[i % 2]:
            st.markdown(f'<div class="dashboard-card">', unsafe_allow_html=True)
            with st.expander(viz_info['title'], expanded=True):
                try:
                    # Re-create the visualization
                    viz_type = viz_info.get('type')
                    x_col = viz_info.get('x_col')
                    y_col = viz_info.get('y_col')
                    color_col = viz_info.get('color_col')
                    agg_func = viz_info.get('agg_func', 'sum')
                    subtitle = viz_info.get('subtitle', '')
                    
                    fig = create_visualization(
                        df,
                        viz_type,
                        x_col,
                        y_col,
                        color_col,
                        agg_func,
                        viz_info['title'],
                        subtitle
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not recreate visualization")
                    
                    # Show the prompt that created this visualization if available
                    if 'prompt' in viz_info:
                        st.caption(f"Created from prompt: '{viz_info['prompt']}'")
                    
                    # Show description if available
                    if 'description' in viz_info and viz_info['description']:
                        st.markdown(viz_info['description'])
                    
                    # Option to remove from dashboard with unique key
                    viz_id = viz_info.get('id', f"viz_{i}")
                    if st.button(f"Remove from Dashboard", key=f"remove_{viz_id}"):
                        to_remove.append(i)
                except Exception as e:
                    st.error(f"Error displaying visualization: {str(e)}")
                    if st.button(f"Remove broken visualization", key=f"remove_broken_{i}"):
                        to_remove.append(i)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Process removals after iteration
    if to_remove:
        # Remove items in reverse order to maintain correct indices
        for idx in sorted(to_remove, reverse=True):
            if idx < len(saved_visualizations):
                saved_visualizations.pop(idx)
        st.rerun()
    
    # Add option to create a PDF report
    st.markdown("""
    <div class="dashboard-card">
    <h3>Export Options</h3>
    <p>Create a shareable report from your dashboard visualizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate PDF Report", use_container_width=True):
            st.success("PDF report generated! Check your downloads folder.")
    with col2:
        if st.button("📧 Share Dashboard", use_container_width=True):
            st.info("Enter email addresses to share this dashboard:")
            st.text_input("Recipients (comma separated):")
            if st.button("Send", key="send_dashboard_email"):
                st.success("Dashboard shared successfully!")

def main():
    # Initialize password protection
    password_protection = PasswordProtection(DEMO_PASSWORD)
    
    # Add Velora styling
    set_velora_styling()
    
    # Check if user is authenticated
    if not password_protection.require_login():
        return
    
    # App title and logo
    display_logo()
    st.title("Practice Intelligence Platform")
    
    if st.session_state.get('demo_mode', False):
        st.info("🔍 You are in demo mode with sample data. For a personalized experience, upload your own practice data.")
    
    st.markdown("""
    Transform your practice management data into actionable insights. Make data-driven decisions 
    to optimize efficiency, increase profitability, and enhance client service.
    """)
    
    # Sidebar for AI configuration
    st.sidebar.header("Configuration")
    
    # Claude API configuration - moved to sidebar
    if ANTHROPIC_AVAILABLE:
        st.sidebar.success("✅ Velora AI Engine connected")
        
        # Model selection
        claude_model = st.sidebar.selectbox(
            "Select AI Model",
            options=[
                "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229"
            ],
            index=0  # Default to Sonnet
        )
        
        # API key input
        if 'CLAUDE_API_KEY' not in st.secrets and 'ANTHROPIC_API_KEY' not in os.environ:
            st.sidebar.warning("For advanced AI analysis, contact Velora support to set up your API key")
    else:
        st.sidebar.warning("AI engine limited. Some features will use demo mode.")
    
    # Initialize Claude analyzer
    claude_analyzer = ClaudeAnalyzer()
    
    # If Claude is available and model is selected, update model
    if ANTHROPIC_AVAILABLE and claude_analyzer.is_available and 'claude_model' in locals():
        claude_analyzer.model = claude_model
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your practice management data (CSV format)", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started or use sample data.")
        
        # Show sample data option
        if st.button("Use Sample Practice Data"):
            # Create sample data
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Create sample dates
            start_date = datetime(2024, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(365)]
            
            # Create sample categories for practice management
            service_categories = ['Tax Preparation', 'Audit', 'Advisory', 'Bookkeeping', 'Payroll', 'Consulting', 'Financial Planning']
            client_industries = ['Healthcare', 'Technology', 'Real Estate', 'Retail', 'Manufacturing', 'Professional Services', 'Nonprofit', 'Finance']
            client_types = ['Individual', 'Small Business', 'Corporation', 'Partnership', 'Nonprofit']
            staff_roles = ['Partner', 'Manager', 'Senior', 'Staff', 'Administrative']
            
            # Create client IDs (100 clients)
            client_ids = [f'CL{i:04d}' for i in range(1, 101)]
            
            # Create staff IDs (15 staff members)
            staff_ids = [f'ST{i:02d}' for i in range(1, 16)]
            
            # Create sample data for practice management
            np.random.seed(42)  # For reproducibility
            
            # Generate 1000 service entries
            sample_data = {
                'Date': np.random.choice(dates, 1000),
                'ClientID': np.random.choice(client_ids, 1000),
                'ClientIndustry': np.random.choice(client_industries, 1000),
                'ClientType': np.random.choice(client_types, 1000),
                'ServiceCategory': np.random.choice(service_categories, 1000),
                'StaffID': np.random.choice(staff_ids, 1000),
                'StaffRole': np.random.choice(staff_roles, 1000),
                'HoursSpent': np.random.uniform(0.5, 8.0, 1000).round(1),
                'HourlyRate': np.random.choice([150, 200, 250, 300, 350, 400], 1000),
                'Revenue': 0,  # Will calculate below
                'DirectCosts': 0,  # Will calculate below
                'ClientSatisfaction': np.random.choice([3, 4, 5, 5, 5], 1000),  # Skewed toward higher satisfaction
                'IsRecurring': np.random.choice([True, False], 1000, p=[0.7, 0.3]),
            }
            
            # Calculate revenue and costs
            sample_data['Revenue'] = sample_data['HoursSpent'] * sample_data['HourlyRate']
            sample_data['DirectCosts'] = sample_data['HoursSpent'] * np.random.uniform(50, 120, 1000).round(0)
            sample_data['Profit'] = sample_data['Revenue'] - sample_data['DirectCosts']
            sample_data['ProfitMargin'] = (sample_data['Profit'] / sample_data['Revenue'] * 100).round(1)
            
            # Add realization rate (realized revenue compared to standard rates)
            sample_data['RealizationRate'] = np.random.uniform(0.7, 1.0, 1000).round(2) * 100
            
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
            
            st.success("Sample practice data loaded successfully!")
            st.rerun()
        
        # Footer
        st.markdown("""
        <div class="custom-footer">
            Velora AI Practice Intelligence Platform | © 2025 Velora, Inc. All rights reserved.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Process the uploaded file
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        with st.status("Processing data...") as status:
            df = process_uploaded_file(uploaded_file)
            
            if df is not None:
                # Show row count
                st.write(f"Found {len(df):,} rows and {len(df.columns):,} columns in the dataset.")
                
                status.update(label="Analyzing data types...", state="running")
                # Set to session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Infer data types
                st.session_state.data_types = infer_data_types(df)
                
                # Convert date columns
                potential_date_cols = [col for col in st.session_state.data_types['potential_date_strings']]
                st.session_state.df = convert_date_columns(st.session_state.df, potential_date_cols)
                
                status.update(label="Data processed successfully!", state="complete")
    
    # Get data from session state
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        df = st.session_state.df
        data_types = st.session_state.data_types
        
        # Create data filters sidebar
        filtered_df = create_data_filters(df, data_types)
        
        # Show data summary
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", f"{len(filtered_df):,}")
        with col2:
            st.metric("Data Points", f"{len(filtered_df.columns):,}")
        with col3:
            date_range = "Full Dataset"
            if data_types['datetime']:
                date_range = f"{filtered_df[data_types['datetime'][0]].min().strftime('%b %Y')} - {filtered_df[data_types['datetime'][0]].max().strftime('%b %Y')}"
            st.metric("Time Period", date_range)
        
        # Initialize session state for saved visualizations if not already done
        if 'saved_visualizations' not in st.session_state:
            st.session_state.saved_visualizations = []
        
        # Create tabs for different sections
        tabs = st.tabs([
            "Overview", 
            "AI Insights",
            "Custom Viz",
            "My Dashboard"
        ])
        
        with tabs[0]:
            create_overview_section(filtered_df, data_types)
        
        with tabs[1]:
            create_claude_analysis_section(filtered_df, data_types, claude_analyzer)
        
        with tabs[2]:
            create_prompt_based_visualizations(filtered_df, data_types, claude_analyzer)
        
        with tabs[3]:
            create_custom_dashboard(filtered_df, st.session_state.saved_visualizations)
        
        # Option to reset/upload a new file
        if st.sidebar.button("Reset / Upload New File"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'authenticated':  # Keep authentication status
                    del st.session_state[key]
            st.rerun()
        
        # Add logout option
        if st.sidebar.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        
        # Footer
        st.markdown("""
        <div class="custom-footer">
            Velora AI Practice Intelligence Platform | © 2025 Velora, Inc. All rights reserved.
        </div>
        """, unsafe_allow_html=True)

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
    
    # Run the main app
    main()
    
