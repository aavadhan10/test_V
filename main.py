import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
import re
from datetime import datetime

# Import Anthropic library with error handling
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    st.warning("Anthropic library not available. Install with: pip install anthropic")

# Set page configuration
st.set_page_config(
    page_title="Data Dashboard Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 28px;}
    div[data-testid="stMetricLabel"] {font-size: 16px;}
</style>
""", unsafe_allow_html=True)

# Define standard color scheme
COLORS = {
    'primary': '#4e8df5',     # Blue
    'secondary': '#4CAF50',   # Green
    'accent': '#FF9800',      # Orange
    'neutral': '#607D8B',     # Blue-grey
    'sequence': px.colors.sequential.Blues,
    'categorical': px.colors.qualitative.Safe,
    'diverging': px.colors.diverging.RdBu,
}

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
                st.sidebar.success("✅ Claude API connected")
            except Exception as e:
                st.sidebar.error(f"Claude API initialization error: {type(e).__name__}")
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
            
            Respond ONLY with valid JSON. Do not include any explanations or text outside the JSON structure.
            """
            
            # User prompt with data
            user_prompt = f"""
            Here is a CSV dataset:
            
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
                    "recommended_visualizations": []
                }
            
        except Exception as e:
            # Handle any API errors
            return {
                "error": f"Error calling Claude API: {str(e)}",
                "summary": "Error in analysis",
                "key_insights": ["API error occurred"],
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
            You are an expert data visualization assistant. Given information about a dataset
            and a natural language request, determine the appropriate visualization specifications.
            
            Return ONLY a JSON object with these fields:
            - "visualization_type": One of "bar", "line", "scatter", "pie", "histogram", "box", "heatmap"
            - "x_column": Column name for x-axis (or null for some chart types)
            - "y_column": Column name for y-axis (or values for pie charts)
            - "color_column": Optional column name for color dimension (or null)
            - "agg_function": Aggregation function to use ("sum", "mean", "count", etc.)
            - "title": Suggested title for the visualization
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
                "title": "Fallback Visualization"
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
        
        # Add a bar chart if we have categorical and numeric columns
        if categorical_cols and numeric_cols:
            visualizations.append({
                "title": f"Sum of {numeric_cols[0]} by {categorical_cols[0]}",
                "description": f"Shows the total {numeric_cols[0]} for each {categorical_cols[0]}",
                "type": "bar",
                "x_column": categorical_cols[0],
                "y_column": numeric_cols[0],
                "color_column": None,
                "agg_function": "sum"
            })
        
        # Add a histogram for numeric data
        if numeric_cols:
            visualizations.append({
                "title": f"Distribution of {numeric_cols[0]}",
                "description": f"Shows the frequency distribution of {numeric_cols[0]}",
                "type": "histogram",
                "x_column": numeric_cols[0],
                "y_column": None,
                "color_column": None,
                "agg_function": None
            })
        
        return {
            "summary": f"Dataset with {len(df)} rows and {len(df.columns)} columns.",
            "key_insights": [
                "This is a demo analysis without Claude API connection.",
                f"The dataset contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.",
                "For actual insights, please connect a valid Claude API key."
            ],
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
            "title": f"Visualization of {y_col} by {x_col}" if x_col and y_col else "Data Visualization",
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

def create_visualization(df, viz_type, x_col, y_col, color_col=None, agg_func="sum", title=None):
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
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig.update_layout(xaxis_tickangle=-45)
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
                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col if color_col else None,
                        title=title,
                        markers=True
                    )
                return fig
            
        elif viz_type == 'scatter':
            if x_col and y_col:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=title
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
                    title=title
                )
                return fig
            
        elif viz_type == 'histogram':
            if x_col:
                fig = px.histogram(
                    df,
                    x=x_col,
                    color=color_col if color_col else None,
                    title=title
                )
                return fig
            
        elif viz_type == 'box':
            if y_col:
                fig = px.box(
                    df,
                    x=x_col if x_col else None,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=title
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
                    color_continuous_scale=COLORS['sequence']
                )
                return fig
        
        # If we get here, return None to indicate failure
        return None
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def create_overview_section(df, data_types):
    """Create an overview section with key metrics and basic visualizations"""
    st.header("Data Overview")
    
    # Basic data information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns):,}")
    with col3:
        st.metric("Numeric Columns", f"{len(data_types['numeric']):,}")
    with col4:
        st.metric("Categorical Columns", f"{len(data_types['categorical']):,}")
    
    # Sample data
    with st.expander("Sample Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Data summary
    with st.expander("Data Summary"):
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
    
    # Show a couple of basic visualizations if data is available
    if len(data_types['numeric']) >= 1:
        col1, col2 = st.columns(2)
        
        # First numeric column for histogram
        with col1:
            with st.expander(f"Distribution of {data_types['numeric'][0]}", expanded=True):
                fig = px.histogram(
                    df, 
                    x=data_types['numeric'][0], 
                    title=f"Distribution of {data_types['numeric'][0]}",
                    color_discrete_sequence=[COLORS['primary']]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # If we have a second numeric column, show a box plot
        if len(data_types['numeric']) >= 2:
            with col2:
                with st.expander(f"Box Plot of {data_types['numeric'][1]}", expanded=True):
                    fig = px.box(
                        df, 
                        y=data_types['numeric'][1], 
                        title=f"Box Plot of {data_types['numeric'][1]}",
                        color_discrete_sequence=[COLORS['secondary']]
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # If we have one numeric and one categorical column, create a bar chart
    if data_types['numeric'] and data_types['categorical']:
        cat_col = data_types['categorical'][0]
        num_col = data_types['numeric'][0]
        
        with st.expander(f"{num_col} by {cat_col}", expanded=True):
            # Group and aggregate
            agg_data = df.groupby(cat_col)[num_col].sum().reset_index()
            agg_data = agg_data.sort_values(num_col, ascending=False).head(10)
            
            fig = px.bar(
                agg_data, 
                x=cat_col, 
                y=num_col,
                title=f"Sum of {num_col} by {cat_col}",
                color_discrete_sequence=[COLORS['primary']]
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

def create_claude_analysis_section(df, data_types, claude_analyzer):
    """Create a section for Claude-powered data analysis"""
    st.header("AI-Powered Data Analysis")
    
    # Let user ask Claude to analyze data
    st.subheader("Ask about your data")
    
    analysis_prompt = st.text_area(
        "What would you like to analyze in your data?",
        height=100,
        placeholder="Example: 'Analyze the relationship between categories and sales' or 'Find interesting patterns in this dataset'"
    )
    
    if st.button("Analyze Data"):
        if not analysis_prompt:
            st.warning("Please enter a prompt for analysis.")
        else:
            with st.spinner("Analyzing your data..."):
                analysis_results = claude_analyzer.analyze_data(df, analysis_prompt)
                
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
                        else:
                            st.warning(f"Could not create {viz_type} visualization with the specified parameters.")
                        
                        # Add option to save this visualization to the dashboard
                        if 'saved_visualizations' not in st.session_state:
                            st.session_state.saved_visualizations = []
                        
                        if st.button(f"Add to Dashboard", key=f"add_viz_{i}"):
                            # Store the visualization info
                            st.session_state.saved_visualizations.append({
                                'title': viz.get('title', f"Visualization {i+1}"),
                                'type': viz_type,
                                'x_col': x_col,
                                'y_col': y_col,
                                'color_col': color_col,
                                'agg_func': agg_func,
                                'description': viz.get('description', '')
                            })
                            st.success(f"Added to your dashboard!")
                
                else:
                    st.write("No visualizations recommended.")

def create_prompt_based_visualizations(df, data_types, claude_analyzer):
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
        with st.spinner("Interpreting your request..."):
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
                
                # Create visualization
                fig = create_visualization(
                    df,
                    viz_type,
                    x_col,
                    y_col,
                    color_col,
                    agg_func,
                    title
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add option to save to dashboard
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
                else:
                    st.error("Could not create visualization. Try a different request.")

def create_dashboard(df, saved_visualizations):
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
                # Re-create the visualization
                viz_type = viz_info.get('type')
                x_col = viz_info.get('x_col')
                y_col = viz_info.get('y_col')
                color_col = viz_info.get('color_col')
                agg_func = viz_info.get('agg_func', 'sum')
                
                fig = create_visualization(
                    df,
                    viz_type,
                    x_col,
                    y_col,
                    color_col,
                    agg_func,
                    viz_info['title']
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not recreate visualization")
                
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
    st.title("Data Dashboard Generator")
    st.markdown("""
    Upload any CSV file to automatically generate visualizations and create your own custom dashboard.
    Ask questions about your data in plain language!
    """)
    
    # Sidebar for AI configuration
    st.sidebar.header("Configuration")
    
    # Claude API configuration - moved to sidebar
    if ANTHROPIC_AVAILABLE:
        st.sidebar.info("Claude API is available for advanced analysis.")
        
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
            st.sidebar.warning("For AI analysis, set up CLAUDE_API_KEY in .streamlit/secrets.toml")
    else:
        st.sidebar.warning("Anthropic library not found. Some features will be limited.")
    
    # Initialize Claude analyzer
    claude_analyzer = ClaudeAnalyzer()
    
    # If Claude is available and model is selected, update model
    if ANTHROPIC_AVAILABLE and claude_analyzer.is_available and 'claude_model' in locals():
        claude_analyzer.model = claude_model
    
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
        
        # Display data overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", f"{len(df.columns):,}")
        with col3:
            st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Initialize session state for saved visualizations if not already done
        if 'saved_visualizations' not in st.session_state:
            st.session_state.saved_visualizations = []
        
        # Create tabs for different sections
        tabs = st.tabs([
            "Overview", 
            "AI Analysis",
            "Prompt-Based Viz",
            "My Dashboard"
        ])
        
        with tabs[0]:
            create_overview_section(df, data_types)
        
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
    
    # Display version info in footer
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #f0f2f6; font-size: 12px;">
        Data Dashboard Generator v1.0 | Created 2025
    </div>
    """, unsafe_allow_html=True)
    
    # Run the main app
    main()
