import pandas as pd
from utils.openai_helpers import get_openai_response

def generate_data_manipulation_prompt(query: str, df: pd.DataFrame) -> str:
    """
    Generates a prompt for data manipulation based on the user query and dataframe structure
    """
    columns_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])

    system_prompt = {
    "role": "system",
    "content": """Generate Python code using pandas and plotly for data visualization.
    Follow these guidelines:

    1. Data Preparation:
        - Use pandas groupby() and agg() for aggregations
        - Always reset_index() after groupby
        - Handle NaN values with dropna()
        - Ensure proper numeric dtypes
        - Sort data if needed using sort_values()

    2. Plotly Visualization:
        - Import: import plotly.express as px
        - For complex plots: import plotly.graph_objects as go
        - Common plots:
            * Line: px.line(df, x='x_col', y='y_col')
            * Bar: px.bar(df, x='x_col', y='y_col')
            * Scatter: px.scatter(df, x='x_col', y='y_col')
            * Box: px.box(df, x='x_col', y='y_col')
            * Histogram: px.histogram(df, x='x_col')
            * Heatmap: px.imshow(df_pivot)
            * Pie: px.pie(df, values='value_col', names='category_col')
        - Add titles, labels, and color schemes
        - Use fig.update_layout() for customization
        - Return the figure object

    Example:
    ```
    import plotly.express as px
    
    # Data preparation
    df = df.dropna(subset=['category', 'value'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df_agg = df.groupby('category')['value'].mean().reset_index()
    
    # Create plot
    fig = px.bar(df_agg, x='category', y='value',
                 title='Category Analysis',
                 labels={'category': 'Categories', 'value': 'Average Value'},
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Customize layout
    fig.update_layout(
        showlegend=True,
        xaxis_tickangle=-45,
        template='plotly_white'
    )
    
    return fig
    ```

    Return only the executable Python code without any explanation.""" }
    user_prompt = {
        "role": "user",
        "content": f"""
        Query: {query}
    
        DataFrame Information:
        1. Columns and Data Types:
        {columns_info}
    
        2. Sample Data (first 3 rows):
        {df.head(3).to_string()}
    
        3. Data Statistics:
        - Total Rows: {len(df)}
        - Missing Values: {df.isnull().sum().to_dict()}
        
        Requirements:
        1. Generate Python code using plotly for the requested visualization
        2. Handle data preparation including:
            - Missing value treatment
            - Proper data type conversions
            - Necessary aggregations
            - Data sorting if needed
        3. Include appropriate:
            - Chart title
            - Axis labels
            - Color schemes
            - Legend (if applicable)
            - Interactive features
        4. Return a plotly figure object
        
        Note: Ensure the code handles numeric columns properly and includes error handling for data type conversions.
        """ }


    response = get_openai_response([system_prompt, user_prompt])
    # Ensure we return only the code part if it's wrapped in backticks
    code = response.strip('`\n ')
    if code.startswith('python'):
        code = code[6:]
    return code
