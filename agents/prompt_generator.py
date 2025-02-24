
import pandas as pd
from utils.openai_helpers import get_openai_response

def generate_data_manipulation_prompt(query: str, df: pd.DataFrame) -> str:
    """
    Generates a prompt for data manipulation based on the user query and dataframe structure
    """
    columns_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])

    system_prompt = {
        "role": "system",
        "content": """Generate Python code using pandas to prepare data for visualization.
        For charts:
        1. If aggregation is needed, use groupby() and agg() functions
        2. Make sure to reset_index() after groupby operations
        3. Handle any NaN values using dropna()
        4. Ensure numeric columns are properly typed

        Example for a plot showing average values:
        ```python
        df = df.dropna(subset=['category_column', 'value_column'])
        df['value_column'] = pd.to_numeric(df['value_column'], errors='coerce')
        df = df.groupby('category_column')['value_column'].mean().reset_index()
        ```

        Return only the valid Python code without any explanation or formatting. assuming df is already declared, give only manipulation and Visualisation code.
        '}
        """
    }

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
        """
    }


    response = get_openai_response([system_prompt, user_prompt])
    # Ensure we return only the code part if it's wrapped in backticks
    code = response.strip('`\n ')
    if code.startswith('python'):
        code = code[6:]
    return code
