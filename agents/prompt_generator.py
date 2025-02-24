import pandas as pd
from utils.openai_helpers import get_openai_response

def generate_data_manipulation_prompt(query: str, df: pd.DataFrame) -> str:
    """
    Generates a prompt for data manipulation based on the user query and dataframe structure
    """
    columns_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])

    system_prompt = {
    "role": "system",
    "content": """
    As a Data Science Expert, generate production-ready Python code for data visualization following these guidelines:

    ## 1. Data Preparation Best Practices
    
    ### Data Cleaning:
    - Handle missing values contextually:
        ```
        # For complete row removal
        df = df.dropna()
        # For specific columns
        df = df.dropna(subset=['important_column'])
        # For filling values
        df = df.fillna({'numeric_col': 0, 'categorical_col': 'Unknown'})
        ```
    
    ### Type Conversion:
    - Ensure proper data types:
        ```
        # Numeric conversion with error handling
        df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')
        # Datetime conversion
        df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce')
        # Category conversion
        df['category_col'] = df['category_col'].astype('category')
        ```

    ## 2. Data Transformation
    
    ### Aggregation Operations:
    ```
    # Single aggregation
    df_agg = df.groupby('category')['value'].mean().reset_index()
    
    # Multiple aggregations
    df_agg = df.groupby('category').agg({
        'value1': 'mean',
        'value2': 'sum',
        'value3': ['min', 'max']
    }).reset_index()
    ```

    ### Data Filtering:
    ```
    # Conditional filtering
    df_filtered = df[df['value'] > df['value'].mean()]
    
    # Multiple conditions
    mask = (df['value'] > 100) & (df['category'] == 'A')
    df_filtered = df[mask]
    ```

    ## 3. Visualization Preparation
    
    ### For Time Series:
    ```
    # Resample and aggregate time data
    df_time = df.set_index('date_column')
    df_daily = df_time.resample('D')['value'].mean()
    ```

    ### For Categorical Analysis:
    ```
    # Value counts with normalization
    df_cats = df['category'].value_counts(normalize=True)
    
    # Cross tabulation
    df_cross = pd.crosstab(df['cat1'], df['cat2'])
    ```

    ## 4. Advanced Operations
    
    ### Feature Engineering:
    ```
    # Creating derived features
    df['ratio'] = df['value1'] / df['value2']
    df['category_encoded'] = pd.get_dummies(df['category'])
    ```

    ### Statistical Preparation:
    ```
    # Calculate percentiles
    df['percentile'] = df['value'].rank(pct=True)
    
    # Z-score normalization
    df['value_normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
    ```

    ## 5. Code Quality Requirements:
    - Include error handling
    - Add data validation steps
    - Use efficient operations
    - Follow PEP 8 style guidelines
    - Add essential comments
    - Return clean, processed data

    Return production-ready Python code that follows these guidelines and includes appropriate error handling and validation steps.
    """
}


    user_prompt = {
        "role": "user",
        "content": f"""
        Query: {query}
    
        DataFrame Analysis Requirements:
        1. Data Overview:
            Columns and Types:
            {columns_info}
            
            Sample Data (10 rows):
            {df.head(10).to_string()}
            
            Dataset Statistics:
            - Total Records: {len(df)}
            - Missing Values: {df.isnull().sum().to_dict()}
            - Numeric Columns: {df.select_dtypes(include=['int64', 'float64']).columns.tolist()}
            - Categorical Columns: {df.select_dtypes(include=['object', 'category']).columns.tolist()}
    
        2. Analysis Steps Required:
            a) Data Cleaning:
                - Handle missing values appropriately (drop/fill)
                - Convert data types if needed
                - Remove duplicates if necessary
                - Handle outliers if relevant
            
            b) Data Transformation:
                - Perform required aggregations
                - Create derived features if needed
                - Apply filters based on query
                - Sort/order data as needed
            
            c) Visualization:
                - Use appropriate plotly chart type
                - Include meaningful title and labels
                - Add interactive features
                - Ensure color schemes are informative
                - Include hover information
                - Format axes and legends properly
    
        3. Validation Checklist:
            - Verify data transformations match query intent
            - Ensure visualization answers the specific question
            - Check for data integrity in final output
            - Confirm all relevant data points are included
            - Validate aggregation logic
    
        Generate production-ready Python code that:
        1. Follows best practices for data analysis
        2. Includes proper error handling
        3. Uses efficient pandas operations
        4. Creates interactive plotly visualizations
        5. Returns a plotly figure object
        
        The code should directly address: {query}
        """
    }


    response = get_openai_response([system_prompt, user_prompt])
    # Ensure we return only the code part if it's wrapped in backticks
    code = response.strip('`\n ')
    if code.startswith('python'):
        code = code[6:]
    return code
