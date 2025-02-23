import pandas as pd

def process_dataframe(manipulation_code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the generated data manipulation code on the dataframe
    """
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Ensure 'House Price' column is numeric
        if 'House Price' in df_copy.columns:
            try:
                df_copy['House Price'] = pd.to_numeric(df_copy['House Price'].str.replace(',', ''), errors='coerce')
            except Exception as e:
                print(f"Error converting House Price to numeric: {str(e)}")

        # Ensure other numeric columns are properly typed
        for col in df_copy.columns:
            if col != 'House Price' and df_copy[col].dtype == object:
                try:
                    # Remove commas and convert to numeric
                    df_copy[col] = pd.to_numeric(df_copy[col].astype(str).str.replace(',', ''), errors='coerce')
                except Exception as e:
                    print(f"Error converting column {col} to numeric: {str(e)}")

        # Drop rows with NaN in House Price if it exists
        if 'House Price' in df_copy.columns:
            df_copy = df_copy.dropna(subset=['House Price'])

        # Create local namespace with only necessary variables
        local_vars = {"df": df_copy, "pd": pd}

        # If no manipulation code provided, return the cleaned dataframe
        if not manipulation_code or manipulation_code.isspace():
            return df_copy

        # Execute the manipulation code
        try:
            exec(manipulation_code, globals(), local_vars)
        except Exception as e:
            print(f"Error executing manipulation code: {str(e)}")
            print(f"Code attempted: {manipulation_code}")
            # Return the cleaned dataframe if manipulation fails
            return df_copy

        # The manipulation code should modify 'df' in the local namespace
        processed_df = local_vars.get('df')

        if processed_df is None:
            print("Warning: Manipulation code did not produce a valid dataframe")
            return df_copy

        return processed_df
    except Exception as e:
        raise Exception(f"Error processing dataframe: {str(e)}\nCode attempted: {manipulation_code}")