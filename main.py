import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
import plotly.express as px
import json

from agents.classifier import classify_query
from agents.prompt_generator import generate_data_manipulation_prompt
from agents.visualization import create_visualization
from utils.data_processor import process_dataframe
from utils.openai_helpers import get_openai_response

st.set_page_config(page_title="GenBI", layout="wide")

def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatOpenAI(temperature=0, model="gpt-4o")

def load_data(uploaded_file):
    """
    Load data from various file formats (CSV, Excel, JSON)
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

        # Convert all numeric columns that might be strings
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='ignore')
            except:
                continue

        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def upload_file():
    uploaded_file = st.file_uploader(
        "Upload your dataset", 
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported formats: CSV, Excel (xlsx/xls), JSON"
    )
    if uploaded_file is not None:
        return load_data(uploaded_file)
    return None

def main():
    st.title("🤖 GenBI")
    initialize_session_state()

    with st.sidebar:
        st.header("Upload Data")
        df = upload_file()

        if df is not None:
            st.session_state.df = df
            st.write("Data Preview:")
            st.dataframe(df)
            st.write(f"Total rows: {len(df)}")
            st.write(f"Columns: {', '.join(df.columns)}")
            #st.session_state.df = df

    if st.session_state.df is not None:
        st.header("Ask Questions About Your Data")
        user_query = st.text_input(
            "Enter your question:", 
            placeholder="e.g., 'Show me a bar plot of sales by category' or 'What is the average age?'"
        )

        if user_query:
            with st.spinner("Analyzing your question..."):
                # Classify the query type
                query_type = classify_query(user_query)

                try:
                    if query_type == "plot":
                        # Generate data manipulation prompt
                        manipulation_prompt = generate_data_manipulation_prompt(user_query, st.session_state.df)
                        #st.write(manipulation_prompt)
                        # Process the dataframe
                        json_input = json.dumps(manipulation_prompt)
                        #st.write(json_input)
                        #st.code(json_input, language='python')

                        processed_df = process_dataframe(manipulation_prompt, st.session_state.df)
                        #st.write(processed_df)
                        # Create visualization
                        fig = create_visualization(processed_df, user_query)
                        st.plotly_chart(fig)

                    elif query_type == "table":
                        agent = create_pandas_dataframe_agent(
                            st.session_state.llm,
                            st.session_state.df,
                            verbose=True,
                            allow_dangerous_code=True
                        )
                        result = agent.run(user_query)
                        st.write(result)

                    else:  # answer
                        agent = create_pandas_dataframe_agent(
                            st.session_state.llm,
                            st.session_state.df,
                            verbose=True,
                            allow_dangerous_code=True
                        )
                        answer = agent.run(user_query)
                        st.success(answer)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("👆 Please upload a dataset file (CSV, Excel, or JSON) to begin analysis.")

if __name__ == "__main__":
    main()
