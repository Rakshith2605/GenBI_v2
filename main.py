import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
import plotly.express as px
import json
import os

from agents.classifier import classify_query
from agents.prompt_generator import generate_data_manipulation_prompt
from agents.visualization import create_visualization
from utils.data_processor import process_dataframe
from utils.openai_helpers import get_openai_response, validate_openai_api_key
from agents.table_generator import get_df

st.set_page_config(page_title="GenBI", layout="wide")

def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False

def setup_api_key():
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password", key="api_key_input")
    
    if api_key and api_key != st.session_state.api_key:
        with st.sidebar:
            with st.spinner("Validating API key..."):
                is_valid, result = validate_openai_api_key(api_key, check_gpt4=True)
                
                if is_valid:
                    st.session_state.api_key = api_key
                    st.session_state.api_key_valid = True
                    os.environ["OPENAI_API_KEY"] = api_key
                    
                    # Initialize LLM with the validated key
                    model = "gpt-4o" if result.get('has_gpt4', False) else "gpt-3.5-turbo"
                    st.session_state.llm = ChatOpenAI(temperature=0, model=model, openai_api_key=api_key)
                    
                    st.success(f"✅ API Key is valid! Using model: {model}")
                    st.write(f"Available models: {len(result['models'])}")
                else:
                    st.session_state.api_key_valid = False
                    st.error(f"❌ API Key validation failed: {result['error']}")
    
    return st.session_state.api_key_valid

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

def process_query(user_query):
    with st.spinner("Analyzing your question..."):
        query_type = classify_query(user_query)
        try:
            if query_type == "plot":
                manipulation_prompt = generate_data_manipulation_prompt(user_query, st.session_state.df)
                processed_df = process_dataframe(manipulation_prompt, st.session_state.df)
                fig = create_visualization(processed_df, user_query)
                return {"type": "plot", "content": fig}

            elif query_type == "table":
                result = get_df(st.session_state.df, user_query)
                return {"type": "text", "content": result}

            else:  # answer
                agent = create_pandas_dataframe_agent(
                    st.session_state.llm,
                    st.session_state.df,
                    verbose=True,
                    allow_dangerous_code=True
                )
                answer = agent.run(user_query)
                return {"type": "text", "content": answer}

        except Exception as e:
            return {"type": "error", "content": f"An error occurred: {str(e)}"}

def main():
    st.title("🤖 GenBI")
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Setup")
        api_key_valid = setup_api_key()
        
        if api_key_valid:
            st.header("Upload Data")
            df = upload_file()

            if df is not None:
                st.session_state.df = df
                st.write("Data Preview:")
                st.dataframe(df.head())
                st.write(f"Total rows: {len(df)}")
                st.write(f"Columns: {', '.join(df.columns)}")

    # Main chat interface
    if not st.session_state.api_key_valid:
        st.info("Please enter a valid OpenAI API key in the sidebar to begin.")
    elif st.session_state.df is not None:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["type"] == "plot":
                    st.plotly_chart(message["content"])
                else:
                    st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask questions about your data..."):
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})

            # Process and display response
            response = process_query(prompt)
            with st.chat_message("assistant"):
                if response["type"] == "plot":
                    st.plotly_chart(response["content"])
                else:
                    st.write(response["content"])
            st.session_state.messages.append({"role": "assistant", **response})

    else:
        st.info("👈 Please upload a dataset file (CSV, Excel, or JSON) to begin analysis.")

if __name__ == "__main__":
    main()
