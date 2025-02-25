from utils.openai_helpers import get_openai_response

def classify_query(query: str) -> str:
    """
    Classifies the user query into one of three types: plot, table, or answer
    """
    prompt = {
        "role": "system",
        "content": """
        Classify the following query into one of three categories:
        - 'plot': If the user is requesting any visualization (charts, graphs, plots, diagrams, maps)
        - 'table': If the user is requesting tabular data display (tables, lists, rankings, matrices)
        - 'answer': If the user is asking a question requiring textual explanation or analysis

        Respond with ONLY one word: 'plot', 'table', or 'answer'.

        Examples:
        - "Create a histogram of temperature distribution" → plot
        - "Show me a scatter plot of height vs weight" → plot
        - "Display the quarterly sales figures" → table
        - "List the top 10 performing stocks" → table
        - "Explain why sales decreased last month" → answer
        - "What factors influenced market growth?" → answer

        For ambiguous queries that could fall into multiple categories, prioritize based on the primary intent:
        1. If visualization is explicitly mentioned → plot
        2. If tabular display is explicitly mentioned → table
        3. If neither is specified but data comparison is requested → table
        4. If analysis or explanation is requested → answer
        """
    }
    
    query_message = {
        "role": "user",
        "content": query
    }
    
    response = get_openai_response([prompt, query_message])
    return response.lower().strip()
