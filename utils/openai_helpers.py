import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv
import os
from pathlib import Path

def load_openai_client():
    # Get the project root directory (where .env is located)
    root_dir = Path(__file__).parent.parent
    
    # Load environment variables from .env file
    load_dotenv(root_dir / '.env')
    
    # Initialize and return the OpenAI client
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a single client instance to be used throughout the application
client = load_openai_client()




# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_openai_response(messages: List[Dict[str, str]]) -> str:
    """
    Gets a response from OpenAI API
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting OpenAI response: {str(e)}")
