from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq   
# llm = ChatGroq(model="openai/gpt-oss-120b") 
groq_api_key = os.getenv("grop_api_key") 
 
llm = ChatGroq(model_name='openai/gpt-oss-120b',
               groq_api_key=groq_api_key)

def llm_response(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content
