from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
from ..config import llm_config as LLMConfig

from langsmith import Client

client = Client()


load_dotenv()

# --------------------------------- CONSTANTS ----------------------------------

LLM_MODEL_NAME = LLMConfig.LLM_MODEL_NAME
LLM_TEMPERATURE = LLMConfig.LLM_TEMPERATURE
EMBEDDING_MODEL_NAME = LLMConfig.EMBEDDING_MODEL_NAME
RAG_PROMPT_HUB_ID = LLMConfig.RAG_PROMPT_HUB_ID

# --------------------------------- MODELS ------------------------------------

llm = ChatGroq(
    model=LLM_MODEL_NAME, 
    temperature=LLM_TEMPERATURE
)
embed_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME,
    output_dimensionality=768)

