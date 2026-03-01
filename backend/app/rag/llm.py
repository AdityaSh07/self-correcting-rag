from dotenv import load_dotenv
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import LLMConfig

load_dotenv()

# --------------------------------- CONSTANTS ----------------------------------

LLM_MODEL_NAME = LLMConfig.LLM_MODEL_NAME
LLM_TEMPERATURE = LLMConfig.LLM_TEMPERATURE
EMBEDDING_MODEL_NAME = LLMConfig.EMBEDDING_MODEL_NAME
RAG_PROMPT_HUB_ID = LLMConfig.RAG_PROMPT_HUB_ID

# --------------------------------- MODELS ------------------------------------

model = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

rag_prompt = hub.pull(RAG_PROMPT_HUB_ID)
rag_chain = rag_prompt | model
