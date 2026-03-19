from pydantic_settings import BaseSettings
from pathlib import Path 

current_file = Path(__file__).resolve()
project_root = current_file.parents[2] 
DOCUMENTS_DIRECTORY = project_root / "docs"

class Settings(BaseSettings):
    database_hostname: str 
    database_port: str
    database_password: str
    database_name: str
    database_username: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int

    GOOGLE_API_KEY : str
    GROQ_API_KEY : str

    LANGCHAIN_TRACING_V2 : bool
    LANGCHAIN_ENDPOINT : str
    LANGCHAIN_API_KEY : str
    LANGCHAIN_PROJECT : str



    class Config:
        env_file = ".env"



settings = Settings()


class LLMConfig(BaseSettings):

    LLM_MODEL_NAME :str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE : float = 0
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    DOCUMENTS_DIR: Path = DOCUMENTS_DIRECTORY
    DOCUMENTS_GLOB: str = "*.txt"
    CHROMA_PERSIST_DIR: Path = project_root / "chroma_db"
    CHROMA_COLLECTION: str = "embeddings"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVER_SEARCH_TYPE: str = "mmr"
    RETRIEVER_K: int = 5
    RETRIEVER_LAMBDA_MULT: float = 0.8

    RAG_PROMPT_HUB_ID: str = "rlm/rag-prompt"


llm_config = LLMConfig()