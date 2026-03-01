from pydantic_settings import BaseSettings

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

    LANGCHAIN_TRACING_V2 : bool
    LANGCHAIN_ENDPOINT : str
    LANGCHAIN_API_KEY : str
    LANGCHAIN_PROJECT : str



    class Config:
        env_file = ".env"



settings = Settings()


class LLMConfig(BaseSettings):

    LLM_MODEL_NAME = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.8
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    DOCUMENTS_DIR = "documents"
    DOCUMENTS_GLOB = "*.txt"
    CHROMA_PERSIST_DIR = "chroma_db"
    CHROMA_COLLECTION = "embeddings"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    RETRIEVER_SEARCH_TYPE = "mmr"
    RETRIEVER_K = 5
    RETRIEVER_LAMBDA_MULT = 0.8

    RAG_PROMPT_HUB_ID = "rlm/rag-prompt"