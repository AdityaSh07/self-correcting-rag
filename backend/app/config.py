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