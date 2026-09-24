"""Runtime configuration loaded exclusively from environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and safe defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    mssql_host: str = "localhost"
    mssql_port: int = 1433
    mssql_database: str = "enterprise_rag"
    mssql_user: str = "sa"
    mssql_sa_password: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "enterprise-rag"
    pinecone_namespace: str = "documents-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_adapter_path: str = "outputs/qlora"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_context_characters: int = 12000
    max_new_tokens: int = 256

    @property
    def sql_url(self) -> str:
        """Build a SQLAlchemy URL without logging the credential."""
        return f"mssql+pyodbc://{self.mssql_user}:{self.mssql_sa_password}@{self.mssql_host}:{self.mssql_port}/{self.mssql_database}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
