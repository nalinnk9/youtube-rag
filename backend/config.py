"""Central configuration, loaded from .env via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # Models
    embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    llm_model: str = "claude-sonnet-4-6"
    whisper_model: str = "small.en"

    # Chunking
    chunk_target_chars: int = 1000
    chunk_overlap_chars: int = 150

    # Retrieval
    top_k_retrieve: int = 15
    top_k_rerank: int = 4

    # Storage
    chroma_path: str = "./chroma_db"
    collection_name: str = "videos"

    # Transcripts
    transcript_languages: str = "en"

    @property
    def language_list(self) -> list[str]:
        return [s.strip() for s in self.transcript_languages.split(",") if s.strip()]


settings = Settings()
