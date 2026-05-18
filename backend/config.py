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

    # Chunking — sliding window (legacy)
    chunk_target_chars: int = 1000
    chunk_overlap_chars: int = 150

    # Chunking — multi-strategy
    enabled_strategies: str = (
        "sliding_window,recursive_token,sentence_window,semantic,fixed_time,parent_child,section_aware"
    )
    default_strategy: str = "sliding_window"
    default_pdf_strategy: str = "section_aware"

    # Strategy-specific params
    recursive_token_size: int = 256
    recursive_token_overlap: int = 50
    sentence_window_size: int = 5
    sentence_window_overlap: int = 1
    semantic_percentile: float = 95.0
    time_window_seconds: float = 60.0
    parent_chunk_chars: int = 1500
    parent_chunk_overlap: int = 200
    child_chunk_chars: int = 300
    section_max_chars: int = 1800       # if a section exceeds this, subdivide
    section_min_chars: int = 200        # tiny tail sections (e.g., "References" stubs) merged into prev

    # PDFs
    pdf_uploads_dir: str = "./uploads"
    pdf_renders_dir: str = "./pdf_renders"
    pdf_assets_dir: str = "./pdf_assets"        # extracted figures/tables
    pdf_max_bytes: int = 50 * 1024 * 1024        # 50 MB per upload
    pdf_default_extractor: str = "pypdfium2"     # "pypdfium2" | "marker"
    pdf_render_scale: float = 1.5                # PNG render scale

    # Retrieval
    top_k_retrieve: int = 15
    top_k_rerank: int = 4

    # Judge
    judge_provider: str = "anthropic"
    judge_model: str = "claude-sonnet-4-6"

    # Storage
    chroma_path: str = "./chroma_db"
    collection_name: str = "videos"

    # Transcripts
    transcript_languages: str = "en"

    @property
    def language_list(self) -> list[str]:
        return [s.strip() for s in self.transcript_languages.split(",") if s.strip()]

    @property
    def strategy_list(self) -> list[str]:
        return [s.strip() for s in self.enabled_strategies.split(",") if s.strip()]


settings = Settings()
