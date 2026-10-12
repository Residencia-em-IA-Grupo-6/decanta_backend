from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configurações centrais da aplicação carregadas de variáveis de ambiente ou .env.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "FactChkBR API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Lista de analisadores ativos separados por vírgula
    # Ex: "claim_extractor,heuristic,fact_check_api,llm_judge"
    ACTIVE_ANALYZERS: str = "claim_extractor,heuristic,fact_check_api,llm_judge"

    # Modo exclusivo de saúde (rejeita temas não relacionados e polêmicas políticas sobre saúde)
    HEALTH_ONLY_MODE: bool = True

    # Classificador Temático e Gatekeeper de Saúde de alta performance (ONNX INT8, ~20ms)
    ONNX_GATEKEEPER_ENABLED: bool = True
    ONNX_MODEL_DIR: str = "models"

    # Motor Epistêmico de Decisão Local: JevK5-Lite (CPU 437M)
    DECISION_MODEL_ENABLED: bool = True
    DECISION_MODEL_PATH: str = "models/jevk5_lite"
    DECISION_MODEL_NAME: str = "alibiserikbay/JevK5-Lite"
    DECISION_THREADS: int = 6

    # Retrocompatibilidade com configurações legadas (Plumb-4B / Laya)
    PLUMB_ENABLED: bool = True
    PLUMB_MODEL_NAME: str = "alibiserikbay/JevK5-Lite"
    PLUMB_DEVICE: str = "cpu"
    LAYA_ENABLED: bool = False
    LAYA_MODEL_NAME: str = "alibiserikbay/JevK5-Lite"
    # Configuração de Provedores de LLM (Groq Cloud, Ollama Local ou OpenAI)
    # Padrão: se GROQ_API_KEY estiver configurado, usa "groq" com Llama 3.3 70B / gpt-oss-120b
    LLM_PROVIDER: str = "groq"  # "groq" | "ollama" | "openai"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3.5"

    # Configuração da Groq (LPU Ultra-Rápido Gratuito: openai/gpt-oss-120b ou qwen/qwen3.8-27b)
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Chaves e integrações com APIs externas (caso LLM_PROVIDER="openai")
    GOOGLE_FACTCHECK_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # Integração com Bot do Telegram (Recebimento via Webhook e Entrega de Mensagens)
    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_WEBHOOK_SECRET: str | None = None
    TELEGRAM_API_BASE_URL: str = "https://api.telegram.org"
    TELEGRAM_AUTO_REPLY: bool = True

    # Banco de Dados Vetorial Persistente (ChromaDB)
    CHROMA_PERSIST_DIR: str = "app/data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "fact_claims"
    VECTOR_SEARCH_ENABLED: bool = True
    VECTOR_SIMILARITY_THRESHOLD: float = 0.70
    VECTOR_SEARCH_BYPASS_THRESHOLD: float = 0.80
    VECTOR_DEDUP_SIMILARITY_THRESHOLD: float = 0.90
    VECTOR_AUTO_INDEX: bool = True

    def get_llm_endpoint(self) -> str:
        """Retorna a URL de chat completions para o provedor configurado."""
        prov = self.LLM_PROVIDER.lower()
        if prov == "groq":
            return f"{self.GROQ_BASE_URL.rstrip('/')}/chat/completions"
        if prov == "ollama":
            base = self.OLLAMA_BASE_URL.rstrip("/")
            return f"{base}/api/chat"
        return f"{self.OPENAI_BASE_URL.rstrip('/')}/chat/completions"

    def get_llm_model(self) -> str:
        """Retorna o identificador do modelo para o provedor ativo."""
        prov = self.LLM_PROVIDER.lower()
        if prov == "groq":
            return self.GROQ_MODEL
        if prov == "ollama":
            return self.OLLAMA_MODEL
        return self.OPENAI_MODEL

    def get_llm_headers(self) -> dict[str, str]:
        """Retorna os headers HTTP para autenticação e formato."""
        headers = {"Content-Type": "application/json"}
        prov = self.LLM_PROVIDER.lower()
        if prov == "groq" and self.GROQ_API_KEY:
            headers["Authorization"] = f"Bearer {self.GROQ_API_KEY}"
        elif prov == "openai" and self.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {self.OPENAI_API_KEY}"
        elif prov == "ollama":
            headers["Authorization"] = "Bearer ollama"
        return headers

    def get_active_analyzers_list(self) -> list[str]:
        """Retorna os nomes dos analisadores ativos como lista."""
        return [item.strip() for item in self.ACTIVE_ANALYZERS.split(",") if item.strip()]



@lru_cache()
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()
