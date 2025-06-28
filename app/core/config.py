import os
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", ".env.diarization"],  # Carrega ambos os arquivos
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
    PROJECT_NAME: str = "Smart Meeting API"
    API_PREFIX: str = "/api"
    
    # Configurações de segurança
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "secret_key_for_development")
    
    # Configurações de CORS - incluindo porta do Vite (5173)
    ALLOWED_ORIGINS: str = os.environ.get(
        "ALLOWED_ORIGINS", 
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
    )
    
    # Configurações do banco de dados
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL", "file:./dev.db"
    )
    
    # Configurações para modelos de IA - PORTUGUÊS BRASILEIRO
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "openai/whisper-large-v3")
    SUMMARIZATION_MODEL: str = os.environ.get("SUMMARIZATION_MODEL", "neuralmind/bert-base-portuguese-cased")
    
    # Modelo único para transcrição PT-BR
    TRANSCRIPTION_MODEL: str = "openai/whisper-large-v3"
    
    # 🎙️ Configurações de Speaker Diarization
    ENABLE_SPEAKER_DIARIZATION: bool = True
    FORCE_DIARIZATION: bool = False  # Força diarização independente do hardware
    MIN_SPEAKERS: int = 1
    MAX_SPEAKERS: int = 10
    MIN_SEGMENT_DURATION: float = 1.0
    
    # Token do HuggingFace (opcional)
    HUGGINGFACE_TOKEN: Optional[str] = None
    
    # Configurações de hardware para diarização
    FORCE_DEVICE: Optional[str] = None  # cuda, mps, cpu
    FORCE_COMPUTE_TYPE: Optional[str] = None  # float16, float32, int8
    
    # Modelo de diarização
    DIARIZATION_MODEL: Optional[str] = "pyannote/speaker-diarization-3.1"

    # Property para obter as origens como lista
    @property
    def cors_origins(self) -> List[str]:
        """Converte ALLOWED_ORIGINS string em lista para uso no CORS"""
        if not self.ALLOWED_ORIGINS:
            return ["http://localhost:5173", "http://localhost:3000"]
        
        # Split por vírgula e remove espaços
        origins = [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
        return [origin for origin in origins if origin]  # Remove strings vazias


settings = Settings() 