from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Union
from enum import Enum


class VoiceGender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class Character(BaseModel):
    name: str = Field(..., description="Nome do personagem")
    voice_id: Optional[str] = Field(None, description="ID específico da voz (para pyttsx3)")
    language: str = Field(default="pt", description="Idioma do personagem")
    rate: int = Field(default=200, description="Velocidade da fala (palavras por minuto)")
    volume: float = Field(default=0.9, description="Volume da voz (0.0 a 1.0)")
    pitch: int = Field(default=0, description="Tom da voz (-50 a 50)")
    gender: VoiceGender = Field(default=VoiceGender.FEMALE, description="Gênero da voz")


class DialogueLine(BaseModel):
    character: str = Field(..., description="Nome do personagem que fala")
    text: str = Field(..., description="Texto da fala", min_length=1, max_length=1000)
    pause_after: float = Field(default=1.0, description="Pausa após a fala em segundos")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Texto simples para conversão", min_length=1, max_length=5000)
    language: str = Field(default="pt", description="Código do idioma")
    slow: bool = Field(default=False, description="Velocidade lenta da fala")


class ConversationTTSRequest(BaseModel):
    characters: List[Character] = Field(..., description="Lista de personagens na conversa", min_items=1)
    dialogue: List[DialogueLine] = Field(..., description="Linhas do diálogo", min_items=1)
    background_music: bool = Field(default=False, description="Adicionar música de fundo")
    output_format: Literal["mp3", "wav"] = Field(default="mp3", description="Formato do arquivo de áudio")


class TTSResponse(BaseModel):
    success: bool = Field(..., description="Status da operação")
    message: str = Field(..., description="Mensagem de retorno")
    audio_url: Optional[str] = Field(None, description="URL para download do áudio gerado")
    filename: Optional[str] = Field(None, description="Nome do arquivo gerado")
    duration: Optional[float] = Field(None, description="Duração do áudio em segundos")
    characters_used: Optional[List[str]] = Field(None, description="Lista de personagens utilizados")


class AvailableVoicesResponse(BaseModel):
    voices: List[dict] = Field(..., description="Lista de vozes disponíveis no sistema")
    total_count: int = Field(..., description="Número total de vozes")


# Modelo unificado que aceita tanto texto simples quanto conversa
class UnifiedTTSRequest(BaseModel):
    # Para texto simples
    simple_text: Optional[str] = Field(None, description="Texto simples para conversão")
    language: Optional[str] = Field("pt", description="Idioma para texto simples")
    slow: Optional[bool] = Field(False, description="Velocidade lenta para texto simples")
    
    # Para conversa entre personagens
    characters: Optional[List[Character]] = Field(None, description="Personagens da conversa")
    dialogue: Optional[List[DialogueLine]] = Field(None, description="Diálogo entre personagens")
    
    # Configurações gerais
    output_format: Literal["mp3", "wav"] = Field(default="mp3", description="Formato de saída")
    background_music: bool = Field(default=False, description="Música de fundo")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "simple_text": "Olá, este é um exemplo de texto simples",
                    "language": "pt",
                    "slow": False
                },
                {
                    "characters": [
                        {
                            "name": "João",
                            "voice_id": "voice_1",
                            "language": "pt",
                            "rate": 180,
                            "gender": "male"
                        },
                        {
                            "name": "Maria",
                            "voice_id": "voice_2", 
                            "language": "pt",
                            "rate": 200,
                            "gender": "female"
                        }
                    ],
                    "dialogue": [
                        {
                            "character": "João",
                            "text": "Olá Maria, como você está hoje?",
                            "pause_after": 1.5
                        },
                        {
                            "character": "Maria", 
                            "text": "Olá João! Estou muito bem, obrigada. E você?",
                            "pause_after": 1.0
                        }
                    ]
                }
            ]
        } 