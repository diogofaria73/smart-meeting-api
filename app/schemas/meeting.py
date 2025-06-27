from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# Importar o schema de transcrição
from app.schemas.transcription import TranscriptionResponse


class MeetingBase(BaseModel):
    title: str = Field(..., description="Título da reunião")
    description: Optional[str] = Field(None, description="Descrição da reunião")
    date: datetime = Field(..., description="Data e hora da reunião")
    participants: List[str] = Field(default_factory=list, description="Lista de participantes da reunião")
    audio_file: Optional[str] = Field(None, description="Arquivo de áudio associada à reunião")


class MeetingCreate(MeetingBase):
    pass


class MeetingUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    date: Optional[datetime] = None
    participants: Optional[List[str]] = None
    audio_file: Optional[str] = None


class MeetingResponse(MeetingBase):
    id: int = Field(..., description="ID da reunião")
    created_at: datetime = Field(..., description="Data de criação do registro")
    updated_at: datetime = Field(..., description="Data da última atualização do registro")
    has_transcription: bool = Field(False, description="Indica se a reunião possui transcrição")
    has_summary: bool = Field(False, description="Indica se a reunião possui resumo")

    class Config:
        from_attributes = True


class MeetingWithTranscriptionsResponse(MeetingBase):
    """Resposta de reunião incluindo suas transcrições"""
    id: int = Field(..., description="ID da reunião")
    created_at: datetime = Field(..., description="Data de criação do registro")
    updated_at: datetime = Field(..., description="Data da última atualização do registro")
    has_transcription: bool = Field(False, description="Indica se a reunião possui transcrição")
    has_summary: bool = Field(False, description="Indica se a reunião possui resumo")
    transcriptions: List[TranscriptionResponse] = Field(default_factory=list, description="Lista de transcrições da reunião")

    class Config:
        from_attributes = True


class MeetingSummary(BaseModel):
    meeting_id: int = Field(..., description="ID da reunião")
    summary: str = Field(..., description="Resumo da reunião")
    topics: List[str] = Field(default_factory=list, description="Tópicos principais discutidos")
    generated_at: datetime = Field(..., description="Data de geração do resumo")

    class Config:
        from_attributes = True


class PaginationParams(BaseModel):
    page: int = 1
    page_size: int = 10


class PaginatedMeetingsResponse(BaseModel):
    meetings: List[MeetingWithTranscriptionsResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


# Novos schemas para estatísticas
class DailyStats(BaseModel):
    date: str = Field(..., description="Data no formato 'YYYY-MM-DD'")
    meetings_count: int = Field(..., description="Número de reuniões realizadas na data")
    transcriptions_count: int = Field(..., description="Número de transcrições realizadas na data")


class DashboardStats(BaseModel):
    total_meetings: int = Field(..., description="Número total de reuniões")
    total_transcriptions: int = Field(..., description="Número total de transcrições")
    completed_transcriptions: int = Field(..., description="Número de transcrições concluídas")
    processing_transcriptions: int = Field(..., description="Número de transcrições em processamento")
    daily_stats: List[DailyStats] = Field(..., description="Estatísticas diárias") 