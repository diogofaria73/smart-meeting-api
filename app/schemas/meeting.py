from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MeetingBase(BaseModel):
    title: str = Field(..., description="Título da reunião")
    description: Optional[str] = Field(None, description="Descrição da reunião")
    date: datetime = Field(..., description="Data e hora da reunião")
    participants: List[str] = Field(default_factory=list, description="Lista de participantes da reunião")


class MeetingCreate(MeetingBase):
    pass


class MeetingResponse(MeetingBase):
    id: int = Field(..., description="ID da reunião")
    created_at: datetime = Field(..., description="Data de criação do registro")
    updated_at: datetime = Field(..., description="Data da última atualização do registro")
    has_transcription: bool = Field(False, description="Indica se a reunião possui transcrição")
    has_summary: bool = Field(False, description="Indica se a reunião possui resumo")

    class Config:
        from_attributes = True


class MeetingSummary(BaseModel):
    meeting_id: int = Field(..., description="ID da reunião")
    summary: str = Field(..., description="Resumo da reunião")
    topics: List[str] = Field(default_factory=list, description="Tópicos principais discutidos")
    generated_at: datetime = Field(..., description="Data de geração do resumo")

    class Config:
        from_attributes = True 