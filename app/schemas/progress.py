from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel

from app.services.progress_service import ProgressStatus, ProgressStep


class ProgressResponse(BaseModel):
    """Resposta com informações de progresso da transcrição"""
    task_id: str
    meeting_id: int
    status: ProgressStatus
    current_step: ProgressStep
    progress_percentage: float
    message: str
    details: Optional[str] = None
    started_at: datetime
    updated_at: datetime
    estimated_remaining_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    # Informações específicas da transcrição
    audio_duration_seconds: Optional[float] = None
    chunks_total: Optional[int] = None
    chunks_processed: Optional[int] = None
    
    class Config:
        # Permite usar enums como valores
        use_enum_values = True


class TranscriptionStartResponse(BaseModel):
    """Resposta ao iniciar uma transcrição"""
    task_id: str
    meeting_id: int
    message: str
    progress_url: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "meeting_id": 1,
                "message": "Transcrição iniciada. Use o task_id para acompanhar o progresso.",
                "progress_url": "/api/transcriptions/progress/abc123-def456-ghi789"
            }
        }


class MeetingProgressSummary(BaseModel):
    """Resumo de progresso de todas as tarefas de uma reunião"""
    meeting_id: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    processing_tasks: int
    latest_task: Optional[ProgressResponse] = None


class ProgressStepInfo(BaseModel):
    """Informações sobre uma etapa específica"""
    step: ProgressStep
    name: str
    description: str
    typical_duration_seconds: Optional[float] = None
    
    class Config:
        use_enum_values = True


class ProgressStepsResponse(BaseModel):
    """Lista de todas as etapas possíveis com descrições"""
    steps: list[ProgressStepInfo]
    total_steps: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "steps": [
                    {
                        "step": "upload_validation",
                        "name": "Validação do Upload",
                        "description": "Verificando tipo e formato do arquivo de áudio"
                    },
                    {
                        "step": "audio_preprocessing", 
                        "name": "Pré-processamento",
                        "description": "Convertendo e normalizando o áudio"
                    }
                ],
                "total_steps": 7
            }
        } 