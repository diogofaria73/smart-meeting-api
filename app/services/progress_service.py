import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel


class ProgressStatus(str, Enum):
    """Status possíveis do progresso"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressStep(str, Enum):
    """Etapas do processo de transcrição"""
    UPLOAD_VALIDATION = "upload_validation"
    AUDIO_PREPROCESSING = "audio_preprocessing"
    MODEL_LOADING = "model_loading"
    TRANSCRIPTION = "transcription"
    POST_PROCESSING = "post_processing"
    DATABASE_SAVE = "database_save"
    COMPLETED = "completed"


class ProgressInfo(BaseModel):
    """Informações detalhadas do progresso"""
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


class ProgressService:
    """Serviço para gerenciar progresso das transcrições"""
    
    def __init__(self):
        # Em produção, isso deveria ser Redis ou similar
        self._progress_data: Dict[str, ProgressInfo] = {}
        
        # Mapeamento de etapas para percentuais
        self._step_percentages = {
            ProgressStep.UPLOAD_VALIDATION: 10.0,
            ProgressStep.AUDIO_PREPROCESSING: 20.0,
            ProgressStep.MODEL_LOADING: 30.0,
            ProgressStep.TRANSCRIPTION: 70.0,  # A maior parte do tempo
            ProgressStep.POST_PROCESSING: 85.0,
            ProgressStep.DATABASE_SAVE: 95.0,
            ProgressStep.COMPLETED: 100.0,
        }
    
    def create_task(self, meeting_id: int) -> str:
        """Cria uma nova tarefa de progresso"""
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        progress_info = ProgressInfo(
            task_id=task_id,
            meeting_id=meeting_id,
            status=ProgressStatus.PENDING,
            current_step=ProgressStep.UPLOAD_VALIDATION,
            progress_percentage=0.0,
            message="Iniciando processamento...",
            started_at=now,
            updated_at=now
        )
        
        self._progress_data[task_id] = progress_info
        return task_id
    
    def update_progress(
        self,
        task_id: str,
        step: ProgressStep,
        message: str,
        details: Optional[str] = None,
        custom_percentage: Optional[float] = None
    ) -> None:
        """Atualiza o progresso de uma tarefa"""
        if task_id not in self._progress_data:
            return
        
        progress = self._progress_data[task_id]
        progress.current_step = step
        progress.message = message
        progress.details = details
        progress.updated_at = datetime.now()
        
        if custom_percentage is not None:
            progress.progress_percentage = custom_percentage
        else:
            progress.progress_percentage = self._step_percentages.get(step, 0.0)
        
        # Calcula tempo estimado restante (estimativa simples)
        if progress.progress_percentage > 0:
            elapsed_time = (progress.updated_at - progress.started_at).total_seconds()
            total_estimated_time = elapsed_time / (progress.progress_percentage / 100.0)
            progress.estimated_remaining_seconds = max(0, total_estimated_time - elapsed_time)
        
        # Atualiza status baseado no step
        if step == ProgressStep.COMPLETED:
            progress.status = ProgressStatus.COMPLETED
        elif progress.status == ProgressStatus.PENDING:
            progress.status = ProgressStatus.PROCESSING
    
    def update_transcription_chunks(
        self,
        task_id: str,
        chunks_total: int,
        chunks_processed: int,
        audio_duration: Optional[float] = None
    ) -> None:
        """Atualiza progresso específico de chunks de transcrição"""
        if task_id not in self._progress_data:
            return
        
        progress = self._progress_data[task_id]
        progress.chunks_total = chunks_total
        progress.chunks_processed = chunks_processed
        
        if audio_duration:
            progress.audio_duration_seconds = audio_duration
        
        # Calcula percentual dentro da etapa de transcrição (30% a 70%)
        base_percentage = 30.0  # Início da transcrição
        transcription_range = 40.0  # 70% - 30% = 40%
        
        if chunks_total > 0:
            chunk_progress = chunks_processed / chunks_total
            current_percentage = base_percentage + (transcription_range * chunk_progress)
            progress.progress_percentage = current_percentage
        
        progress.message = f"Transcrevendo chunk {chunks_processed}/{chunks_total}"
        progress.updated_at = datetime.now()
    
    def mark_failed(self, task_id: str, error_message: str) -> None:
        """Marca uma tarefa como falhada"""
        if task_id not in self._progress_data:
            return
        
        progress = self._progress_data[task_id]
        progress.status = ProgressStatus.FAILED
        progress.error_message = error_message
        progress.updated_at = datetime.now()
    
    def mark_completed(self, task_id: str) -> None:
        """Marca uma tarefa como concluída"""
        if task_id not in self._progress_data:
            return
        
        progress = self._progress_data[task_id]
        progress.status = ProgressStatus.COMPLETED
        progress.current_step = ProgressStep.COMPLETED
        progress.progress_percentage = 100.0
        progress.message = "Transcrição concluída com sucesso!"
        progress.updated_at = datetime.now()
        progress.estimated_remaining_seconds = 0.0
    
    def get_progress(self, task_id: str) -> Optional[ProgressInfo]:
        """Obtém informações de progresso de uma tarefa"""
        return self._progress_data.get(task_id)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove tarefas antigas (mais de X horas)"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        tasks_to_remove = []
        
        for task_id, progress in self._progress_data.items():
            if progress.started_at.timestamp() < cutoff_time:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self._progress_data[task_id]
        
        return len(tasks_to_remove)
    
    def get_all_tasks_for_meeting(self, meeting_id: int) -> Dict[str, ProgressInfo]:
        """Obtém todas as tarefas de uma reunião específica"""
        return {
            task_id: progress 
            for task_id, progress in self._progress_data.items() 
            if progress.meeting_id == meeting_id
        }


# Instância singleton do serviço de progresso
progress_service = ProgressService() 