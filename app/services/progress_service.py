import time
import uuid
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union
from pydantic import BaseModel
import threading


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
    
    def _notify_progress_async(self, progress: ProgressInfo):
        """Envia notificação WebSocket de forma assíncrona"""
        try:
            # Import lazy para evitar circular imports
            from app.core.events import notify_transcription_progress
            
            def run_notification():
                """Executa a notificação em um novo event loop"""
                try:
                    import asyncio
                    
                    # Verifica se já existe um event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Se existe um loop, agenda a tarefa
                        loop.create_task(notify_transcription_progress(
                            meeting_id=progress.meeting_id,
                            task_id=progress.task_id,
                            progress_data={
                                "status": progress.status.value,
                                "step": progress.current_step.value,
                                "progress_percentage": progress.progress_percentage,
                                "message": progress.message,
                                "details": progress.details,
                                "estimated_remaining_seconds": progress.estimated_remaining_seconds
                            }
                        ))
                    except RuntimeError:
                        # Não há loop rodando, cria um novo
                        asyncio.run(notify_transcription_progress(
                            meeting_id=progress.meeting_id,
                            task_id=progress.task_id,
                            progress_data={
                                "status": progress.status.value,
                                "step": progress.current_step.value,
                                "progress_percentage": progress.progress_percentage,
                                "message": progress.message,
                                "details": progress.details,
                                "estimated_remaining_seconds": progress.estimated_remaining_seconds
                            }
                        ))
                except Exception as e:
                    print(f"❌ Erro interno na notificação WebSocket: {e}")
            
            # Executa em uma thread separada para não bloquear
            thread = threading.Thread(target=run_notification, daemon=True)
            thread.start()
            
        except Exception as e:
            # Não falha o progresso se notificação falhar
            print(f"❌ Erro ao configurar notificação WebSocket: {e}")
    
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
        # Envia notificação inicial
        self._notify_progress_async(progress_info)
        return task_id
    
    def update_progress(
        self,
        task_id: str,
        step: Union[ProgressStep, str],
        message: str,
        details: Optional[str] = None,
        custom_percentage: Optional[float] = None
    ) -> None:
        """Atualiza o progresso de uma tarefa"""
        if task_id not in self._progress_data:
            return
        
        progress = self._progress_data[task_id]
        
        # Converte string para enum se necessário
        if isinstance(step, str):
            # Mapeia strings comuns para enums
            step_mapping = {
                "upload_validation": ProgressStep.UPLOAD_VALIDATION,
                "audio_preprocessing": ProgressStep.AUDIO_PREPROCESSING,
                "audio_processing": ProgressStep.AUDIO_PREPROCESSING,
                "model_loading": ProgressStep.MODEL_LOADING,
                "initialization": ProgressStep.MODEL_LOADING,
                "transcription": ProgressStep.TRANSCRIPTION,
                "transcription_processing": ProgressStep.TRANSCRIPTION,
                "speaker_diarization": ProgressStep.TRANSCRIPTION,
                "enhanced_transcription_start": ProgressStep.TRANSCRIPTION,
                "post_processing": ProgressStep.POST_PROCESSING,
                "alignment": ProgressStep.POST_PROCESSING,
                "database_save": ProgressStep.DATABASE_SAVE,
                "completed": ProgressStep.COMPLETED,
                "error": ProgressStep.COMPLETED,  # Erro será tratado pelo status
                "speaker_diarization_error": ProgressStep.TRANSCRIPTION
            }
            step = step_mapping.get(step, ProgressStep.TRANSCRIPTION)
        
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
        
        # Envia notificação WebSocket
        self._notify_progress_async(progress)
    
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
        
        # Envia notificação WebSocket
        self._notify_progress_async(progress)
    
    def mark_failed(self, task_id: str, error_message: str) -> None:
        """Marca uma tarefa como falhada"""
        if task_id not in self._progress_data:
            return
        
        progress = self._progress_data[task_id]
        progress.status = ProgressStatus.FAILED
        progress.current_step = ProgressStep.COMPLETED  # Define step final
        progress.error_message = error_message
        progress.updated_at = datetime.now()
        
        # Envia notificação de erro
        self._notify_progress_async(progress)
    
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
        
        # Envia notificação de conclusão
        self._notify_progress_async(progress)
    
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