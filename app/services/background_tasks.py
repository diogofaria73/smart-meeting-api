"""
🚀 BACKGROUND TASKS SERVICE
Processamento assíncrono de transcrições e análises
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import UploadFile
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from app.services.transcription_service import TranscriptionService
from app.services.progress_service import progress_service, ProgressStep, ProgressStatus
from app.core.events import (
    notify_transcription_started,
    notify_transcription_progress,
    notify_transcription_completed,
    notify_transcription_failed,
    notify_analysis_completed
)


class BackgroundTaskService:
    """Serviço para executar tarefas em background"""
    
    def __init__(self):
        self.transcription_service = TranscriptionService()
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def process_transcription_async(
        self,
        task_id: str,
        meeting_id: int,
        file_content: bytes,
        filename: str,
        content_type: str,
        enable_diarization: bool = True
    ) -> None:
        """
        🎙️ PROCESSA TRANSCRIÇÃO DE FORMA ASSÍNCRONA
        
        Esta função é executada em background e notifica o progresso via WebSocket
        """
        try:
            logger.info(f"🚀 Iniciando processamento assíncrono da transcrição")
            logger.info(f"   Task ID: {task_id}")
            logger.info(f"   Meeting ID: {meeting_id}")
            logger.info(f"   Arquivo: {filename}")
            
            # Notifica início do processamento
            await notify_transcription_started(meeting_id, task_id, filename)
            
            # Cria arquivo temporário
            file_extension = ""
            if filename:
                file_extension = Path(filename).suffix.lower()
            elif content_type:
                extension_map = {
                    "audio/wav": ".wav",
                    "audio/mp3": ".mp3", 
                    "audio/mpeg": ".mp3",
                    "audio/mp4": ".m4a",
                    "audio/m4a": ".m4a",
                    "audio/flac": ".flac",
                    "audio/ogg": ".ogg",
                    "audio/webm": ".webm"
                }
                file_extension = extension_map.get(content_type, ".wav")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Configura callback de progresso
                progress_callback = self._create_progress_callback(task_id, meeting_id)
                
                # Processa transcrição diretamente do arquivo temporário
                logger.info(f"🎙️ Executando transcrição principal...")
                
                # Usa método interno que trabalha com path de arquivo
                from fastapi import UploadFile
                import io
                
                # Cria UploadFile a partir do conteúdo
                file_like = io.BytesIO(file_content)
                upload_file = UploadFile(
                    file=file_like,
                    filename=filename,
                    headers={"content-type": content_type}
                )
                
                result = await self.transcription_service.transcribe_audio(
                    meeting_id=meeting_id,
                    file=upload_file,
                    enable_diarization=enable_diarization,
                    task_id=task_id
                )
                
                logger.info(f"✅ Transcrição concluída com sucesso!")
                logger.info(f"   ID: {result.id}")
                logger.info(f"   Speakers: {result.speakers_count}")
                logger.info(f"   Caracteres: {len(result.content)}")
                
                # Notifica conclusão
                await notify_transcription_completed(
                    meeting_id=meeting_id,
                    task_id=task_id,
                    transcription_id=result.id,
                    speakers_count=result.speakers_count
                )
                
                # Agenda análise inteligente se transcrição foi bem-sucedida
                if len(result.content.strip()) > 100:  # Só analisa se há conteúdo suficiente
                    logger.info(f"📊 Agendando análise inteligente...")
                    asyncio.create_task(
                        self._process_analysis_async(task_id, meeting_id, result.id)
                    )
                
            finally:
                # Remove arquivo temporário
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"❌ Erro no processamento assíncrono: {e}")
            
            # Marca como falhado
            progress_service.mark_failed(task_id, str(e))
            
            # Notifica falha
            await notify_transcription_failed(meeting_id, task_id, str(e))
            
        finally:
            # Remove da lista de tarefas ativas
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _process_analysis_async(self, task_id: str, meeting_id: int, transcription_id: int):
        """Processa análise inteligente em background"""
        try:
            logger.info(f"📊 Iniciando análise inteligente para reunião {meeting_id}")
            
            # Executa análise/resumo
            analysis_result = await self.transcription_service.generate_summary(meeting_id)
            
            logger.info(f"✅ Análise inteligente concluída!")
            
            # Notifica conclusão da análise
            analysis_data = {
                "transcription_id": analysis_result.id,
                "summary_length": len(analysis_result.summary or ""),
                "topics_count": len(analysis_result.topics),
                "has_analysis": analysis_result.analysis is not None
            }
            
            await notify_analysis_completed(meeting_id, analysis_data)
            
        except Exception as e:
            logger.error(f"❌ Erro na análise inteligente: {e}")
            # Análise falha não é crítica, transcrição já foi salva
    
    def _create_progress_callback(self, task_id: str, meeting_id: int):
        """Cria callback para notificar progresso via WebSocket"""
        
        async def progress_callback():
            """Callback que verifica progresso e notifica via WebSocket"""
            try:
                progress_info = progress_service.get_progress(task_id)
                if progress_info:
                    # Garante conversão segura para string
                    status_value = progress_info.status.value if hasattr(progress_info.status, 'value') else str(progress_info.status)
                    step_value = progress_info.current_step.value if hasattr(progress_info.current_step, 'value') else str(progress_info.current_step)
                    
                    await notify_transcription_progress(meeting_id, task_id, {
                        "status": status_value,
                        "step": step_value,
                        "progress_percentage": progress_info.progress_percentage,
                        "message": progress_info.message,
                        "details": progress_info.details,
                        "estimated_remaining_seconds": progress_info.estimated_remaining_seconds
                    })
            except Exception as e:
                logger.error(f"❌ Erro no callback de progresso: {e}")
        
        return progress_callback
    
    def start_transcription_task(
        self,
        task_id: str,
        meeting_id: int,
        file_content: bytes,
        filename: str,
        content_type: str,
        enable_diarization: bool = True
    ) -> asyncio.Task:
        """Inicia uma tarefa de transcrição em background"""
        
        logger.info(f"🚀 Criando tarefa de transcrição em background: {task_id}")
        
        # Cria e inicia a tarefa
        task = asyncio.create_task(
            self.process_transcription_async(
                task_id=task_id,
                meeting_id=meeting_id,
                file_content=file_content,
                filename=filename,
                content_type=content_type,
                enable_diarization=enable_diarization
            )
        )
        
        # Adiciona à lista de tarefas ativas
        self.running_tasks[task_id] = task
        
        return task
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retorna status de uma tarefa"""
        progress_info = progress_service.get_progress(task_id)
        
        if not progress_info:
            return None
        
        # Garante conversão segura para string
        status_value = progress_info.status.value if hasattr(progress_info.status, 'value') else str(progress_info.status)
        step_value = progress_info.current_step.value if hasattr(progress_info.current_step, 'value') else str(progress_info.current_step)
        
        return {
            "task_id": task_id,
            "meeting_id": progress_info.meeting_id,
            "status": status_value,
            "current_step": step_value,
            "progress_percentage": progress_info.progress_percentage,
            "message": progress_info.message,
            "details": progress_info.details,
            "started_at": progress_info.started_at.isoformat(),
            "updated_at": progress_info.updated_at.isoformat(),
            "estimated_remaining_seconds": progress_info.estimated_remaining_seconds,
            "error_message": progress_info.error_message,
            "is_running": task_id in self.running_tasks
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancela uma tarefa em execução"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            if not task.done():
                task.cancel()
                logger.info(f"🛑 Tarefa {task_id} cancelada")
                return True
        
        return False
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Retorna todas as tarefas ativas"""
        active_tasks = {}
        
        for task_id in list(self.running_tasks.keys()):
            task_status = self.get_task_status(task_id)
            if task_status:
                active_tasks[task_id] = task_status
            else:
                # Remove tarefa órfã
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
        
        return active_tasks


# Instância singleton do serviço de background tasks
background_task_service = BackgroundTaskService() 