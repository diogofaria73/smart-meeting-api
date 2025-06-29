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
        logger.info(f"🚀 Iniciando tarefa de transcrição em background: {task_id}")
        
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
        
        logger.info(f"✅ Tarefa de transcrição {task_id} adicionada à lista de execução")
        return task

    def start_enhanced_analysis_task(
        self,
        task_id: str,
        meeting_id: int,
        transcription_text: str,
        custom_config: Optional[Dict] = None
    ) -> asyncio.Task:
        """
        🚀 Inicia uma tarefa de análise OTIMIZADA em background
        
        Usa o enhanced_summary_service para melhor performance e qualidade
        """
        logger.info(f"🧠 Iniciando tarefa de análise otimizada em background: {task_id}")
        
        task = asyncio.create_task(
            self._process_enhanced_analysis_async(
                task_id=task_id,
                meeting_id=meeting_id,
                transcription_text=transcription_text,
                custom_config=custom_config or {}
            )
        )
        
        # Adiciona à lista de tarefas ativas
        self.running_tasks[task_id] = task
        
        logger.info(f"✅ Tarefa de análise otimizada {task_id} adicionada à lista de execução")
        return task

    async def _process_enhanced_analysis_async(
        self,
        task_id: str,
        meeting_id: int,
        transcription_text: str,
        custom_config: Dict
    ) -> None:
        """
        🧠 Processa análise OTIMIZADA de forma assíncrona com progresso detalhado
        """
        try:
            logger.info(f"🔍 Iniciando processamento de análise otimizada")
            logger.info(f"   Task ID: {task_id}")
            logger.info(f"   Meeting ID: {meeting_id}")
            logger.info(f"   Texto: {len(transcription_text)} caracteres")
            
            # Notifica início da análise
            progress_service.update_progress(
                task_id, 
                'enhanced_analysis_start', 
                'Iniciando análise inteligente otimizada...', 
                5
            )
            
            # Configura serviço otimizado
            from app.services.enhanced_summary_service import enhanced_summary_service
            
            # Configuração otimizada para background
            config = {
                'cache_enabled': True,
                'parallel_processing': True,
                'min_confidence': 0.6,
                'max_chunk_size': 1200,  # Chunks maiores para background
                'chunk_overlap': 150,
                **custom_config
            }
            
            # Executa análise otimizada
            logger.info(f"🤖 Executando análise com enhanced_summary_service...")
            
            analysis_result = await enhanced_summary_service.analyze_meeting_async(
                meeting_id=meeting_id,
                transcription_text=transcription_text,
                task_id=task_id,
                custom_config=config
            )
            
            logger.info(f"✅ Análise otimizada concluída em {analysis_result.processing_time:.2f}s!")
            logger.info(f"   • Participantes: {len(analysis_result.participants)}")
            logger.info(f"   • Tópicos: {len(analysis_result.main_topics)}")
            logger.info(f"   • Ações: {len(analysis_result.action_items)}")
            logger.info(f"   • Decisões: {len(analysis_result.key_decisions)}")
            logger.info(f"   • Confiança: {analysis_result.confidence_score:.2f}")
            
            # Salva resultados no banco de dados
            await self._save_enhanced_analysis_results(meeting_id, analysis_result)
            
            # Atualiza progresso final
            progress_service.update_progress(
                task_id, 
                'enhanced_analysis_save', 
                'Salvando resultados da análise...', 
                95
            )
            
            # Notifica conclusão da análise otimizada
            analysis_summary = {
                "meeting_id": meeting_id,
                "participants_count": len(analysis_result.participants),
                "topics_count": len(analysis_result.main_topics),
                "actions_count": len(analysis_result.action_items),
                "decisions_count": len(analysis_result.key_decisions),
                "confidence_score": analysis_result.confidence_score,
                "processing_time": analysis_result.processing_time,
                "summary_length": len(analysis_result.summary or ""),
                "has_sentiment": analysis_result.sentiment_analysis is not None
            }
            
            await notify_analysis_completed(meeting_id, analysis_summary)
            
            # Marca como concluída
            progress_service.mark_completed(task_id)
            
            logger.info(f"🎯 Análise otimizada completa para reunião {meeting_id}")
            
        except Exception as e:
            logger.error(f"❌ Erro na análise otimizada assíncrona: {e}")
            
            # Marca como falhado
            progress_service.mark_failed(task_id, f"Erro na análise: {str(e)}")
            
            # Notifica falha
            await notify_transcription_failed(meeting_id, task_id, f"Falha na análise: {str(e)}")
            
        finally:
            # Remove da lista de tarefas ativas
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    async def _save_enhanced_analysis_results(
        self, 
        meeting_id: int, 
        analysis_result
    ) -> None:
        """
        💾 Salva resultados da análise otimizada no banco de dados
        """
        try:
            from app.db.client import get_db
            import json
            
            async with get_db() as db:
                # Gera resumo tradicional se não houver
                summary = analysis_result.summary if analysis_result.summary and len(analysis_result.summary) > 50 else "Resumo gerado automaticamente"
                topics = [topic.title for topic in analysis_result.main_topics] if analysis_result.main_topics else []
                
                # Salva o resumo tradicional
                await db.summary.create(
                    data={
                        "meeting_id": meeting_id,
                        "content": summary,
                        "topics": json.dumps(topics, ensure_ascii=False),
                    }
                )
                
                # Salva a análise inteligente completa
                analysis_data = {
                    "meeting_id": meeting_id,
                    "participants": json.dumps([p.dict() for p in analysis_result.participants], ensure_ascii=False),
                    "main_topics": json.dumps([t.dict() for t in analysis_result.main_topics], ensure_ascii=False),
                    "action_items": json.dumps([a.dict() for a in analysis_result.action_items], ensure_ascii=False),
                    "key_decisions": json.dumps([d.dict() for d in analysis_result.key_decisions], ensure_ascii=False),
                    "summary": analysis_result.summary,
                    "confidence_score": analysis_result.confidence_score
                }
                
                # Adiciona análise de sentimento se disponível
                if analysis_result.sentiment_analysis:
                    analysis_data["sentiment_analysis"] = json.dumps(
                        analysis_result.sentiment_analysis.dict(), ensure_ascii=False
                    )
                
                await db.meetinganalysis.create(data=analysis_data)
                
                # Atualiza status da transcrição e reunião
                await db.transcription.update_many(
                    where={"meeting_id": meeting_id},
                    data={
                        "is_summarized": True,
                        "is_analyzed": True
                    }
                )
                
                await db.meeting.update(
                    where={"id": meeting_id},
                    data={
                        "has_summary": True,
                        "has_analysis": True
                    }
                )
                
            logger.info("✅ Resultados da análise otimizada salvos com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar resultados da análise: {e}")
            raise
    
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