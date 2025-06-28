import asyncio
from typing import Dict, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, BackgroundTasks, Query

from app.schemas.transcription import TranscriptionResponse, SpeakerSegment, ParticipantInfo
from app.services.transcription_service import TranscriptionService, transcription_service
from app.services.meeting_analysis_service import meeting_analysis_service
import logging

# 🎙️ NOVA FUNCIONALIDADE: Importar serviço de diarização para endpoint de teste
try:
    from app.services.enhanced_transcription_service import enhanced_transcription_service
    ENHANCED_TRANSCRIPTION_AVAILABLE = True
except ImportError:
    ENHANCED_TRANSCRIPTION_AVAILABLE = False
    enhanced_transcription_service = None

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio_async(
    meeting_id: int = Query(..., description="ID da reunião"),
    file: UploadFile = File(..., description="Arquivo de áudio para transcrição"),
    enable_diarization: bool = Query(True, description="Habilitar identificação de speakers/participantes")
):
    """
    🚀 TRANSCRIÇÃO ASSÍNCRONA - Inicia processamento em background
    
    **NOVO COMPORTAMENTO ASSÍNCRONO:**
    - ✅ Retorna `task_id` imediatamente (não bloqueia o frontend)
    - 🔄 Processamento continua em background
    - 📡 Notificações via WebSocket em tempo real
    - 📊 Status disponível via `/transcribe/status/{task_id}`
    
    **Funcionalidades:**
    - 🎤 Transcrição automática de áudio para texto
    - 🇧🇷 Otimizado para português brasileiro
    - 🎙️ Identificação automática de speakers/participantes (diarização)
    - ⚡ Configurações adaptativas baseadas na duração do áudio
    - 📝 Pós-processamento de texto para melhor qualidade
    - 📊 Estatísticas detalhadas de participação por speaker
    - 🤖 Análise inteligente automática após transcrição
    
    **Como usar:**
    1. Faça upload do áudio → receba `task_id`
    2. Conecte WebSocket em `/ws/meeting/{meeting_id}` para notificações
    3. Ou consulte status em `/transcribe/status/{task_id}`
    4. Quando concluído, busque resultado em `/transcriptions/{meeting_id}`
    """
    try:
        from app.services.background_tasks import background_task_service
        from app.services.progress_service import progress_service
        
        logger.info(f"🚀 Iniciando transcrição ASSÍNCRONA para reunião {meeting_id}")
        logger.info(f"📁 Arquivo: {file.filename} ({file.content_type})")
        
        # Validar tipo de arquivo
        allowed_types = [
            "audio/wav", "audio/mp3", "audio/mpeg", "audio/mp4", 
            "audio/m4a", "audio/flac", "audio/ogg", "audio/webm"
        ]
        
        if file.content_type and file.content_type not in allowed_types:
            error_msg = f"Tipo de arquivo não suportado: {file.content_type}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=f"{error_msg}. Tipos suportados: {', '.join(allowed_types)}"
            )
        
        # Validar se reunião existe
        from app.db.client import get_db
        async with get_db() as db:
            meeting = await db.meeting.find_unique(where={"id": meeting_id})
            if not meeting:
                raise HTTPException(
                    status_code=404,
                    detail=f"Reunião com ID {meeting_id} não encontrada"
                )
        
        # Lê o conteúdo do arquivo
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Arquivo de áudio está vazio"
            )
        
        # Cria task de progresso
        task_id = progress_service.create_task(meeting_id)
        
        logger.info(f"✅ Task criada: {task_id}")
        
        # Inicia processamento em background
        background_task_service.start_transcription_task(
            task_id=task_id,
            meeting_id=meeting_id,
            file_content=file_content,
            filename=file.filename or "audio",
            content_type=file.content_type or "audio/wav",
            enable_diarization=enable_diarization
        )
        
        logger.info(f"🚀 Processamento iniciado em background para task {task_id}")
        
        return {
            "message": "Processamento iniciado com sucesso",
            "task_id": task_id,
            "meeting_id": meeting_id,
            "status": "processing",
            "filename": file.filename,
            "enable_diarization": enable_diarization,
            "websocket_url": f"/ws/meeting/{meeting_id}",
            "status_url": f"/api/transcriptions/status/{task_id}",
            "result_url": f"/api/transcriptions/{meeting_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar transcrição assíncrona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/status/{task_id}")
async def get_transcription_status(task_id: str):
    """
    📊 Consulta status de uma tarefa de transcrição assíncrona
    
    Retorna informações detalhadas sobre o progresso, incluindo:
    - Status atual (pending, processing, completed, failed)
    - Porcentagem de progresso (0-100)
    - Etapa atual do processamento
    - Tempo estimado restante
    - Mensagens de erro (se houver)
    """
    try:
        from app.services.background_tasks import background_task_service
        
        task_status = background_task_service.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=404,
                detail=f"Tarefa {task_id} não encontrada"
            )
        
        return {
            "task_id": task_id,
            "status": task_status["status"],
            "progress": {
                "percentage": task_status["progress_percentage"],
                "current_step": task_status["current_step"],
                "message": task_status["message"],
                "details": task_status["details"],
                "estimated_remaining_seconds": task_status["estimated_remaining_seconds"]
            },
            "meeting_id": task_status["meeting_id"],
            "timestamps": {
                "started_at": task_status["started_at"],
                "updated_at": task_status["updated_at"]
            },
            "error": task_status.get("error_message"),
            "is_running": task_status["is_running"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao consultar status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.delete("/cancel/{task_id}")
async def cancel_transcription_task(task_id: str):
    """
    🛑 Cancela uma tarefa de transcrição em execução
    """
    try:
        from app.services.background_tasks import background_task_service
        
        success = background_task_service.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Tarefa {task_id} não encontrada ou já finalizada"
            )
        
        return {
            "message": f"Tarefa {task_id} cancelada com sucesso",
            "task_id": task_id,
            "cancelled": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao cancelar tarefa: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/tasks/active")
async def get_active_transcription_tasks():
    """
    📋 Lista todas as tarefas de transcrição ativas
    
    Útil para monitoramento e debug do sistema
    """
    try:
        from app.services.background_tasks import background_task_service
        
        active_tasks = background_task_service.get_active_tasks()
        
        return {
            "total_active_tasks": len(active_tasks),
            "tasks": active_tasks
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao listar tarefas ativas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/summary/{meeting_id}", response_model=TranscriptionResponse)
async def generate_summary(meeting_id: int):
    """
    Gera resumo inteligente da transcrição com análise completa.
    
    Extrai automaticamente:
    - 👥 Participantes da reunião
    - 📋 Tópicos principais discutidos  
    - 📝 Itens de ação e tarefas
    - ⚖️ Decisões importantes tomadas
    - 😊 Análise de sentimento
    - 📄 Resumo estruturado em português brasileiro
    """
    try:
        logger.info(f"📝 Gerando resumo inteligente para reunião {meeting_id}")
        
        result = await transcription_service.generate_summary(meeting_id)
        
        logger.info(f"✅ Resumo inteligente gerado para reunião {meeting_id}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ Erro de validação: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Erro ao gerar resumo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/{meeting_id}", response_model=TranscriptionResponse)
async def get_transcription(meeting_id: int):
    """
    Obtém a transcrição e resumo de uma reunião específica.
    """
    try:
        from app.db.client import get_db
        
        async with get_db() as db:
            # Busca a transcrição
            transcription = await db.transcription.find_first(
                where={"meeting_id": meeting_id}
            )
            
            if not transcription:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Transcrição não encontrada para reunião {meeting_id}"
                )
            
            # Busca o resumo se existir
            summary_record = await db.summary.find_first(
                where={"meeting_id": meeting_id}
            )
            
            # Busca a análise se existir
            analysis_record = await db.meetinganalysis.find_first(
                where={"meeting_id": meeting_id}
            )
            
            # Prepara a resposta
            response_data = {
                "id": transcription.id,
                "meeting_id": transcription.meeting_id,
                "content": transcription.content,
                "created_at": transcription.created_at,
                "updated_at": transcription.updated_at,
                "is_summarized": transcription.is_summarized,
                "is_analyzed": getattr(transcription, 'is_analyzed', False),
                # 🎙️ Dados de diarização
                "speakers_count": getattr(transcription, 'speakers_count', 0) or 0,
                "diarization_method": getattr(transcription, 'diarization_method', None),
                "speaker_segments": [],
                "participants": [],
                "processing_details": None
            }
            
            # Adiciona dados de diarização do banco se disponível
            import json
            if hasattr(transcription, 'speaker_segments') and transcription.speaker_segments:
                try:
                    response_data["speaker_segments"] = json.loads(transcription.speaker_segments)
                except:
                    logger.warning("⚠️ Erro ao carregar speaker_segments do banco")
            
            if hasattr(transcription, 'participants') and transcription.participants:
                try:
                    response_data["participants"] = json.loads(transcription.participants)
                except:
                    logger.warning("⚠️ Erro ao carregar participants do banco")
            
            if hasattr(transcription, 'processing_details') and transcription.processing_details:
                try:
                    response_data["processing_details"] = json.loads(transcription.processing_details)
                except:
                    logger.warning("⚠️ Erro ao carregar processing_details do banco")
            
            # Adiciona resumo se existir
            if summary_record:
                response_data["summary"] = summary_record.content
                if summary_record.topics:
                    import json
                    try:
                        response_data["topics"] = json.loads(summary_record.topics)
                    except:
                        response_data["topics"] = []
            
            # Adiciona análise se existir
            if analysis_record:
                import json
                try:
                    from app.schemas.transcription import MeetingAnalysisResult, ParticipantInfo, TopicInfo, ActionItem, KeyDecision, SentimentAnalysis
                    
                    # Reconstrói a análise
                    participants = [ParticipantInfo(**p) for p in json.loads(analysis_record.participants)]
                    main_topics = [TopicInfo(**t) for t in json.loads(analysis_record.main_topics)]
                    action_items = [ActionItem(**a) for a in json.loads(analysis_record.action_items)]
                    key_decisions = [KeyDecision(**d) for d in json.loads(analysis_record.key_decisions)]
                    
                    sentiment_analysis = None
                    if analysis_record.sentiment_analysis:
                        sentiment_data = json.loads(analysis_record.sentiment_analysis)
                        sentiment_analysis = SentimentAnalysis(**sentiment_data)
                    
                    analysis_result = MeetingAnalysisResult(
                        participants=participants,
                        main_topics=main_topics,
                        action_items=action_items,
                        key_decisions=key_decisions,
                        summary=analysis_record.summary,
                        sentiment_analysis=sentiment_analysis,
                        confidence_score=analysis_record.confidence_score
                    )
                    
                    response_data["analysis"] = analysis_result
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao carregar análise: {e}")
            
            return TranscriptionResponse(**response_data)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao obter transcrição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/test-diarization")
async def test_speaker_diarization(
    file: UploadFile = File(..., description="Arquivo de áudio para teste de diarização"),
    enable_diarization: bool = Query(True, description="Habilitar identificação de speakers")
):
    """
    🎙️ ENDPOINT DE TESTE - Transcrição com Identificação de Speakers
    
    Testa a nova funcionalidade de speaker diarization que:
    - 🎯 Identifica quantos speakers estão presentes no áudio
    - ⏱️ Segmenta temporalmente quem está falando quando
    - 📝 Combina transcrição do Whisper com identificação de speakers
    - 📊 Fornece estatísticas detalhadas de participação
    
    **Nota**: Esta é uma funcionalidade experimental que requer pyannote.audio
    """
    if not ENHANCED_TRANSCRIPTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Serviço de diarização não disponível. Instale: pip install pyannote.audio"
        )
    
    try:
        logger.info(f"🎙️ Testando diarização para arquivo: {file.filename}")
        
        # Carrega o áudio de forma similar ao endpoint principal
        import tempfile
        import os
        from pathlib import Path
        import numpy as np
        
        # Validar tipo de arquivo
        allowed_types = [
            "audio/wav", "audio/mp3", "audio/mpeg", "audio/mp4", 
            "audio/m4a", "audio/flac", "audio/ogg", "audio/webm"
        ]
        
        if file.content_type and file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de arquivo não suportado: {file.content_type}"
            )
        
        # Detectar extensão do arquivo
        file_extension = ""
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
        elif file.content_type:
            extension_map = {
                "audio/wav": ".wav", "audio/mp3": ".mp3", "audio/mpeg": ".mp3",
                "audio/mp4": ".m4a", "audio/m4a": ".m4a", "audio/flac": ".flac",
                "audio/ogg": ".ogg", "audio/webm": ".webm"
            }
            file_extension = extension_map.get(file.content_type, ".wav")
        
        # Salva arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Carrega áudio usando o método robusto do serviço principal
            audio_data, sample_rate = transcription_service._load_audio_robust(temp_file_path)
            
            # Verifica duração
            duration = len(audio_data) / sample_rate
            logger.info(f"📊 Áudio carregado: {duration:.2f}s, {sample_rate}Hz")
            
            if duration < 5.0:
                logger.warning(f"⚠️ Áudio muito curto ({duration:.2f}s) para diarização confiável")
            
            # Executa transcrição aprimorada
            result = await enhanced_transcription_service.transcribe_with_speakers(
                audio_data=audio_data,
                sample_rate=sample_rate,
                enable_diarization=enable_diarization
            )
            
            # Formata resposta para teste
            response = {
                "status": "success",
                "audio_info": {
                    "filename": file.filename,
                    "duration_seconds": duration,
                    "sample_rate": sample_rate,
                    "content_type": file.content_type
                },
                "transcription": {
                    "text": result["transcription"],
                    "confidence": result.get("confidence", 0.0),
                    "method": result.get("method", "unknown")
                },
                "speakers": {
                    "count": result.get("speakers_count", 0),
                    "segments": [
                        {
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "speaker_id": seg.speaker_id,
                            "text": seg.text,
                            "confidence": seg.confidence
                        }
                        for seg in result.get("speaker_segments", [])
                    ][:10]  # Limita a 10 segmentos para teste
                },
                "participants": [
                    {
                        "name": p.name,
                        "speaker_id": p.speaker_id,
                        "speaking_time": p.speaking_time,
                        "segments_count": p.segments_count,
                        "confidence": p.confidence
                    }
                    for p in result.get("participants", [])
                ],
                "processing": {
                    "total_time": result.get("total_processing_time", 0.0),
                    "details": result.get("processing_details", {})
                },
                "service_info": enhanced_transcription_service.get_service_info()
            }
            
            logger.info(f"✅ Teste de diarização concluído para {file.filename}")
            logger.info(f"   - Speakers identificados: {result.get('speakers_count', 0)}")
            logger.info(f"   - Método: {result.get('method', 'N/A')}")
            logger.info(f"   - Tempo: {result.get('total_processing_time', 0):.2f}s")
            
            return response
            
        finally:
            # Remove arquivo temporário
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro no teste de diarização: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro no teste de diarização: {str(e)}"
        ) 