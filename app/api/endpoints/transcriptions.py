import asyncio
from typing import Dict, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, BackgroundTasks, Query

from app.schemas.transcription import TranscriptionResponse, SpeakerSegment, ParticipantInfo
from app.services.transcription_service import TranscriptionService, transcription_service
from app.services.meeting_analysis_service import meeting_analysis_service
import logging
import json

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


@router.post("/summary/{meeting_id}/async", response_model=dict)
async def generate_summary_async(meeting_id: int):
    """
    🚀 Gera resumo inteligente ASSÍNCRONO com progresso em tempo real via WebSocket.
    
    Funcionalidades otimizadas:
    - ⚡ Processamento em background (não bloqueia)
    - 📊 Progresso detalhado via WebSocket  
    - 🧠 Pipeline de análise avançado
    - ⚡ Cache inteligente de resultados
    - 🔄 Chunking otimizado para textos longos
    - 🎯 Análise semântica aprimorada
    
    Retorna task_id para acompanhar progresso via WebSocket /ws/meeting/{meeting_id}
    """
    try:
        logger.info(f"🚀 Iniciando sumarização ASSÍNCRONA otimizada para reunião {meeting_id}")
        
        # Valida se existe transcrição
        from app.db.client import get_db
        async with get_db() as db:
            transcription = await db.transcription.find_first(
                where={"meeting_id": meeting_id}
            )
            
            if not transcription:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Transcrição não encontrada para reunião {meeting_id}"
                )
            
            # Verifica se já foi analisada recentemente
            existing_analysis = await db.meetinganalysis.find_first(
                where={"meeting_id": meeting_id},
                order={"generated_at": "desc"}
            )
            
            if existing_analysis:
                # Verifica se análise é recente (menos de 1 hora)
                from datetime import datetime, timedelta
                if existing_analysis.generated_at > datetime.utcnow() - timedelta(hours=1):
                    logger.info("✅ Análise recente encontrada, retornando resultado existente")
                    return {
                        "message": "Análise já disponível (recente)",
                        "meeting_id": meeting_id,
                        "status": "completed",
                        "analysis_id": existing_analysis.id,
                        "websocket_url": f"/ws/meeting/{meeting_id}",
                        "result_url": f"/api/transcriptions/{meeting_id}"
                    }
        
        # Cria task de progresso para análise
        from app.services.progress_service import progress_service
        task_id = progress_service.create_task(
            meeting_id, 
            initial_step='analysis_queue',
            initial_message='Análise adicionada à fila de processamento...'
        )
        
        logger.info(f"✅ Task de análise criada: {task_id}")
        
        # Inicia processamento em background
        from app.services.background_tasks import background_task_service
        background_task_service.start_enhanced_analysis_task(
            task_id=task_id,
            meeting_id=meeting_id,
            transcription_text=transcription.content
        )
        
        logger.info(f"🚀 Análise assíncrona iniciada em background para task {task_id}")
        
        return {
            "message": "Análise inteligente iniciada com sucesso",
            "task_id": task_id,
            "meeting_id": meeting_id,
            "status": "processing",
            "estimated_duration": "30-60 segundos",
            "websocket_url": f"/ws/meeting/{meeting_id}",
            "status_url": f"/api/transcriptions/status/{task_id}",
            "result_url": f"/api/transcriptions/{meeting_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar análise assíncrona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/summary/{meeting_id}", response_model=TranscriptionResponse)
async def generate_summary(meeting_id: int):
    """
    Gera resumo inteligente da transcrição com análise completa.
    
    ⚠️ OTIMIZADO: Agora usa enhanced_summary_service para melhor performance
    
    Extrai automaticamente:
    - 👥 Participantes da reunião
    - 📋 Tópicos principais discutidos  
    - 📝 Itens de ação e tarefas
    - ⚖️ Decisões importantes tomadas
    - 😊 Análise de sentimento
    - 📄 Resumo estruturado em português brasileiro
    """
    try:
        logger.info(f"📝 Gerando resumo inteligente OTIMIZADO para reunião {meeting_id}")
        
        # Usa o serviço otimizado
        from app.services.enhanced_summary_service import enhanced_summary_service
        
        # Busca a transcrição
        from app.db.client import get_db
        async with get_db() as db:
            transcription = await db.transcription.find_first(
                where={"meeting_id": meeting_id}
            )
            
            if not transcription:
                error_msg = f"Transcrição para reunião com ID {meeting_id} não encontrada"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Verifica se já foi analisada
            if transcription.is_summarized and transcription.is_analyzed:
                logger.info("Transcrição já possui análise completa, retornando dados existentes")
                existing_analysis = await transcription_service._get_existing_analysis(meeting_id)
                existing_summary = await transcription_service._get_existing_summary(meeting_id)
                existing_topics = await transcription_service._get_existing_topics(meeting_id)
                
                return TranscriptionResponse(
                    id=transcription.id,
                    meeting_id=transcription.meeting_id,
                    content=transcription.content,
                    created_at=transcription.created_at,
                    updated_at=transcription.updated_at,
                    is_summarized=transcription.is_summarized,
                    is_analyzed=transcription.is_analyzed,
                    summary=existing_summary,
                    topics=existing_topics,
                    analysis=existing_analysis
                )
        
        logger.info(f"📄 Transcrição encontrada: {len(transcription.content)} caracteres")
        
        # 🚀 USA O SERVIÇO OTIMIZADO
        logger.info("🤖 Iniciando análise com serviço otimizado")
        analysis_result = await enhanced_summary_service.analyze_meeting_async(
            meeting_id=meeting_id,
            transcription_text=transcription.content,
            custom_config={
                'cache_enabled': True,
                'parallel_processing': True,
                'min_confidence': 0.6
            }
        )
        logger.info(f"✅ Análise otimizada concluída em {analysis_result.processing_time:.2f}s")
        
        # Usa o resumo da análise inteligente
        summary = analysis_result.summary if analysis_result.summary and len(analysis_result.summary) > 50 else "Resumo não disponível"
        topics = [topic.title for topic in analysis_result.main_topics] if analysis_result.main_topics else []
        
        logger.info(f"📊 RESULTADOS DA ANÁLISE OTIMIZADA:")
        logger.info(f"   • Resumo: {len(summary)} caracteres")
        logger.info(f"   • Participantes: {len(analysis_result.participants)}")
        logger.info(f"   • Tópicos: {len(analysis_result.main_topics)}")
        logger.info(f"   • Itens de ação: {len(analysis_result.action_items)}")
        logger.info(f"   • Decisões: {len(analysis_result.key_decisions)}")
        logger.info(f"   • Confiança: {analysis_result.confidence_score:.2f}")
        logger.info(f"   • Tempo de processamento: {analysis_result.processing_time:.2f}s")
        
        # Salva os resultados no banco
        async with get_db() as db:
            # Salva o resumo tradicional
            await db.summary.create(
                data={
                    "meeting_id": meeting_id,
                    "content": summary,
                    "topics": json.dumps(topics, ensure_ascii=False),
                }
            )
            
            # 🆕 Salva a análise inteligente completa
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
            
            # Atualiza o status da transcrição e da reunião
            await db.transcription.update(
                where={"id": transcription.id},
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
        
        logger.info("✅ Resumo e análise otimizada salvos com sucesso")
        
        # Retorna a transcrição atualizada com análise completa
        async with get_db() as db:
            updated_transcription = await db.transcription.find_unique(
                where={"id": transcription.id}
            )
        
        return TranscriptionResponse(
            id=updated_transcription.id,
            meeting_id=updated_transcription.meeting_id,
            content=updated_transcription.content,
            created_at=updated_transcription.created_at,
            updated_at=updated_transcription.updated_at,
            is_summarized=updated_transcription.is_summarized,
            is_analyzed=updated_transcription.is_analyzed,
            summary=summary,
            topics=topics,
            analysis=analysis_result
        )
        
    except ValueError as e:
        logger.error(f"❌ Erro de validação: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Erro ao gerar resumo otimizado: {str(e)}")
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