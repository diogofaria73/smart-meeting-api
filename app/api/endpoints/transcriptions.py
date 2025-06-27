import asyncio
from typing import Dict
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, BackgroundTasks, Query

from app.schemas.transcription import TranscriptionResponse
from app.services.transcription_service import TranscriptionService, transcription_service
from app.services.meeting_analysis_service import meeting_analysis_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    meeting_id: int = Query(..., description="ID da reunião"),
    file: UploadFile = File(..., description="Arquivo de áudio para transcrição")
):
    """
    Transcreve áudio para texto usando Whisper otimizado para português brasileiro.
    
    Funcionalidades:
    - 🎤 Transcrição automática de áudio para texto
    - 🇧🇷 Otimizado para português brasileiro
    - ⚡ Configurações adaptativas baseadas na duração do áudio
    - 📝 Pós-processamento de texto para melhor qualidade
    """
    try:
        logger.info(f"🎤 Iniciando transcrição para reunião {meeting_id}")
        logger.info(f"📁 Arquivo: {file.filename} ({file.content_type})")
        
        result = await transcription_service.transcribe_audio(meeting_id, file)
        
        logger.info(f"✅ Transcrição concluída para reunião {meeting_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro na transcrição: {str(e)}")
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
                "is_analyzed": getattr(transcription, 'is_analyzed', False)
            }
            
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