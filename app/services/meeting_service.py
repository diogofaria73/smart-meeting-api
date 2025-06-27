import json
import math
from datetime import datetime, timedelta
from typing import List, Optional

from app.db.client import get_db
from app.schemas.meeting import MeetingCreate, MeetingResponse, MeetingSummary, MeetingWithTranscriptionsResponse, PaginatedMeetingsResponse, MeetingUpdate, DashboardStats, DailyStats
from app.schemas.transcription import TranscriptionResponse
from prisma import Prisma


class MeetingService:
    def __init__(self):
        pass
    
    async def create_meeting(self, meeting_data: MeetingCreate) -> MeetingResponse:
        """
        Cria uma nova reunião no banco de dados.
        """
        async with get_db() as db:
            # Converte a lista de participantes para JSON
            participants_json = json.dumps(meeting_data.participants)
            
            # Cria a reunião no banco de dados
            meeting = await db.meeting.create(
                data={
                    "title": meeting_data.title,
                    "description": meeting_data.description,
                    "date": meeting_data.date,
                    "participants": participants_json,
                }
            )
            
            # Converte o resultado de volta para o formato esperado
            return MeetingResponse(
                id=meeting.id,
                title=meeting.title,
                description=meeting.description,
                date=meeting.date,
                participants=json.loads(meeting.participants),
                audio_file=None,  # Campo do schema base
                created_at=meeting.created_at,
                updated_at=meeting.updated_at,
                has_transcription=meeting.has_transcription,
                has_summary=meeting.has_summary,
            )
    
    async def get_all_meetings(self) -> List[MeetingResponse]:
        """
        Retorna todas as reuniões cadastradas.
        """
        async with get_db() as db:
            meetings = await db.meeting.find_many(
                include={"transcriptions": True},
                order={"created_at": "desc"}
            )
            
            return [
                MeetingResponse(
                    id=meeting.id,
                    title=meeting.title,
                    description=meeting.description,
                    date=meeting.date,
                    participants=json.loads(meeting.participants),
                    audio_file=None,  # Campo do schema base
                    created_at=meeting.created_at,
                    updated_at=meeting.updated_at,
                    has_transcription=meeting.has_transcription,
                    has_summary=meeting.has_summary,
                )
                for meeting in meetings
            ]
    
    async def get_all_meetings_with_transcriptions(self) -> List[MeetingWithTranscriptionsResponse]:
        """
        Retorna todas as reuniões cadastradas incluindo suas transcrições.
        """
        async with get_db() as db:
            # Busca todas as reuniões com suas transcrições relacionadas
            meetings = await db.meeting.find_many(
                include={
                    "transcriptions": True
                }
            )
            
            result = []
            for meeting in meetings:
                # Converte as transcrições para o formato esperado
                transcriptions = []
                if meeting.transcriptions:
                    for transcription in meeting.transcriptions:
                        transcriptions.append(
                            TranscriptionResponse(
                                id=transcription.id,
                                meeting_id=transcription.meeting_id,
                                content=transcription.content,
                                created_at=transcription.created_at,
                                updated_at=transcription.updated_at,
                                is_summarized=transcription.is_summarized,
                                is_analyzed=transcription.is_analyzed,
                                summary=None,  # Pode ser expandido no futuro
                                topics=[],      # Pode ser expandido no futuro
                                analysis=None   # Pode ser expandido no futuro
                            )
                        )
                
                # Adiciona a reunião com suas transcrições
                result.append(
                    MeetingWithTranscriptionsResponse(
                        id=meeting.id,
                        title=meeting.title,
                        description=meeting.description,
                        date=meeting.date,
                        participants=json.loads(meeting.participants),
                        audio_file=None,  # Campo do schema base
                        created_at=meeting.created_at,
                        updated_at=meeting.updated_at,
                        has_transcription=meeting.has_transcription,
                        has_summary=meeting.has_summary,
                        transcriptions=transcriptions
                    )
                )
            
            return result
    
    async def get_meetings_paginated(self, page: int = 1, page_size: int = 10) -> PaginatedMeetingsResponse:
        """
        Retorna reuniões paginadas incluindo suas transcrições.
        """
        async with get_db() as db:
            # Calcula offset
            offset = (page - 1) * page_size
            
            # Busca o total de reuniões
            total = await db.meeting.count()
            
            # Busca as reuniões com paginação
            meetings = await db.meeting.find_many(
                include={
                    "transcriptions": True
                },
                skip=offset,
                take=page_size,
                order={"created_at": "desc"}  # Mais recentes primeiro
            )
            
            # Converte para o formato esperado
            meetings_data = []
            for meeting in meetings:
                # Converte as transcrições
                transcriptions = []
                if meeting.transcriptions:
                    for transcription in meeting.transcriptions:
                        transcriptions.append(
                            TranscriptionResponse(
                                id=transcription.id,
                                meeting_id=transcription.meeting_id,
                                content=transcription.content,
                                created_at=transcription.created_at,
                                updated_at=transcription.updated_at,
                                is_summarized=transcription.is_summarized,
                                is_analyzed=transcription.is_analyzed,
                                summary=None,
                                topics=[],
                                analysis=None
                            )
                        )
                
                meetings_data.append(
                    MeetingWithTranscriptionsResponse(
                        id=meeting.id,
                        title=meeting.title,
                        description=meeting.description,
                        date=meeting.date,
                        participants=json.loads(meeting.participants),
                        audio_file=None,  # Campo do schema base
                        created_at=meeting.created_at,
                        updated_at=meeting.updated_at,
                        has_transcription=meeting.has_transcription,
                        has_summary=meeting.has_summary,
                        transcriptions=transcriptions
                    )
                )
            
            # Calcula informações de paginação
            total_pages = math.ceil(total / page_size)
            has_next = page < total_pages
            has_prev = page > 1
            
            return PaginatedMeetingsResponse(
                meetings=meetings_data,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_next=has_next,
                has_prev=has_prev
            )
    
    async def get_meeting_by_id(self, meeting_id: int) -> Optional[MeetingResponse]:
        """
        Retorna uma reunião específica pelo ID.
        """
        async with get_db() as db:
            meeting = await db.meeting.find_unique(where={"id": meeting_id})
            
            if not meeting:
                return None
            
            return MeetingResponse(
                id=meeting.id,
                title=meeting.title,
                description=meeting.description,
                date=meeting.date,
                participants=json.loads(meeting.participants),
                audio_file=None,  # Campo do schema base
                created_at=meeting.created_at,
                updated_at=meeting.updated_at,
                has_transcription=meeting.has_transcription,
                has_summary=meeting.has_summary,
            )
    
    async def get_meeting_summary(self, meeting_id: int) -> Optional[MeetingSummary]:
        """
        Retorna o resumo de uma reunião específica.
        """
        async with get_db() as db:
            summary = await db.summary.find_first(where={"meeting_id": meeting_id})
            
            if not summary:
                return None
            
            return MeetingSummary(
                meeting_id=summary.meeting_id,
                summary=summary.content,
                topics=json.loads(summary.topics),
                generated_at=summary.generated_at,
            )
    
    async def delete_meeting(self, meeting_id: int) -> bool:
        """
        Exclui uma reunião pelo ID.
        """
        async with get_db() as db:
            try:
                await db.meeting.delete(where={"id": meeting_id})
                return True
            except Exception:
                return False
    
    async def get_dashboard_stats(self, days: int = 30) -> DashboardStats:
        """Busca estatísticas para o dashboard"""
        try:
            async with get_db() as db:
                # Estatísticas gerais
                total_meetings = await db.meeting.count()
                total_transcriptions = await db.transcription.count()
                
                # Transcrições por status (baseado nos campos existentes)
                # Consideramos "completed" as transcrições que foram criadas (todas)
                # e "processing" como 0 já que não temos esse status no schema atual
                completed_transcriptions = total_transcriptions
                processing_transcriptions = 0
                
                # Estatísticas diárias dos últimos X dias
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                daily_stats = []
                current_date = start_date
                
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y-%m-%d')
                    next_date = current_date + timedelta(days=1)
                    
                    # Conta reuniões do dia
                    meetings_count = await db.meeting.count(
                        where={
                            "created_at": {
                                "gte": current_date,
                                "lt": next_date
                            }
                        }
                    )
                    
                    # Conta transcrições do dia
                    transcriptions_count = await db.transcription.count(
                        where={
                            "created_at": {
                                "gte": current_date,
                                "lt": next_date
                            }
                        }
                    )
                    
                    daily_stats.append(DailyStats(
                        date=date_str,
                        meetings_count=meetings_count,
                        transcriptions_count=transcriptions_count
                    ))
                    
                    current_date = next_date
                
                return DashboardStats(
                    total_meetings=total_meetings,
                    total_transcriptions=total_transcriptions,
                    completed_transcriptions=completed_transcriptions,
                    processing_transcriptions=processing_transcriptions,
                    daily_stats=daily_stats
                )
                
        except Exception as e:
            print(f"Erro ao buscar estatísticas do dashboard: {e}")
            raise e

# Instância singleton do serviço
meeting_service = MeetingService()
 