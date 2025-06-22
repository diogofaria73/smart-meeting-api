import json
from datetime import datetime
from typing import List, Optional

from app.db.client import get_db
from app.schemas.meeting import MeetingCreate, MeetingResponse, MeetingSummary


class MeetingService:
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
            meetings = await db.meeting.find_many()
            
            return [
                MeetingResponse(
                    id=meeting.id,
                    title=meeting.title,
                    description=meeting.description,
                    date=meeting.date,
                    participants=json.loads(meeting.participants),
                    created_at=meeting.created_at,
                    updated_at=meeting.updated_at,
                    has_transcription=meeting.has_transcription,
                    has_summary=meeting.has_summary,
                )
                for meeting in meetings
            ]
    
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