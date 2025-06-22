from fastapi import APIRouter, HTTPException, status

from app.schemas.meeting import MeetingCreate, MeetingResponse
from app.services.meeting_service import MeetingService

router = APIRouter()


@router.post("/", response_model=MeetingResponse, status_code=status.HTTP_201_CREATED)
async def create_meeting(meeting: MeetingCreate):
    """
    Cria uma nova reunião.
    
    Campos obrigatórios:
    - title: Título da reunião
    - description: Descrição (opcional)
    - date: Data e hora da reunião
    - participants: Lista de participantes (JSON)
    """
    meeting_service = MeetingService()
    return await meeting_service.create_meeting(meeting)


@router.get("/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(meeting_id: int):
    """
    Retorna uma reunião específica pelo ID.
    """
    meeting_service = MeetingService()
    meeting = await meeting_service.get_meeting_by_id(meeting_id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reunião com ID {meeting_id} não encontrada",
        )
    return meeting 