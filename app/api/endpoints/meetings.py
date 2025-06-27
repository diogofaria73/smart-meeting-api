from fastapi import APIRouter, HTTPException, status, Query
from typing import List

from app.schemas.meeting import (
    MeetingCreate, 
    MeetingResponse, 
    MeetingWithTranscriptionsResponse, 
    PaginationParams,
    PaginatedMeetingsResponse,
    DashboardStats
)
from app.services.meeting_service import meeting_service

router = APIRouter()


@router.post("/", response_model=MeetingResponse)
async def create_meeting(meeting: MeetingCreate):
    """
    Cria uma nova reunião.
    
    Campos obrigatórios:
    - title: Título da reunião
    - description: Descrição (opcional)
    - date: Data e hora da reunião
    - participants: Lista de participantes (JSON)
    """
    try:
        return await meeting_service.create_meeting(meeting)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao criar reunião: {str(e)}")


@router.get("/", response_model=PaginatedMeetingsResponse)
async def get_meetings(
    page: int = Query(1, ge=1, description="Número da página"),
    page_size: int = Query(10, ge=1, le=100, description="Itens por página")
):
    """
    Retorna reuniões paginadas incluindo suas transcrições (quando existirem).
    
    Parâmetros:
    - page: Número da página (padrão: 1)
    - page_size: Itens por página (padrão: 10, máximo: 100)
    
    Resposta inclui:
    - meetings: Lista de reuniões com transcrições
    - total: Total de reuniões
    - page: Página atual
    - page_size: Itens por página
    - total_pages: Total de páginas
    - has_next: Se há próxima página
    - has_prev: Se há página anterior
    """
    try:
        return await meeting_service.get_meetings_paginated(page=page, page_size=page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar reuniões: {str(e)}")

@router.get("/all", response_model=List[MeetingWithTranscriptionsResponse])
async def get_all_meetings():
    """
    Retorna TODAS as reuniões cadastradas incluindo suas transcrições (sem paginação).
    
    Resposta inclui:
    - Dados básicos da reunião (título, descrição, data, participantes)
    - Lista de transcrições associadas à reunião
    - Metadados (datas de criação/atualização, flags de status)
    """
    try:
        return await meeting_service.get_all_meetings_with_transcriptions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar reuniões: {str(e)}")

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    days: int = Query(30, ge=1, le=365, description="Número de dias para estatísticas diárias")
):
    """Retorna estatísticas do dashboard"""
    try:
        return await meeting_service.get_dashboard_stats(days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar estatísticas: {str(e)}")

@router.get("/{meeting_id}", response_model=MeetingWithTranscriptionsResponse)
async def get_meeting(meeting_id: int):
    """
    Retorna uma reunião específica pelo ID.
    """
    try:
        meeting = await meeting_service.get_meeting_by_id(meeting_id)
        if not meeting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reunião com ID {meeting_id} não encontrada",
            )
        return meeting
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar reunião: {str(e)}")

@router.delete("/{meeting_id}")
async def delete_meeting(meeting_id: int):
    """Exclui uma reunião"""
    try:
        success = await meeting_service.delete_meeting(meeting_id)
        if not success:
            raise HTTPException(status_code=404, detail="Reunião não encontrada")
        return {"message": "Reunião excluída com sucesso"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao excluir reunião: {str(e)}") 