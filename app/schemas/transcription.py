from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    ALTA = "alta"
    MEDIA = "média"
    BAIXA = "baixa"


class ActionItemStatus(str, Enum):
    PENDENTE = "pendente"
    EM_ANDAMENTO = "em_andamento"
    CONCLUIDA = "concluida"


class ActionItem(BaseModel):
    """Item de ação/tarefa extraído da reunião"""
    task: str = Field(..., description="Descrição da tarefa")
    assignee: Optional[str] = Field(None, description="Responsável pela tarefa")
    due_date: Optional[str] = Field(None, description="Prazo mencionado")
    priority: Priority = Field(Priority.MEDIA, description="Prioridade da tarefa")
    confidence: float = Field(0.8, description="Confiança na extração (0-1)")


class SpeakerSegment(BaseModel):
    """Segmento de fala com identificação de speaker"""
    start_time: float = Field(..., description="Tempo de início em segundos")
    end_time: float = Field(..., description="Tempo de fim em segundos")
    speaker_id: str = Field(..., description="ID do speaker (e.g., 'SPEAKER_00')")
    text: str = Field(..., description="Texto transcrito do segmento")
    confidence: float = Field(0.8, description="Confiança na identificação do speaker (0-1)")


class ParticipantInfo(BaseModel):
    """Informações de um participante"""
    name: str = Field(..., description="Nome do participante")
    speaker_id: Optional[str] = Field(None, description="ID do speaker identificado (e.g., 'SPEAKER_00')")
    mentions: int = Field(0, description="Número de vezes mencionado")
    speaking_time: float = Field(0.0, description="Tempo total de fala em segundos")
    segments_count: int = Field(0, description="Número de segmentos de fala")
    role: Optional[str] = Field(None, description="Função/cargo identificado")
    confidence: float = Field(0.8, description="Confiança na identificação (0-1)")


class TopicInfo(BaseModel):
    """Informações de um tópico principal"""
    title: str = Field(..., description="Título do tópico")
    summary: str = Field(..., description="Resumo do tópico")
    keywords: List[str] = Field(default_factory=list, description="Palavras-chave relacionadas")
    importance: float = Field(0.5, description="Importância do tópico (0-1)")
    duration_mentioned: Optional[str] = Field(None, description="Tempo aproximado discutido")


class KeyDecision(BaseModel):
    """Decisão importante tomada na reunião"""
    decision: str = Field(..., description="Descrição da decisão")
    context: str = Field(..., description="Contexto da decisão")
    impact: str = Field("média", description="Impacto estimado: alta, média, baixa")
    confidence: float = Field(0.8, description="Confiança na extração (0-1)")


class SentimentAnalysis(BaseModel):
    """Análise de sentimento da reunião"""
    overall: str = Field("neutro", description="Sentimento geral: positivo, neutro, negativo")
    topics: Dict[str, str] = Field(default_factory=dict, description="Sentimento por tópico")
    confidence: float = Field(0.7, description="Confiança na análise (0-1)")


class MeetingAnalysisResult(BaseModel):
    """Resultado completo da análise inteligente da reunião"""
    participants: List[ParticipantInfo] = Field(default_factory=list, description="Participantes identificados")
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list, description="Segmentos de fala por speaker")
    speakers_count: int = Field(0, description="Número total de speakers identificados")
    main_topics: List[TopicInfo] = Field(default_factory=list, description="Tópicos principais")
    action_items: List[ActionItem] = Field(default_factory=list, description="Tarefas e ações identificadas")
    key_decisions: List[KeyDecision] = Field(default_factory=list, description="Decisões importantes")
    summary: str = Field(..., description="Resumo geral estruturado")
    sentiment_analysis: Optional[SentimentAnalysis] = Field(None, description="Análise de sentimento")
    confidence_score: float = Field(0.8, description="Confiança geral da análise (0-1)")
    processing_time: Optional[float] = Field(None, description="Tempo de processamento em segundos")


class TranscriptionBase(BaseModel):
    meeting_id: int = Field(..., description="ID da reunião associada")
    content: str = Field(..., description="Conteúdo da transcrição")


class TranscriptionCreate(TranscriptionBase):
    pass


class TranscriptionResponse(TranscriptionBase):
    id: int = Field(..., description="ID da transcrição")
    created_at: datetime = Field(..., description="Data de criação da transcrição")
    updated_at: datetime = Field(..., description="Data da última atualização da transcrição")
    is_summarized: bool = Field(False, description="Indica se a transcrição já foi resumida")
    is_analyzed: bool = Field(False, description="Indica se a transcrição já foi analisada")
    summary: Optional[str] = Field(None, description="Resumo gerado a partir da transcrição")
    topics: List[str] = Field(default_factory=list, description="Tópicos principais identificados")
    analysis: Optional[MeetingAnalysisResult] = Field(None, description="Análise inteligente completa")
    
    # 🎙️ NOVOS CAMPOS: Informações de diarização
    speakers_count: int = Field(0, description="Número total de speakers identificados")
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list, description="Segmentos de fala por speaker")
    participants: List[ParticipantInfo] = Field(default_factory=list, description="Lista de participantes identificados")
    diarization_method: Optional[str] = Field(None, description="Método usado para diarização")
    processing_details: Optional[Dict[str, Any]] = Field(None, description="Detalhes do processamento")

    class Config:
        from_attributes = True


class MeetingAnalysisResponse(BaseModel):
    """Resposta da API para análise de reunião"""
    id: int = Field(..., description="ID da análise")
    meeting_id: int = Field(..., description="ID da reunião")
    analysis: MeetingAnalysisResult = Field(..., description="Resultado da análise")
    generated_at: datetime = Field(..., description="Data de geração da análise")
    updated_at: datetime = Field(..., description="Data da última atualização")

    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    """Requisição para análise de reunião"""
    meeting_id: int = Field(..., description="ID da reunião a ser analisada")
    include_sentiment: bool = Field(True, description="Incluir análise de sentimento")
    extract_participants: bool = Field(True, description="Extrair participantes")
    extract_action_items: bool = Field(True, description="Extrair itens de ação")
    min_confidence: float = Field(0.6, description="Confiança mínima para extração (0-1)") 