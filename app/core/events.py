"""
🔔 SISTEMA DE EVENTOS E NOTIFICAÇÕES
Gerencia WebSocket e eventos para notificações em tempo real
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Tipos de eventos do sistema"""
    TRANSCRIPTION_STARTED = "transcription_started"
    TRANSCRIPTION_PROGRESS = "transcription_progress"
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    TRANSCRIPTION_FAILED = "transcription_failed"
    
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    
    SYSTEM_NOTIFICATION = "system_notification"


class NotificationManager:
    """Gerenciador de notificações WebSocket"""
    
    def __init__(self):
        # Conexões WebSocket ativas por meeting_id
        self.active_connections: Dict[int, List[WebSocket]] = {}
        # Conexões globais (para notificações gerais)
        self.global_connections: List[WebSocket] = []
        
    async def connect_to_meeting(self, websocket: WebSocket, meeting_id: int):
        """Conecta um WebSocket a uma reunião específica"""
        await websocket.accept()
        
        if meeting_id not in self.active_connections:
            self.active_connections[meeting_id] = []
        
        self.active_connections[meeting_id].append(websocket)
        logger.info(f"🔌 Cliente conectado à reunião {meeting_id}. Total: {len(self.active_connections[meeting_id])}")
        
        # Envia confirmação de conexão
        await self._send_to_websocket(websocket, {
            "event_type": "connection_established",
            "meeting_id": meeting_id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Conectado à reunião {meeting_id}"
        })
    
    async def connect_global(self, websocket: WebSocket):
        """Conecta um WebSocket para notificações globais"""
        await websocket.accept()
        self.global_connections.append(websocket)
        logger.info(f"🌐 Cliente conectado globalmente. Total: {len(self.global_connections)}")
        
        await self._send_to_websocket(websocket, {
            "event_type": "global_connection_established",
            "timestamp": datetime.now().isoformat(),
            "message": "Conectado para notificações globais"
        })
    
    def disconnect_from_meeting(self, websocket: WebSocket, meeting_id: int):
        """Desconecta um WebSocket de uma reunião"""
        if meeting_id in self.active_connections:
            if websocket in self.active_connections[meeting_id]:
                self.active_connections[meeting_id].remove(websocket)
                logger.info(f"🔌 Cliente desconectado da reunião {meeting_id}")
                
                # Remove lista vazia
                if not self.active_connections[meeting_id]:
                    del self.active_connections[meeting_id]
    
    def disconnect_global(self, websocket: WebSocket):
        """Desconecta um WebSocket das notificações globais"""
        if websocket in self.global_connections:
            self.global_connections.remove(websocket)
            logger.info(f"🌐 Cliente desconectado globalmente")
    
    async def notify_meeting(self, meeting_id: int, event_type: EventType, data: Dict[str, Any]):
        """Envia notificação para todos os clientes de uma reunião"""
        if meeting_id not in self.active_connections:
            logger.debug(f"📭 Nenhum cliente conectado à reunião {meeting_id}")
            return
        
        message = {
            "event_type": event_type.value,
            "meeting_id": meeting_id,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        
        # Cria cópia da lista para evitar problemas de concorrência
        connections = self.active_connections[meeting_id].copy()
        
        logger.info(f"📢 Enviando {event_type.value} para {len(connections)} clientes da reunião {meeting_id}")
        
        # Envia para todos os clientes conectados
        disconnected = []
        for websocket in connections:
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.warning(f"❌ Falha ao enviar para cliente: {e}")
                disconnected.append(websocket)
        
        # Remove conexões que falharam
        for websocket in disconnected:
            self.disconnect_from_meeting(websocket, meeting_id)
    
    async def notify_global(self, event_type: EventType, data: Dict[str, Any]):
        """Envia notificação global para todos os clientes"""
        if not self.global_connections:
            logger.debug("📭 Nenhum cliente conectado globalmente")
            return
        
        message = {
            "event_type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        
        connections = self.global_connections.copy()
        logger.info(f"🌐 Enviando {event_type.value} para {len(connections)} clientes globais")
        
        disconnected = []
        for websocket in connections:
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.warning(f"❌ Falha ao enviar para cliente global: {e}")
                disconnected.append(websocket)
        
        # Remove conexões que falharam
        for websocket in disconnected:
            self.disconnect_global(websocket)
    
    async def _send_to_websocket(self, websocket: WebSocket, message: Dict[str, Any]):
        """Envia mensagem para um WebSocket específico"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"❌ Erro ao enviar WebSocket: {e}")
            raise
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das conexões"""
        meeting_stats = {
            meeting_id: len(connections) 
            for meeting_id, connections in self.active_connections.items()
        }
        
        return {
            "global_connections": len(self.global_connections),
            "meeting_connections": meeting_stats,
            "total_meeting_connections": sum(meeting_stats.values()),
            "total_connections": len(self.global_connections) + sum(meeting_stats.values())
        }


# Instância singleton do gerenciador de notificações
notification_manager = NotificationManager()


# 🚀 FUNÇÕES DE CONVENIÊNCIA PARA NOTIFICAÇÕES DE TRANSCRIÇÃO

async def notify_transcription_started(meeting_id: int, task_id: str, filename: str):
    """Notifica início de transcrição"""
    await notification_manager.notify_meeting(meeting_id, EventType.TRANSCRIPTION_STARTED, {
        "task_id": task_id,
        "filename": filename,
        "message": f"Iniciando transcrição do arquivo {filename}"
    })


async def notify_transcription_progress(meeting_id: int, task_id: str, progress_data: Dict[str, Any]):
    """Notifica progresso de transcrição"""
    await notification_manager.notify_meeting(meeting_id, EventType.TRANSCRIPTION_PROGRESS, {
        "task_id": task_id,
        "progress": progress_data
    })


async def notify_transcription_completed(meeting_id: int, task_id: str, transcription_id: int, speakers_count: int = 0):
    """Notifica conclusão de transcrição"""
    await notification_manager.notify_meeting(meeting_id, EventType.TRANSCRIPTION_COMPLETED, {
        "task_id": task_id,
        "transcription_id": transcription_id,
        "speakers_count": speakers_count,
        "message": f"Transcrição concluída! {speakers_count} speakers identificados."
    })


async def notify_transcription_failed(meeting_id: int, task_id: str, error_message: str):
    """Notifica falha de transcrição"""
    await notification_manager.notify_meeting(meeting_id, EventType.TRANSCRIPTION_FAILED, {
        "task_id": task_id,
        "error": error_message,
        "message": f"Falha na transcrição: {error_message}"
    })


async def notify_analysis_completed(meeting_id: int, analysis_data: Dict[str, Any]):
    """Notifica conclusão de análise"""
    await notification_manager.notify_meeting(meeting_id, EventType.ANALYSIS_COMPLETED, {
        "analysis": analysis_data,
        "message": "Análise inteligente concluída!"
    }) 