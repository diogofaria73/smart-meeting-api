from fastapi import APIRouter

from app.api.endpoints import meetings, transcriptions

api_router = APIRouter()

api_router.include_router(meetings.router, prefix="/meetings", tags=["meetings"])
api_router.include_router(transcriptions.router, prefix="/transcriptions", tags=["transcriptions"])

# 🚀 NOVO: WebSocket endpoints para notificações em tempo real
from fastapi import WebSocket, WebSocketDisconnect
from app.core.events import notification_manager
import logging

logger = logging.getLogger(__name__)

@api_router.websocket("/ws/meeting/{meeting_id}")
async def websocket_meeting_notifications(websocket: WebSocket, meeting_id: int):
    """
    📡 WebSocket para notificações de uma reunião específica
    
    Conecte-se aqui para receber notificações em tempo real sobre:
    - Início de transcrição
    - Progresso de processamento  
    - Conclusão de transcrição
    - Falhas de processamento
    - Conclusão de análise inteligente
    
    Exemplo de uso no JavaScript:
    ```javascript
    const ws = new WebSocket(`ws://localhost:8000/api/ws/meeting/${meetingId}`);
    ws.onmessage = (event) => {
        const notification = JSON.parse(event.data);
        console.log('Notificação:', notification);
    };
    ```
    """
    try:
        await notification_manager.connect_to_meeting(websocket, meeting_id)
        
        # Mantém conexão viva
        while True:
            # Aguarda mensagem do cliente (ou timeout)
            try:
                data = await websocket.receive_text()
                # Cliente pode enviar ping para manter conexão
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            
    except WebSocketDisconnect:
        logger.info(f"🔌 Cliente desconectado da reunião {meeting_id}")
    except Exception as e:
        logger.error(f"❌ Erro no WebSocket da reunião {meeting_id}: {e}")
    finally:
        notification_manager.disconnect_from_meeting(websocket, meeting_id)


@api_router.websocket("/ws/global")
async def websocket_global_notifications(websocket: WebSocket):
    """
    🌐 WebSocket para notificações globais do sistema
    
    Conecte-se aqui para receber notificações gerais do sistema
    """
    try:
        await notification_manager.connect_global(websocket)
        
        # Mantém conexão viva
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info("🌐 Cliente desconectado das notificações globais")
    except Exception as e:
        logger.error(f"❌ Erro no WebSocket global: {e}")
    finally:
        notification_manager.disconnect_global(websocket)


@api_router.get("/ws/stats")
async def get_websocket_stats():
    """
    📊 Estatísticas das conexões WebSocket ativas
    """
    try:
        stats = notification_manager.get_connection_stats()
        return {
            "websocket_connections": stats,
            "status": "active"
        }
    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas WebSocket: {e}")
        return {
            "error": str(e),
            "status": "error"
        } 