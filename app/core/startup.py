"""
🚀 EVENTOS DE INICIALIZAÇÃO DA APLICAÇÃO
Gerencia startup e shutdown da aplicação FastAPI
"""

from typing import Callable
from fastapi import FastAPI
from app.db.client import close_prisma, init_prisma


def create_start_app_handler(app: FastAPI) -> Callable:
    """
    Função para inicializar recursos na inicialização da aplicação.
    """
    async def start_app() -> None:
        # Inicializa a conexão com o banco de dados
        await init_prisma()
    
    return start_app


def create_stop_app_handler(app: FastAPI) -> Callable:
    """
    Função para liberar recursos ao encerrar a aplicação.
    """
    async def stop_app() -> None:
        # Fecha a conexão com o banco de dados
        await close_prisma()
    
    return stop_app


def register_startup_events(app: FastAPI) -> None:
    """
    Registra os eventos de inicialização e encerramento da aplicação.
    """
    app.add_event_handler("startup", create_start_app_handler(app))
    app.add_event_handler("shutdown", create_stop_app_handler(app)) 