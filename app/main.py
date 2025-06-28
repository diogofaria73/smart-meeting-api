from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.routes import api_router
from app.core.config import settings
from app.core.startup import register_startup_events

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API para transcrição e resumo automático de reuniões",
    version="0.1.0",
)

# Configuração de CORS
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173", 
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

logger.info(f"Configurando CORS com origens: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Registra os eventos de inicialização e encerramento
register_startup_events(app)

# Inclui as rotas da API
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {"message": "Smart Meeting API - Bem-vindo!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/test-cors")
async def test_cors():
    """Endpoint para testar se o CORS está funcionando"""
    return {
        "message": "CORS está funcionando!",
        "allowed_origins": settings.cors_origins,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/debug-cors")
async def debug_cors():
    """Endpoint para debug das configurações de CORS"""
    return {
        "ALLOWED_ORIGINS_raw": settings.ALLOWED_ORIGINS,
        "cors_origins_processed": settings.cors_origins,
        "cors_config": {
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }
    } 