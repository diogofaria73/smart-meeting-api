from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import settings
from app.core.events import register_events

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API para transcrição e resumo automático de reuniões",
    version="0.1.0",
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registra os eventos de inicialização e encerramento
register_events(app)

# Inclui as rotas da API
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {"message": "Smart Meeting API - Bem-vindo!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"} 