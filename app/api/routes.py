from fastapi import APIRouter

from app.api.endpoints import meetings, transcriptions

api_router = APIRouter()

api_router.include_router(meetings.router, prefix="/meetings", tags=["meetings"])
api_router.include_router(transcriptions.router, prefix="/transcriptions", tags=["transcriptions"]) 