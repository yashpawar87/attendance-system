from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import attendance, camera, people, recognition
from app.core.config import get_settings
from app.services.camera_feed import CameraFrameStore


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title=settings.app_name)
    application.state.settings = settings
    application.state.camera_store = CameraFrameStore()
    application.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    application.include_router(people.router)
    application.include_router(recognition.router)
    application.include_router(camera.router)
    application.include_router(attendance.router)

    @application.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    return application


app = create_app()
