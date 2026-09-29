import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import attendance, camera, demo, people, recognition
from app.core.config import get_settings
from app.db.database import setup_db
from app.services.camera_feed import CameraFrameStore
from app.services.recognition_service import RecognitionState
from app.vision.detector import YuNetFaceDetector
from app.vision.embedder import SFaceEmbedder
from app.vision.liveness import MiniFASNetLivenessDetector
from app.vision.pipeline import RecognitionPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_db(settings)

    model_dir = Path(settings.model_dir)
    paths = [model_dir / settings.detection_model, model_dir / settings.embedding_model, model_dir / settings.liveness_model]

    if all(path.exists() for path in paths):
        detector = YuNetFaceDetector(paths[0])
        embedder = SFaceEmbedder(paths[1])
        liveness = MiniFASNetLivenessDetector(paths[2], live_class_index=settings.liveness_live_class_index)
        app.state.recognition_pipeline = RecognitionPipeline(detector, liveness, embedder, settings)
        app.state.recognition_state = RecognitionState()
    else:
        logging.warning("Recognition models missing. Recognition will be unavailable.")
        app.state.recognition_pipeline = None
        app.state.recognition_state = None

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title=settings.app_name, lifespan=lifespan)
    application.state.settings = settings
    application.state.camera_store = CameraFrameStore()
    application.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    application.include_router(people.router)
    application.include_router(recognition.router)
    application.include_router(camera.router)
    application.include_router(attendance.router)
    application.include_router(demo.router)

    @application.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    return application


app = create_app()
