from pathlib import Path

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings
from app.db.database import get_db
from app.services.attendance_service import AttendanceService
from app.services.enrollment import EnrollmentService
from app.services.recognition_service import RecognitionService
from app.vision.matcher import PgVectorFaceMatcher


def get_settings_from_request(request: Request) -> Settings:
    return request.app.state.settings


def get_recognition_service(request: Request, db=Depends(get_db)) -> RecognitionService:
    settings = request.app.state.settings
    pipeline = getattr(request.app.state, "recognition_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Recognition models are not installed")
    state = getattr(request.app.state, "recognition_state", None)
    return RecognitionService(pipeline, lambda: PgVectorFaceMatcher(db, settings), settings, state=state)


def get_enrollment_service(request: Request) -> EnrollmentService:
    settings = request.app.state.settings
    model_dir = Path(settings.model_dir)
    detector_path = model_dir / settings.detection_model
    embedder_path = model_dir / settings.embedding_model
    if not detector_path.exists() or not embedder_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Recognition models are not installed")
    return EnrollmentService(YuNetFaceDetector(detector_path), SFaceEmbedder(embedder_path), settings)


def get_attendance_service(request: Request) -> AttendanceService:
    return AttendanceService(request.app.state.settings)
