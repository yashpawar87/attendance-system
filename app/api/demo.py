from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from app.core.security import require_api_token

router = APIRouter(prefix="/api/v1/demo", tags=["demo"], dependencies=[Depends(require_api_token)])
VIDEO_PATH = Path(__file__).resolve().parents[2] / "create_syn_data" / "P1E_S1_C1" / "chokepoint_P1E_S1_C1.mp4"


@router.get("/video")
def demo_video(request: Request):
    if not request.app.state.settings.demo_replay_enabled:
        raise HTTPException(status_code=403, detail="Replay demo mode is disabled")
    if not VIDEO_PATH.is_file():
        raise HTTPException(status_code=404, detail="Demo video is not installed")
    return FileResponse(VIDEO_PATH, media_type="video/mp4", filename=VIDEO_PATH.name)
