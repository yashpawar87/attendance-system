from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.core.security import require_api_token

router = APIRouter(prefix="/api/v1/camera", tags=["camera"], dependencies=[Depends(require_api_token)])


@router.post("/frame")
async def publish_frame(
    request: Request,
    image: UploadFile = File(...),
    source_id: str = Form(default="door-camera"),
):
    content = await image.read()
    if not content or len(content) > request.app.state.settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail="Invalid or oversized image")
    request.app.state.camera_store.publish(source_id, content)
    return {"accepted": True, "source_id": source_id}


@router.get("/feed")
def camera_feed(request: Request, source_id: str = Query(default="door-camera", min_length=1, max_length=100)):
    store = request.app.state.camera_store
    return StreamingResponse(
        store.stream(source_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.get("/status")
def camera_status(request: Request, source_id: str = Query(default="door-camera", min_length=1, max_length=100)):
    return request.app.state.camera_store.status(source_id)
