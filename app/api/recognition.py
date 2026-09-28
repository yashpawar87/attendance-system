from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.dependencies import get_recognition_service
from app.api.schemas import IdentifyResponse
from app.core.security import require_api_token
from app.vision.overlay import draw_face_overlays

router = APIRouter(prefix="/api/v1/recognition", tags=["recognition"], dependencies=[Depends(require_api_token)])


async def _decode_upload(image: UploadFile):
    import cv2
    import numpy as np

    content = await image.read()
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


@router.post("/identify", response_model=IdentifyResponse)
async def identify(
    request: Request,
    image: UploadFile = File(...),
    source_id: str = Form(default="default"),
    service=Depends(get_recognition_service),
):
    frame = await _decode_upload(image)
    result, observations = service.identify_with_observations(frame, source_id=source_id)
    request.app.state.camera_store.record_intruders(source_id, sum(1 for observation in observations if not observation.matched))
    import cv2
    import numpy as np
    annotated = draw_face_overlays(frame, observations)
    encoded_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if encoded_ok:
        request.app.state.camera_store.publish(source_id, encoded.tobytes())
    if result is None:
        return IdentifyResponse(matched=False)
    return IdentifyResponse(matched=True, person_id=result.person_id, similarity=result.similarity, liveness=result.liveness)


@router.post("/demo-identify", response_model=IdentifyResponse)
async def demo_identify(
    request: Request,
    image: UploadFile = File(...),
    source_id: str = Form(default="demo-replay"),
    service=Depends(get_recognition_service),
):
    if not request.app.state.settings.demo_replay_enabled:
        raise HTTPException(status_code=403, detail="Replay demo mode is disabled")
    frame = await _decode_upload(image)
    result, observations = service.identify_with_observations(frame, source_id=source_id, demo_replay=True)
    request.app.state.camera_store.record_intruders(source_id, sum(1 for observation in observations if not observation.matched))
    import cv2
    annotated = draw_face_overlays(frame, observations)
    encoded_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if encoded_ok:
        request.app.state.camera_store.publish(source_id, encoded.tobytes())
    if result is None:
        return IdentifyResponse(matched=False)
    return IdentifyResponse(matched=True, person_id=result.person_id, similarity=result.similarity, liveness=result.liveness)
