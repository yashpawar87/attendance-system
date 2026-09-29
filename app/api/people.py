from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.api.dependencies import get_enrollment_service
from app.api.schemas import PersonCreate, PersonResponse
from app.core.audit import record
from app.core.security import require_api_token
from app.db import repositories
from app.db.database import get_db

router = APIRouter(prefix="/api/v1/people", tags=["people"], dependencies=[Depends(require_api_token)])


@router.get("", response_model=list[PersonResponse])
def list_people(db=Depends(get_db)):
    return [
        PersonResponse.model_validate(person).model_copy(update={"embedding_count": embedding_count})
        for person, embedding_count in repositories.list_people_with_embedding_counts(db)
    ]


@router.post("", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
def create_person(payload: PersonCreate, db=Depends(get_db)):
    try:
        person = repositories.create_person(db, payload.employee_code, payload.name, payload.email)
        record("create", "person", person.id)
        return person
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Employee code already exists") from exc


@router.post("/{person_id}/enroll", response_model=dict)
async def enroll_person(
    person_id: int,
    images: list[UploadFile] = File(...),
    db=Depends(get_db),
    service=Depends(get_enrollment_service),
):
    person = repositories.get_person(db, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    import cv2
    import numpy as np

    decoded = []
    for upload in images:
        content = await upload.read()
        if not content or len(content) > 8 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Invalid or oversized image")
        if not (content.startswith(b'\xff\xd8\xff') or content.startswith(b'\x89PNG\r\n\x1a\n')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        decoded.append(image)
    try:
        rows = service.enroll(db, person, decoded)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record("enroll", "person", person_id)
    return {"person_id": person_id, "enrolled": len(rows)}


@router.get("/{person_id}/photo")
def get_person_photo(person_id: int, db=Depends(get_db)):
    person = repositories.get_person(db, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    if not person.profile_photo:
        raise HTTPException(status_code=404, detail="Profile photo not found")
    return Response(content=person.profile_photo, media_type=person.profile_photo_content_type or "image/jpeg")


@router.post("/{person_id}/photo", response_model=PersonResponse)
async def upload_person_photo(
    person_id: int,
    image: UploadFile = File(...),
    db=Depends(get_db),
):
    person = repositories.get_person(db, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    content = await image.read()
    if not content or len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Invalid or oversized image")
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    encoded_ok, encoded = cv2.imencode(".jpg", decoded, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not encoded_ok:
        raise HTTPException(status_code=400, detail="Could not encode image")
    person = repositories.set_profile_photo(db, person, encoded.tobytes())
    record("update_profile_photo", "person", person_id)
    return person


@router.delete("/{person_id}/embeddings", response_model=dict)
def delete_person_embeddings(person_id: int, db=Depends(get_db)):
    if repositories.get_person(db, person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    deleted = repositories.delete_embeddings(db, person_id)
    record("delete_biometric_data", "person", person_id)
    return {"person_id": person_id, "deleted_embeddings": deleted}
