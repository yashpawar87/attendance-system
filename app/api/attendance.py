import csv
import io
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_attendance_service
from app.api.schemas import AttendanceMarkRequest, AttendanceResponse, MarkResponse, PersonResponse
from app.core.audit import record
from app.core.security import require_api_token
from app.db import repositories
from app.db.database import get_db

router = APIRouter(prefix="/api/v1/attendance", tags=["attendance"], dependencies=[Depends(require_api_token)])


def serialize(row):
    return AttendanceResponse(
        id=row.id, person_id=row.person_id, person_name=row.person.name if row.person else None, door_id=row.door_id,
        event_type=row.event_type, event_at=row.event_at, attendance_date=row.attendance_date,
        similarity=row.similarity, liveness_score=row.liveness_score, created_at=row.created_at,
    )


@router.post("/mark", response_model=MarkResponse)
def mark_attendance(payload: AttendanceMarkRequest, db=Depends(get_db), service=Depends(get_attendance_service)):
    if repositories.get_person(db, payload.person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        row, marked = service.mark(db, person_id=payload.person_id, similarity=payload.similarity, liveness_score=payload.liveness_score, door_id=payload.door_id, event_at=payload.event_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if marked:
        record("mark", "attendance", row.id)
    return MarkResponse(marked=marked, attendance=serialize(row))


@router.post("/demo-mark", response_model=MarkResponse)
def mark_demo_attendance(request: Request, payload: AttendanceMarkRequest, db=Depends(get_db), service=Depends(get_attendance_service)):
    if not request.app.state.settings.demo_replay_enabled:
        raise HTTPException(status_code=403, detail="Replay demo mode is disabled")
    if repositories.get_person(db, payload.person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        row, marked = service.mark(
            db,
            person_id=payload.person_id,
            similarity=payload.similarity,
            liveness_score=payload.liveness_score,
            door_id=payload.door_id,
            event_at=payload.event_at,
            event_type="demo_replay",
            allow_replay_demo=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if marked:
        record("demo_mark", "attendance", row.id)
    return MarkResponse(marked=marked, attendance=serialize(row))


@router.get("", response_model=list[AttendanceResponse])
def get_attendance(attendance_date: date | None = Query(default=None), person_id: int | None = Query(default=None), db=Depends(get_db)):
    return [serialize(row) for row in repositories.list_attendance(db, attendance_date, person_id)]


@router.get("/today", response_model=list[AttendanceResponse])
def today(db=Depends(get_db), service=Depends(get_attendance_service)):
    from datetime import datetime, timezone
    local_date = datetime.now(timezone.utc).astimezone(service.timezone).date()
    return [serialize(row) for row in repositories.list_attendance(db, local_date)]


@router.get("/absentees", response_model=list[PersonResponse])
def absentees(db=Depends(get_db), service=Depends(get_attendance_service)):
    from datetime import datetime, timezone
    local_date = datetime.now(timezone.utc).astimezone(service.timezone).date()
    return repositories.list_absentees(db, local_date)


@router.get("/export")
def export_csv(attendance_date: date | None = Query(default=None), db=Depends(get_db)):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "person_id", "person_name", "door_id", "event_type", "event_at", "attendance_date", "similarity", "liveness_score"])
    for row in repositories.list_attendance(db, attendance_date):
        writer.writerow([row.id, row.person_id, row.person.name if row.person else "", row.door_id, row.event_type, row.event_at.isoformat(), row.attendance_date.isoformat(), row.similarity, row.liveness_score])
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=attendance.csv"})


@router.get("/{person_id}", response_model=list[AttendanceResponse])
def person_attendance(person_id: int, db=Depends(get_db)):
    return [serialize(row) for row in repositories.list_attendance(db, person_id=person_id)]
