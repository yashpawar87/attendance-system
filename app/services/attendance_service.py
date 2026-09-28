from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.core.config import Settings
from app.db.repositories import insert_attendance


class AttendanceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.timezone = ZoneInfo(settings.attendance_timezone)

    def mark(
        self,
        db,
        *,
        person_id: int,
        similarity: float,
        liveness_score: float,
        door_id: int | None = None,
        event_at: datetime | None = None,
        event_type: str | None = None,
        allow_replay_demo: bool = False,
    ):
        if similarity < self.settings.recognition_threshold:
            raise ValueError("Recognition score is below the configured threshold")
        if not allow_replay_demo and liveness_score < self.settings.liveness_threshold:
            raise ValueError("Liveness score is below the configured threshold")
        timestamp = event_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp_utc = timestamp.astimezone(timezone.utc)
        local_date = timestamp_utc.astimezone(self.timezone).date()
        return insert_attendance(
            db,
            person_id=person_id,
            door_id=door_id,
            event_type=event_type or self.settings.attendance_event_type,
            event_at=timestamp_utc,
            attendance_date=local_date,
            similarity=similarity,
            liveness_score=liveness_score,
        )
