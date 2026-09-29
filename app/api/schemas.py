from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class PersonCreate(BaseModel):
    employee_code: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=255)
    email: str | None = Field(default=None, max_length=320)


class PersonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employee_code: str
    name: str
    email: str | None
    active: bool
    embedding_count: int = 0


class FaceDetectionResponse(BaseModel):
    box: tuple[float, float, float, float]
    matched: bool
    person_id: int | None = None
    similarity: float | None = None


class RecognitionMatchResponse(BaseModel):
    person_id: int
    similarity: float
    liveness: float


class IdentifyResponse(BaseModel):
    matched: bool
    person_id: int | None = None
    similarity: float | None = None
    liveness: float | None = None
    detections: list[FaceDetectionResponse] = Field(default_factory=list)
    matches: list[RecognitionMatchResponse] = Field(default_factory=list)


class AttendanceMarkRequest(BaseModel):
    person_id: int
    similarity: float = Field(ge=-1, le=1)
    liveness_score: float = Field(ge=0, le=1)
    door_id: int | None = None
    event_at: datetime | None = None


class AttendanceResponse(BaseModel):
    id: int
    person_id: int
    person_name: str | None = None
    door_id: int | None
    event_type: str
    event_at: datetime
    attendance_date: date
    similarity: float
    liveness_score: float
    created_at: datetime | None


class MarkResponse(BaseModel):
    marked: bool
    attendance: AttendanceResponse
