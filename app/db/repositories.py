from datetime import date, datetime

import numpy as np
from sqlalchemy import Select, delete, desc, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.db.models import Attendance, FaceEmbedding, Person


def create_person(db: Session, employee_code: str, name: str, email: str | None) -> Person:
    person = Person(employee_code=employee_code, name=name, email=email)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


def get_person(db: Session, person_id: int) -> Person | None:
    return db.get(Person, person_id)


def list_people_with_embedding_counts(db: Session) -> list[tuple[Person, int]]:
    query = (
        select(Person, func.count(FaceEmbedding.id))
        .outerjoin(FaceEmbedding, FaceEmbedding.person_id == Person.id)
        .group_by(Person.id)
        .order_by(Person.employee_code)
    )
    return [(person, int(count)) for person, count in db.execute(query)]


def set_profile_photo(db: Session, person: Person, content: bytes, content_type: str = "image/jpeg") -> Person:
    person.profile_photo = content
    person.profile_photo_content_type = content_type
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


def add_embedding(db: Session, person_id: int, embedding: list[float], model_name: str, model_version: str) -> FaceEmbedding:
    row = FaceEmbedding(person_id=person_id, embedding=embedding, model_name=model_name, model_version=model_version)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_embeddings(db: Session, person_id: int) -> int:
    result = db.execute(delete(FaceEmbedding).where(FaceEmbedding.person_id == person_id))
    db.commit()
    return int(result.rowcount or 0)


from sqlalchemy.exc import OperationalError, ProgrammingError


def find_best_matches(db: Session, embedding: list[float], limit: int = 2) -> list[tuple[int, float]]:
    """Use pgvector cosine distance in production and a deterministic Python fallback in SQLite tests."""
    try:
        vector_literal = "[" + ",".join(f"{float(v):.9g}" for v in embedding) + "]"
        rows = db.execute(
            text(
                "SELECT person_id, 1 - MIN(embedding <=> CAST(:embedding AS vector)) AS similarity "
                "FROM face_embeddings JOIN people ON people.id = face_embeddings.person_id "
                "WHERE people.active = true GROUP BY person_id ORDER BY similarity DESC LIMIT :limit"
            ),
            {"embedding": vector_literal, "limit": limit},
        ).all()
        return [(int(row[0]), float(row[1])) for row in rows]
    except (OperationalError, ProgrammingError):
        db.rollback()
        query = select(FaceEmbedding.person_id, FaceEmbedding.embedding).join(Person).where(Person.active.is_(True))
        scores: dict[int, float] = {}
        target = np.asarray(embedding, dtype=np.float32)
        target_norm = np.linalg.norm(target)
        if target_norm == 0:
            return []
        for person_id, stored in db.execute(query):
            candidate = np.asarray(stored, dtype=np.float32)
            denominator = target_norm * np.linalg.norm(candidate)
            score = float(np.dot(target, candidate) / denominator) if denominator else -1.0
            scores[int(person_id)] = max(scores.get(int(person_id), -1.0), score)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def insert_attendance(
    db: Session,
    *,
    person_id: int,
    door_id: int | None,
    event_type: str,
    event_at: datetime,
    attendance_date: date,
    similarity: float,
    liveness_score: float,
) -> tuple[Attendance, bool]:
    row = Attendance(
        person_id=person_id,
        door_id=door_id,
        event_type=event_type,
        event_at=event_at,
        attendance_date=attendance_date,
        similarity=similarity,
        liveness_score=liveness_score,
    )
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
        return row, True
    except IntegrityError:
        db.rollback()
        existing = db.scalar(
            select(Attendance).where(
                Attendance.person_id == person_id,
                Attendance.attendance_date == attendance_date,
                Attendance.event_type == event_type,
            )
        )
        if existing is None:
            raise
        return existing, False


def list_attendance(db: Session, attendance_date: date | None = None, person_id: int | None = None) -> list[Attendance]:
    query: Select[tuple[Attendance]] = select(Attendance).options(joinedload(Attendance.person)).order_by(desc(Attendance.event_at))
    if attendance_date:
        query = query.where(Attendance.attendance_date == attendance_date)
    if person_id:
        query = query.where(Attendance.person_id == person_id)
    return list(db.scalars(query).unique())


def list_absentees(db: Session, attendance_date: date) -> list[Person]:
    query = select(Person).where(
        Person.active.is_(True),
        ~Person.id.in_(select(Attendance.person_id).where(Attendance.attendance_date == attendance_date)),
    )
    return list(db.scalars(query))
